"""MindSpeed FSDP+EP distributed backend for LlamaFactory v1.

DeviceMesh 权责：
  - init_process_group：由 LlamaFactory DistributedInterface 负责（run_sft 入口）
  - ep / efsdp / fsdp mesh：由 MindSpeedLite 内部 ParallelState 负责（shard_model 内触发）

Model loading 权责：
  - init 模式（meta/rank0/default）：ModelEngine._init_model() 决定
  - 权重广播/加载：MindSpeedEngine.shard_model() 按 init_mode 处理

dist_config 示例：
{
    "name": "mindspeed",
    "fully_shard_parallel_size": 4,      # 非 MoE 层 FSDP
    "expert_parallel_size": 4,           # MoE 专家层 EP
    "expert_fully_shard_parallel_size": 2,  # per-expert 权重 FSDP
    "ep_apply_modules": ["*mlp.experts"],
    "fsdp_apply_modules": {"model.layers.*": {"reshard_after_forward": true}},
    "fsdp_ignored_modules": ["*mlp.experts*"],
    "dispatcher": "fused",
    "dcp_path": null
}
"""

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

from ....utils.logging import get_logger
from ....utils.types import HFModel, Processor
from .fsdp2 import FSDP2Engine, save_model as fsdp2_save_model

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _build_mindspeed_config(dist_config: dict):
    """将 LlamaFactory dist_config 字典转换为 MindSpeedLiteConfig。"""
    from mindspeed.lite.mindspeed_lite_config import (
        MindSpeedLiteConfig, FSDPPlanConfig, EPPlanConfig,
    )
    dc = dist_config
    return MindSpeedLiteConfig(
        fully_shard_parallel_size=dc.get("fully_shard_parallel_size", 1),
        # SP / TP 均走 LlamaFactory 原有机制，这里保持 1
        tensor_parallel_size=1,
        context_parallel_size=1,
        ulysses_parallel_size=1,
        # EP
        expert_parallel_size=dc.get("expert_parallel_size", 1),
        expert_fully_shard_parallel_size=dc.get("expert_fully_shard_parallel_size", 1),
        expert_data_parallel_size=dc.get("expert_data_parallel_size", 1),
        # Recompute
        recompute=dc.get("recompute", False),
        recompute_plan=dc.get("recompute_plan", None),
        # Plan configs
        fsdp_plan=FSDPPlanConfig(
            ignored_modules=dc.get("fsdp_ignored_modules", []),
            apply_modules=dc.get("fsdp_apply_modules", {"model.layers.*": {}}),
        ),
        ep_plan=EPPlanConfig(
            apply_modules=dc.get("ep_apply_modules", []),
            dispatcher=dc.get("dispatcher", "fused"),
        ),
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MindSpeedEngine(FSDP2Engine):
    """MindSpeed FSDP+EP backend.

    继承 FSDP2Engine，复用：
      - materialize_and_load()
      - _load_weights_from_hf_checkpoint()
      - _load_from_dcp()
      - _copy_weights()                    ← DTensor shard-aware
      - _save/_restore_non_persistent_buffers()
      - _resolve_hf_checkpoint_dir()

    Override：
      - prepare_model()   ← 替换为 MindSpeedLite
      - shard_model()     ← 解包路径改为 wrapper.model
    """

    # ------------------------------------------------------------------
    # Override: prepare_model
    # ------------------------------------------------------------------

    def prepare_model(self, model: HFModel) -> HFModel:
        """用 MindSpeedLite 替代 FSDP2Engine 的 fully_shard 逻辑。

        MindSpeedLite 内部会触发 ParallelState（建 ep/efsdp/fsdp mesh）
        并依次执行 EP 结构改造 + eFSDP wrap + FSDP wrap。
        """
        from mindspeed.lite import MindSpeedLite

        if self.rank == 0:
            logger.info("[MindSpeed] Applying MindSpeedLite (EP + eFSDP + FSDP)...")

        config = _build_mindspeed_config(self.dist_config)
        wrapper = MindSpeedLite(config, model)  # 就地改造，返回透传 wrapper

        if self.rank == 0:
            logger.info("[MindSpeed] MindSpeedLite wrap complete.")
        return wrapper

    # ------------------------------------------------------------------
    # Override: shard_model
    # ------------------------------------------------------------------

    def shard_model(self, model: HFModel) -> HFModel:
        """按 model._init_mode 分发，结构与 FSDP2Engine.shard_model 对称。

        关键差异：prepare_model 返回的是 MindSpeedLite wrapper，
        权重操作（broadcast / load）需作用在 wrapper.model（内层 HF 模型）。
        """
        init_mode = getattr(model, "_init_mode", "init_on_default")

        if init_mode == "init_on_default":
            # 模型权重已在 NPU 上 → 直接 wrap
            return self._shard_default(model)

        elif init_mode == "init_on_rank0":
            # rank0 CPU 有完整权重，其余 meta → wrap 后广播
            return self._shard_rank0(model)

        elif init_mode == "init_on_meta":
            # 全卡 meta → wrap 后从 checkpoint 加载
            return self._shard_meta(model)

        else:
            raise ValueError(f"[MindSpeed] Unknown init_mode: {init_mode}")

    # ------------------------------------------------------------------
    # Private: three init_mode branches
    # ------------------------------------------------------------------

    def _shard_default(self, model: HFModel) -> HFModel:
        """权重已在卡上，直接 EP/FSDP wrap。"""
        return self.prepare_model(model)

    def _shard_rank0(self, model: HFModel) -> HFModel:
        """rank0 持有完整权重 → wrap → broadcast 到各 EP rank 分片。"""
        # 1. 保存 rank0 权重 & non-persistent buffers（复用父类）
        if self.rank == 0:
            full_sd = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            full_sd = {}
        saved_bufs = self._save_non_persistent_buffers(model) if self.rank == 0 else {}

        # 2. 结构改造（meta tensor 下安全）
        wrapper = self.prepare_model(model)

        # 3. 物化 meta → device
        device = _current_device()
        wrapper.model.to_empty(device=device)

        # 4. 广播权重（复用 FSDP2Engine 的 StateDictOptions 路径）
        options = StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True)
        set_model_state_dict(wrapper.model, full_sd, options=options)

        # 5. 广播 & 恢复 non-persistent buffers（复用父类）
        bufs = [saved_bufs]
        dist.broadcast_object_list(bufs, src=0)
        self._restore_non_persistent_buffers(wrapper.model, bufs[0])

        if self.rank == 0:
            logger.info("[MindSpeed] rank0 broadcast complete.")
        return wrapper

    def _shard_meta(self, model: HFModel) -> HFModel:
        """全卡 meta → wrap → 从 HF checkpoint / DCP 加载权重。"""
        # 1. 保存 non-persistent buffers（复用父类）
        saved_bufs = self._save_non_persistent_buffers(model)

        # 2. 结构改造（meta tensor 下安全）
        wrapper = self.prepare_model(model)

        # 3. 物化 + 加载权重（复用父类 materialize_and_load，作用于内层模型）
        wrapper.model = self.materialize_and_load(
            wrapper.model,
            hf_model_path=model.config.name_or_path,
            dcp_path=self.dist_config.get("dcp_path"),
        )

        # 4. 恢复 non-persistent buffers（复用父类）
        self._restore_non_persistent_buffers(wrapper.model, saved_bufs)

        if self.rank == 0:
            logger.info("[MindSpeed] meta load complete.")
        return wrapper


# ---------------------------------------------------------------------------
# save_model：解包 wrapper.model 后复用 fsdp2 的 save 逻辑
# ---------------------------------------------------------------------------

def save_model(model: HFModel, output_dir: str, processor: Processor) -> None:
    """保存 MindSpeedLite 包装后的模型。
    
    MindSpeedLite 是透传 wrapper（model.model = 真正的 HF 模型），
    解包后走 fsdp2 相同的 full state dict 保存路径。
    """
    inner = model.model if hasattr(model, "model") else model
    fsdp2_save_model(inner, output_dir, processor)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _current_device() -> torch.device:
    """获取当前加速器 device（兼容 NPU / CUDA）。"""
    try:
        import torch_npu  # noqa: F401
        return torch.device(f"npu:{torch.npu.current_device()}")
    except ImportError:
        return torch.device(f"cuda:{torch.cuda.current_device()}")
