# MindSpeed FSDP+EP Backend for LlamaFactory v1

## Overview

This document describes the integration of [MindSpeed](https://github.com/Huawei-Ascend/MindSpeed) expert parallelism (EP) into LlamaFactory v1's distributed training framework.

**Scope**: EP (All-to-All) + eFSDP (per-expert weight sharding) for MoE expert layers. Non-MoE layers continue to use existing FSDP2 logic.

---

## Architecture

### DeviceMesh Ownership

| Mesh | Owner | Created When |
|---|---|---|
| `dp`, `cp` | `DistributedInterface` (LlamaFactory) | `run_sft()` entry |
| `mp_shard`, `mp_replicate` | `DistributedInterface` (LlamaFactory) | `run_sft()` entry — **unused** with mindspeed backend |
| `fsdp` | `ParallelState` (MindSpeed) | Inside `shard_model()` |
| `ep`, `efsdp`, `edp` | `ParallelState` (MindSpeed) | Inside `shard_model()` |

`init_process_group` is called **once** by `DistributedInterface`. `ParallelState` detects `is_initialized()` and skips re-initialization.

### SP + EP Co-existence

Sequence parallelism (SP) continues to use LlamaFactory's existing `cp_size` path (`SequenceParallelModelPlugin`). MindSpeed's `ulysses_parallel_size` is hardcoded to `1` to avoid double-patching.

```
LlamaFactory                      MindSpeed (mindspeed backend)
─────────────────────────         ─────────────────────────────
data preprocessing (cp_size)      EP all2all dispatch
SequenceParallelModelPlugin       eFSDP per-expert sharding
SequenceParallelLossPlugin        Gradient divide hook (1/ep_size)
DTensor-aware grad clip
```

### Model Loading

`ModelEngine._init_model()` decides the init mode. `MindSpeedEngine.shard_model()` handles weight loading per mode, mirroring `FSDP2Engine.shard_model()`:

| `init_mode` | Model state on entry | Loading strategy |
|---|---|---|
| `init_on_default` | Full weights on NPU | Direct wrap |
| `init_on_rank0` | rank0=CPU, others=meta | Wrap → broadcast via `set_model_state_dict` |
| `init_on_meta` | All meta | Wrap → `materialize_and_load` (HF ckpt / DCP) |

---

## Changed Files

### [NEW] `src/llamafactory/v1/plugins/trainer_plugins/distributed/mindspeed.py`

Inherits `FSDP2Engine`, reusing weight-loading utilities. Only two methods are overridden:

- `prepare_model()` — replaces `fully_shard` with `MindSpeedLite(config, model)`
- `shard_model()` — same three-branch structure as FSDP2, but operates on `wrapper.model`

### [MODIFY] `src/llamafactory/v1/plugins/trainer_plugins/distributed/hub.py`

Append at the end of file:

```python
@DistributedPlugin("mindspeed").register()
def shard_model_mindspeed(model: HFModel, dist_config: PluginConfig, **kwargs) -> HFModel:
    from .mindspeed import MindSpeedEngine
    return MindSpeedEngine(dist_config).shard_model(model)


@DistributedPlugin("mindspeed").register("save_model")
def save_model_mindspeed(model: HFModel, output_dir: str, processor: Processor) -> None:
    from .mindspeed import save_model
    return save_model(model, output_dir, processor)
```

### [MODIFY] `src/llamafactory/v1/core/base_trainer.py`

Line 316 — add `"mindspeed"` to the dist backend check:

```python
# Before
if self.args.dist_config is not None and self.args.dist_config.name in ("deepspeed", "fsdp2"):
# After
if self.args.dist_config is not None and self.args.dist_config.name in ("deepspeed", "fsdp2", "mindspeed"):
```

---

## Usage

### dist_config (JSON)

```json
{
    "name": "mindspeed",
    "fully_shard_parallel_size": 4,
    "expert_parallel_size": 4,
    "expert_fully_shard_parallel_size": 2,
    "ep_apply_modules": ["*mlp.experts"],
    "fsdp_apply_modules": {"model.layers.*": {"reshard_after_forward": true}},
    "fsdp_ignored_modules": ["*mlp.experts*"],
    "dispatcher": "fused",
    "dcp_path": null
}
```

**World size constraint**: `dp × fsdp = world_size` and `edp × efsdp × ep = world_size` must both hold.

### Training Command

```bash
torchrun --nproc_per_node=8 \
    -m llamafactory.v1.trainers.sft_trainer \
    --model /path/to/qwen3-moe \
    --train_dataset data/sft_demo.yaml \
    --dist_config '{"name":"mindspeed","expert_parallel_size":4,...}'
```

---

## Known Issues / TODO

| Item | Priority |
|---|---|
| `clip_grad_norm_` on eFSDP DTensor params — use DTensor-aware path unconditionally | Medium |
| `init_on_rank0` / `init_on_meta` e2e validation | Medium |
| `_current_device()` NPU/CUDA branch in `mindspeed.py` | Low |
