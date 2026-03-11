# LlamaFactory v1：从 torchtitan 借鉴 FSDP2 + EP 的思路分析

## 背景与目标

**Expert Parallelism (EP)** 是专为 MoE（Mixture-of-Experts）模型设计的并行策略：不同 GPU 持有不同的专家权重，token 通过 All-to-All 通信路由到拥有对应专家的 GPU，然后在本地计算，最后再把结果 All-to-All 汇合。

核心挑战：和 FSDP2 组合时，**专家参数（experts）** 和 **非专家参数（dense层）** 所在的 device mesh 大小不一样，需要分别处理。

---

## torchtitan 的设计架构（参考基准）

### 1. 双 Mesh 设计：`dense_mesh` vs `sparse_mesh`

torchtitan 的 [ParallelDims](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/distributed/parallel_dims.py#18-374) 构建了两套 mesh：

```
dense_mesh:  [pp, dp_replicate, fsdp, tp]         <-- 普通 Dense 层使用
sparse_mesh: [pp, dp_replicate, efsdp, ep, etp]   <-- MoE 专家层使用
```

关键公式：
```python
fsdp = dp_shard * cp
efsdp = fsdp * tp // (etp * ep)   # 专家层的 FSDP shard 维度变小了
```

**直觉**：EP 把专家从 [ep](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/distributed/parallel_dims.py#346-349) 个 GPU 角度分配，所以每个 EP rank 上已经只有 `1/ep` 的专家了；FSDP 再在 `efsdp` 个 GPU 上对这些专家做参数 sharding。两者相乘等于原来 dense 的 FSDP 总 GPU 数，梯度归约系数保持一致。

### 2. [ExpertParallel](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/distributed/expert_parallel.py#74-178) 类：`distribute_module` 三件套

[expert_parallel.py](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/distributed/expert_parallel.py) 中的 [ExpertParallel](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/distributed/expert_parallel.py#74-178) 通过 `distribute_module` 注入三个函数：

| 函数 | 作用 |
|------|------|
| [_partition_fn](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/distributed/expert_parallel.py#326-334) | 把专家权重在 expert 维度（dim=0）用 `Shard(0)` 做 DTensor 切分 |
| [_token_dispatch](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/distributed/expert_parallel.py#303-325) (`input_fn`) | 前向：All-to-All 分发 token 到对应专家所在的 GPU |
| [_token_combine](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/distributed/expert_parallel.py#39-44) ([output_fn](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/distributed/expert_parallel.py#263-275)) | 前向：All-to-All 收集各 GPU 计算结果回来 |

```python
def _partition_fn(self, name, mod, device_mesh):
    for param_name, param in mod.named_parameters(recurse=False):
        dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
        mod.register_parameter(param_name, dist_param)

def _apply(self, module, device_mesh):
    return distribute_module(
        module, device_mesh,
        partition_fn=self._partition_fn,
        input_fn=self._token_dispatch,
        output_fn=self._token_combine,
    )
```

### 3. FSDP 分层 Wrapping

[apply_fsdp](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/models/llama4/parallelize.py#347-511) 中，当 EP 启用时专家层单独用 `edp_mesh` 做 FSDP：

```python
if transformer_block.moe_enabled and ep_degree > 1:
    # 专家层：用更小的 efsdp mesh
    fully_shard(transformer_block.moe.experts, mesh=edp_mesh, ...)

# Dense部分（包含 router 等）：用正常的 dp_mesh
fully_shard(transformer_block, mesh=dp_mesh, ...)
```

还有一个细节：当 `efsdp * ep > num_experts` 时，应切 dim=1 而非 dim=0 来避免 FSDP 和 EP 在同一维度重复切分：
```python
if edp_mesh["efsdp"].size() * ep_degree > transformer_block.moe.experts.num_experts:
    _experts_shard_placement_fn = lambda param: Shard(1)
```

### 4. EP + TP 组合（高级）

[ExpertTensorParallel](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/distributed/expert_parallel.py#181-222) 进一步把专家权重在 `[ep, etp]` 二维 mesh 上切分（expert 维度 + 列式/行式 TP）。token dispatch/combine 只在 ep 维度做 All-to-All，权重 AllReduce 在 etp 维度完成。

---

## LlamaFactory v1 现状分析

### 现有结构

- **[DistributedInterface](file:///Users/humphrey/Documents/ep/LlamaFactory/src/llamafactory/v1/accelerator/interface.py#109-254)**：管理 [(mp_replicate, mp_shard)](file:///Users/humphrey/Documents/ep/LlamaFactory/src/llamafactory/v1/accelerator/interface.py#45-52) 和 [(dp, cp)](file:///Users/humphrey/Documents/ep/LlamaFactory/src/llamafactory/v1/accelerator/interface.py#45-52) 两套 2D mesh，**没有 ep 维度**
- **[FSDP2Engine](file:///Users/humphrey/Documents/ep/LlamaFactory/src/llamafactory/v1/plugins/trainer_plugins/distributed/fsdp2.py#68-479)**：用 `fsdp_mesh = data_device_mesh["dp"]` 对所有层统一 `fully_shard`，**对专家层无特殊处理**
- **无 EP 相关的 token dispatch/combine 机制**

### 与 MoE 的差距

| 能力 | torchtitan | LlamaFactory v1 |
|------|-----------|----------------|
| EP mesh 定义 | ✅ `efsdp/ep/etp` | ❌ 无 ep 维度 |
| 专家权重 EP 切分 | ✅ `Shard(0)` via DTensor | ❌ 无 |
| Token All-to-All dispatch | ✅ [ExpertParallel](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/distributed/expert_parallel.py#74-178) | ❌ 无 |
| 专家层单独 FSDP mesh | ✅ `edp_mesh` | ❌ 统一用 dp mesh |
| MoE 模型感知 | ✅ 按 `moe_enabled` 分支 | ❌ 无 |

---

## 实现思路：如何移植到 LlamaFactory v1

### Step 1：扩展 [DistributedStrategy](file:///Users/humphrey/Documents/ep/LlamaFactory/src/llamafactory/v1/accelerator/interface.py#54-107)，增加 EP 维度

在 [interface.py](file:///Users/humphrey/Documents/ep/LlamaFactory/src/llamafactory/v1/accelerator/interface.py) 中增加 `ep_size` 参数，构建包含 ep 的 mesh：

```python
class Dim(StrEnum):
    MP_REPLICATE = "mp_replicate"
    MP_SHARD = "mp_shard"
    DP = "dp"
    CP = "cp"
    EP = "ep"       # 新增
    EFSDP = "efsdp" # 新增：专家层的 FSDP 维度
```

mesh 构建逻辑参考 torchtitan：
```
efsdp = dp_shard // ep_size
sparse_mesh = init_device_mesh([dp_replicate, efsdp, ep_size])
```

注意：初期可以简化，只支持 `dp × ep` 两个维度，不引入 [cp](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/distributed/parallel_dims.py#326-329) 和 [etp](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/distributed/parallel_dims.py#350-353)。

### Step 2：在 [FSDP2Engine](file:///Users/humphrey/Documents/ep/LlamaFactory/src/llamafactory/v1/plugins/trainer_plugins/distributed/fsdp2.py#68-479) 中识别 MoE 层并分层处理

在 [prepare_model](file:///Users/humphrey/Documents/ep/LlamaFactory/src/llamafactory/v1/plugins/trainer_plugins/distributed/fsdp2.py#116-194) 中，增加对专家层的识别：

```python
def _get_expert_module(self, module):
    """识别给定 transformer block 中的专家聚合模块。"""
    # 常见命名：.mlp.experts, .moe.experts, .block_sparse_moe.experts
    for attr in ["mlp.experts", "moe.experts", "block_sparse_moe.experts"]:
        try:
            return module.get_submodule(attr)
        except AttributeError:
            continue
    return None
```

在 per-layer wrap 循环中：
```python
for name, module in model.named_modules():
    if type(module) in transformer_layer_cls_to_wrap:
        if self.ep_size > 1:
            experts = self._get_expert_module(module)
            if experts is not None:
                # 1. 先单独 wrap 专家层，用小的 efsdp mesh
                fully_shard(experts, mesh=self.efsdp_mesh, ...)
        # 2. 再 wrap 整个 transformer block，用正常 dp mesh
        fully_shard(module, mesh=self.fsdp_mesh, ...)
```

### Step 3：实现 [ExpertParallel](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/distributed/expert_parallel.py#74-178) 的 token dispatch/combine

核心：在 [_partition_fn](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/distributed/expert_parallel.py#326-334) 里把专家权重用 `Shard(0)` 切分；在 `input_fn/output_fn` 里注入 All-to-All 通信。可以直接从 torchtitan 复制/适配 [ExpertParallel](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/distributed/expert_parallel.py#74-178)：

```python
# 新文件：v1/plugins/trainer_plugins/distributed/expert_parallel.py
from torch.distributed._functional_collectives import all_to_all_single_autograd
from torch.distributed.tensor import distribute_module, distribute_tensor, Shard
from torch.distributed.tensor.parallel import ParallelStyle

class ExpertParallel(ParallelStyle):
    def _partition_fn(self, name, mod, device_mesh):
        for param_name, param in mod.named_parameters(recurse=False):
            mod.register_parameter(
                param_name,
                nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            )

    def _token_dispatch(self, mod, inputs, device_mesh):
        # All-to-All：把 token 按专家分发到对应 EP rank
        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]
        # ... (直接复用 torchtitan 逻辑)
        return routed_input, num_tokens_per_expert_group

    def _token_combine(self, mod, routed_output, device_mesh):
        # All-to-All：把结果收集回来
        routed_output = all_to_all_single_autograd(...)
        return routed_output

    def _apply(self, module, device_mesh):
        return distribute_module(module, device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine)
```

### Step 4：增加 `apply_ep` 入口函数

在 [fsdp2.py](file:///Users/humphrey/Documents/ep/LlamaFactory/src/llamafactory/v1/plugins/trainer_plugins/distributed/fsdp2.py) 或新文件里增加：
```python
def apply_ep(model, ep_mesh):
    """对模型中所有 MoE 专家层应用 Expert Parallelism。"""
    from torch.distributed.tensor.parallel import parallelize_module
    expert_plan = ExpertParallel()
    for name, module in model.named_modules():
        experts = _get_expert_module(module)
        if experts is not None:
            parallelize_module(experts, ep_mesh, expert_plan)
```

调用顺序（在 [shard_model](file:///Users/humphrey/Documents/ep/LlamaFactory/src/llamafactory/v1/plugins/trainer_plugins/distributed/fsdp2.py#246-265) 里）：
```
1. apply_ep(model, ep_mesh)          # EP 注册 input_fn/output_fn + 权重 Shard(0)
2. prepare_model(model)              # FSDP wrap（专家层用 efsdp_mesh，其余用 fsdp_mesh）
3. materialize_and_load(...)         # 加载权重（_copy_weights 已支持 DTensor）
```

### Step 5：梯度除法系数校正

EP 启用后专家层的 FSDP mesh 变小（efsdp < dp_shard），但梯度仍需除以完整 DP size，需禁用 FSDP 自动 gradient division 并手动统一：

```python
# 参考 torchtitan 的 disable_fsdp_gradient_division
for module in model.modules():
    if isinstance(module, FSDPModule):
        module.set_gradient_divide_factor(1.0)

# 在优化器步骤前手动 all-reduce 梯度并除以 dp_replicate * dp_shard
```

---

## 整体实现路线图

```
Phase 1: 基础 EP 支持（无 FSDP+EP 组合）
├── 扩展 DistributedStrategy 增加 ep_size
├── 实现 ExpertParallel（token dispatch/combine + 权重 Shard）
└── 在 shard_model 前调用 apply_ep

Phase 2: FSDP2 + EP 正确组合
├── 构建 efsdp mesh（dp_shard / ep_size）
├── 修改 prepare_model：MoE 专家层单独 fully_shard 到 efsdp_mesh
└── 修正梯度系数（disable auto grad division）

Phase 3: 权重加载适配
├── _copy_weights 支持 EP DTensor（Shard(0) 在 ep 维度）
└── checkpoint save/load 适配（per-expert sharding）

Phase 4（可选）: 高级功能
├── ExpertTensorParallel（EP × TP 组合）
├── DeepEP 替换（高性能 token dispatch kernel）
└── 显式 prefetch 优化（EP All-to-All 阻止 FSDP 隐式 prefetch）
```

---

## 关键注意事项

> [!IMPORTANT]
> **模型接口兼容性**：EP 的 [_token_dispatch](file:///Users/humphrey/Documents/ep/torchtitan/torchtitan/distributed/expert_parallel.py#303-325) 接收 [(routed_input, num_tokens_per_expert)](file:///Users/humphrey/Documents/ep/LlamaFactory/src/llamafactory/v1/accelerator/interface.py#45-52) 作为输入，对应专家层的 **forward 接口必须匹配**（如 Qwen MoE、Mixtral、DeepSeek-V3 各有不同接口）。建议先选定一个 HuggingFace MoE 模型作为目标，再适配接口。

> [!WARNING]
> **FSDP + EP 双重 Shard 冲突**：EP 已在 dim=0（expert 数量维度）做了 DTensor Shard；如果 FSDP 也在 dim=0 做 sharding，且 `efsdp × ep > num_experts`，就会产生问题。此时需要把 FSDP 的 sharding 切换到 dim=1，参考 `torchtitan` 的 `shard_placement_fn = lambda param: Shard(1)`。

> [!NOTE]
> **[_copy_weights](file:///Users/humphrey/Documents/ep/LlamaFactory/src/llamafactory/v1/plugins/trainer_plugins/distributed/fsdp2.py#433-479) 已基本就绪**：LlamaFactory 现有的 [_copy_weights](file:///Users/humphrey/Documents/ep/LlamaFactory/src/llamafactory/v1/plugins/trainer_plugins/distributed/fsdp2.py#433-479) 方法已经支持 DTensor，能按 Shard placement 正确切分权重。EP 引入后（Shard(0) 在 ep 维度），只要正确传递 `ep_mesh` 给 `distribute_tensor`，加载代码基本不需大改。

> [!TIP]
> **先从 `ep_size=1` 验证正确性**：实现后，在 `ep_size=1` 的配置下跑 dense 模型需要和之前完全等价，这是最简单的回归测试。再逐步增加 ep_size 和 MoE 模型验证。
