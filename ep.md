# FSDP2 + Expert Parallelism (EP) 完整实现解析

---

## 0. 先解释：什么是 EP？我们要达到什么目的？

MoE 模型（Mixture of Experts，混合专家）的核心结构是这样的：
- 每个 Transformer 层里，不是一个 FFN（前馈层），而是有 **N 个专家**（Expert），每个专家是一个独立的小 FFN。
- 路由器（Router/Gate）负责对每个 Token 决策：送给哪几个专家去处理。
- 这样每个 Token 只经过少数几个专家，**计算量不变，但模型容量大幅扩大**。

**问题**：如果有 128 个专家，把这 128 个专家的参数都放在每张卡上，**显存根本放不下**。

**Expert Parallelism（EP）的解决方案**：把这 128 个专家拆分到多张卡上，每张卡只存一部分专家。比如 8 张卡，每张卡存 16 个专家。

**随之而来的问题**：Token 要找自己的专家，但专家现在在不同的 GPU 上！所以 Token 必须通过网络通信，**从自己所在的 GPU 发送到专家所在的 GPU 去计算，计算完再发送回来**。这就是 EP 最核心的通信操作：**All-to-All**。

---

## 1. 系统架构鸟瞰

整个 EP 实现分 4 个模块，关系如下：

```
训练入口
  │
  ▼
[interface.py] ─── 负责：规划所有 GPU 的分组方式（Mesh 拓扑）
  │ 告诉系统：哪些卡是 EP 组，哪些卡是 FSDP 组
  │
  ▼
[fsdp2.py] ─────── 负责：协调整个模型的包装流程
  │ 先调用适配器打 patch，再用 fully_shard 切分参数
  │
  ├──► [ep_adapters/] ─ 负责：认识具体的 MoE 模型结构，替换 forward 方法
  │        │
  │        └──► [expert_parallel.py] ─ 负责：实际执行 All-to-All 通信路由
  │
  (训练时每个 forward 自动触发 All-to-All)
```

---

## 2. 第一关：`interface.py` — GPU 分组（Device Mesh）

### 在做什么？
在多 GPU 训练前，需要告诉 PyTorch：这些 GPU 该如何分组协作？

**核心数据结构**：`DistributedStrategy`

假设有 8 张卡，`ep_size=4`，`dp_size=8`：

```
sparse_mesh_shape = (cp_size=1, dp_size/ep_size=2, ep_size=4)

物理布局：
  [卡0, 卡1, 卡2, 卡3]  → 这 4 张卡互为 EP 组（共同存放所有专家）
  [卡4, 卡5, 卡6, 卡7]  → 这 4 张卡互为 EP 组（共同存放所有专家）
  
  [卡0, 卡4] → 这 2 张卡互为 EFSDP 组（专家参数的 FSDP 数据并行组）
  [卡1, 卡5] → EFSDP 组
  [卡2, 卡6] → EFSDP 组
  [卡3, 卡7] → EFSDP 组
```

三个重要的 Mesh 维度名称：
- **`ep`**：EP 通信组 — 同一组内的卡共同持有全部专家，需要 All-to-All 交换 Token。
- **`efsdp`**：Expert FSDP 组 — 专家参数被切分存放在这个组的各卡上（参数级别的显存共享）。
- **`dp`/`cp`**：普通数据并行 / 序列并行，非专家层用这个。

### 为什么这样做？
FSDP2 和 EP 必须知道"哪张卡和哪张卡是朋友"，后续所有的参数切分、通信都依赖这个分组信息。`interface.py` 是唯一的事实来源。

---

## 3. 第二关：`expert_parallel.py` — Token 跨卡路由（核心通信）

这是 EP 的核心引擎，解决的问题是：**Token 如何从自己所在的卡，跑到专家所在的卡？**

### `TokenPermuteBackend`（抽象基类）

这是一个接口（Abstract Class），规定了两个操作必须被实现：

```python
permute()   # Token 发送：把 token 按专家分发到各卡
unpermute() # Token 接收：把计算结果收回来
```

把通信逻辑抽象成接口的好处是：**未来可以随时替换通信实现（比如换成 CUDA Fused 版本），不需要改任何业务代码**。

---

### `DefaultTokenPermuteBackend.permute()` — Token 发送，分步解析

假设当前有 4 卡（ep_size=4），每卡 2 个专家（共 8 个专家），当前卡有 100 个 Token：

**Step 1：统计每个专家分配了多少 Token**
```python
num_tokens_per_expert = [15, 12, 8, 20, 10, 11, 14, 10]  # 长度=8（总专家数）
```

**Step 2：用 All-to-All 交换"我要发多少"的信息**
```python
# 每张卡广播自己的 num_tokens_per_expert，收集别人的
num_tokens_per_expert_group = all_to_all_single(num_tokens_per_expert, ...)
# 现在知道："卡0 要给我发多少 token，卡1 要给我发多少 token..."
```
这一步**只交换元数据（数量），不交换实际 Token**，所以开销很小，且在 `torch.no_grad()` 里。

**Step 3：计算每张卡之间互发多少**
```python
input_splits  = [27, 30, 21, 13]   # 我要分别发给 卡0/卡1/卡2/卡3 多少
output_splits = [18, 22, 25, 31]   # 我分别从 卡0/卡1/卡2/卡3 收多少
```

**Step 4：实际发送 Token（核心 All-to-All）**
```python
routed_input = all_to_all_single_autograd(
    routed_input,
    output_splits_list,   # 我要收多少
    input_splits_list,    # 我要发多少
    group=ep_group
)
```
用 `autograd` 版本是因为**反向传播时需要自动计算梯度**（等价于把梯度反向 All-to-All 回去）。

**Step 5：重排 Token 顺序（Permutation）**

All-to-All 之后，收到的 Token 是按"来源卡"排列的，但专家计算时需要"按专家 ID"排列。`permutation` 索引用于将 Token 重新排好队：

```
All-to-All收到：[卡0给我的token(expert0), 卡1给我的token(expert0), 卡0给我的(expert1), ...]
重排后：        [所有expert0的token, 所有expert1的token, ...]
```

这个重排索引 `dispatch_permutation` 被保存在 `state` 里，`unpermute` 阶段要用它反向还原。

---

### `DefaultTokenPermuteBackend.unpermute()` — Token 接收回来

两个对称操作：
1. 用保存的 `combine_permutation`（是 `dispatch_permutation` 的逆）把 Token 顺序还原
2. 再做一次反向 All-to-All，把计算结果发回 Token 原来的卡

---

### `ExpertParallel(ParallelStyle)` — 参数切分 + 挂钩子

PyTorch 的 `parallelize_module` 接受一个 `ParallelStyle` 对象，会调用它的 `_apply()` 方法来改造模块。`ExpertParallel` 就是这个改造规则：

```python
def _apply(self, module, device_mesh):
    return distribute_module(
        module,
        device_mesh,
        partition_fn=self._partition_fn,   # 切分参数
        input_fn=self._token_dispatch,      # forward 前执行 permute
        output_fn=self._token_combine,      # forward 后执行 unpermute
    )
```

- **`_partition_fn`**：把 `gate_up_proj`（形状 `[N_experts, 2*I, H]`）和 `down_proj`（`[N_experts, H, I]`）沿第 0 维（专家数量维）切开，每张卡只保留自己负责的专家的权重。
- **`_token_dispatch`**（input hook）：在 forward 计算**之前**自动触发 `permute()`，Token 发出去。
- **`_token_combine`**（output hook）：在 forward 计算**之后**自动触发 `unpermute()`，结果收回来。

**效果**：业务代码完全不需要知道 EP 的存在。调用 `self.experts(x, counts)` 时，框架在前后自动处理 All-to-All。

---

## 4. 第三关：`ep_adapters/` — 认识具体的 MoE 模型

### 为什么需要适配层？

Hugging Face 的不同模型，MoE 的实现方式完全不同：
- Qwen3MoE：`mlp.experts.gate_up_proj [E, 2I, H]`（堆叠矩阵，可直接切）
- Mixtral：可能用 `nn.ModuleList`（一个 list 包 N 个独立 FFN）
- DeepSeekV3：可能有共享专家 + 路由专家混合结构

每种模型的 `forward()` 写法、专家参数的组织方式都不同，但切分和通信的核心逻辑完全一样。**适配层负责"翻译"，核心层负责"干活"**。

---

### `BaseEPAdapter`（抽象基类）

规定了每个适配器必须实现的接口：

| 方法 | 作用 |
|---|---|
| `get_expert_module(module)` | 给我一个 Transformer 层，告诉我专家模块在哪 |
| `get_num_experts(experts)` | 这个专家模块里有多少个专家 |
| `prepare_and_apply_ep(model, ep_mesh)` | 找到所有 MoE 块，打 patch，调用 `parallelize_module` |

---

### `Qwen3MoeEPAdapter`（当前唯一实现）

**`_ep_moe_block_forward()`** — 替换原始的 MoE Block forward

原始 HF 的 `forward()` 是逐个专家循环的，非常慢也不支持 EP。新的版本：

```
① 过路由器 → 得到 routing_weights 和 selected_experts
② 统计每个专家有多少 Token → num_tokens_per_expert
③ 把 Token 按专家 ID 排序 → routed_input（已按专家分组的展平 Tensor）
④ 调用 self.experts(routed_input, num_tokens_per_expert)
   ↑ ExpertParallel 自动在这前后做 All-to-All
⑤ 用 routing_weights 加权，scatter_add_ 还原回 [B, T, H]
```

**`_ep_experts_forward()`** — 替换原始的 Experts 模块 forward

替换后接受"已排好序的展平 Token"和每个专家的 Token 数量：
```python
for local_expert i in range(num_local_experts):
    x_e = x[offset:offset+count]           # 取出属于这个专家的 token
    gate, up = linear(x_e, gate_up[i]).chunk(2)
    out = linear(act_fn(gate) * up, down[i])
    outputs.append(out)
```
这里的 `num_local_experts` 已经是 All-to-All 之后、当前卡负责的专家数量（比如从 128 变成 16）。

**`prepare_and_apply_ep()`** — 主入口

```python
for each MoE block in model:
    # 1. 替换 experts 的 forward 方法
    experts.forward = _ep_experts_forward
    # 2. 替换整个 MoE block 的 forward 方法
    moe_block.forward = _ep_moe_block_forward
    # 3. 注册 ExpertParallel 策略（切分参数 + 挂 dispatch/combine 钩子）
    parallelize_module(experts, ep_mesh, ExpertParallel())
```

---

## 5. 第四关：`fsdp2.py` — 总指挥

这是整个流程的入口，`shard_model()` 是核心：

```python
def shard_model(model):
    # 第一步：EP 处理（必须在 FSDP 之前！）
    model = self.apply_ep(model)
    
    # 第二步：FSDP2 参数切分包装
    model = self.prepare_model(model)
    
    # 第三步：加载权重（从 HF checkpoint 或 DCP）
    model = self.materialize_and_load(model, ...)
```

### 为什么 EP 必须在 FSDP 之前？

`parallelize_module` 会把专家参数变成 `DTensor`（分布式张量），之后 FSDP 的 `fully_shard` 要再进一步把这些 DTensor 切片到各卡。必须先确定"专家归属（EP 切分）"，FSDP 才知道还剩多大的显存需要切分。

### `prepare_model()` 中的双层包装

```python
for each transformer layer:
    experts = adapter.get_expert_module(module)
    if experts is not None:
        # 专家用 efsdp_mesh 包 → 专家参数在 efsdp 组内切分（比全局小）
        fully_shard(experts, mesh=self.efsdp_mesh, ...)
        experts.set_gradient_divide_factor(float(self.world_size))
    
    # 整个 Transformer 层（含非专家部分）用全局 fsdp_mesh 包
    fully_shard(module, mesh=self.fsdp_mesh, ...)
```

**为什么专家用 `efsdp_mesh`，非专家用 `fsdp_mesh`？**

- 非专家参数（Attention、LayerNorm 等）在所有卡上都是一样的（数据并行），用全局 `fsdp_mesh` 切分，梯度要在所有卡上同步汇总。
- 专家参数只存在于 EP 组内（每张卡不同），梯度只需在 `efsdp` 组内同步汇总，如果错用全局 Mesh 会导致通信错误。

**为什么要 `set_gradient_divide_factor(world_size)`？**

FSDP 在 reduce-scatter 后会把梯度除以 Mesh 尺寸（`efsdp_size`），使得梯度代表该组的平均值。但 Loss 是算过全局 batch 的平均的，所以梯度的有效除数应该是 `world_size`（全局卡数），而不是 `efsdp_size`（组内卡数）。设置这个值让梯度的缩放比例正确。

---

## 6. 完整数据流：一个 Token 的 EP 旅程

```
[Trainer 触发 forward()]
       │
       ▼
[Transformer 层 forward()]
       │
       ├──► Attention 计算（正常，全局 FSDP 参数）
       │
       └──► MoE Block forward（已被 patch 为 _ep_moe_block_forward）
               │
               ├─ 1. 路由：Gate([B,T,H]) → routing_weights, selected_experts
               ├─ 2. 排序：Token 按 expert ID 排列 → routed_input [N_tokens, H]
               ├─ 3. 统计：num_tokens_per_expert [N_experts]
               │
               └─ 4. 调用 self.experts(routed_input, num_tokens_per_expert)
                       │
                       ├─ [ExpertParallel input_hook] permute()
                       │     ├─ all_to_all "我要发多少" (元数据)
                       │     └─ all_to_all_autograd 实际 Token → 跨卡发出
                       │
                       ├─ [_ep_experts_forward] 在本卡上计算本卡负责的专家
                       │
                       └─ [ExpertParallel output_hook] unpermute()
                             ├─ 逆排列还原顺序
                             └─ all_to_all_autograd 结果 → 跨卡收回
               │
               ├─ 5. 加权：output * routing_weights
               └─ 6. scatter_add_ 还原 [B, T, H]
```

---

## 7. 如何扩展新 MoE 模型（简明步骤）

1. 在 `ep_adapters/` 下新建 `your_model_moe.py`
2. 继承 `BaseEPAdapter`，实现 `get_expert_module`、`get_num_experts`、`prepare_and_apply_ep`
3. 写一个 `_ep_moe_block_forward`（参考 Qwen3 的写法，重点：排序 Token、调 `self.experts`、加权还原）
4. 写一个 `_ep_experts_forward`（按 expert 切片处理展平 Token，避免 Python `for` 循环嵌套）
5. 在 `ep_adapters/__init__.py` 的 `_EP_ADAPTER_REGISTRY` 里注册
6. **改进 `_infer_adapter_name`**（当前直接返回 `"qwen3_moe"`，需要按 `model.config.architectures` 条件判断）

## 8. 如何替换通信方式（简明步骤）

1. 在 `expert_parallel.py` 里继承 `TokenPermuteBackend`，实现 `permute` 和 `unpermute`
2. 调用 `register_token_permute_backend("your_name", YourBackendClass)`
3. 在 `ExpertParallel(token_permute_backend="your_name")` 处传入名字即可
4. 不需要修改 Adapter 或 FSDP2 Engine 任何代码
