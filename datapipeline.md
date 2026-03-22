# RLVR Pipeline 主控逻辑与 TransferQueue 适配分析

> 对应代码路径：`ROLL/roll/pipeline/rlvr/`

---

## 目录

1. [原始主控逻辑](#一原始主控逻辑)
2. [流转数据类型与字段](#二流转数据类型与字段)
3. [TransferQueue 适配分析](#三transferqueue-适配分析)

---

## 一、原始主控逻辑

`RLVRPipeline` 是整个 RLVR（Reinforcement Learning from Verifiable Rewards）的驱动器，定义在 `rlvr_pipeline.py` 中，继承自 `BasePipeline`。训练流程以 `global_step` 为单位循环推进。

### 1.1 初始化阶段 `__init__`

按顺序完成以下事项：

1. **加载数据集**：读取训练集（可含多 domain），按 `domain_interleave_probs` 分域过滤，并对每个 domain 创建独立的 `DynamicSamplingScheduler`；可选地加载验证集。
2. **数据预处理**：调用 `get_encode_function` / `preprocess_dataset` 对 prompt 做 tokenize（使用 chat template），并过滤超长序列；补全 `domain` 字段。
3. **KL 控制器**：根据配置实例化 `kl_ctrl`，用于后续 KL reward 衰减调节。
4. **创建 Cluster**：
   - `actor_train`：训练用 Actor（对应 `ActorWorker` / `ActorPGWorker`）
   - `actor_infer`：推理用 Actor（`InferWorker`，跑 vLLM 生成）
   - `reference`（可选）：参考模型，用于 KL 计算
   - `critic`（可选，GAE 时）：价值网络
   - `rewards`：多 domain 奖励 Worker
   - `reward_model_cluster`（可选）：LLM-as-judge 奖励模型
5. **下载模型权重**：调用 `download_models` 一次性下载所有 cluster 的权重。
6. **初始化各 Cluster**：依次并发调用 `cluster.initialize()`。
7. **设置 model_update_pair**：每隔 `model_update_frequency` 步将 actor_train 权重同步到 actor_infer。
8. **初始化指标统计**（`RunningMoments`）：用于滚动计算每个 domain 的 reward 均值和标准差。

---

### 1.2 主训练循环 `run()`

每个 `global_step` 依次执行以下六个阶段：

#### 阶段 ①：模型同步（Model Update）

```
actor_train.offload_states()
actor_infer.offload_states(other_params)   # 仅 async 模式
model_update(global_step)                  # actor_train → actor_infer 权重同步
actor_infer.load_states()
```

- 在训练开始前先将权重从训练侧广播到推理侧，保证本步推理使用最新策略。
- `async_pipeline` 模式下：先暂停采样（`pause_sampling`），再做权重同步，再恢复推理服务。

---

#### 阶段 ②：Rollout 生成（Generate）与异步数据流

```python
# RLVRPipeline 向所有 Domain Scheduler 并发要数据
for domain, scheduler in self.generate_schedulers.items():
    scheduler_refs[domain] = scheduler.get_batch.remote(..., batch_size=self.domain_batch_size[domain])
```

这一阶段是整个强化学习大循环中最核心的**“发车捡件”**环节。每个 `DynamicSamplingScheduler` 内部通过两个核心组件异步协作完成数据收集：

1. **后台永动采样（发送请求）**：
   Scheduler 内部的 `sending_request` 后台协程作为一个死循环，只要系统允许，就会调用 `get_next_dataset_item()` **从该 Domain 专属的数据集中抽取一条 Prompt**。随后通过 `RouterManager` 发给 vLLM（`actor_infer`）并发生成 `num_return_sequences` 个 Responses，再送到对应的 `RewardWorker` 打分。最终成品打包成 `ExperienceItem` 塞入 `ReplayBuffer`（成品仓库）。
2. **数据凑批与“陈旧数据”等待（ReplayBuffer 机制）**：
   当主控下达 `DynamicSamplingScheduler.get_batch` 时，真正的阻塞提取发生在底层：
   `await self.replay_buffer.get_batch(expected_samples=...)`

   **【核心追溯 1：ReplayBuffer 怎么拿够数据？】**
   `ReplayBuffer.get_batch` 内部并不是自己生产数据，而是根据步数（由旧到新的 `min_step` 规则）遍历其内部缓存的 `ItemsGroup`（按 global_step 拆分的子仓库）。真正的取药动作是：
   `finished_prompts = await group.get_batch(expected_samples=...)`
   这个 `group.get_batch` 是一个条件等待。如果该 group 里已经 `commit` 完的算完分数的产品够数，马上返回；如果不够，它会循环执行 `await asyncio.sleep(0)` 挂起，把进程控制权让给后台协程，等后台生成完了塞进 group，它再被唤醒拿去凑批。**程序绝不复用过往的假数据/老数据。**

   **【核心追溯 2：真实的推理拿数据接口在哪里？】**
   后台协程是如何产生数据的呢？它源于那个死循环 `sending_request()`。核心调用链如下：
   - 抽一条提示词，开启生命周期管控：`RolloutContext.process_new_prompt(scheduler, prompt_id)`
   - 真正向推理引擎索要回答的底层网关接口：
     `await self._scheduler.router_client.generate_request(req=req, ...)`
   - 这个调用通过 Ray 的 Router，真实触发了远端 vLLM Worker (`actor_infer`) 的自回归生成。
   - 等拿回 `responses`，马上调 `compute_rewards.remote` 让奖励节点给分。全部完成后，调用 `scheduler.replay_buffer.commit()` 把这批完整数据入库，唤醒上面那个正在 `await group.get_batch` 焦急等待的人。
3. **数据组合拼接（DataConcat）**：
   数据的拼接是由下而上、分两层物理合并的：
   - **第一层（Scheduler 同域合并）**：在 Scheduler 凑齐该领域的 `batch_size` 个样本后，调用内部方法，通过 `DataProto.concat` 将张量在 Batch 维度拼接成一个领域专属的小 Batch（若是 TQ 模式则合并 `KVBatchMeta` 的 keys）。
   - **第二层（Pipeline 跨域合并）**：主控 `ray.get()` 收集齐所有 Domain 的小 Batch 后，进行最后一次跨域的全量拼接：`DataProto.concat(domain_batches.values())`。产生一个既包含代码又包含数学、整体大小严格等于 `rollout_batch_size` 的超级大矩阵 `generate_output`，交由后续网络计算优势函数。

---

#### 阶段 ③：计算参考日志概率（Ref Log Probs）

```
# 如果 enable_reference:
reference.compute_log_probs(batch) → ref_log_probs
# 合并：batch.union(ref_log_probs)，将 log_probs 重命名为 ref_log_probs
```

- 如果是 LoRA 训练，则 `use_ref_model=False`，直接用 `actor_train` 加 `disable_adapter=True` 来计算。
- 如果不开 reference，则直接用 `old_log_probs` 代替 `ref_log_probs`。

---

#### 阶段 ④：计算旧日志概率与价值（Old Log Probs & Values）

```
# 如果 enable_old_logprobs_recompute:
actor_train.compute_log_probs(batch) → old_log_probs
# 如果 adv_estimator == "gae":
critic.compute_values(batch) → values
```

- `old_log_probs`：本策略在生成时产生的 token 级别对数概率，用于后续重要性采样比率 `ratio = exp(log_probs_new - log_probs_old)`。
- `values`：critic 对每个 token/step 的价值估计，GAE 模式才用。

---

#### 阶段 ⑤：本地计算 Reward、Advantage（主控所在机器）

按 domain 分组，对每个 domain batch 顺序做：

1. **`get_sample_level_mask`**：根据长度、难度等策略生成 `final_response_mask`，过滤或下权重。
2. **`reward_postprocess`**：对 `scores` 做 normalize（减均值 / 除标准差），group-level GRPO 处理等。
3. **`compute_token_reward`**：将 response-level reward 分发到 token 级别，叠加 KL penalty，生成 `token_level_rewards`。
4. **`compute_advantage`**：根据 `adv_estimator`（GRPO / GAE 等）计算 `advantages` 和 `returns`（GAE 时需要 `values`）。

以上均为纯本地 CPU/GPU 计算，不涉及 Ray remote call。

---

#### 阶段 ⑥：训练步（Train Step）

```
# 可选：
critic.train_step(batch)
# 主要：
actor_train.train_step(batch) → actor_train_metrics
```

`ActorPGWorker.loss_func` 在 worker 侧计算 policy gradient loss：

- 根据 `pg_variant` 选择 PPO / GRPO / TIS / TOPR / CISPO / Kimi15 等变体。
- 计算 `ratio = exp(log_probs - old_log_probs)`，结合 `advantages` 计算 pg_loss。
- 加上 KL loss（`k3`）、entropy loss（可选）。
- 返回训练 metrics。

---

#### 步末处理

- 保存 checkpoint（`do_checkpoint`）
- 上报 metrics（`tracker.log`）
- 打印示例 prompt / response

---

## 二、流转数据类型与字段

整个流程中，数据以 `DataProto` 为核心容器在 pipeline、scheduler、worker 之间传递。

### 2.1 DataProto 结构

```python
class DataProto:
    batch: TensorDict       # 张量数据，batch 维度对齐
    non_tensor_batch: dict  # 非张量数据（字符串、列表等），numpy object array
    meta_info: dict         # 元信息，全步共享的标量 / 配置
```

### 2.2 各阶段数据字段汇总

| 阶段 | 新增 / 使用的 `batch` 字段 | 说明 |
|------|--------------------------|------|
| **数据采样** | `input_ids`, `attention_mask`, `position_ids` | tokenize 后的 prompt（padding 到 `prompt_length`） |
| **生成后** | `prompts`, `responses`, `response_mask`, `eos_mask` | 拼接后的完整序列、response 部分 mask、EOS 标记 |
| **奖励计算后** | `scores` | response 级别标量得分，shape `[B]` |
| **非张量字段** | `non_tensor_batch["tag"]`, `non_tensor_batch["domain"]` | 数据来源标签（用于 group/filter） |
| **ref log probs** | `ref_log_probs` | 参考模型对 response 的对数概率，shape `[B, L]` |
| **old log probs** | `old_log_probs` | 生成策略的 token 级对数概率，shape `[B, L]` |
| **values（GAE）** | `values` | critic 的价值预测，shape `[B, L]` |
| **mask 后** | `final_response_mask` | 过滤后实际参与 loss 的 token mask |
| **reward 处理后** | `response_level_rewards`, `token_level_rewards` | response/token 级别的最终 reward（含 KL penalty） |
| **advantage 后** | `advantages`, `returns` | 优势函数值和回报（GAE 时两者独立） |
| **train 时** | `prompt_id`, `sample_uuid` | 样本标识，用于 DP 负载均衡和日志 |

### 2.3 meta_info 关键字段

| 字段 | 类型 | 用途 |
|------|------|------|
| `global_step` | int | 当前训练步 |
| `generation_config` | dict | vLLM 生成配置（温度、max_new_tokens 等） |
| `is_offload_states` | bool | 控制 worker 是否在调用前 offload 模型权重 |
| `disable_adapter` | bool | LoRA 模式下，让 actor_train 临时关闭 adapter（作参考模型用） |
| `agg_entropy` | float | 全批次平均 entropy，用于日志 |
| `loss_mask_keys` | list | 训练时保护的 mask key 列表 |
| `metrics` | dict | 各阶段采集的性能指标 |

### 2.4 ExperienceItem（调度器内部）

```python
@dataclass
class ExperienceItem:
    prompt_id: int                    # 唯一 prompt 标识
    domain: str                       # 数据域（"math_rule" 等）
    sampling_start_step: int          # 采样开始的 training step（off-policy 判断）
    data: DataProto | KVBatchMeta     # 单条经验数据（TQ 开启后为 KVBatchMeta）
    score: float                      # response 级别得分（快速 filter 用）
```

---

## 三、TransferQueue 适配分析

TransferQueue（TQ）是一个**基于共享内存/高速传输通道的张量存储系统**，目标是消除各阶段 Ray RPC 搬运大块张量 tensor 时的序列化 / 反序列化开销。

### 3.1 核心抽象与 TQ 键值（Key-Value）结构

> **`KVBatchMeta` 里的 `Key` 和 `Value` 到底长什么样？**
> 对于在不同模块间飞来飞去的 `KVBatchMeta` 提货单：
> - **Key (键)**: 每生成一个单独的 prompt-response 样本，系统都会为其分配一串唯一的随机字符串 ID（本质上是类似 `uuid.uuid4().hex` 生成的全局字典键）。
> - **Value (值)**: 在底层 TQ 后台共享内存/显存中，保存着这一条特定样本所拥有的**完整张量数据切片**（也就是这个 UUID 对应的 `input_ids`、`responses`、`scores` 等张量）。
> 
> 因此，`KVBatchMeta` 虽然叫 Batch，但它实际上就是一个捏在手里的 **“包含了一组张量切片存储指针的列表”**，非常轻巧。

| 类型 | 作用 |
|------|------|
| `KVBatchMeta` | 轻量级元数据句柄（也就是上述的提货单），包含 `keys`（上述 UUID 列表）、`partition_id`、`fields`（当前 TQ 里存了哪些张量字段）、`extra_info`（非张量字符串或字典）|
| `BatchMeta` | TQ 内部低层句柄，由 `@tqbridge` 装饰器使用（含 `global_indexes`、`partition_ids`、`field_names`）|
| `@tqbridge` | 装饰 worker RPC 函数：输入 `BatchMeta` 时自动从 TQ 拉取数据为 `DataProto`，输出时将结果写回 TQ 并返回 `BatchMeta` |

---

### 3.2 关键工具函数

| 函数 | 位置 | 作用 |
|------|------|------|
| `init_tq(config)` | `tq_pipeline_utils.py` | 初始化 TQ，pipeline 主进程和 scheduler 进程分别调用一次 |
| `dataproto_to_kv_batch_meta(data, partition_id)` | `tq_pipeline_utils.py` | 将 DataProto 的全部 batch 张量写入 TQ，返回 `KVBatchMeta` 句柄 |
| `kv_batch_meta_to_dataproto(meta, fields)` | `tq_pipeline_utils.py` | 从 TQ 按 fields 读回张量，组装为 `DataProto` |
| `kv_batch_meta_put_fields(meta, data, field_keys)` | `tq_pipeline_utils.py` | 将 DataProto 中指定字段写回 TQ（追加/覆盖），更新 `meta.fields` |
| `merge_kv_batch_metas(kv_metas)` | `tq_pipeline_utils.py` | 合并多个 domain 各自返回的 `KVBatchMeta`，拼接 keys/tags |
| `kv_batch_meta2batch_meta(meta)` | `transferqueue_utils.py` | `KVBatchMeta` → `BatchMeta`（供 `@tqbridge` 内部使用） |
| `batch_meta2kv_batch_meta(meta)` | `transferqueue_utils.py` | `BatchMeta` → `KVBatchMeta`（worker 写回后转换） |
| `kv_batch_meta_put_tensordict(meta, td)` | `transferqueue_utils.py` | 底层：将 TensorDict 写回 TQ 指定 KVBatchMeta 对应的 offset |

---

### 3.3 各阶段 TQ 适配详解

#### 阶段 ②：Rollout 生成 → 提交到 TQ

**原始路径**
```
ExperienceItem.data = DataProto  →  ReplayBuffer.commit()
# 最终通过 DataProto.concat 返回给 pipeline
```

**TQ 适配路径**（`_commit_tq_responses`）
```python
# 生成 + reward 完成后：
combined = DataProto.concat(responses)
kv_meta = dataproto_to_kv_batch_meta(combined, partition_id="train", tags=tags)
# 每条样本单独切分一个 single-key KVBatchMeta，存入 ExperienceItem
ExperienceItem.data = KVBatchMeta(keys=[kv_meta.keys[i]], ...)
scheduler.replay_buffer.commit(prompt_id, items)
```

- **关键点**：张量在 rollout 完成后立即写入 TQ，`ExperienceItem` 存的是轻量 `KVBatchMeta`（只有一个 key handle），不再携带完整tensor。
- **`non_tensor_batch`**（tag、domain 等字符串字段）通过 `extra_info` 携带，不进 TQ 共享内存。

---

#### 阶段 ②（合并）：Scheduler 返回 `KVBatchMeta`

**原始路径**（`_collect_items_as_dataproto`）
```python
batch = DataProto.concat([...])  # 全量张量合并，通过 Ray RPC 返回 pipeline
```

**TQ 适配路径**（`_collect_items_as_kv_batch_meta`）
```python
# 合并各 domain 的 keys/tags，构建跨域 KVBatchMeta
merged_kv_meta = KVBatchMeta(keys=all_keys, tags=all_tags, ...)
merged_kv_meta.extra_info["metrics"] = metrics
merged_kv_meta.extra_info["non_tensor_batch"] = dict(merged_non_tensor)
return merged_kv_meta  # 通过 Ray RPC 只传 meta，张量留在 TQ
```

**Pipeline 侧**：
```python
# 各 domain scheduler 返回 KVBatchMeta
domain_kv_meta = ray.get(scheduler_ref)
kv_meta = merge_kv_batch_metas(list(domain_kv_metas.values()))
# 拉回全量张量，组装 DataProto（用于本地 advantage 计算）
batch = kv_batch_meta_to_dataproto(kv_meta)
```

> **权衡**：pipeline driver 仍需一次性从 TQ 读取全量 batch 用于本地计算（advantage 等），但避免了经 Ray Actor 做 Python pickle 的大包序列化。

---

#### 阶段 ③：Ref Log Probs

**原始路径**
```python
ref_log_probs: DataProto = reference.compute_log_probs(batch)  # 完整 DataProto via RPC
batch = batch.union(ref_log_probs)
```

**TQ 适配路径**
```python
# reference / actor_train worker 的 compute_log_probs 被 @tqbridge 装饰
# pipeline 传 KVBatchMeta（含 extra_info=batch.meta_info）
self.reference.compute_log_probs(kv_meta, blocking=True)
# worker 内部：@tqbridge 自动拉取张量，forward，将 log_probs 写回 TQ

# pipeline 只读取需要的字段：
ref_fields = kv_batch_meta_to_dataproto(kv_meta, fields=["log_probs"])
ref_fields.rename("log_probs", "ref_log_probs")
# 同时写回 kv_meta，令后续步骤共用：
kv_batch_meta_put_fields(kv_meta, ref_fields, field_keys=["ref_log_probs"])
batch.batch["ref_log_probs"] = ref_fields.batch["ref_log_probs"]
```

---

#### 阶段 ④：Old Log Probs & Values

**TQ 适配路径**
```python
self.actor_train.compute_log_probs(kv_meta, blocking=True)
# @tqbridge：从 TQ 拉 → forward → log_probs 写回 TQ

old_fields = kv_batch_meta_to_dataproto(kv_meta, fields=["log_probs", "entropy"])
# 本地聚合 entropy
agg_entropy = agg_loss(old_fields.batch["entropy"], ...)
batch.meta_info["agg_entropy"] = agg_entropy
batch.batch["old_log_probs"] = old_fields.batch["log_probs"]
# 重命名并写回 TQ
old_fields.batch["old_log_probs"] = old_fields.batch["log_probs"]
kv_batch_meta_put_fields(kv_meta, old_fields, field_keys=["old_log_probs"])

# critic values（GAE）：
self.critic.compute_values(kv_meta, blocking=True)
values_data = kv_batch_meta_to_dataproto(kv_meta, fields=["values"])
batch = batch.union(values_data)
```

---

#### 阶段 ⑤：本地 Advantage 计算 → 写回 TQ

Advantage（优势函数、归一化的 Token 级奖励）等计算发生在纯本地（Pipeline 驱动进程的主 CPU 上）。主控完成这部分极其核心的 RL 运算后，立刻将结果字段写回 TQ 存储结构。

> **为什么要放回 TQ 呢？（承上启下）**
> 因为紧接着到了最后的第 ⑥ 阶段（`train_step`），就要启动位于 GPU 集群上的 `Actor_Train` 工作节点来跑真正吃性能的**“网络权重反向传播梯度下降”**了。
> 而 `train_step` 那个工作节点的 RPC 引脚是被 `@tqbridge` 装饰的。这意味着，它在启动时**不吃物理张量，只接收你传过去的这张 `KVBatchMeta` 提货单**。
> 如果 Pipeline 在本地算完了 `advantages` 这些惩罚奖励数据，没有立刻用 `kv_batch_meta_put_fields` 放回 TQ 仓库里，那么 Worker 拿着旧的提货单去 TQ 后台拉货时，就会发现根本拉不到算损失所需的核心参数，直接报错崩溃。因此，这一步是为了给下游的反向传播“备料入库”。

```python
# 写回 TQ 的字段：
tq_writeback_fields = [
    "advantages", "returns",
    "token_level_rewards", "final_response_mask",
    "train_infer_is_weight",
    # 如果 enable_old_logprobs_recompute，还包括 loss_mask_keys
]
kv_batch_meta_put_fields(kv_meta, batch, field_keys=existing_fields)
kv_meta.extra_info = batch.meta_info  # 同步 meta 信息
```

---

#### 阶段 ⑥：Train Step

**原始路径**
```python
# 完整 DataProto（含 advantages、ref_log_probs、old_log_probs 等所有字段）通过 RPC 传给 worker
actor_train.train_step(batch)
```

**TQ 适配路径**
```python
# 只传 KVBatchMeta，worker 侧 @tqbridge 自动拉取所有字段
actor_train.train_step(kv_meta, blocking=True)
```

- Worker `train_step` 的 `@tqbridge` 装饰器会从 TQ 一次性拉取该 worker 所负责的 DP 分片数据，调用 `loss_func`，并将梯度更新完成的结果（如有）写回。

---

### 3.4 数据流对比总览

```
【原始】
Dataset → Scheduler(DataProto) ──RPC──→ Pipeline(DataProto)
  → Reference(DataProto) ──RPC──→ Pipeline(ref_log_probs)
  → Actor(DataProto) ──RPC──→ Pipeline(old_log_probs)
  → [本地 advantage 计算]
  → Actor/Critic train_step(DataProto) ──RPC──→ metrics

【TQ 适配】
Dataset → Scheduler(DataProto) → [写入 TQ] → KVBatchMeta ──RPC──→ Pipeline
  Pipeline → [从 TQ 读全量 batch，本地计算] → 写部分字段到 TQ
  → Reference(@tqbridge, KVBatchMeta) → [TQ 内拉数据、写结果]
  → Actor(@tqbridge, KVBatchMeta) → [TQ 内拉数据、写结果]
  → [本地 advantage 计算] → 写 advantages 等到 TQ
  → Actor/Critic train_step(@tqbridge, KVBatchMeta) → [TQ 拉最终数据训练]
```

**Ray RPC 搬运的从大 tensor 变为轻量 key-handle，减少 Ray object store 压力和序列化耗时。**

---

### 3.5 `@tqbridge` 装饰器工作原理

```python
@tqbridge(dispatch_mode=Dispatch.MEGATRON_COMPUTE)
def compute_log_probs(self, data: DataProto) -> DataProto:
    ...
```

1. **检测入参**：若发现 `BatchMeta` 类型的参数（由 `KVBatchMeta` 转换而来），进入 TQ 路径。
2. **拉取数据**：调用 `tq_client.async_get_data(meta)` 从 TQ 拉回 `TensorDict`，包装为 `DataProto`。
3. **调用原函数**：以正常的 `DataProto` 参数调用原始业务逻辑。
4. **写回结果**：若输出 `DataProto` 有非空 batch，调用 `tq_client.async_put(output.batch, metadata=meta)` 将结果写回 TQ，返回更新后的 `BatchMeta`。
5. **zero-overhead fallback**：若无 `BatchMeta` 参数，装饰器完全透明，不做任何额外操作。

---

### 3.6 TQ 相关配置

| 配置项 | 位置 | 默认 | 说明 |
|--------|------|------|------|
| `transfer_queue.enable` | `RLVRConfig` | `False` | 全局开关，控制是否启用 TQ 路径 |
| `partition_id` | 代码写死 `"train"` | — | TQ partition，隔离不同用途的数据 |

初始化由 `init_tq(pipeline_config.transfer_queue)` 驱动，**pipeline 主进程**和**每个 scheduler 进程**都需独立调用一次（各自在自己的进程空间初始化 TQ client）。

---

*文档生成时间：2026-03-22*
