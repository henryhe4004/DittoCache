# DittoCache TP 负载均衡当前结论

更新时间：2026-09-16

## 一句话结论

DittoCache 的 TP head mapping 已在 Qwen2.5-14B-1M、TP2、256K、BSZ=1、TOPK=0.10 上观察到正向端到端收益；它不是跨模型通用的静态优化。相同的 profiling、候选筛选和 fresh paired A/B 流程在 Llama-3-8B 上没有找到可接受的收益，原因是 Llama 的稳态可搬运流量和可重排的 rank 不均衡显著更小。

当前可以支持的工程决策是：**将 TP mapping 作为由 profile 门控的优化，而不是默认启用的静态策略。**

## 目标与边界

DittoCache 的 offloading 路径中，每个 TP rank 只处理一部分 KV heads。`DITTO_TP_KV_HEAD_ORDER_FILE` 决定全局 KV heads 到两个 rank 的分组。优化目标不是改变注意力语义或减少总 KV，而是：

1. 基于实际 decode profile，把高传输或高关键路径的 heads 分配到更合适的 TP rank。
2. 降低两个 rank 中较慢一侧的 byte-critical transfer 时间。
3. 在不改变模型、TP 度、top-k、resident 配置、CUDA Graph 设置和输入数据的前提下，改善 decode latency 与 TPS。

该优化只对**持续存在的、可重排的 rank 传输不均衡**有作用。它不能消除 cold-start 时的总搬运量，也不能自动改善由计算、cache 管理或 CPU 调度主导的延迟。

## 统一候选算法：RSCTRO

当前使用 `robust split-constrained trust-region refinement`（RSCTRO）：

1. 在目标序列长度上单独采集 linear mapping 的双 rank 传输 profile；不从其他长度外推。
2. 丢弃前 10 个 decode step，仅用后续稳态样本。
3. 对每层枚举等大小的 TP head partition，并以 byte-critical transfer 代价评分。
4. 将 profile 切为 `odd/even` 与 `first-half/second-half` 四个子集。
5. 仅接受四个子集都改善的层；按最弱 split 的改善排序，最多改动 5 层。
6. 候选必须经过 fresh-process 的 candidate/linear paired A/B；未通过则回退 linear。

实现：`test/ditto/speedup/build_tp_head_mapping_robust_refine.py`。

## 实验环境

| 项目 | 设置 |
| --- | --- |
| 硬件 | 2 x NVIDIA A40，GPU 0/1，NV4，NUMA 0 |
| 并行 | TP2，PP1，BSZ=1 |
| 稀疏设置 | TOPK=0.10，Ditto CUDA Graph 开，SGLang CUDA Graph 关 |
| decode 规范 | 固定 50 steps；Llama 使用 `ignore_eos=true` |
| 测量原则 | profile 与性能测量分离；性能 A/B 使用 fresh process |

## 结果汇总

| 模型与长度 | 传输 profile | RSCTRO 候选 | Fresh paired A/B 结论 |
| --- | --- | --- | --- |
| Qwen2.5-14B-1M，256K | rank 0/1 平均 H2D：349.24 / 391.62 MB/step | 5 层，预测节省 461.26 us | 接受：decode latency `51.865 -> 49.090 ms`，`-5.35%`；TPS `19.291 -> 20.371`，`+5.60%` |
| Llama-3-8B，256K | rank 0/1 平均 H2D：52.51 / 55.07 MB/step | 默认阈值下 0 层；放宽后仅 layer 8，预测节省 7.27 us | 拒绝：latency `26.627 -> 26.946 ms`，`+1.20%`；TPS `37.971 -> 37.592`，`-1.00%` |
| Llama-3-8B，512K | 后 41 个 decode step 中仅 9 个存在约 26.17 MB H2D，其余为 0 | 默认阈值下 0 层；放宽后 layer 14、16，合计预测节省 29.08 us | 未进行 mapping A/B：理论绝对收益低于 decode 测量噪声，不将其作为性能候选 |

## Qwen：有收益的证据

Qwen 的 256K 候选来自独立的 256K linear profile，不是 128K 外推。profile 在每个 TP rank 均记录了 51 个 decode step；优化器丢弃前 10 个后使用 41 个样本。

RSCTRO 从已有多长度 mapping 出发，选择 layer `21, 44, 19, 43, 20`。这 5 层均在 odd/even 与前/后半段改善：

| Layer | Full-profile 改善 | 最弱 split 改善 | 代理节省 |
| ---: | ---: | ---: | ---: |
| 21 | 26.86% | 20.45% | 102.93 us |
| 44 | 14.14% | 11.73% | 153.77 us |
| 19 | 10.68% | 7.00% | 65.29 us |
| 43 | 8.93% | 4.53% | 88.48 us |
| 20 | 6.55% | 4.06% | 50.79 us |

性能测量中，candidate 与 linear 均为 fresh process、相同模型/输入/TP/top-k/resident 设置。3 个 candidate epoch 为 `48.999, 49.264, 49.006 ms`；3 个 linear epoch 为 `52.718, 50.174, 52.704 ms`。candidate 的每个 epoch 都快于最快的 linear epoch；prefill 均值仅相差 `-0.0052 s`，故观测到的改善落在 decode 路径，而不是 prefill 差异。

**证据限制：**每边只有 3 个 epoch。保守的 df=2、95% latency difference interval 为 `[-6.434, +0.882] ms`，仍跨 0。因此应表述为“方向一致且端到端可观测的收益”，而不是已完成统计显著性证明。

原始结果：

- `cost-model/timeline-v2/RESULTS.md`
- `cost-model/timeline-v2/perf-256K-robust-top5-paired-20260907/comparison.json`

## Llama：没有收益的证据

### 256K：同流程端到端拒绝

Llama-3-8B 只有 32 层和 8 个全局 KV heads；其可重排空间与绝对 H2D 流量均明显小于 Qwen。默认 RSCTRO 配置（最弱 split 至少改善 4%，最多改动 5 层）选择 0 层。

为检验边界情形，将阈值放宽到 0% 且只允许 1 层。候选仅修改 layer 8，其 byte-critical proxy 的绝对节省只有 `7.27 us`。它在 even steps 与 second-half 的改善均为 0%，不满足默认稳健门槛。

该候选仍进行了 fresh paired A/B：

| Variant | Mean decode latency | Mean TPS |
| --- | ---: | ---: |
| Layer-8 candidate | 26.946 ms | 37.592 |
| Linear | 26.627 ms | 37.971 |

候选慢 `0.319 ms`（`+1.20%` latency），TPS 下降 `1.00%`；两组 epoch 区间重叠，接受规则失败。因此 Llama-256K 的 mapping 结论是 **reject / linear rollback**。

### 512K：缺少稳态 mapping 信号

512K 使用独立 profile，输入为 511,922 tokens。正式 epoch 的 decode 为 `56.407 ms/step`、`17.728 tok/s`。profile 显示搬运主要集中在 cold-start：第 1、2 个 decode step 的单 rank H2D 分别可达约 3.25 GB、1.83 GB；跳过前 10 步后，41 个稳态样本中只有 9 个 step 有约 26.17 MB H2D，其余为 0。

因此静态 head mapping 不能减少 cold-start 的总传输，只能重新安排少量稳态搬运发生在哪个 rank。默认 RSCTRO 无候选；0% 边界候选仅含 layer 14、16，合计预测节省 `29.08 us`，相对于 56.407 ms decode latency 约为 `0.052%`。它没有足够的信号进入昂贵的 mapping A/B。

### 512K extra-resident 探索：不作为 mapping 证据

曾尝试将空余 GPU 预算用于每层额外常驻一个 local KV head。该实验不是纯 mapping 对照：candidate 有 35 个 resident heads（约 10.07 GB cache），自动 resident 对照有 13 个（约 5.12 GB cache）。结果为 candidate `97.514 ms/step, 10.255 tok/s`，对照 `68.434 ms/step, 14.613 tok/s`。

该结果只说明“在此 Llama 512K 配置中，更激进的 resident placement 没有带来性能”，不能用于归因 static head mapping，也不能替代同 resident placement 的 mapping A/B。

原始结果：

- `cost-model/robust-unified/llama3-8b-256K-20260907/RESULTS.md`
- `cost-model/robust-unified/llama3-8b-256K-20260907/comparison.json`
- `cost-model/robust-unified/llama3-8b-512K-20260912-r2/profile-linear/`
- `cost-model/robust-unified/llama3-8b-512K-joint-resident-20260912/`

## 为什么 Qwen 与 Llama 的结果不同

| 条件 | Qwen 256K | Llama 256K/512K | 对 mapping 的含义 |
| --- | --- | --- | --- |
| 稳态 H2D 规模 | 数百 MB/step | 256K 约 53-55 MB/step；512K 大多为 0 | Qwen 有足够绝对节省空间，Llama 没有 |
| 可变层数 | 5 个层通过四 split 门槛 | 默认 0 层；放宽后最多 1-2 层 | Qwen 的不均衡持续，Llama 的信号稀疏 |
| 候选代理节省 | 461.26 us | 7.27 us / 29.08 us | Llama 代理收益低于端到端噪声 |
| 端到端验证 | 方向一致为正 | 256K 为负并拒绝 | 不允许将 Qwen mapping 直接迁移至 Llama |

不是“模型不准确”这一单一原因。模型只负责在已观察到的 profile 上排序候选；Llama 的核心限制是目标工作负载里可由静态 rank 分组优化的稳态传输本身极少。将低信号候选拒绝，是该方法设计的一部分。

## 当前交付状态

- 已支持：针对目标模型、目标长度、目标 resident policy 的 profile-gated TP mapping。
- 已支持：双 rank profile、四 split 稳健筛选、fresh paired A/B 和 linear rollback。
- Qwen-256K：保留 robust top-5 mapping 作为候选默认值，但拓扑、BSZ、top-k、resident policy 或模型变化后必须重新验证。
- Llama-256K/512K：保持 linear mapping；不启用 Qwen 派生 mapping。
- 内核兼容性：为 Llama TP2 的 16 local query heads 增加了 `HEAD_SWITCH` 的 `NumHead=16` 特化，并已做直接 CUDA smoke test。

## 下一步：如何把结论提升为更强证据

1. 对 Qwen-256K 进行至少 10 次交替顺序的 paired trial（AB/BA），报告 bootstrap CI 或配对置信区间；当前 3 epoch 的正向结果不足以声称显著。
2. 在 Qwen 上补 4K、8K、16K、32K、64K、128K、256K 的独立 profile 与相同 A/B，形成“可优化稳态 H2D vs speedup”的相关性图。
3. 对 Llama 若继续投入，不应复用静态 mapping；应做 cold-start/首 token 专用的自适应 prefetch，并将 step 1-2 与 step >= 11 分开计分。
4. 若要评估 Llama resident placement，必须固定 resident 数量和显存预算，仅改变 head identity；当前 35-vs-13 resident 的探索性对照不能用于归因。
