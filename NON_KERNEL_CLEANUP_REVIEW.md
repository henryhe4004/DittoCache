# 非 sgl-kernel 清理待确认清单

基准：`feature/ditto-tp-pp-graph` / `251df0902`（已推送）。**用户已批准并完成 A01–A04、B01–B09 删除；C/D/E 尚未批准、未执行。** `sgl-kernel/` 整个目录不在范围内。

A、B 类均已处理。C 类必须先合并、验证，不能直接删除。D 类是历史实验记录，建议移出默认源码目录而非丢弃。

[逐文件清单与 SHA256](docs/ditto-open-source-audit/non-kernel-cleanup-candidates.json) 保存本次具体路径、大小、Git 状态和分组；A01–A04、B01–B09 状态已更新为获批并删除；其他项目仍待确认，没有可自动执行的删除命令。

## A：已获批准并删除，4 个文件

| 编号 | 文件 | 为什么列入 | 删除后保留的能力 / 配套修改 |
| --- | --- | --- | --- |
| A01 | `test/ditto/speedup/ditto_server_override.json` | 无仓库代码引用；当前 `ditto_server.sh:110` 内联生成配置，`:178` 将变量传给服务 | 当前服务入口不读取该文件；若你手工引用过这个 JSON，需要先保留参数 |
| A02 | `test/ditto/archive/legacy/export_head_threshold.py` | 已归档的一次性检查片段，写死 `/speedup/...` CSV，仅打印单个 head；无活跃调用 | 当前模型推理、测试不依赖；同步更新 archive 索引和目录说明 |
| A03 | `test/ditto/archive/legacy/serve_llama3_ditto.sh` | 已归档的客户端 sweep，写死 `/data3` 路径；无活跃调用 | 保留 `speedup/ditto_server.sh`、`launch_client.py`；同步更新 archive 索引和目录说明 |
| A04 | `test/ditto/archive/legacy/overlap_script.sh` | 已归档的 GPU7 固定包装；无活跃调用 | 保留 `test/ditto/test_overlap.sh`；同步更新 archive 索引和目录说明 |

A01–A04 的当前副本已删除，原内容可从远程 feature 提交 `251df0902` 找回。`archive/manifest.json` 已将 A02–A04 标记为 `removed_after_user_approval`，归档 README 与目录说明已同步更新。

## B：已获批准并删除，9 个文件

以下文件已按用户批准删除。删除前逐个核对 SHA256，与审核版本一致；此前未发现其他活跃源码/脚本按文件名或静态 Python import 调用。表中保留删除依据与替代入口，原文件可从 Git 提交 `251df0902` 恢复。

所有路径均相对 `test/ditto/`：

| 编号 | 文件 | 判断 | 替代入口 / 影响 |
| --- | --- | --- | --- |
| B01 | `run_q_similarity_awq_gpu1.sh` | 固定 GPU1、模型、数据与 `/auxiliary/` 路径 | 保留 `auxiliary/attn_pattern/profile_heads_cosine.py`，将原参数移到示例 |
| B02 | `run_q_similarity_fp16_gpu2.sh` | 固定 GPU2；与 B01 差异主要为设备及模型 | 与 B01 共用一个文档示例或参数化包装 |
| B03 | `run_infinitebench_offloading_gpu4_7.sh` | 设置一组默认参数后调用通用脚本 | 保留 `test_accuracy_infinitebench.sh`；若常用这组参数，保留 preset 即可 |
| B04 | `run_longbench_v2_4nway_pp2.sh` | 4 worker / PP2 实验包装 | 保留 `run_longbench_v2_nway.sh`，记录 worker 与 TP/PP 参数 |
| B05 | `run_qwen_4gpu_aime24_gpqa_math500_aime25_top03_s1024.sh` | 一次性命令组合；文件名写 top03/s1024，内容却是 TOPK=0.50、SELECTIVE_START_LEN=2048 | 保留其调用的四个准确率入口；移除这个过时组合最直接 |
| B06 | `run_qwen25_14b_1m_offloading_32k.sh` | 32K 模型实验包装；末尾调用 `test_accuracy.sh` | 保留 `test_accuracy.sh`；迁移其特定参数后可删除包装 |
| B07 | `run_when_model_idle.sh` | 等待指定模型进程退出后启动任务 | 你若仍用它排队，建议保留到实验工具目录；它不是推理运行依赖 |
| B08 | `run_when_results_ready.sh` | 等待指定结果文件完成后启动任务 | 与 B07 等待条件不同，不是重复实现；是否保留取决于你的实验流程 |
| B09 | `eval_math.py` | 独立旧评测入口，未被当前评测链调用；与 `math_grader.py` 部分重复，但答案提取和归一化并不完全等价 | 保留 `eval_longbench_infinitebench.py → math_grader.py`；如需旧评分口径，应先归档而不是直接替换 |

A+B 已累计删除 **13 个文件，26,132 bytes（约 25.5 KiB）**。收益主要是减少入口混乱，不是释放磁盘。

## C：建议合并，暂不直接删除，14 个脚本

| 编号 | 文件组 | 冗余证据 | 后续动作 |
| --- | --- | --- | --- |
| C01 | `test/ditto/multi_news_e_nway`、`test/ditto/run_gov_report_nway` | 行相似度 98.72%；仅任务名、恢复结果名、默认 GPU 三处差异 | 共用一个任务参数化 runner；覆盖两种任务后再删旧入口 |
| C02 | `test/ditto/test_accuracy_math500.sh`、`test_accuracy_multinews.sh`、`test_accuracy_infinitebench.sh` | 两两行相似度约 90%–97% | 抽取公共模型启动/预算/路径/输出部分，保留各任务的数据适配；原 B03 包装已删除 |
| C03 | 下列 9 个 `speedup/run_n2n_*` 脚本 | 都包装 `n2n_offloading.py`；部分脚本行相似度约 90%–91%，但新版 TP/PP 和预热处理有差异 | 整理为模型、方法、batch/长度 sweep 参数；比较生成的命令与配置，验证后再移除旧入口 |

C03 的精确范围，均位于 `test/ditto/speedup/`：

```text
run_n2n_llama_fullattn_bsz.sh
run_n2n_llama_fullattn_seqlen.sh
run_n2n_llama_offloading_bsz.sh
run_n2n_llama_offloading_seqlen.sh
run_n2n_qwen_fullattn_bsz.sh
run_n2n_qwen_fullattn_seqlen.sh
run_n2n_qwen_offloading_bsz.sh
run_n2n_qwen_offloading_seqlen.sh
run_n2n_qwen_offloading_perf_from32k.sh
```

`run_n2n_qwen_ablation_bsz.sh` 有单独的消融定义，不在 C03 的删除范围。相似度只说明代码复制较多，不代表不同任务可不经测试直接互换。

## D：建议单独存放的历史结果，20 个已跟踪文件

均位于 `test/ditto/speedup/head-mapping-ab/`：

| 编号 | 目录 | 已跟踪文件数 | 建议 |
| --- | --- | ---: | --- |
| D01 | `perf/` | 9 | 保留一份汇总，原始 benchmark 输出迁到实验产物位置 |
| D02 | `multilen-final/` | 4 | 同上；保留用于论文/复现实验的记录 |
| D03 | `multilen-rerun-20260906/` | 5 | 同上 |
| D04 | `resident-formal/` | 2 | 同上 |

以上 20 个文件合计 29,798 bytes。它们不是默认推理代码，但有人可能手工拿来比较实验；本轮没有授权删除任何结果。若仅批准“从 Git 移除、保留本地”，会按该方式处理，不等同于物理删除。

**不整体处理 `head-mapping-ab/` 或其 `cost-model/`：** 里面混有 head/resident mapping、profile 输入、算法说明、脚本和结果，必须逐项分离。尤其 `qsac-*.json`、`resident-placement/*.json` 及 `candidate*.json` 等不能凭名字当成无用输出。

## E：本地生成物，不计入仓库删减

- E01：`test/ditto/speedup/online_client_results.jsonl` 当前 0 bytes，**未被 Git 跟踪，已忽略**。`launch_client.py:484–485` 创建/清空它，删除后可重新生成；不需要修改代码中的输出路径。
- 其他本地 results/history、虚拟环境和缓存本轮不列入执行范围。已提交的 `test/ditto/results/tp-pp-graph/validation-final.json` 是验证摘要，建议保留。

## 本轮明确保留

- `python/sglang/ditto/`、`python/sglang/srt/models/ditto/` 的运行链及 TP/PP 修改。
- 三个 `run_tp_pp_*.py`、`test_pp_partition.py`、`test_cuda_graph_decode.py`。
- `run_pred.py`、`dataloader.py`、`utils.py`、现行评测器和 `math_grader.py`。
- head mapping 的 cost model / timeline / robust refine 实现与对应单测。
- `auxiliary/` 的输入生成脚本、hash 权重和 attention pattern 输入。
- `config/`：文件名由运行参数动态拼接；先改配置机制才能减少 presets。
- `sgl-model-gateway/`、上游 `examples/`、`benchmark/`、`docs/`、`python/sglang/` 其他模块：它们是上游功能范围，不能仅凭本次未使用而当成冗余。
- `sgl-kernel/`：按你的要求完全排除。

## 如何确认

A、B 已完成。剩余 C/D/E 可以按编号决定，例如：`C 先不动；D 保留；E01 删除`。这只是回复格式示例，不是已获批准的方案。

A、B 类删除前已核对 SHA256，删除后更新了清理记录；A 类涉及的归档索引已更新。C/D/E 保持待确认；若批准 C 类，会先完成合并和适当验证，再移除旧脚本。本轮改动尚未提交或推送。
