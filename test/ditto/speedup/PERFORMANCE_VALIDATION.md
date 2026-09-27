# 性能指南实测记录（2026-09-21）

README 的主要测试流程已在本机的 **Qwen2.5-14B-Instruct-1M 和 Llama-3-8B-Instruct-Gradient-1048k** 两种模型上跑通。下方初始记录为 Qwen，Llama 的补测范围见文末。测试期间发现并修复了吞吐脚本和离线批量计数问题，也修正了指南中的配置说明。**这是一轮缩短参数的流程验证，不是完整性能矩阵或正式加速比评测。**

## 环境与测试范围

- 模型：`/jhe/Qwen2.5-14B-Instruct-1M`；GPU：8×NVIDIA A40 48 GiB。
- Python 3.13.9、PyTorch 2.9.1+cu128；使用当前仓库的 `python/` 和 `sgl-kernel/python/`。
- RULER：`/jhe/myTransformer/speedup/data/` 下的 Qwen 8K / 16K JSONL；离线实际 prompt 分别为 7710 / 15864 tokens。
- 辅助输入：`/jhe/myTransformer/auxiliary`；ShareGPT：`/jhe/ShareGPT.json`。自动延迟包装使用其默认 `/jhe/vllm/ShareGPT_V3_unfiltered_cleaned_split.json`。
- 离线预热 1 次、测量 1 次、32 个 decode 步（每请求共 33 个输出 tokens）；在线输出 32 tokens，吞吐各并发点持续提交 5 秒；手动延迟预热 1 次、重复 1 次。
- 不同 GPU 上并行验证了部分流程，因此此次耗时不用于正式性能比较。

| 流程 | 实际配置 | 结果 |
| --- | --- | --- |
| 四卡离线 | TP=2、PP=2，offloading Graph / eager、SGLang 原生 full-attention | 3/3 通过；修复计数后复测，每轮输出 33 tokens |
| 长度扫描 | TP=2、PP=2、Ditto Graph，8K / 16K | 2/2 通过 |
| 累计消融 | batch=1，fullattn、b0–b6 | 8/8 生成 `status=ok`；b4–b6 配置中内部 Graph 开启 |
| 批量消融 | batch=2，fullattn、b0、b1 | 3/3 通过；修复后每轮总输出 66 tokens |
| 在线吞吐 | Ditto offloading / Ditto full-attention，并发 1/2/4 | 6/6 通过；每种方法合计 8/8 请求成功，零失败，每请求输出 32 tokens |
| 手动在线延迟 | 两种服务，并发 1/2/4，输入 8192 tokens | 6/6 通过；核对实际完成请求数及输入/输出 tokens |
| 自动延迟包装 | full→ditto，并发 1/2，输入 1024 tokens，重复 1 次 | 4/4 通过；完成服务启动、切换、测试和退出 |
| CSV 导出 | 显式传入上述 16 个离线 JSON | 输出 16 行 |
| 新增 CPU 回归 | 非默认端口、持续模式/失败计数、批量流式计数和计时 | 6/6 通过 |

在线最终复测中，两种服务均使用 `DITTO_GPU_MEMORY_BUDGET=12`、`DITTO_MAX_BATCH_SIZE=4`、`DITTO_TARGET_SEQ_LEN=16384`，内部 Graph 开启、外层 Graph 关闭。服务日志中可见 `ditto cuda graph: True`。四卡离线 full-attention 是 SGLang 原生 eager 基线，不能将它称为 Ditto Graph full-attention。

未遍历 README 列出的全部 Llama/Qwen 包装、32K 以上长上下文、所有 batch/TP/PP 组合、默认 128 步×3 次重复或可选绘图/分析脚本。本轮也不重新证明生成质量；性能流程成功与准确率评测是两件事。

## 实测发现与修复

1. `sweep_ruler_throughput.py` 之前没有把 `--server-base-url` 传给客户端，非默认端口会连接错误。现在正确传递服务根 URL，客户端自行追加 `/generate`。
2. 持续吞吐模式用 `total=0` 表示不限请求数，但旧汇总把它当作实际总数，导致失败计数错误。现在使用实际记录数；零成功、存在失败或子进程异常均使 sweep 非零退出，仍保留 CSV 和日志。
3. sweep 原始请求记录改为输出目录内独立的 `raw_c*.jsonl`，不再依赖共享的 `online_client_results.jsonl`；相对输出路径会先转为绝对路径。
4. 离线 batch>1 的流式响应按请求交错返回。旧脚本只用最后一条响应计数，也可能在较早请求结束时停止计时。现在按请求 ID/index 汇总累计 token 数，计时到最后一个请求的最后 token；流式间隔按请求分别计算。JSON 新增 `epoch_completion_tokens` 和 `num_decode_steps`。
5. Qwen 长度扫描增加 `GPU_MEMORY_BUDGET` 透传，并去掉重复的 `--pp-size`。
6. 指南补充 `DITTO_ROOT` 辅助路径解析、离线 N 个 decode 步对应 N+1 输出 tokens、离线/在线 full-attention 实现区别。Qwen14B 的 full-attention 16K×4 容量需要 12 GiB KV cache，照搬 offloading 示例的 4 GiB 会在首个请求时报错。

未修改 `sgl-kernel`，未提交或推送这些改动。

## 结果与复现入口

本机结果根目录：

```text
/jhe/sglang-litecache/test/ditto/results/perf-guide-validation-20260921/
```

主要文件（结果目录按仓库规则被 Git 忽略，仍保留在本机）：

- `validation.json`：逐类核验结果和关键源文件 SHA256。
- `summary.csv`：16 组离线汇总。
- `final-offline/`：修复后的四卡离线日志、JSON；`seqlen/`：长度扫描。
- `ablation/`、`ablation-batch2/`：消融结果；`ablation-batch2/` 是计数修复后重跑的结果。
- `final-online/`：最终两种服务日志、客户端命令、吞吐 CSV、请求原始 JSONL、延迟结果。最终在线结果以此目录为准。
- `auto-latency/`：自动服务管理和延迟结果。
- `initial-failures/`：首次失败和旧计数结果，保留用于排查，不纳入最终汇总。
- `regression.log`：6 项 CPU 回归结果。

复现命令和驱动脚本已保留：`offline-final.sh`、`seqlen.sh`、`ablation.sh`、`ablation-batch2.sh`、`online-final.py`、`auto-latency.sh`。它们记录的是本机路径、GPU 和端口；再次运行会覆盖对应结果，应先修改输出目录，并确保所用 GPU/端口空闲。正常性能测试入口及其可配置参数见 [README](README.md)。

核验已有结果和重新导出 CSV：

```bash
cd /jhe/sglang-litecache
python3 test/ditto/results/perf-guide-validation-20260921/verify_results.py
```

运行新增回归测试：

```bash
PYTHONPATH=python:sgl-kernel/python python3 -m pytest \
  test/ditto/speedup/test_n2n_stream_accounting.py \
  test/ditto/speedup/test_sweep_ruler_throughput.py -q
```


## Llama 补测与双模型覆盖

同日追加了 Llama-3-8B-Instruct-Gradient-1048k 验证；使用 `/jhe/Llama-3-8B-Instruct-Gradient-1048k`、对应 RULER JSONL 和 `/jhe/myTransformer/auxiliary` 下的 Llama 专用权重及 profile。8K / 16K 文件的离线实际输入为 **7926 / 15845 tokens**。沿用预热 1 次、测量 1 次、32 个 decode 步；在线输出 32 tokens。

| 流程 | Qwen14B | Llama8B |
| --- | --- | --- |
| TP=2、PP=2，Graph / eager / 原生 full-attention | 3/3 | 3/3 |
| b0–b6 单卡累计消融及 full-attention | 8/8 | 8/8 |
| batch=2 | full、b0、b1：3/3 | full、b0、b1、默认 offloading Graph：4/4 |
| 额外长度验证 | 四卡 8K/16K sweep：2/2 | 单卡 16K Ditto Graph：1/1；8K 已在上述测试覆盖 |
| 在线吞吐：full/ditto × 并发 1/2/4 | 6/6，合计 16/16 请求成功 | 6/6，合计 22/22 请求成功 |
| 手动延迟：full/ditto × 并发 1/2/4 | 6/6 | 6/6 |
| 自动延迟：full/ditto × 并发 1/2 | 4/4 | 4/4 |
| 离线 CSV | 16 行 | 16 行 |

Llama 四卡离线使用 16K 容量配置；额外单卡消融、批量和 16K 输入使用 32K 容量配置，给输入及生成留足空间。其在线服务使用 16K×4 容量、12 GiB KV 预算，内部 Graph 开启、外层关闭。真实请求长度和配置容量不是同一概念；这些结果用于验证流程，不能直接作为两模型的性能排名。

Llama 的 batch=1 每轮输出 33 tokens，batch=2 每轮总输出 66 tokens；在线每个成功请求输出 32 tokens。核验脚本同时检查状态、实际 token 数、Graph 配置、在线日志中 Graph 实际启用及成功/失败请求数。已校验关键测试脚本 SHA256 与前面的 Qwen 验证一致，本次无需额外修改模型或 kernel 代码。所有测试服务已经退出，8 张 GPU 均已释放。

Llama 日志与复现驱动保存在：

```text
/jhe/sglang-litecache/test/ditto/results/perf-guide-validation-20260921/llama/
├── validation.json       # 逐项核验结果
├── summary.csv           # 16 组离线汇总
├── offline.sh            # 四卡三组对照
├── extra.py              # 单卡消融、批量和 16K 输入
├── online.py             # 两种服务的吞吐及延迟
├── auto-latency.sh        # 自动服务管理流程
├── verify_results.py     # 核验并导出 CSV
├── offline/              # 四卡日志与 JSON
├── extra/                # 单卡日志、命令与 JSON
├── online/               # 服务日志、客户端原始结果、吞吐及延迟
└── auto-latency/          # 自动延迟结果
```

```bash
cd /jhe/sglang-litecache
python3 test/ditto/results/perf-guide-validation-20260921/llama/verify_results.py
```

复现驱动记录了本机路径和 GPU 分配，再次运行前请调整输出目录，避免覆盖本次证据。Llama 单卡扩展测试通过通用 Python 入口执行，没有将 Qwen 专用 shell 包装假装成 Llama 包装。双模型验证仍未覆盖所有长上下文、并行布局、完整重复次数或生成质量评测。
