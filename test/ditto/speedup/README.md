# Ditto 性能测试指南

这个目录提供离线耗时、在线吞吐、在线延迟和消融实验入口。下面命令均从**仓库根目录**执行；模型、数据、辅助权重必须替换为你实际使用的路径。首次先跑一个配置，确认 JSON 中 `status=ok`、实际输入和输出 token 数正确，再扩大 sweep。

2026-09-21 已完成 Qwen14B 和 Llama8B 的主要流程实测（含 TP2/PP2、Ditto Graph、批量、消融、在线吞吐与延迟）；范围、修复与日志位置见 [实测记录](PERFORMANCE_VALIDATION.md)。

## 1. 选哪个脚本

| 想测什么 | 入口 | 是否需要提前启动服务 | 结果 |
| --- | --- | --- | --- |
| 单配置耗时、TP/PP、Graph 对照 | [`n2n_offloading.py`](n2n_offloading.py) | 不需要，脚本创建 Engine | `--result-json` 指定的 JSON |
| 多组长度 / batch size | `run_n2n_*_seqlen.sh` / `run_n2n_*_bsz.sh` | 不需要 | `LOG_DIR` 内的 `.log`、`.json` |
| 并发请求吞吐 | [`sweep_ruler_throughput.py`](sweep_ruler_throughput.py) | 需要 | `ruler_throughput.csv`、可选图片、请求日志 |
| TTFT / TPOT 随并发数变化 | [`run_latency_benchmark.py`](run_latency_benchmark.py) | 需要 | `<output-dir>/<method>/summary.csv` |
| 自动启动服务并比较延迟 | [`run_latency_experiment.sh`](run_latency_experiment.sh) | 不需要，脚本管理服务 | `RESULT_ROOT` 内分方法结果和服务日志 |
| 传输、预取、复用、Graph、驻留头的累计消融 | [`run_n2n_qwen_ablation_bsz.sh`](run_n2n_qwen_ablation_bsz.sh) | 不需要 | `LOG_DIR` 内分 stage 的 JSON |
| 汇总离线 JSON | [`export_speedup_csv.py`](export_speedup_csv.py) | 不需要 | CSV |

`--batch_size` 是离线批量；`--concurrency` / `--concurrencies` 是在线请求并发，不是同一个参数。
`flash_server.sh` / `full_latency_server.sh` 启动的是 **Ditto full-attention 基线**，不要将名字解释为另一套独立推理引擎。

功能正确性先看 [`../run_tp_pp_graph.py`](../run_tp_pp_graph.py) 和 [TP/PP 验证报告](../../../DITTO_TP_PP.md)。它们不用于比较吞吐量。

## 2. 准备环境和输入

```bash
cd /jhe/sglang-litecache  # 换成你的仓库目录
export PERF_REPO="$PWD"
export PYTHONPATH="$PERF_REPO/python:$PERF_REPO/sgl-kernel/python${PYTHONPATH:+:$PYTHONPATH}"
export PERF_PYTHON="$(command -v python3)"
export PERF_CPUSET="$("$PERF_PYTHON" -c 'import os; print(",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))))')"
export PERF_MODEL="/jhe/Qwen2.5-14B-Instruct-1M"
export PERF_AUX="/jhe/myTransformer/auxiliary"
export PERF_DATA="/path/to/RULER-Qwen2.5-14B-Instruct-1M-8K.jsonl"
export PERF_OUT="$PERF_REPO/test/ditto/results/perf-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$PERF_OUT"
```

需要当前仓库对应的 SGLang、PyTorch/CUDA 与 sgl-kernel 扩展。当前机器的默认 Python 可用；其他机器先完成安装。若仅需重建 Ditto gather/prefetch 扩展，可参考 [`../../../scripts/build_ditto_gather.py`](../../../scripts/build_ditto_gather.py)。GDR 传输还依赖可用的 GDRCopy 环境。

Hash offloading 的辅助数据要匹配模型：

```text
$PERF_AUX/hash_weights/<模型目录名>-256/hash_weight_layer_*.pt
$PERF_AUX/attn_pattern/<模型目录名>/heads_cosine_similarity.csv
$PERF_AUX/attn_pattern/<模型目录名>/k_heads_importance.tsv
$PERF_AUX/attn_pattern/<模型目录名>/q_heads_importance.tsv
```

离线 `--data` 接收 JSONL，每行可包含 `input`、`prompt` 或 `text`。脚本取**第一条有效 prompt**，当 batch size 大于 1 时复制这一条输入，因此这是固定工作负载测试，不是完整数据集吞吐评测。在线 RULER 测试建议使用真实 RULER JSONL。

文件名中的 `8K` 不保证真实 token 数恰好为 8192。离线日志中的 `[DATA] prompt_tokens=...` 和 JSON 的 `input_tokens` 才是实际值。上下文容量要容纳输入和输出；改变 tokenizer 后应重新检查。

## 3. 推荐起点：离线单配置，TP=2、PP=2、Ditto Graph

下面使用四张 GPU、batch=1、执行 128 个 decode 步（加上 prefill 的首 token，共生成 129 tokens）；预热 1 次、测量 3 次。示例采用 16K 配置，给 8K 输入留出生成空间。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 DITTO_TP_ENABLE_CUDA_GRAPH=1 \
"$PERF_PYTHON" test/ditto/speedup/n2n_offloading.py \
  --model "$PERF_MODEL" \
  --config_file test/ditto/config/hata_offloading/Qwen2.5-14B-Instruct-1M-16K.yaml \
  --data "$PERF_DATA" --method offloading --offloading-method hash \
  --aux-data-path "$PERF_AUX/hash_weights/$(basename "$PERF_MODEL")-256" \
  --attn-pattern-path "$PERF_AUX/attn_pattern/$(basename "$PERF_MODEL")" \
  --rbits 256 --topk 0.10 --batch_size 1 \
  --tp-size 2 --pp-size 2 \
  --max_seq_len 16384 --max-total-tokens 16384 --max-running-requests 1 \
  --gpu-memory-budget 4 \
  --num_decode_steps 128 --ignore-eos --warmup 1 --epoch 3 \
  --disable-cuda-graph --ditto-enable-cuda-graph \
  --result-json "$PERF_OUT/offloading-tp2-pp2-graph.json"
```

- **eager 对照**：将 `--ditto-enable-cuda-graph` 改为 `--ditto-disable-cuda-graph`，换一个结果文件名，其余保持一致。
- **full-attention 对照**：使用 `--method flashattn` 和 `config/full_attn/Qwen2.5-14B-Instruct-1M-16K.yaml`；其余 batch、TP/PP、长度和测量次数保持一致。这个离线 `flashattn` 分支使用 SGLang 原生模型，不启用 Ditto，`--ditto-enable-cuda-graph` 对它无效；保留 `--disable-cuda-graph` 时是 eager 基线。它与第 5 节 server shell 的 Ditto full-attention 实现不同，不应直接混作同一组基线。full-attention 不需要 hash 辅助输入，可省略对应参数。
- **单卡 / 其他布局**：修改 `CUDA_VISIBLE_DEVICES`、`--tp-size` 和 `--pp-size`，需要的 GPU 数为 TP×PP。权重和 cache 必须能放入每张 GPU。
- **改模型**：同时更换配置、辅助输入和数据文件；不要给 Llama 使用 Qwen 的配置/权重。

### Graph 开关的区别

| 参数 / 环境变量 | 控制对象 |
| --- | --- |
| `--ditto-enable-cuda-graph` / `--ditto-disable-cuda-graph` | Ditto 内部 Graph |
| `DITTO_TP_ENABLE_CUDA_GRAPH=1` | TP 模式下显式允许 Ditto 内部 Graph |
| `--disable-cuda-graph` / `--enable-cuda-graph` | SGLang 外层 Graph |

已验证的组合是**内部开启、外层关闭**。需要检查实际 capture/replay 时使用 `DITTO_CUDA_GRAPH_DEBUG=1` 或功能验证脚本；正式计时关掉详细调试日志。外层和内部同时开启不在现有验证范围。

### 如何读离线结果

| JSON 字段 | 含义 |
| --- | --- |
| `status` | `ok` 才是完成测量；部分条件可能得到 skip，不能当成性能结果 |
| `input_tokens`、`epoch_completion_tokens`、`epoch_elapsed_s`、`epoch_tokens_per_s` | 实际输入长度、每轮总输出 token 数、每轮耗时、每轮输出吞吐 |
| `avg_elapsed_s` / `overall_tokens_per_s` | 请求测量窗口内的平均总耗时 / 总输出 tokens 除以总耗时，包含 prefill 影响 |
| `avg_prefill_latency_s` | 从调用到首个流式输出的时间，含调度和返回开销 |
| `avg_decode_latency_ms_per_step` / `avg_decode_tokens_per_s` | 基于流式返回时间间隔的 decode 估计；间隔足够多时剔除前 10 个 |
| `avg_internal_*` | 内部 forward 计时；不可用时会警告并退回流式估计，不能一律解释为 GPU kernel 耗时 |
| `runtime_meta` | 实际运行配置；消融配置在 `runtime_meta.ablation_config` |

不要把流式估计、内部 forward 计时和在线服务指标混在同一组 speedup 中。离线 `--num_decode_steps N` 请求 `N+1` 个输出 tokens（包括首 token）；在线 `--max-new-tokens N` / `--output-len N` 请求 N 个。EOS 会导致实际生成量不同，固定工作负载时保留 `--ignore-eos`。

需要分析传输时加 `--record-transfer-stats`；TP/PP 会产生分 rank 记录。统计/详细日志可能影响耗时，正式性能对照要在两组采用相同设置。

## 4. 批量扫描长度和 batch size

Qwen 长度 sweep 的例子：

```bash
MODEL_PATH="$PERF_MODEL" \
DATA_ROOT="$(dirname "$PERF_DATA")" \
CONFIG_ROOT="$PERF_REPO/test/ditto/config/hata_offloading" \
LOG_DIR="$PERF_OUT/seqlen" \
PYTHON_BIN="$PERF_PYTHON" CUDA_DEVICE="0,1,2,3" \
TP_SIZE=2 PP_SIZE=2 CPUSET="" \
SEQ_LIST="8000 16000" BSZ=1 WARMUP=1 EPOCH=3 DECODE_STEPS=128 \
DITTO_CUDA_GRAPH=1 SGLANG_CUDA_GRAPH=0 GPU_MEMORY_BUDGET=4 \
bash test/ditto/speedup/run_n2n_qwen_offloading_seqlen.sh
```

如果辅助数据位于 `/jhe/myTransformer/auxiliary`，可先 `export DITTO_ROOT=/jhe/myTransformer`，使 YAML 的 `../auxiliary/...` 解析到实际位置。

运行前按下面的规则准备数据和配置。该包装从配置读取辅助路径，**不会读取上面自定义的 `PERF_AUX` 变量**；先确认 YAML 中的路径可解析。

| 设置 | 说明 |
| --- | --- |
| `SEQ_LIST="8000 16000"` | 空格分隔；包装按 `seq / 1000` 选择 `8K`、`16K` 文件 |
| `DATA_ROOT` | 本例查找 `RULER-Qwen2.5-14B-Instruct-1M-8K.jsonl` 等 |
| `CONFIG_ROOT` | 本例按长度桶×batch 选择 `<模型>-<桶大小>K.yaml`；batch>1 会需要更大的配置 |
| `LOG_DIR` | 每组 `.log` 和 `.json` 的保存目录 |
| `BSZ` / `BSZ_LIST` | 长度 sweep 用 `BSZ`；batch sweep 用空格分隔的 `BSZ_LIST` |
| `CPUSET` | 传给 `taskset -c` 的 CPU 范围；应符合 `taskset -pc $$` 显示的允许范围 |
| `CUDA_DEVICE` | 包装脚本会用它设置 `CUDA_VISIBLE_DEVICES`；只设置外部 `CUDA_VISIBLE_DEVICES` 可能被覆盖 |

`run_n2n_qwen_offloading_seqlen.sh` 支持 `CPUSET=""` 关闭绑核。部分其他脚本用 `${CPUSET:-默认值}`，空值仍会恢复机器专用默认值，应显式设置合法范围，例如 `CPUSET="$PERF_CPUSET"`。

其余包装入口：

- Qwen：`run_n2n_qwen_{offloading,fullattn}_{bsz,seqlen}.sh`。
- Llama：`run_n2n_llama_{offloading,fullattn}_{bsz,seqlen}.sh`。
- `run_n2n_qwen_offloading_perf_from32k.sh`：较长输入扫描，结束后可自动导出 CSV；`EXPORT_CSV=0` 关闭，`CSV_SUMMARY` 改输出位置。

不同包装的参数并未完全统一：**TP/PP 首选第 3 节的 Python 入口或上述 Qwen offloading 长度入口**，不要假设每个 shell 都支持 `TP_SIZE`、`PP_SIZE` 或 Graph 开关。部分包装遇到文件缺失会 skip，退出码为 0 也应检查结果数量和 `status`。

## 5. 在线服务和吞吐

在终端 A 启动服务。下面先用一张 GPU、最多 4 个请求，显式缩小默认 cache 预算；能否容纳模型取决于显存。

```bash
MODEL_PATH="$PERF_MODEL" DITTO_AUX_ROOT="$PERF_AUX" \
PYTHON_BIN="$PERF_PYTHON" CUDA_DEVICE=0 PORT=30000 \
DITTO_GPU_MEMORY_BUDGET=4 DITTO_MAX_BATCH_SIZE=4 \
DITTO_TARGET_SEQ_LEN=16384 SGLANG_MAX_RUNNING_REQUESTS=4 \
DITTO_ENABLE_CUDA_GRAPH=true \
bash test/ditto/speedup/ditto_server.sh
```

在终端 B 设置第 2 节的环境变量，等待服务就绪后运行：

```bash
curl --noproxy '*' -f http://127.0.0.1:30000/v1/models

"$PERF_PYTHON" test/ditto/speedup/sweep_ruler_throughput.py \
  --server-base-url http://127.0.0.1:30000 \
  --model-path "$PERF_MODEL" --data-file "$PERF_DATA" \
  --ruler-len 8K --concurrencies 1,2,4 \
  --max-seq-len 16384 --max-total-tokens 16384 \
  --max-new-tokens 128 --duration-sec 30 --warmup-requests 4 \
  --output-dir "$PERF_OUT/online-ditto" --no-plot
```

`--concurrencies` 在此处为**逗号分隔**。`--duration-sec 30` 在每个并发档位持续补充请求；省略时按请求数运行。默认固定输出长度，`--no-force-output-len` 允许提前结束。加 `--dry-run` 可先检查将执行的客户端命令，去掉 `--no-plot` 可生成图片（需要 matplotlib）。

看 `ruler_throughput.csv` 的 `output_tok_per_s`、`request_per_s` 和 `latency_*`；`total_tok_per_s` 含 prompt tokens，与输出吞吐不是同一个指标。保留请求原始日志，核对成功/失败数和实际 token 数。

测试 full-attention 时，在终端 A 停止当前服务，再启动 `flash_server.sh`，客户端输出改为另一个目录。Qwen14B 的 16K×4 容量需要 **12 GiB KV cache**，应将 `DITTO_GPU_MEMORY_BUDGET=12`；照搬 offloading 的 4 GiB 会在首个请求时失败。full-attention 的 KV 需求约为 `2 × 层数 × KV heads × head_dim × dtype字节 × 总token容量`，还需给模型权重和工作区留显存。当前两个 server shell **没有暴露 TP_SIZE/PP_SIZE**；不要以为设置环境变量或传入多个 GPU 编号就启用了 TP/PP。

sweep 会将 `--server-base-url` 传到客户端，原始请求记录独立保存在每次输出目录的 `raw_c*.jsonl`；持续模式的总请求数按实际记录统计，存在失败或零成功请求时以非零状态退出。

`launch_client.py` 是单次客户端入口，支持 `--server http://127.0.0.1:30000`、`--concurrency`、`--data-file`、`--log-file` 等；sweep 会调用它。直接调用客户端而不传 `--log-file` 时，默认 `online_client_results.jsonl` 会被重写。

## 6. 在线延迟：TTFT / TPOT

沿用第 5 节已启动的服务，指定实际存在的 ShareGPT JSON 路径：

```bash
"$PERF_PYTHON" test/ditto/speedup/run_latency_benchmark.py \
  --method ditto --base-url http://127.0.0.1:30000 \
  --model-path "$PERF_MODEL" \
  --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --python-bin "$PERF_PYTHON" \
  --concurrencies 1 2 4 --input-len 8192 --output-len 128 \
  --warmup-requests 4 --repeats 3 --output-dir "$PERF_OUT/latency"
```

这里 `--concurrencies` 是**空格分隔**，与吞吐 sweep 不同。`--method ditto|full` 用于结果标签和默认预热设置，**不会替你切换或启动服务器**。full 对照需要先启动 full-attention 服务，再以 `--method full` 测量。

脚本调用 `sglang.bench_serving` 的 `random` 工作负载：每轮请求数等于该轮并发数，指定固定输入/输出长度。它不是持续满载吞吐测试。输出包括 `ditto/raw/`、`ditto/logs/`、`ditto/records.jsonl` 和 `ditto/summary.csv`，重点看 `mean_ttft_ms`、`mean_tpot_ms` 与跨轮标准差。

已有合法结果会被复用。更换模型、Graph、TP/PP 或服务设置时使用新输出目录，避免误复用另一配置的结果。

自动化脚本 `run_latency_experiment.sh` 可以依次启动 full/ditto 服务、等 GPU 空闲、测试、关闭服务并重试。主要参数是 `RUN_METHODS="full ditto"`、`CONCURRENCIES="1 2 4"`、`INPUT_LEN`、`OUTPUT_LEN`、`REPEATS`、`RESULT_ROOT`、`CUDA_DEVICE`、`PORT`。
**当前自动化包装没有透传 `--dataset-path`，且子 benchmark 的 Python 默认是 `/opt/conda/bin/python3`**；换机器或数据位置时，优先使用上面显式传参的手动两终端方式。自动化的 full/ditto 默认最大请求数也不同，公平对比时显式统一 `DITTO_MAX_BATCH_SIZE` 和 `SGLANG_MAX_RUNNING_REQUESTS`。

## 7. 累计消融

可直接使用第 3 节的 Python 命令，加 `--ablation-stage b0_memcpy` 等参数并分别保存结果。要比较 Graph stage 时，去掉第 3 节显式的 `--ditto-enable-cuda-graph`，让 stage 决定开关；外层仍保留 `--disable-cuda-graph`。

| Stage | 传输 | 获取 KV 的方式 | 复用 | 阈值 | Ditto Graph | 驻留头 |
| --- | --- | --- | --- | --- | --- | --- |
| `b0_memcpy` | CUDA memcpy | 同层按需 | 总是 gather | 固定 0.8 | 关 | 无 |
| `b1_gdr` | GDR | 同层按需 | 总是 gather | 固定 0.8 | 关 | 无 |
| `b2_prefetch` | GDR | 跨层预取 | 总是 gather | 固定 0.8 | 关 | 无 |
| `b3_qsac_fixed` | GDR | 跨层预取 | QSAC | 固定 0.8 | 关 | 无 |
| `b4_cudagraph` | GDR | 跨层预取 | QSAC | 固定 0.8 | 开 | 无 |
| `b5_adaptive` | GDR | 跨层预取 | QSAC | 按 head 从离线 profile 导出 | 开 | 无 |
| `b6_resident` | GDR | 跨层预取 | QSAC | 按 head 从离线 profile 导出 | 开 | profile 选择 |

这是**累计增加功能**的对照；profile-adaptive 不是在线每 token 更新阈值。full-attention 基线单独用 `--method flashattn`，不传 `--ablation-stage fullattn`。

Qwen 的 batch sweep 示例（单卡）：

```bash
MODEL_PATH="$PERF_MODEL" DATA_ROOT="$(dirname "$PERF_DATA")" \
ATTN_PATTERN_PATH="$PERF_AUX/attn_pattern/$(basename "$PERF_MODEL")" \
PYTHON_BIN="$PERF_PYTHON" CUDA_DEVICE=0 \
CPUSET="$PERF_CPUSET" \
STAGES="fullattn b0_memcpy b1_gdr" BSZ_LIST="1 2" SEQ_LIST="8000" \
WARMUP=1 EPOCH=3 DECODE_STEPS=128 GPU_MEMORY_BUDGET=4 \
LOG_DIR="$PERF_OUT/ablation" \
bash test/ditto/speedup/run_n2n_qwen_ablation_bsz.sh
```

包装中的 hash 权重路径来自配置，不会读取 `PERF_AUX`；需要先修正配置中的路径，或设置 `DITTO_ROOT` 为包含 `auxiliary/` 的目录。`RECORD_TRANSFER_STATS` 默认 1。包装目前没有传入 `--ignore-eos`，需严格固定生成长度时用 Python 入口显式传该选项。

## 8. 汇总、分析与排查

```bash
"$PERF_PYTHON" test/ditto/speedup/export_speedup_csv.py \
  --input-dir "$PERF_OUT" --output-csv "$PERF_OUT/summary.csv"
```

将 `--input-dir` 指向同一次离线实验目录，避免混入在线结果或旧模型数据。`export_speedup_csv.py` 默认输入为 `logs-perf-from32k/`；上面显式指定输出到 `test/ditto/results/`。

- `parse_decode_step_times.py`：分析 decode 日志；`plot_serving_gen_throughput_8k.py`：绘制已有在线结果。
- [`../analyze_overlap_stats.py`](../analyze_overlap_stats.py)、`analyze_recall.py`：分析已有 overlap/recall 记录，不能替代耗时测试。参数以各自 `--help` 为准。
- `build_tp_head_mapping_*.py`、`run_robust_mapping_experiment.sh`：head mapping 优化实验，先完成普通配置测试再使用；它们需要 profile 输入。

| 现象 | 优先检查 |
| --- | --- |
| 找不到数据 / 配置，或只有 skip | `DATA_ROOT`、模型 basename、长度桶×batch 对应文件是否存在 |
| 找不到辅助权重 / profile | Hash 位数、层数、模型是否匹配；明确覆盖 `--aux-data-path` / `--attn-pattern-path` |
| `taskset` 报错 | 显式设置容器允许的 CPU 范围，或使用不绑核的 Python 入口 |
| OOM | 先降低 batch/并发、序列长度或 cache 预算；多卡配置核对 TP×PP 和实际传参 |
| Graph 实际没启用 | 区分内外层开关；TP 需 `DITTO_TP_ENABLE_CUDA_GRAPH=1`，通过功能脚本核验 capture/replay |
| 吞吐变化异常 | 检查 EOS、实际输入长度、成功请求数、是否开了调试/统计、是否有其他 GPU 工作负载 |
| 延迟测试没有重新测量 | 该目录命中了 resume；更换输出目录 |

这份文档说明现有脚本的用法和限制，不包含新的性能结论。A/B 清理移除的旧包装不再作为入口；本文引用的运行脚本仍保留。
