# Ditto 脚本入口

从仓库根目录运行。新测试产物统一放在 `test/ditto/results/`，历史结果在
`archive/history/`。原 `archive/legacy/` 中的 3 个旧脚本已按用户批准删除；
迁移位置、校验值、移除状态和 Git 恢复版本见 `archive/manifest.json`。

性能测量的运行命令、参数与结果解读见 [性能测试指南](speedup/README.md)，
包括离线 TP/PP Graph、在线吞吐、TTFT/TPOT 和累计消融。

## TP + PP 与 Ditto Graph

`run_tp_pp_graph.py` 测试 **Ditto 内部 CUDA Graph**。它设置
`DITTO_TP_ENABLE_CUDA_GRAPH=1`，传入 `--ditto-enable-cuda-graph`，并禁用外层
SGLang CUDA Graph。不能只看配置开关判断通过：脚本会检查每个 rank 的 capture、
replay 日志、实际 offloading 传输、两次请求的一致性，以及与同一 TP/PP 配置的 eager
输出逐 token 一致性，同时检查缓存长度逐步递增及 top-k 预算与 eager 一致。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test/ditto/run_tp_pp_graph.py \
  --model-path /jhe/Llama-3-8B-Instruct-Gradient-1048k \
  --output-dir test/ditto/results/my-graph-run
```

默认测试 `1x1,2x1,1x2,2x2`（TP×PP），每种模式生成 32 tokens、重复两次。
只测融合配置用 `--cases 2x2`；加每层 head 重排用 `--permute-heads`。
Llama 的预取 k 从 0 增长边界用 `--cases 1x1,2x1 --prompt-repetitions 46`。
辅助文件默认从 `/jhe/myTransformer/auxiliary` 查找，可用 `--aux-root` 或
`--aux-data-path`、`--attn-pattern-path` 覆盖。每次使用新的输出目录，避免覆盖证据。
每个 case 保存 `command.json`、`run.log`、`output.json`、各 rank 的 transfer 文件；
`summary.json` 只记录已通过的 case，**应以脚本退出码 0 和最终 PASS 行判断整轮通过**。

其他验证入口：

| 文件 | 用途 |
| --- | --- |
| `run_tp_pp_smoke.py` | Full attention 的单卡、TP、PP、TP+PP、head 重排五种配置对照；eager 模式 |
| `run_tp_pp_batch_smoke.py` | 两个不同长度请求、4-token 分块 prefill、两次调用；支持 `--reference` 对照单卡结果 |
| `test_pp_partition.py` | PP 分层、索引、head 映射、跨层预取和分块 prefill 单元测试 |
| `test_cuda_graph_decode.py` | 用小型 CUDA 运算验证 warmup/capture/replay 每一步输出，覆盖 fullattn/offloading、中间/末阶段、动态 top-k 缓冲区和 host 长度 |
| `../../scripts/build_ditto_gather.py` | 从本仓库源码重建 gather/prefetch 扩展；`--install` 会先备份再替换二进制 |

```bash
PYTHONPATH=python:sgl-kernel/python python3 -m pytest \
  test/ditto/test_pp_partition.py test/ditto/test_cuda_graph_decode.py \
  test/ditto/speedup/test_build_tp_head_mapping_*.py -q

CUDA_VISIBLE_DEVICES=0 python3 test/ditto/run_tp_pp_batch_smoke.py \
  --model-path /jhe/Qwen2.5-7B-Instruct-1M --tp-size 1 --pp-size 1 \
  --output-file test/ditto/results/batch/reference.json
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test/ditto/run_tp_pp_batch_smoke.py \
  --model-path /jhe/Qwen2.5-7B-Instruct-1M --tp-size 2 --pp-size 2 \
  --reference test/ditto/results/batch/reference.json \
  --output-file test/ditto/results/batch/tp2-pp2.json

python3 scripts/build_ditto_gather.py --cuda-arch 8.6 --install
```

2026-09-21 的最终结果见 [`DITTO_TP_PP.md`](../../DITTO_TP_PP.md) 和
[`validation-final.json`](results/tp-pp-graph/validation-final.json)：40 项回归通过，
Llama 四布局、短输入 k=0 增长边界、Qwen14B TP2/PP2 head 重排对照全部通过。

## 保留的实验脚本

这些入口不是重复文件；按用途选择，不需要全部运行。

| 类别 | 入口 | 依赖/说明 |
| --- | --- | --- |
| 离线耗时 | `speedup/n2n_offloading.py`、`run_n2n_*.sh` | 多组模型/长度/batch sweep；需模型、配置、数据；旧 `/models/...` 默认值可用 `MODEL_PATH` 覆盖 |
| 在线服务/请求 | `speedup/ditto_server.sh`、`flash_server.sh`、`launch_client.py`、`sweep_ruler_throughput.py` | 服务启动与客户端分开；需要额外请求数据 |
| 延迟测试 | `speedup/*latency*` | 默认输出改为 `results/latency/` |
| 消融 | `speedup/run_n2n_qwen_ablation_bsz.sh` | 比较传输、预取、复用、Graph 和 resident heads 的累计阶段 |
| TP head 优化 | `speedup/build_qsac_tp_head_mapping.py`、`build_tp_head_mapping_*.py` | 成本模型、时间线模型、局部优化算法；有相互导入，不能只留一个 |
| 优化实验 | `speedup/run_robust_mapping_experiment.sh`、`summarize_robust_mapping_experiment.py` | 自动采样→构建映射→对比；虚拟环境失效时回退到当前 `python3` |
| 结果分析 | `analyze_overlap_stats.py`、`analyze_recall.py`、`speedup/export_speedup_csv.py`、`parse_decode_step_times.py`、`plot_serving_gen_throughput_8k.py` | 读取已有实验结果，独立 CLI 无内部引用也不代表无用 |
| 准确率 | `run_pred.py`、`test_accuracy*.sh`、`run_*bench*.sh`、`run_math500_nway.sh`、`eval_*.py`、`summarize_accuracy.py` | 评测和数据处理链；本次只做功能验证，未重跑完整准确率评测 |
| 数据/配置 | `dataloader.py`、`utils.py`、`math_grader.py`、`config/`、`auxiliary/` | 被评测或 benchmark 调用，保留 |
| 映射/驻留输入 | `speedup/head-mapping-ab/` | 同时含可直接使用的 JSON 配置、profile 和实验记录；不能整目录删除 |

目录审计与可清理项详见仓库根目录 `DIRECTORY_AUDIT.md`。
