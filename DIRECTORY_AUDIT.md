# 目录整理记录（2026-09-20）

范围：整个 `/jhe/sglang-litecache` 的目录体积与运行依赖，重点检查 Ditto 的源码、
测试、benchmark 和历史结果。判断依据是实际导入、脚本调用、配置引用、失效路径和
文件内容；没有把“未发现内部引用”直接当成可删除，也没有按修改时间判定脚本过期。

## 已整理

| 原位置 | 当前处理 | 原因 |
| --- | --- | --- |
| `test/ditto/speedup/export_head_threshold.py` | 移入 `test/ditto/archive/legacy/` | 写死 `/speedup/...` 输入，仅打印 L0H3 阈值；没有命令行参数和仓库内调用者 |
| `test/ditto/speedup/serve_llama3_ditto.sh` | 移入 `archive/legacy/` | 名称像服务端，实际是写死机器路径的客户端调用片段 |
| `test/ditto/overlap_script.sh` | 移入 `archive/legacy/` | 固定 GPU7 和模型的旧包装，已有可配置的 `test_overlap.sh` |
| `test/ditto/speedup/latency_results/` | 移入 `archive/history/` | 2026-08-30 的旧输出；两个延迟入口的新输出默认改到 `test/ditto/results/latency/` |
| `test/ditto/speedup/results-ruler-qwen14b-20260902/` | 移入 `archive/history/` | 历史 PP 诊断/benchmark 结果，无源码消费路径 |

上述归档共 **304 文件、12.31 MiB**；2026-09-21 再次逐文件核对 SHA256 和字节数，全部通过。没有删除原始数据。
索引与恢复位置见 [`test/ditto/archive/manifest.json`](test/ditto/archive/manifest.json)。
历史结果中记录的旧绝对路径保留原样。

新增统一入口说明 [`test/ditto/README.md`](test/ditto/README.md)，脚本和运行结果分开。
Graph 回归、此前临时保存的批量回归和 gather 编译都已保留为仓库脚本。

同时修正：

- `n2n_offloading.py` 重复注册 `--pp-size`，导致 argparse 启动失败。
- `speedup/README.md` 中过时的“仅 batch=1”和 `/speedup` 工作目录说明。
- `run_robust_mapping_experiment.sh` 默认使用断链虚拟环境的问题；优先有效旧环境，否则回退当前 `python3`。

## 可以清理/重建，但本次保留的较大目录

| 位置 | 扫描文件大小 | 判断与处理 |
| --- | ---: | --- |
| `.venv-py313/` | 10,030.45 MiB | Python 链接指向不存在的 `/root/.local/share/uv/python/...`，当前不能启动；仍有大量依赖和历史构建引用，保留。确认不再修复该环境后可整目录移除/重建 |
| `sgl-kernel/build/` | 1,210.36 MiB | 构建产物和依赖缓存，当前 CMakeCache 仍引用旧 Python；不是在线推理入口，但完整重建可能需要下载依赖，保留 |
| `.ruff_cache/`、`__pycache__/`、`.pytest_cache/` | 可再生成 | 无业务数据，属清理候选；测试运行也会重新生成 |

这些大小按普通文件逻辑大小统计，不能当成删除后必然回收的物理磁盘空间。
归档是移动文件，不释放磁盘空间。

## 不能当作无用文件删除

- `sgl-kernel/python/sgl_kernel/kvlib_cpu_gather.so`、`flash_ops.abi3.so`、`spatial_ops.abi3.so` 和架构 `common_ops`：运行依赖。
  **A40 实际走 `sm100/common_ops` 兼容分支**（见 `load_utils.py`），不能因目录名是 sm100 就删。
- `speedup/head-mapping-ab/`（11.30 MiB）：包括仍被 handoff、优化脚本引用的 head 映射和 resident-head JSON，必须保留。
- `build_tp_head_mapping_cost_model.py`、`timeline.py`、`robust_refine.py`：不同算法且相互依赖；对应单测也应保留。
- `python/sglang/ditto/` 与 `python/sglang/srt/models/ditto/`：分别负责 cache 和模型适配，名称相近不代表重复。
  `modeling_*_offloading_duohead.py`、Loki/InfiniGen/Quest 分支有注册/导入，不是闲置备份。
- `test/ditto/config/`、`auxiliary/`：模型配置、统计和哈希权重，实验输入，不是可丢弃输出。
- `test/ditto/speedup/online_client_results.jsonl`：目前 0 bytes，但客户端默认追加目标；可以再生成，不是废弃源码。
- `docs/`、`examples/`、`benchmark/`、`sgl-model-gateway/`：部分不是此次 TP/PP 的执行路径，但属于 SGLang 的其他功能；本次不裁剪产品范围。

Ditto 源码/测试中的 `.py`、`.sh` 文件按 SHA256 检查，**没有内容完全相同的重复文件**。
带 `/models/...` 默认值的 sweep 通常支持 `MODEL_PATH` 覆盖，属于需配置后运行的脚本，
不是失效即无用。详细脚本分类见 `test/ditto/README.md`。
