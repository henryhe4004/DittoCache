# 开源前的依赖与冗余审计（2026-09-21）

执行进展：清理清单 A01–A04、B01–B09 已经用户批准并删除，见
[非 kernel 清理记录](NON_KERNEL_CLEANUP_REVIEW.md)。下文保留清理前的审计依据。

本次重点是减少维护负担和发布噪声，不仅是腾出磁盘空间。默认按“保留完整 SGLang fork，精简 Ditto 实验层”评估；若改成只发布 LiteCache，需要额外裁剪上游功能及验证安装链。

**本轮只新增审计脚本和清单，没有删除、移动或改写运行代码。** 上次的历史归档记录仍在 [DIRECTORY_AUDIT.md](DIRECTORY_AUDIT.md)。

## 证据与判定方法

- [逐文件 CSV](docs/ditto-open-source-audit/inventory.csv)：Ditto cache、模型适配、测试脚本、配置及 KVLib 自有源码；分别记录导入它的文件数、它导入的本地文件数、代码中的文件名引用、文档引用、直接入口和机器路径线索。
- [引用证据 JSON](docs/ditto-open-source-audit/evidence.json)：具体引用文件、行号、20 组覆盖定义的位置、脚本相似度。
- [第三方裁剪候选](docs/ditto-open-source-audit/vendor-trim-candidates.txt)：逐文件列出未落在当前 CMake include 目录中的附带工程，保留 LICENSE/NOTICE/COPYING 前缀的文件与目录及 README.md。这是候选列表，不是删除脚本。
- [重新扫描脚本](scripts/audit_ditto_dependencies.py)：`python3 scripts/audit_ditto_dependencies.py`。只使用 Python 标准库和 Git，不导入待审计模块、不运行模型、不删除文件。

本轮扫描 3925 个自有文本文件，输出 213 个重点文件的明细；Python 解析失败为 0。扫描使用 Git 已跟踪文件和未忽略的新文件。引用源排除历史归档、结果、head-mapping 实验产物和第三方源码；第三方另外按 CMake 的 include 路径审查。**静态零引用只表示未发现仓库内调用，不保证没有用户直接运行、动态加载或外部 API 使用。** 同名文件的文件名引用标记为歧义，不能直接算实际调用。脚本相似度是原始行序列相似度，不是行为等价证明。条件分支内的 import 也计数。

## A. 优先移除或排除发布的明确候选

| 对象 | 证据 | 建议 | 置信度/验证 |
| --- | --- | --- | --- |
| `sgl-kernel/csrc/kvlib/kvlib_bindings.cc` | 当前 `sgl-kernel/CMakeLists.txt:375` 只编译 `csrc/kvlib_bindings.cc`；内层旧文件还引用仓库中缺失的 `tl_operator.h`，内容与实际编译版本有差异 | 删除内层旧副本，保留外层实际构建版本 | 高；依据当前 CMake，清理后做干净构建 |
| `sgl-kernel/csrc/kvlib/cuda_utils.cuh` | 没有 include；只定义 `CUDA_CALL`，自有 KVLib 源码无该宏调用 | 可删除 | 高；不要连带删 `cp_async.cuh`，后者被 `ham_dist.cu` 和 `prefetch.cu` include |
| `sgl-kernel/python/sgl_kernel/kvlib.py` 的前一批同名函数 | 20 个顶层函数之后被无条件重新定义，前一批共 272 行；无 decorator | 删除被覆盖的旧定义，保留最终定义及对外 API | 高；**不能直接删文件前半段**，其中 `create_cpu_gather_engine_v3` 和 `__all__` 仍需保留/整理；随后验证导出符号、gather 加载和 Graph 回归 |
| `test/ditto/speedup/ditto_server_override.json` | 零 import/文件名代码引用；`ditto_server.sh` 在脚本中生成 `JSON_MODEL_OVERRIDE`，没有读取它；内容写死 `/auxiliary/` 和实验预算 | 删除孤立快照，或改成正式的参数模板并让 launcher 显式使用 | 高，针对当前入口；用户手工 `--json-model-override-args` 的用法不在静态证据内 |
| `sgl-kernel/log` | 已跟踪的 39 KiB 构建日志，记录 `/tmp/.../build`；不是构建输入 | 从源码发布中移除 | 高 |
| `sgl-kernel/_deps/json-src/` | 已跟踪 53 文件、约 1.79 MiB；仓库 CMake/源码没有指向该目录的引用，旧日志使用的是临时 build 下另一个 json-src | 排除提交的下载缓存 | 中高；在空 build 目录验证依赖仍能正常获取/构建 |
| `test/ditto/speedup/online_client_results.jsonl` | 空文件；client/sweep 默认输出目标，运行时创建/追加 | 删除仓库中的空输出占位、忽略新产物；保留代码里的输出路径 | 高 |

没有发现 Ditto 自有 `.py`/`.sh` 整文件字节级重复；主要冗余是**函数覆盖、脚本复制和数据/工程边界混杂**。

## B. 依赖少、适合移出默认入口的文件

下列候选在扫描中均没有来自其他文件的 import 或文件名代码引用。它们仍可能是人工直接运行的 CLI；建议从默认源码发布中排除或放到独立的 `experiments/legacy/`，不是自动认定功能无效。

| 文件（均在 `test/ditto/` 下） | 具体原因 | 保留替代入口/处理 |
| --- | --- | --- |
| `run_q_similarity_awq_gpu1.sh`、`run_q_similarity_fp16_gpu2.sh` | 固定 GPU1/2，固定 `/auxiliary/`、`/models/`、`/datasets/`，差异主要是模型和设备 | 保留 `auxiliary/attn_pattern/profile_heads_cosine.py`，把这些运行参数放进文档示例 |
| `run_infinitebench_offloading_gpu4_7.sh` | 为一组设备和实验设置的包装 | 当前直接调用的 `test_accuracy_infinitebench.sh`（多 worker 调度另用 `run_infinitebench_nway.sh`） |
| `run_longbench_v2_4nway_pp2.sh` | 固定 4 worker / PP2 的包装 | 通用 `run_longbench_v2_nway.sh` |
| `run_qwen_4gpu_aime24_gpqa_math500_aime25_top03_s1024.sh` | 多个参数固定的一次性实验组合；文件名 top03_s1024，但实际命令是 TOPK=0.50、SELECTIVE_START_LEN=2048 | `test_accuracy.sh` 与任务选择参数 |
| `run_qwen25_14b_1m_offloading_32k.sh` | 固定模型和 32K 实验配置 | 当前直接调用的 `test_accuracy.sh` / `run_pred.py` |
| `run_when_model_idle.sh`、`run_when_results_ready.sh` | 本地实验排队辅助，不是推理或评测算法；两者是不同等待条件，**不是相同实现** | 如团队仍使用，集中放 `experiments/tools/`；否则不作为开源核心入口 |
| `eval_math.py` | 零仓库调用，独立旧 CLI；`--input` 默认 `/preds/...`；与 `math_grader.py` 有 19 个同名函数，但只有 3 个 AST 完全一致 | 优先归档；当前评测链使用 `eval_longbench_infinitebench.py → math_grader.py`。不同答案抽取/归一化语义须用样例比较后才能合并 |

上次已经移到 `test/ditto/archive/legacy/` 的 `export_head_threshold.py`、`serve_llama3_ditto.sh`、`overlap_script.sh` 可以留在本地历史包，不必再放进默认开源目录。

## C. 有用但重复，应该合并的脚本

| 组 | 证据 | 建议的收敛方式 |
| --- | --- | --- |
| `multi_news_e_nway` 与 `run_gov_report_nway` | **98.72%** 行相似；diff 只有 `TARGET_TASK`、`BASE_SEED_TAG` 和默认 GPU 列表三处 | 一个支持任务/恢复文件/设备参数的 runner，保留两份参数示例；当前无扩展名文件也统一命名 |
| `test_accuracy_math500.sh` 与 `test_accuracy_multinews.sh` | **96.95%** 行相似 | 共用模型启动、预算、路径、输出逻辑；任务和数据适配独立 |
| `test_accuracy_infinitebench.sh` 与上述两份 | 分别 **91.40% / 90.37%** | 与上一组合并公共 launcher；不要丢失 InfiniteBench 的数据解析 |
| `speedup/run_n2n_llama_fullattn_bsz.sh`、`run_n2n_llama_offloading_bsz.sh`、`run_n2n_qwen_fullattn_bsz.sh` | 部分两两 **90.14%**；共同调用 `n2n_offloading.py` | 用模型、方法和 batch sweep 参数表达差异 |
| `speedup/run_n2n_llama_fullattn_seqlen.sh` 与 `run_n2n_llama_offloading_seqlen.sh` | **91.30%** | 与 batch sweep 共用一个 `--sweep batch|seqlen` 入口 |
| 其余 `run_n2n_qwen_*`、`*_perf_from32k.sh` | 方法/模型/序列长度的包装较多，但新版 TP/PP、预热及消融逻辑并不完全相同 | 先建立参数表再合并；保留 `run_n2n_qwen_ablation_bsz.sh` 的消融定义，不能用行数较短的旧脚本覆盖它 |
| `config/` 64 个已跟踪 JSON/YAML | 19 JSON、20 full-attention YAML、25 offloading YAML；模型/长度/预算枚举多，JSON/YAML 两套兼容读取并存 | 长期改成模型模板 + 方法模板 + CLI overrides，保留少量能直接复现的 presets |

配置文件名由 `test_accuracy.sh:272–278` 根据模型、方法和长度动态拼接。CSV 中某个配置是零引用，**不等于可以删除**；要先改配置生成/选择逻辑。

`test_accuracy_aime24.sh`、`aime25.sh`、`gpqa.sh`、`mmlu_pro.sh` 这类很短的任务别名，维护成本小：可以保留，也可以只在文档给出等价参数。优先去除大段复制，而不是为了减少文件数删除所有便利入口。

## D. 最大的文件数量来源：第三方完整工程

`sgl-kernel/csrc/kvlib/3rdparty/` 已跟踪 1,665 个文件。当前 CMake 只加入：

- `raft/cpp/include`
- `rmm/include`
- `spdlog/include`

这三者没有被作为独立工程 `add_subdirectory`。这些 include 目录之外、保留 LICENSE/NOTICE/COPYING 前缀的元信息及 README.md 后，有 **872 个候选文件，合计约 4.20 MiB**：RAFT 680，RMM 131，spdlog 61。常见内容是 `python/`、`tests/`、`benchmarks/`、`docs/`、`.github/`、`ci/`、`conda/`、独立项目构建脚本。

这是缩减默认源码目录的高收益项。建议保留当前需要的完整 include 子树、来源/版本信息和许可文件，再验证干净构建。`raft/cpp/include/raft/thirdparty/` 里的依赖嵌套先整体保留，不按目录名继续删。此轮没有编译器级头文件闭包分析，也没有在裁剪副本上重建，故这 **872 个是待验证候选，不是已证明全部安全可删**。

`sgl-kernel/_deps/json-src/` 的 53 文件另计。不能由此推断主仓库 `3rdparty/` 或所有外部构建依赖都可删除。

## E. 数据与产物应拆开，不应整目录删除

| 路径 | 建议 |
| --- | --- |
| `test/ditto/results/`、`archive/history/` | 本地诊断记录；发布保留一份简短验证报告，详细日志作为附加产物 |
| `speedup/head-mapping-ab/perf/`、`multilen-final/`、`multilen-rerun-20260906/`、`resident-formal/` | profile/benchmark 历史结果，可移到 artifacts；其中有些被成本模型实验用作输入，移动时同步修改实验参数 |
| `speedup/head-mapping-ab/qsac-*.json`、`resident-placement/*.json`、优化算法生成的 mapping JSON | 实际可用的映射/驻留配置，提取代表性版本到 `configs/head_mapping/`；不要跟结果一起删除 |
| `test/ditto/auxiliary/hash_weights/` 的 `.pt` | 80 个已跟踪权重文件，约 40 MiB，是算法输入；可作为独立下载包/发布资产，但必须保留获取方式、模型对应关系与校验值 |
| `test/ditto/auxiliary/attn_pattern/` 的统计文件 | 推理需要的输入；保留或提供可靠的生成/下载途径 |
| `.venv-py313/`、`sgl-kernel/build/`、Python/pytest/ruff 缓存 | 不随源码发布；已经忽略的大目录不是 Git 文件数量的主要问题，详情见旧审计 |
| native `.so` | 当前机器运行依赖，但属于构建产物；源码开源需要可复现构建或配套 wheel，不能简单删掉当前机器所用文件后宣称运行不受影响 |

## F. 不能因为低引用或名称相似就删

- `python/sglang/ditto/` 的 cache 文件均有静态导入路径。`kvcache_hash.py` 是 offloading/hash 路径，`kvcache_offloading_hash.py` 是另一条 duohead/hash 路径。它们并非两份可互换副本。
- Loki、InfiniGen、Quest 在 `ditto.py`、包导出和 duohead 模型中引用。删文件前必须同时收缩注册、配置和公开选项；“本次只测 hash”不代表其他后端死代码。
- `sglang/srt/models/registry.py` 动态扫描模型包。上游模型的入度低不能作为删模型依据。
- `build_tp_head_mapping_cost_model.py`、`timeline.py`、`robust_refine.py` 各有独立算法和测试；后两者有相互/公共依赖，不能只留一个文件名看起来最通用的版本。
- `run_pred.py`、`dataloader.py`、`utils.py`、`eval_longbench_infinitebench.py`、`math_grader.py` 是完整评测链的组成部分。
- `auxiliary/` 下 build_dataset、learn_hash_weights、profile_heads_cosine 及其模型适配是准备算法输入的入口。它们通常不被推理程序 import，依然有复现价值。
- `run_tp_pp_graph.py`、`run_tp_pp_smoke.py`、`run_tp_pp_batch_smoke.py` 和 `test_*.py` 是 CLI/pytest 入口。零代码入度属于正常情况，应保留。
- 完整 fork 下保留 `python/sglang/srt/`、`sgl-kernel/`、许可证及上游功能；`sgl-model-gateway/`、`benchmark/`、`examples/`、`docs/` 是范围选择，不是已证实的废代码。

## 建议的开源结构与落地顺序

保留上游目录布局，让本项目新增内容集中且容易找到：

```text
python/sglang/ditto/              cache 与 TP head 映射
python/sglang/srt/models/ditto/   模型适配
sgl-kernel/                      实际构建源码 + 精简后 vendor headers
scripts/ditto/                   通用服务、benchmark、辅助输入生成入口
configs/ditto/                   少量模板、presets 和 head mapping 示例
test/ditto/                      单测及 TP/PP/Graph 功能验证
docs/ditto/                      安装、复现、算法和验证说明
experiments/                     可选评测/消融，不混入运行包
```

1. 先处理 A 类旧副本、覆盖定义、孤立快照和输出文件。这一层对功能面影响最小。
2. 将 B 类机器实验搬出默认入口，合并 C 类公共 runner；对比旧/新 runner 生成的命令和配置，再跑代表性用例。
3. 用独立裁剪副本验证 D 类 vendor 清理；原生扩展干净构建成功后，跑已有 40 项单测和 Ditto Graph 对照。
4. 拆出 E 类大数据/历史结果，保证一个新用户仍可获取必需输入并复现实验。
5. 最后重写项目 README 的 quickstart；`python/sglang/ditto/__init__.py` 仍写着“未接入默认 runtime”，需同步更新。新测试的辅助输入默认路径 `/jhe/myTransformer/auxiliary` 也应改成文档明确的可配置路径。仓库当前只跟踪各模型部分 attn_pattern 统计，需核对全套输入是否能由公开生成脚本产出。内部交接 `DITTO_TP_HANDOFF.md` 可移到开发文档，保留 `DITTO_TP_PP.md` 中经过验证的用户说明。

审计时的 TP/PP 修复、实验代码和本报告已一并保存到 `feature/ditto-tp-pp-graph`。实际清理时应基于这个 feature 版本做独立发布副本/分支，避免从旧 `TP_support` 导出后遗漏改动。本报告没有执行后续裁剪或承诺裁剪版已经通过测试。
