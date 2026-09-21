# 历史归档（2026-09-20）

本目录保留原始文件，没有删除或重写历史实验数据。`manifest.json` 记录原路径、
现路径、归档理由、字节数和每个文件的 SHA256。共迁移 304 个文件，12,908,418 bytes。

- `legacy/export_head_threshold.py`：一次性打印某个 head 阈值的片段，硬编码失效的 `/speedup/...` 输入，不是导出 CLI。
- `legacy/serve_llama3_ditto.sh`：实际是客户端 sweep；写死 `/data3` 模型和机器路径，不是启动服务器。
- `legacy/overlap_script.sh`：写死 GPU7 和模型路径的简短包装；日常使用 `../test_overlap.sh`。
- `history/latency_results/`：2026-08-30 延迟实验，新的输出使用 `../results/latency/`。
- `history/results-ruler-qwen14b-20260902/`：2026-09-02 的 PP 调试和 benchmark 结果。

`legacy/` 中的脚本按原样保留，只作历史参考；相对路径和绝对路径没有为归档位置改写。
如需恢复，按 manifest 中的 source/destination 将文件移回原路径；先检查原路径没有新文件，
不要覆盖。`history/` 是本机保留的产物目录，已加入 `.gitignore`。
