# 历史归档

最初归档 304 个文件、12,908,418 bytes。经用户确认，清理清单 A02–A04
对应的 3 个旧脚本已删除；当前保留 **301 个历史结果文件、12,907,191 bytes**。

`manifest.json` 保留原路径、归档路径、理由、字节数和 SHA256：

- `status=retained`：文件仍在本地归档中。
- `status=removed_after_user_approval`：旧脚本已删除，保留记录供追溯；
  可从 `recovery_commit` 指定的 Git 提交取回，不能再从当前归档目录恢复。

已删除的旧脚本为 `legacy/export_head_threshold.py`、
`legacy/serve_llama3_ditto.sh`、`legacy/overlap_script.sh`。
当前保留的目录为：

- `history/latency_results/`：2026-08-30 的延迟实验。
- `history/results-ruler-qwen14b-20260902/`：2026-09-02 的 PP 调试和 benchmark 结果。

历史结果内容没有改写，目录已加入 `.gitignore`；新测试输出位于 `../results/`。
