#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent


COLUMNS = [
    "file",
    "status",
    "reason",
    "method",
    "variant",
    "offloading_method",
    "batch_size",
    "max_seq_len",
    "seq_k",
    "input_tokens",
    "warmup",
    "epoch",
    "avg_elapsed_s",
    "p50_elapsed_s",
    "avg_tokens_per_s",
    "overall_tokens_per_s",
    "avg_prefill_latency_s",
    "avg_decode_latency_ms_per_step",
    "avg_decode_tokens_per_s",
    "ditto_enabled",
    "sglang_cuda_graph_enabled",
    "ditto_cuda_graph_enabled",
    "transfer_stats_enabled",
    "transfer_epochs",
    "transfer_total_selected_tokens",
    "transfer_total_recalled_tokens",
    "transfer_total_hit_tokens",
    "transfer_overall_hit_rate",
    "topk_hint",
    "data",
    "config_file",
    "json_path",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate Ditto speedup JSON outputs into one CSV. "
            "By default it scans logs-perf-from32k/*.json."
        )
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=THIS_DIR / "logs-perf-from32k",
        help="Directory that stores benchmark JSON files.",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="*.json",
        help="Glob pattern under --input-dir (default: *.json).",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Use rglob instead of glob when scanning --input-dir.",
    )
    p.add_argument(
        "--input-json",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit JSON file list. If set, --input-dir/--glob are ignored.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: <input-dir>/summary.csv",
    )
    return p.parse_args()


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _bool_to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return 1 if value else 0
    return None


def _extract_topk_hint(path: Path) -> float | None:
    m = re.search(r"topk([0-9]+(?:\.[0-9]+)?)", path.stem)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _extract_transfer_summary(raw: dict[str, Any]) -> dict[str, Any]:
    summary = raw.get("transfer_stats_summary")
    if isinstance(summary, dict):
        return summary

    epoch_stats = raw.get("epoch_decode_transfer_stats")
    if not isinstance(epoch_stats, list) or not epoch_stats:
        return {}

    total_selected = 0
    total_recalled = 0
    total_hit = 0
    epochs = 0
    for item in epoch_stats:
        if not isinstance(item, dict):
            continue
        epochs += 1
        total_selected += int(item.get("total_selected_tokens", 0))
        total_recalled += int(item.get("total_recalled_tokens", 0))
        total_hit += int(item.get("total_hit_tokens", 0))

    if epochs == 0:
        return {}
    return {
        "epochs": epochs,
        "total_selected_tokens": total_selected,
        "total_recalled_tokens": total_recalled,
        "total_hit_tokens": total_hit,
        "overall_hit_rate": (total_hit / total_selected) if total_selected > 0 else 0.0,
    }


def _build_row(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {k: None for k in COLUMNS}
    row["file"] = path.name
    row["json_path"] = str(path.resolve())
    row["topk_hint"] = _extract_topk_hint(path)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        row["status"] = "invalid_json"
        row["reason"] = str(e)
        return row

    if not isinstance(raw, dict):
        row["status"] = "invalid_json_root"
        row["reason"] = f"expected object, got {type(raw).__name__}"
        return row

    runtime = raw.get("runtime_meta")
    if not isinstance(runtime, dict):
        runtime = {}
    transfer = _extract_transfer_summary(raw)

    seq_len = _to_int(raw.get("max_seq_len"))

    row.update(
        {
            "status": raw.get("status"),
            "reason": raw.get("reason"),
            "method": raw.get("method"),
            "variant": runtime.get("variant"),
            "offloading_method": runtime.get("offloading_method"),
            "batch_size": _to_int(raw.get("batch_size")),
            "max_seq_len": seq_len,
            "seq_k": int(seq_len / 1000) if seq_len else None,
            "input_tokens": _to_int(raw.get("input_tokens")),
            "warmup": _to_int(raw.get("warmup")),
            "epoch": _to_int(raw.get("epoch")),
            "avg_elapsed_s": _to_float(raw.get("avg_elapsed_s")),
            "p50_elapsed_s": _to_float(raw.get("p50_elapsed_s")),
            "avg_tokens_per_s": _to_float(raw.get("avg_tokens_per_s")),
            "overall_tokens_per_s": _to_float(raw.get("overall_tokens_per_s")),
            "avg_prefill_latency_s": _to_float(raw.get("avg_prefill_latency_s")),
            "avg_decode_latency_ms_per_step": _to_float(
                raw.get("avg_decode_latency_ms_per_step")
            ),
            "avg_decode_tokens_per_s": _to_float(raw.get("avg_decode_tokens_per_s")),
            "ditto_enabled": _bool_to_int(runtime.get("ditto_enabled")),
            "sglang_cuda_graph_enabled": _bool_to_int(
                runtime.get("sglang_cuda_graph_enabled")
            ),
            "ditto_cuda_graph_enabled": _bool_to_int(
                runtime.get("ditto_cuda_graph_enabled")
            ),
            "transfer_stats_enabled": _bool_to_int(runtime.get("transfer_stats_enabled")),
            "transfer_epochs": _to_int(transfer.get("epochs")),
            "transfer_total_selected_tokens": _to_int(
                transfer.get("total_selected_tokens")
            ),
            "transfer_total_recalled_tokens": _to_int(
                transfer.get("total_recalled_tokens")
            ),
            "transfer_total_hit_tokens": _to_int(transfer.get("total_hit_tokens")),
            "transfer_overall_hit_rate": _to_float(transfer.get("overall_hit_rate")),
            "data": raw.get("data"),
            "config_file": raw.get("config_file"),
        }
    )
    return row


def _discover_jsons(args: argparse.Namespace) -> list[Path]:
    if args.input_json:
        result = [p.resolve() for p in args.input_json]
        return [p for p in result if p.is_file()]

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        return []

    if args.recursive:
        paths = list(input_dir.rglob(args.glob))
    else:
        paths = list(input_dir.glob(args.glob))
    return sorted([p for p in paths if p.is_file()])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    json_files = _discover_jsons(args)

    if args.output_csv is not None:
        output_csv = args.output_csv.resolve()
    else:
        output_base = args.input_dir.resolve()
        output_csv = output_base / "summary.csv"

    rows = [_build_row(p) for p in json_files]
    rows.sort(
        key=lambda r: (
            r.get("max_seq_len") is None,
            r.get("max_seq_len") or 0,
            r.get("file") or "",
        )
    )
    write_csv(output_csv, rows)

    print(f"[csv] input_json_count={len(json_files)}")
    print(f"[csv] output={output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
