from __future__ import annotations

import csv
from copy import deepcopy
import json
import os
from pathlib import Path
import statistics
from typing import Any


LAYER_CSV_COLUMNS = [
    "step",
    "attn_tp_rank",
    "attn_tp_size",
    "seq_len",
    "prefetch_k",
    "layer_idx",
    "prefetch_heads",
    "offloaded_heads",
    "prefetch_h2d_bytes",
    "offload_d2h_bytes",
    "total_bytes",
    "selected_tokens",
    "recalled_tokens",
    "hit_tokens",
    "hit_rate",
    "overlap_mean_precision",
    "overlap_mean_recall",
    "overlap_mean_jaccard",
    "overlap_union_precision",
    "overlap_union_recall",
    "overlap_union_jaccard",
    "overlap_active_heads",
    "overlap_prefetch_k",
    "overlap_union_real_count",
    "overlap_union_prefetch_count",
    "overlap_union_intersection_count",
    "overlap_sum_real_tokens",
    "overlap_sum_prefetch_tokens",
    "overlap_sum_hit_tokens",
    "overlap_recall_tokens",
    "overlap_prefetch_h2d_bytes_est",
    "overlap_recall_h2d_bytes_est",
    "step_h2d_bytes",
    "step_d2h_bytes",
    "step_total_bytes",
    "step_selected_tokens",
    "step_recalled_tokens",
    "step_hit_tokens",
    "step_hit_rate",
    "hit_source",
]

HEAD_CSV_COLUMNS = [
    "step",
    "attn_tp_rank",
    "attn_tp_size",
    "seq_len",
    "prefetch_k",
    "layer_idx",
    "active_head_slot",
    "active_bh_index",
    "corr",
    "overlap_head_recall",
    "overlap_head_precision",
    "overlap_head_jaccard",
    "overlap_head_real_count",
    "overlap_head_prefetch_count",
    "overlap_head_intersection_count",
]


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return default
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return default
        return float(value)
    except Exception:
        return default


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _resolve_csv_path(json_path_value: str | None) -> Path | None:
    csv_path_env = os.environ.get("DITTO_TRANSFER_STATS_CSV_FILE")
    if csv_path_env:
        return _with_tp_rank_suffix(Path(csv_path_env))
    if not json_path_value:
        return None

    json_path = Path(json_path_value)
    if json_path.suffix.lower() in {".json", ".jsonl"}:
        return _with_tp_rank_suffix(json_path.with_suffix(".csv"))
    return _with_tp_rank_suffix(Path(str(json_path) + ".csv"))


def _resolve_head_csv_path(layer_csv_path: Path | None) -> Path | None:
    head_csv_env = os.environ.get("DITTO_TRANSFER_STATS_PER_HEAD_CSV_FILE")
    if head_csv_env:
        return _with_tp_rank_suffix(Path(head_csv_env))
    if layer_csv_path is None:
        return None
    return layer_csv_path.with_name(f"{layer_csv_path.stem}_per_head.csv")


def _detect_tp_rank_suffix() -> tuple[int, int]:
    try:
        from sglang.srt.layers.dp_attention import (  # pylint: disable=import-outside-toplevel
            get_attention_tp_rank,
            get_attention_tp_size,
        )

        return int(get_attention_tp_rank()), int(get_attention_tp_size())
    except Exception:
        return 0, 1


def _with_tp_rank_suffix(path: Path) -> Path:
    rank, size = _detect_tp_rank_suffix()
    suffix = f".tp{rank:02d}" if size > 1 else ""
    try:
        from sglang.srt.distributed import get_pp_group

        pp_group = get_pp_group()
        if pp_group.world_size > 1:
            suffix = f".pp{pp_group.rank_in_group:02d}" + suffix
    except (AssertionError, RuntimeError):
        pass
    if not suffix:
        return path
    if path.stem.endswith(suffix):
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _empty_payload() -> dict[str, Any]:
    return {
        "enabled": False,
        "decode_steps": 0,
        "total_bytes": 0,
        "total_h2d_bytes": 0,
        "total_d2h_bytes": 0,
        "total_selected_tokens": 0,
        "total_recalled_tokens": 0,
        "total_hit_tokens": 0,
        "overall_hit_rate": 0.0,
        "avg_bytes_per_step": 0.0,
        "p50_bytes_per_step": 0.0,
        "max_bytes_per_step": 0,
        "min_bytes_per_step": 0,
        "per_step_total_bytes": [],
        "per_step_h2d_bytes": [],
        "per_step_d2h_bytes": [],
        "per_step_selected_tokens": [],
        "per_step_recalled_tokens": [],
        "per_step_hit_tokens": [],
        "per_step_hit_rates": [],
        "overlap_steps": 0,
        "avg_overlap_mean_recall": 0.0,
        "avg_overlap_mean_precision": 0.0,
        "avg_overlap_mean_jaccard": 0.0,
        "avg_overlap_union_recall": 0.0,
        "avg_overlap_union_precision": 0.0,
        "avg_overlap_union_jaccard": 0.0,
        "per_step_overlap_mean_recall": [],
        "per_step_overlap_mean_precision": [],
        "per_step_overlap_mean_jaccard": [],
        "per_step_overlap_union_recall": [],
        "per_step_overlap_union_precision": [],
        "per_step_overlap_union_jaccard": [],
        "steps": [],
    }


def _build_layer_rows(steps: list[dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in steps:
        layer_h2d = _safe_list(step.get("layer_h2d_bytes"))
        layer_d2h = _safe_list(step.get("layer_d2h_bytes"))
        layer_prefetch_heads = _safe_list(step.get("layer_prefetch_heads"))
        layer_offloaded_heads = _safe_list(step.get("layer_offloaded_heads"))
        layer_selected_tokens = _safe_list(step.get("layer_selected_tokens"))
        layer_recalled_tokens = _safe_list(step.get("layer_recalled_tokens"))
        layer_hit_tokens = _safe_list(step.get("layer_hit_tokens"))
        layer_hit_rates = _safe_list(step.get("layer_hit_rates"))
        layer_overlap_sum_real = _safe_list(step.get("layer_overlap_sum_real_tokens"))
        layer_overlap_sum_prefetch = _safe_list(step.get("layer_overlap_sum_prefetch_tokens"))
        layer_overlap_sum_hit = _safe_list(step.get("layer_overlap_sum_hit_tokens"))
        layer_overlap_recall_tokens = _safe_list(step.get("layer_overlap_recall_tokens"))
        layer_overlap_prefetch_h2d_est = _safe_list(step.get("layer_overlap_prefetch_h2d_bytes_est"))
        layer_overlap_recall_h2d_est = _safe_list(step.get("layer_overlap_recall_h2d_bytes_est"))

        overlap_layers: dict[int, dict[str, Any]] = {}
        overlap = step.get("overlap")
        if isinstance(overlap, dict):
            for layer_metrics in _safe_list(overlap.get("layers")):
                if not isinstance(layer_metrics, dict):
                    continue
                layer_idx = _to_int(layer_metrics.get("layer_idx"), -1)
                if layer_idx >= 0:
                    overlap_layers[layer_idx] = layer_metrics

        max_layer_count = max(
            len(layer_h2d),
            len(layer_d2h),
            len(layer_prefetch_heads),
            len(layer_offloaded_heads),
            len(layer_selected_tokens),
            len(layer_recalled_tokens),
            len(layer_hit_tokens),
            len(layer_hit_rates),
            len(layer_overlap_sum_real),
            len(layer_overlap_sum_prefetch),
            len(layer_overlap_sum_hit),
            len(layer_overlap_recall_tokens),
            len(layer_overlap_prefetch_h2d_est),
            len(layer_overlap_recall_h2d_est),
            max(overlap_layers.keys(), default=-1) + 1,
        )
        if max_layer_count <= 0:
            continue

        for layer_idx in range(max_layer_count):
            overlap_layer = overlap_layers.get(layer_idx, {})
            rows.append(
                {
                    "step": _to_int(step.get("step")),
                    "attn_tp_rank": _to_int(step.get("attn_tp_rank")),
                    "attn_tp_size": _to_int(step.get("attn_tp_size"), 1),
                    "seq_len": _to_int(step.get("seq_len")),
                    "prefetch_k": _to_int(step.get("prefetch_k")),
                    "layer_idx": layer_idx,
                    "prefetch_heads": _to_int(
                        layer_prefetch_heads[layer_idx] if layer_idx < len(layer_prefetch_heads) else 0
                    ),
                    "offloaded_heads": _to_int(
                        layer_offloaded_heads[layer_idx] if layer_idx < len(layer_offloaded_heads) else 0
                    ),
                    "prefetch_h2d_bytes": _to_int(
                        layer_h2d[layer_idx] if layer_idx < len(layer_h2d) else 0
                    ),
                    "offload_d2h_bytes": _to_int(
                        layer_d2h[layer_idx] if layer_idx < len(layer_d2h) else 0
                    ),
                    "total_bytes": _to_int(
                        (layer_h2d[layer_idx] if layer_idx < len(layer_h2d) else 0)
                        + (layer_d2h[layer_idx] if layer_idx < len(layer_d2h) else 0)
                    ),
                    "selected_tokens": _to_int(
                        layer_selected_tokens[layer_idx] if layer_idx < len(layer_selected_tokens) else 0
                    ),
                    "recalled_tokens": _to_int(
                        layer_recalled_tokens[layer_idx] if layer_idx < len(layer_recalled_tokens) else 0
                    ),
                    "hit_tokens": _to_int(
                        layer_hit_tokens[layer_idx] if layer_idx < len(layer_hit_tokens) else 0
                    ),
                    "hit_rate": _to_float(
                        layer_hit_rates[layer_idx] if layer_idx < len(layer_hit_rates) else 0.0
                    ),
                    "overlap_mean_precision": _to_float(overlap_layer.get("mean_precision")),
                    "overlap_mean_recall": _to_float(overlap_layer.get("mean_recall")),
                    "overlap_mean_jaccard": _to_float(overlap_layer.get("mean_jaccard")),
                    "overlap_union_precision": _to_float(overlap_layer.get("union_precision")),
                    "overlap_union_recall": _to_float(overlap_layer.get("union_recall")),
                    "overlap_union_jaccard": _to_float(overlap_layer.get("union_jaccard")),
                    "overlap_active_heads": _to_int(overlap_layer.get("active_heads")),
                    "overlap_prefetch_k": _to_int(overlap_layer.get("prefetch_k")),
                    "overlap_union_real_count": _to_int(overlap_layer.get("union_real_count")),
                    "overlap_union_prefetch_count": _to_int(
                        overlap_layer.get("union_prefetch_count")
                    ),
                    "overlap_union_intersection_count": _to_int(
                        overlap_layer.get("union_intersection_count")
                    ),
                    "overlap_sum_real_tokens": _to_int(
                        layer_overlap_sum_real[layer_idx] if layer_idx < len(layer_overlap_sum_real) else 0
                    ),
                    "overlap_sum_prefetch_tokens": _to_int(
                        layer_overlap_sum_prefetch[layer_idx]
                        if layer_idx < len(layer_overlap_sum_prefetch)
                        else 0
                    ),
                    "overlap_sum_hit_tokens": _to_int(
                        layer_overlap_sum_hit[layer_idx] if layer_idx < len(layer_overlap_sum_hit) else 0
                    ),
                    "overlap_recall_tokens": _to_int(
                        layer_overlap_recall_tokens[layer_idx]
                        if layer_idx < len(layer_overlap_recall_tokens)
                        else 0
                    ),
                    "overlap_prefetch_h2d_bytes_est": _to_int(
                        layer_overlap_prefetch_h2d_est[layer_idx]
                        if layer_idx < len(layer_overlap_prefetch_h2d_est)
                        else 0
                    ),
                    "overlap_recall_h2d_bytes_est": _to_int(
                        layer_overlap_recall_h2d_est[layer_idx]
                        if layer_idx < len(layer_overlap_recall_h2d_est)
                        else 0
                    ),
                    "step_h2d_bytes": _to_int(step.get("h2d_bytes")),
                    "step_d2h_bytes": _to_int(step.get("d2h_bytes")),
                    "step_total_bytes": _to_int(step.get("total_bytes")),
                    "step_selected_tokens": _to_int(step.get("selected_tokens")),
                    "step_recalled_tokens": _to_int(step.get("recalled_tokens")),
                    "step_hit_tokens": _to_int(step.get("hit_tokens")),
                    "step_hit_rate": _to_float(step.get("hit_rate")),
                    "hit_source": step.get("hit_source") or "",
                }
            )
    return rows


def _write_layer_csv(path: Path, steps: list[dict]) -> None:
    rows = _build_layer_rows(steps)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LAYER_CSV_COLUMNS)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _build_head_rows(steps: list[dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in steps:
        overlap = step.get("overlap")
        if not isinstance(overlap, dict):
            continue
        for layer_metrics in _safe_list(overlap.get("layers")):
            if not isinstance(layer_metrics, dict):
                continue
            layer_idx = _to_int(layer_metrics.get("layer_idx"), -1)
            if layer_idx < 0:
                continue
            head_slots = _safe_list(layer_metrics.get("active_bh_indices"))
            head_recall = _safe_list(layer_metrics.get("per_head_recall"))
            head_precision = _safe_list(layer_metrics.get("per_head_precision"))
            head_jaccard = _safe_list(layer_metrics.get("per_head_jaccard"))
            head_real_count = _safe_list(layer_metrics.get("per_head_real_count"))
            head_prefetch_count = _safe_list(layer_metrics.get("per_head_prefetch_count"))
            head_intersection_count = _safe_list(layer_metrics.get("per_head_intersection_count"))
            row_count = max(
                len(head_slots),
                len(head_recall),
                len(head_precision),
                len(head_jaccard),
                len(head_real_count),
                len(head_prefetch_count),
                len(head_intersection_count),
            )
            if row_count <= 0:
                continue
            for active_head_slot in range(row_count):
                recall = _to_float(
                    head_recall[active_head_slot] if active_head_slot < len(head_recall) else 0.0
                )
                rows.append(
                    {
                        "step": _to_int(step.get("step")),
                        "attn_tp_rank": _to_int(step.get("attn_tp_rank")),
                        "attn_tp_size": _to_int(step.get("attn_tp_size"), 1),
                        "seq_len": _to_int(step.get("seq_len")),
                        "prefetch_k": _to_int(step.get("prefetch_k")),
                        "layer_idx": layer_idx,
                        "active_head_slot": active_head_slot,
                        "active_bh_index": _to_int(
                            head_slots[active_head_slot]
                            if active_head_slot < len(head_slots)
                            else active_head_slot
                        ),
                        "corr": recall,
                        "overlap_head_recall": recall,
                        "overlap_head_precision": _to_float(
                            head_precision[active_head_slot]
                            if active_head_slot < len(head_precision)
                            else 0.0
                        ),
                        "overlap_head_jaccard": _to_float(
                            head_jaccard[active_head_slot]
                            if active_head_slot < len(head_jaccard)
                            else 0.0
                        ),
                        "overlap_head_real_count": _to_int(
                            head_real_count[active_head_slot]
                            if active_head_slot < len(head_real_count)
                            else 0
                        ),
                        "overlap_head_prefetch_count": _to_int(
                            head_prefetch_count[active_head_slot]
                            if active_head_slot < len(head_prefetch_count)
                            else 0
                        ),
                        "overlap_head_intersection_count": _to_int(
                            head_intersection_count[active_head_slot]
                            if active_head_slot < len(head_intersection_count)
                            else 0
                        ),
                    }
                )
    return rows


def _write_head_csv(path: Path, steps: list[dict]) -> None:
    rows = _build_head_rows(steps)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEAD_CSV_COLUMNS)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


class _TransferStatsRecorder:
    def __init__(self) -> None:
        self.enabled = False
        self.steps: list[dict] = []

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.reset()

    def reset(self) -> None:
        self.steps = []
        self._flush_to_file()

    def record_step(self, step: dict) -> None:
        if not self.enabled:
            return
        self.steps.append(deepcopy(step))
        self._flush_to_file()

    def snapshot(self) -> list[dict]:
        return deepcopy(self.steps)

    def _flush_to_file(self) -> None:
        raw_json_path = os.environ.get("DITTO_TRANSFER_STATS_FILE")
        csv_path = _resolve_csv_path(raw_json_path)
        head_csv_path = _resolve_head_csv_path(csv_path)
        if not raw_json_path and csv_path is None and head_csv_path is None:
            return

        if self.enabled:
            payload = summarize_transfer_stats(self.steps)
        else:
            payload = _empty_payload()

        if raw_json_path:
            json_path = _with_tp_rank_suffix(Path(raw_json_path))
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        if csv_path is not None:
            _write_layer_csv(csv_path, self.steps if self.enabled else [])
        if head_csv_path is not None:
            _write_head_csv(head_csv_path, self.steps if self.enabled else [])


_RECORDER = _TransferStatsRecorder()


def set_transfer_stats_enabled(enabled: bool) -> None:
    _RECORDER.set_enabled(enabled)


def transfer_stats_enabled() -> bool:
    return _RECORDER.enabled


def reset_transfer_stats() -> None:
    _RECORDER.reset()


def record_decode_transfer_step(step: dict) -> None:
    _RECORDER.record_step(step)


def get_transfer_stats_snapshot() -> list[dict]:
    return _RECORDER.snapshot()


def summarize_transfer_stats(steps: list[dict]) -> dict:
    total_bytes_per_step = [int(step.get("total_bytes", 0)) for step in steps]
    h2d_bytes_per_step = [int(step.get("h2d_bytes", 0)) for step in steps]
    d2h_bytes_per_step = [int(step.get("d2h_bytes", 0)) for step in steps]
    selected_tokens_per_step = [int(step.get("selected_tokens", 0)) for step in steps]
    recalled_tokens_per_step = [int(step.get("recalled_tokens", 0)) for step in steps]
    hit_tokens_per_step = []
    hit_rates_per_step = []
    for step, selected_tokens, recalled_tokens in zip(
        steps, selected_tokens_per_step, recalled_tokens_per_step
    ):
        hit_tokens = int(step.get("hit_tokens", max(selected_tokens - recalled_tokens, 0)))
        hit_tokens = max(hit_tokens, 0)
        hit_tokens_per_step.append(hit_tokens)
        if "hit_rate" in step:
            hit_rate = float(step.get("hit_rate", 0.0))
        else:
            hit_rate = (hit_tokens / selected_tokens) if selected_tokens > 0 else 0.0
        hit_rates_per_step.append(float(hit_rate))
    overlap_steps = [step.get("overlap") for step in steps if isinstance(step.get("overlap"), dict)]
    overlap_mean_recall = [float(step.get("mean_recall", 0.0)) for step in overlap_steps]
    overlap_mean_precision = [float(step.get("mean_precision", 0.0)) for step in overlap_steps]
    overlap_mean_jaccard = [float(step.get("mean_jaccard", 0.0)) for step in overlap_steps]
    overlap_union_recall = [float(step.get("union_recall", 0.0)) for step in overlap_steps]
    overlap_union_precision = [float(step.get("union_precision", 0.0)) for step in overlap_steps]
    overlap_union_jaccard = [float(step.get("union_jaccard", 0.0)) for step in overlap_steps]

    if total_bytes_per_step:
        avg_total_bytes = statistics.mean(total_bytes_per_step)
        p50_total_bytes = statistics.median(total_bytes_per_step)
        max_total_bytes = max(total_bytes_per_step)
        min_total_bytes = min(total_bytes_per_step)
    else:
        avg_total_bytes = 0.0
        p50_total_bytes = 0.0
        max_total_bytes = 0
        min_total_bytes = 0

    def _mean_or_zero(values: list[float]) -> float:
        return float(statistics.mean(values)) if values else 0.0

    return {
        "enabled": True,
        "decode_steps": len(steps),
        "total_bytes": int(sum(total_bytes_per_step)),
        "total_h2d_bytes": int(sum(h2d_bytes_per_step)),
        "total_d2h_bytes": int(sum(d2h_bytes_per_step)),
        "total_selected_tokens": int(sum(selected_tokens_per_step)),
        "total_recalled_tokens": int(sum(recalled_tokens_per_step)),
        "total_hit_tokens": int(sum(hit_tokens_per_step)),
        "overall_hit_rate": float(
            (sum(hit_tokens_per_step) / sum(selected_tokens_per_step))
            if sum(selected_tokens_per_step) > 0
            else 0.0
        ),
        "avg_bytes_per_step": float(avg_total_bytes),
        "p50_bytes_per_step": float(p50_total_bytes),
        "max_bytes_per_step": int(max_total_bytes),
        "min_bytes_per_step": int(min_total_bytes),
        "per_step_total_bytes": [int(v) for v in total_bytes_per_step],
        "per_step_h2d_bytes": [int(v) for v in h2d_bytes_per_step],
        "per_step_d2h_bytes": [int(v) for v in d2h_bytes_per_step],
        "per_step_selected_tokens": [int(v) for v in selected_tokens_per_step],
        "per_step_recalled_tokens": [int(v) for v in recalled_tokens_per_step],
        "per_step_hit_tokens": [int(v) for v in hit_tokens_per_step],
        "per_step_hit_rates": [float(v) for v in hit_rates_per_step],
        "overlap_steps": int(len(overlap_steps)),
        "avg_overlap_mean_recall": _mean_or_zero(overlap_mean_recall),
        "avg_overlap_mean_precision": _mean_or_zero(overlap_mean_precision),
        "avg_overlap_mean_jaccard": _mean_or_zero(overlap_mean_jaccard),
        "avg_overlap_union_recall": _mean_or_zero(overlap_union_recall),
        "avg_overlap_union_precision": _mean_or_zero(overlap_union_precision),
        "avg_overlap_union_jaccard": _mean_or_zero(overlap_union_jaccard),
        "per_step_overlap_mean_recall": [float(v) for v in overlap_mean_recall],
        "per_step_overlap_mean_precision": [float(v) for v in overlap_mean_precision],
        "per_step_overlap_mean_jaccard": [float(v) for v in overlap_mean_jaccard],
        "per_step_overlap_union_recall": [float(v) for v in overlap_union_recall],
        "per_step_overlap_union_precision": [float(v) for v in overlap_union_precision],
        "per_step_overlap_union_jaccard": [float(v) for v in overlap_union_jaccard],
        "steps": deepcopy(steps),
    }
