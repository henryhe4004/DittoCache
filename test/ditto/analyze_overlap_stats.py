#!/usr/bin/env python3
"""Analyze Ditto overlap stats from JSON or JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _to_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _looks_like_layer_row(obj: Any) -> bool:
    return isinstance(obj, dict) and "layer_idx" in obj and (
        "mean_recall" in obj
        or "mean_precision" in obj
        or "mean_jaccard" in obj
        or "union_recall" in obj
        or "union_precision" in obj
        or "union_jaccard" in obj
    )


def _load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty file: {path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line {lineno} in {path}: {exc}"
                ) from exc
        if not rows:
            raise ValueError(f"No valid JSON objects found in {path}")
        return rows


def _enrich_layer_row(row: dict[str, Any], step_meta: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["step"] = step_meta.get("step")
    out["seq_len"] = step_meta.get("seq_len")
    out["step_prefetch_k"] = step_meta.get("prefetch_k")
    if out.get("prefetch_k") is None:
        out["prefetch_k"] = step_meta.get("prefetch_k")

    ui = out.get("union_intersection_count")
    ur = out.get("union_real_count")
    up = out.get("union_prefetch_count")
    if isinstance(ui, (int, float)) and isinstance(ur, (int, float)):
        out["union_miss_count"] = ur - ui
    else:
        out["union_miss_count"] = None
    if isinstance(ui, (int, float)) and isinstance(up, (int, float)):
        out["union_extra_prefetch_count"] = up - ui
    else:
        out["union_extra_prefetch_count"] = None

    if isinstance(ui, (int, float)) and isinstance(ur, (int, float)) and isinstance(
        up, (int, float)
    ):
        den = ur + up - ui
        out["recomputed_union_jaccard"] = _safe_div(float(ui), float(den))
    else:
        out["recomputed_union_jaccard"] = None
    return out


def _collect_rows(data: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    layer_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    root_meta: dict[str, Any] = {}

    def consume_step(step: dict[str, Any], step_index: int) -> None:
        step_meta = {
            "step": step.get("step", step_index + 1),
            "seq_len": step.get("seq_len"),
            "prefetch_k": step.get("prefetch_k"),
        }
        overlap = step.get("overlap")
        if isinstance(overlap, dict):
            step_rows.append(
                {
                    "step": step_meta["step"],
                    "seq_len": step_meta["seq_len"],
                    "prefetch_k": step_meta["prefetch_k"],
                    "matched_layers": overlap.get("matched_layers"),
                    "mean_recall": overlap.get("mean_recall"),
                    "mean_precision": overlap.get("mean_precision"),
                    "mean_jaccard": overlap.get("mean_jaccard"),
                    "union_recall": overlap.get("union_recall"),
                    "union_precision": overlap.get("union_precision"),
                    "union_jaccard": overlap.get("union_jaccard"),
                }
            )
            for layer in overlap.get("layers", []):
                if _looks_like_layer_row(layer):
                    layer_rows.append(_enrich_layer_row(layer, step_meta))

    if isinstance(data, dict):
        if isinstance(data.get("steps"), list):
            for key in (
                "enabled",
                "decode_steps",
                "overlap_steps",
                "total_bytes",
                "total_h2d_bytes",
                "total_d2h_bytes",
                "avg_bytes_per_step",
                "p50_bytes_per_step",
                "max_bytes_per_step",
                "min_bytes_per_step",
                "avg_overlap_mean_recall",
                "avg_overlap_mean_precision",
                "avg_overlap_mean_jaccard",
                "avg_overlap_union_recall",
                "avg_overlap_union_precision",
                "avg_overlap_union_jaccard",
            ):
                if key in data:
                    root_meta[key] = data[key]
            for i, step in enumerate(data["steps"]):
                if isinstance(step, dict):
                    consume_step(step, i)
        elif _looks_like_layer_row(data):
            layer_rows.append(_enrich_layer_row(data, step_meta={}))
        elif isinstance(data.get("layers"), list):
            for layer in data["layers"]:
                if _looks_like_layer_row(layer):
                    layer_rows.append(_enrich_layer_row(layer, step_meta={}))
        elif isinstance(data.get("overlap"), dict):
            consume_step(data, 0)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if _looks_like_layer_row(item):
                layer_rows.append(_enrich_layer_row(item, step_meta={}))
            elif isinstance(item, dict) and isinstance(item.get("layers"), list):
                for layer in item["layers"]:
                    if _looks_like_layer_row(layer):
                        layer_rows.append(_enrich_layer_row(layer, step_meta={}))
            elif isinstance(item, dict) and (
                isinstance(item.get("overlap"), dict) or isinstance(item.get("steps"), list)
            ):
                sub_layers, sub_steps, _ = _collect_rows(item)
                layer_rows.extend(sub_layers)
                step_rows.extend(sub_steps)

    return layer_rows, step_rows, root_meta


def _aggregate_by_layer(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        layer_idx = row.get("layer_idx")
        if isinstance(layer_idx, int):
            groups[layer_idx].append(row)

    result = []
    for layer_idx, items in sorted(groups.items()):
        inter_total = sum(
            float(x["union_intersection_count"])
            for x in items
            if isinstance(x.get("union_intersection_count"), (int, float))
        )
        real_total = sum(
            float(x["union_real_count"])
            for x in items
            if isinstance(x.get("union_real_count"), (int, float))
        )
        pref_total = sum(
            float(x["union_prefetch_count"])
            for x in items
            if isinstance(x.get("union_prefetch_count"), (int, float))
        )

        mean_j = _mean([x for x in (_to_float(i.get("mean_jaccard")) for i in items) if x is not None])
        union_j = _mean([x for x in (_to_float(i.get("union_jaccard")) for i in items) if x is not None])
        mean_r = _mean([x for x in (_to_float(i.get("mean_recall")) for i in items) if x is not None])
        union_r = _mean([x for x in (_to_float(i.get("union_recall")) for i in items) if x is not None])
        mean_p = _mean([x for x in (_to_float(i.get("mean_precision")) for i in items) if x is not None])
        union_p = _mean([x for x in (_to_float(i.get("union_precision")) for i in items) if x is not None])
        active_heads_avg = _mean(
            [x for x in (_to_float(i.get("active_heads")) for i in items) if x is not None]
        )

        result.append(
            {
                "layer_idx": layer_idx,
                "count": len(items),
                "active_heads_avg": active_heads_avg,
                "mean_recall_avg": mean_r,
                "mean_precision_avg": mean_p,
                "mean_jaccard_avg": mean_j,
                "union_recall_avg": union_r,
                "union_precision_avg": union_p,
                "union_jaccard_avg": union_j,
                "weighted_union_recall": _safe_div(inter_total, real_total),
                "weighted_union_precision": _safe_div(inter_total, pref_total),
                "weighted_union_jaccard": _safe_div(
                    inter_total, (real_total + pref_total - inter_total)
                ),
            }
        )
    return result


def _print_rank(
    rows: list[dict[str, Any]], metric_key: str, topk: int, reverse: bool, title: str
) -> None:
    picked = [r for r in rows if isinstance(r.get(metric_key), (int, float))]
    picked.sort(key=lambda x: x[metric_key], reverse=reverse)
    picked = picked[:topk]
    print(title)
    if not picked:
        print("  (empty)")
        return
    for row in picked:
        print(
            f"  layer {row['layer_idx']:>3} | {metric_key}={_fmt(row[metric_key])} "
            f"| count={row['count']} | active_heads_avg={_fmt(row['active_heads_avg'], 2)}"
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Ditto overlap stats file.")
    p.add_argument("input", type=Path, help="Path to overlap stats json/jsonl")
    p.add_argument("--topk", type=int, default=10, help="Top/Bottom K layers to print")
    p.add_argument(
        "--metric",
        default="mean_jaccard_avg",
        choices=[
            "mean_recall_avg",
            "mean_precision_avg",
            "mean_jaccard_avg",
            "union_recall_avg",
            "union_precision_avg",
            "union_jaccard_avg",
            "weighted_union_recall",
            "weighted_union_precision",
            "weighted_union_jaccard",
        ],
        help="Metric used for Top/Bottom ranking",
    )
    p.add_argument("--layer", type=int, action="append", help="Filter by layer_idx (repeatable)")
    p.add_argument("--step", type=int, action="append", help="Filter by step (repeatable)")
    p.add_argument("--export-csv", type=Path, help="Export filtered layer rows to CSV")
    p.add_argument("--export-json", type=Path, help="Export aggregated layer summary to JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = _load_json_or_jsonl(args.input)
    layer_rows, step_rows, root_meta = _collect_rows(data)

    if not layer_rows:
        print("No layer-level overlap rows found.")
        return

    filtered = layer_rows
    if args.layer:
        layer_set = set(args.layer)
        filtered = [r for r in filtered if r.get("layer_idx") in layer_set]
    if args.step:
        step_set = set(args.step)
        filtered = [r for r in filtered if r.get("step") in step_set]

    print(f"Input: {args.input}")
    if root_meta:
        print(
            "Root stats: "
            f"enabled={root_meta.get('enabled')} "
            f"decode_steps={root_meta.get('decode_steps')} "
            f"overlap_steps={root_meta.get('overlap_steps')} "
            f"total_bytes={root_meta.get('total_bytes')}"
        )
    print(
        f"Rows: total_layer_rows={len(layer_rows)} "
        f"filtered_rows={len(filtered)} "
        f"step_rows={len(step_rows)}"
    )
    if not filtered:
        print("No rows left after filtering.")
        return

    layer_agg = _aggregate_by_layer(filtered)
    print(
        f"Unique layers in filtered data: {len(layer_agg)} "
        f"(layer_idx min={min(x['layer_idx'] for x in layer_agg)}, "
        f"max={max(x['layer_idx'] for x in layer_agg)})"
    )

    if len(filtered) == 1:
        row = filtered[0]
        print("Single row details:")
        for k in sorted(row.keys()):
            print(f"  {k}: {row[k]}")

    _print_rank(
        layer_agg,
        metric_key=args.metric,
        topk=args.topk,
        reverse=True,
        title=f"Top {args.topk} layers by {args.metric}:",
    )
    _print_rank(
        layer_agg,
        metric_key=args.metric,
        topk=args.topk,
        reverse=False,
        title=f"Bottom {args.topk} layers by {args.metric}:",
    )

    if args.export_csv:
        _write_csv(args.export_csv, filtered)
        print(f"Exported filtered rows CSV: {args.export_csv}")
    if args.export_json:
        _write_json(args.export_json, layer_agg)
        print(f"Exported aggregated layer summary JSON: {args.export_json}")


if __name__ == "__main__":
    main()
