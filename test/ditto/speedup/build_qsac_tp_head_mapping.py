#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a per-layer TP2 head mapping from QSAC miss masks."
    )
    parser.add_argument("--transfer-json", action="append", required=True)
    parser.add_argument("--threshold-csv", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-overlapped-heads", type=int, default=3)
    parser.add_argument("--skip-steps", type=int, default=10)
    parser.add_argument("--skip-layers", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128000)
    parser.add_argument("--topk-ratio", type=float, default=0.10)
    parser.add_argument(
        "--objective",
        choices=("aggregate", "stepwise"),
        default="aggregate",
        help=(
            "Balance per-layer aggregate QSAC misses by default. The stepwise "
            "mode is useful for analysis but can synchronize host/GDR bursts."
        ),
    )
    parser.add_argument(
        "--resident-policy",
        choices=("heuristic", "none"),
        default="heuristic",
        help=(
            "Use the existing hard-head overlap heuristic, or optimize mappings "
            "assuming every non-skip head is explicitly offloaded."
        ),
    )
    return parser.parse_args()


def load_thresholds(paths: list[str]) -> dict[tuple[int, int], dict[str, Any]]:
    thresholds: dict[tuple[int, int], dict[str, Any]] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8", newline="") as file_obj:
            for row in csv.DictReader(file_obj):
                key = (int(row["layer_idx"]), int(row["head_idx"]))
                thresholds[key] = {
                    "hard": bool(int(row["hard_to_reuse"])),
                    "difficulty": float(row["reuse_difficulty"]),
                }
    return thresholds


def load_qsac_misses(
    paths: list[str], skip_steps: int
) -> tuple[list[list[int]], list[list[int]], list[list[list[int]]]]:
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    num_layers = 0
    all_heads: set[int] = set()
    for payload in payloads:
        for step in payload.get("steps", []):
            masks = step.get("layer_prefetch_head_masks")
            if not isinstance(masks, list):
                continue
            num_layers = max(num_layers, len(masks))
            layer_orders = step.get("local_kv_head_ids_by_layer")
            if layer_orders is None:
                global_order = step.get("local_kv_head_ids")
                layer_orders = [global_order for _ in masks]
            all_heads.update(
                int(head) for layer_order in layer_orders for head in layer_order
            )
    if num_layers <= 0 or not all_heads:
        raise ValueError("No per-head QSAC masks were found in the transfer profiles")

    num_heads = max(all_heads) + 1
    miss_counts = [[0] * num_heads for _ in range(num_layers)]
    sample_counts = [[0] * num_heads for _ in range(num_layers)]
    step_misses: dict[int, list[list[int]]] = {}
    step_samples: dict[int, list[list[int]]] = {}
    for payload in payloads:
        for step_pos, step in enumerate(payload.get("steps", [])):
            if step_pos < skip_steps:
                continue
            masks = step.get("layer_prefetch_head_masks")
            if not isinstance(masks, list):
                continue
            layer_orders = step.get("local_kv_head_ids_by_layer")
            if layer_orders is None:
                global_order = step.get("local_kv_head_ids")
                layer_orders = [global_order for _ in masks]
            step_idx = int(step.get("step", step_pos))
            current_misses = step_misses.setdefault(
                step_idx, [[0] * num_heads for _ in range(num_layers)]
            )
            current_samples = step_samples.setdefault(
                step_idx, [[0] * num_heads for _ in range(num_layers)]
            )
            for layer_idx, (layer_mask, layer_order) in enumerate(
                zip(masks, layer_orders)
            ):
                for miss, original_head in zip(layer_mask, layer_order):
                    head_idx = int(original_head)
                    miss_value = int(bool(miss))
                    miss_counts[layer_idx][head_idx] += miss_value
                    sample_counts[layer_idx][head_idx] += 1
                    current_misses[layer_idx][head_idx] += miss_value
                    current_samples[layer_idx][head_idx] += 1
    for layer_idx, layer_samples in enumerate(sample_counts):
        if not all(layer_samples):
            raise ValueError(
                f"Layer {layer_idx} does not contain samples for every original head: "
                f"{layer_samples}"
            )
    for step_idx, layer_samples in step_samples.items():
        for layer_idx, head_samples in enumerate(layer_samples):
            if not all(value == 1 for value in head_samples):
                raise ValueError(
                    f"Step {step_idx} layer {layer_idx} does not have exactly one "
                    f"sample per original head: {head_samples}"
                )
    ordered_step_misses = [step_misses[key] for key in sorted(step_misses)]
    return miss_counts, sample_counts, ordered_step_misses


def resident_heads(
    layer_idx: int,
    group: tuple[int, ...],
    thresholds: dict[tuple[int, int], dict[str, Any]],
    num_overlapped_heads: int,
) -> tuple[int, ...]:
    hard_heads = [head for head in group if thresholds[(layer_idx, head)]["hard"]]
    resident_count = max(len(hard_heads) - num_overlapped_heads, 0)
    hard_heads.sort(
        key=lambda head: (-thresholds[(layer_idx, head)]["difficulty"], group.index(head))
    )
    return tuple(hard_heads[:resident_count])


def choose_groups(
    layer_idx: int,
    layer_misses: list[int],
    layer_step_misses: list[list[int]],
    thresholds: dict[tuple[int, int], dict[str, Any]],
    num_overlapped_heads: int,
    objective_mode: str,
    resident_policy: str,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    num_heads = len(layer_misses)
    local_heads = num_heads // 2
    all_heads = tuple(range(num_heads))
    best = None
    for rank0 in itertools.combinations(all_heads, local_heads):
        if 0 not in rank0:
            continue
        rank1 = tuple(head for head in all_heads if head not in rank0)
        if resident_policy == "none":
            resident0 = ()
            resident1 = ()
        else:
            resident0 = resident_heads(
                layer_idx, rank0, thresholds, num_overlapped_heads
            )
            resident1 = resident_heads(
                layer_idx, rank1, thresholds, num_overlapped_heads
            )
        miss0 = sum(layer_misses[head] for head in rank0 if head not in resident0)
        miss1 = sum(layer_misses[head] for head in rank1 if head not in resident1)
        step_critical = 0
        step_imbalance = 0
        for step_miss in layer_step_misses:
            step_load0 = sum(
                step_miss[head] for head in rank0 if head not in resident0
            )
            step_load1 = sum(
                step_miss[head] for head in rank1 if head not in resident1
            )
            step_critical += max(step_load0, step_load1)
            step_imbalance += abs(step_load0 - step_load1)
        if objective_mode == "stepwise":
            qsac_objective = (
                step_critical,
                step_imbalance,
                max(miss0, miss1),
                abs(miss0 - miss1),
            )
        else:
            qsac_objective = (
                max(miss0, miss1),
                abs(miss0 - miss1),
            )
        objective = (
            abs(len(resident0) - len(resident1)),
            max(len(resident0), len(resident1)),
            *qsac_objective,
            miss0 + miss1,
            rank0,
        )
        candidate = (
            objective,
            rank0,
            rank1,
            {
                "resident_heads": [list(resident0), list(resident1)],
                "offloaded_qsac_misses": [miss0, miss1],
                "step_layer_critical_misses": step_critical,
                "step_layer_imbalance_misses": step_imbalance,
            },
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError(f"No TP2 partition found for layer {layer_idx}")
    _, rank0, rank1, details = best
    return rank0, rank1, details


def main() -> None:
    args = parse_args()
    thresholds = load_thresholds(args.threshold_csv)
    misses, samples, step_misses = load_qsac_misses(
        args.transfer_json, args.skip_steps
    )
    num_layers = len(misses)
    num_heads = len(misses[0])
    expected_keys = {
        (layer_idx, head_idx)
        for layer_idx in range(num_layers)
        for head_idx in range(num_heads)
    }
    missing_keys = sorted(expected_keys - thresholds.keys())
    if missing_keys:
        raise ValueError(f"Missing threshold records: {missing_keys[:16]}")

    orders: list[list[int]] = []
    layer_details: list[dict[str, Any]] = []
    cumulative_load = [0.0, 0.0]
    dense_to_sparse_ratio = 1.0 / args.topk_ratio
    for layer_idx, layer_misses in enumerate(misses):
        if layer_idx < args.skip_layers:
            rank0 = tuple(range(num_heads // 2))
            rank1 = tuple(range(num_heads // 2, num_heads))
            details = {
                "resident_heads": [list(rank0), list(rank1)],
                "offloaded_qsac_misses": [0, 0],
            }
        else:
            rank0, rank1, details = choose_groups(
                layer_idx,
                layer_misses,
                [step[layer_idx] for step in step_misses],
                thresholds,
                args.num_overlapped_heads,
                args.objective,
                args.resident_policy,
            )

        steady_steps = max(samples[layer_idx])
        effective = [
            float(details["offloaded_qsac_misses"][rank])
            + len(details["resident_heads"][rank])
            * steady_steps
            * dense_to_sparse_ratio
            for rank in range(2)
        ]
        if abs((cumulative_load[0] + effective[1]) - (cumulative_load[1] + effective[0])) < abs(
            (cumulative_load[0] + effective[0]) - (cumulative_load[1] + effective[1])
        ):
            rank0, rank1 = rank1, rank0
            details["resident_heads"].reverse()
            details["offloaded_qsac_misses"].reverse()
            effective.reverse()
        cumulative_load[0] += effective[0]
        cumulative_load[1] += effective[1]

        orders.append(list(rank0 + rank1))
        layer_details.append(
            {
                "layer_idx": layer_idx,
                "rank_heads": [list(rank0), list(rank1)],
                "qsac_miss_counts": layer_misses,
                "sample_counts": samples[layer_idx],
                **details,
            }
        )

    output = {
        "orders": orders,
        "metadata": {
            "objective": (
                f"{'resident-aware' if args.resident_policy == 'heuristic' else 'no-resident'} "
                f"per-layer {args.objective} QSAC critical-rank load"
            ),
            "resident_policy": args.resident_policy,
            "num_overlapped_heads": args.num_overlapped_heads,
            "skip_steps": args.skip_steps,
            "skip_layers": args.skip_layers,
            "seq_len": args.seq_len,
            "topk_ratio": args.topk_ratio,
            "profile_steps": len(step_misses),
            "transfer_json": args.transfer_json,
            "threshold_csv": args.threshold_csv,
            "cumulative_effective_load": cumulative_load,
        },
        "layers": layer_details,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {num_layers} per-layer mappings to {output_path}")


if __name__ == "__main__":
    main()
