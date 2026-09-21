#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StepSample:
    profile: str
    step: int
    seq_len: int
    prefetch_k: int
    masks: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class CostConfig:
    h2d_gbps: tuple[float, float]
    d2h_gbps: tuple[float, float]
    h2d_launch_us: float
    d2h_launch_us: float
    active_head_us: float
    shrinkage: float
    tail_quantile: float
    tail_weight: float
    imbalance_weight: float
    worst_profile_weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a TP2 per-layer KV-head mapping with a byte-aware critical-rank "
            "cost model. Transfer profiles must be recorded with "
            "DITTO_RECORD_HEAD_MASKS=1."
        )
    )
    parser.add_argument("--transfer-json", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resident-heads-file")
    parser.add_argument("--reference-mapping")
    parser.add_argument("--skip-steps", type=int, default=10)
    parser.add_argument("--skip-layers", type=int, default=1)
    parser.add_argument("--h2d-gbps", type=float, default=22.0)
    parser.add_argument("--rank0-h2d-gbps", type=float)
    parser.add_argument("--rank1-h2d-gbps", type=float)
    parser.add_argument("--d2h-gbps", type=float, default=22.0)
    parser.add_argument("--rank0-d2h-gbps", type=float)
    parser.add_argument("--rank1-d2h-gbps", type=float)
    parser.add_argument("--h2d-launch-us", type=float, default=2.0)
    parser.add_argument("--d2h-launch-us", type=float, default=1.0)
    parser.add_argument("--active-head-us", type=float, default=0.5)
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=0.5,
        help="Blend observed stepwise critical cost toward aggregate rank cost.",
    )
    parser.add_argument("--tail-quantile", type=float, default=0.95)
    parser.add_argument("--tail-weight", type=float, default=0.10)
    parser.add_argument("--imbalance-weight", type=float, default=0.05)
    parser.add_argument(
        "--worst-profile-weight",
        type=float,
        default=0.25,
        help="Penalize a partition that is good on average but bad for one length bucket.",
    )
    return parser.parse_args()


def _profile_key(path: str) -> str:
    return re.sub(r"\.tp\d+(?=\.[^.]+$)", "", str(Path(path).resolve()))


def _validate_fraction(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(math.ceil(quantile * len(ordered)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def _layer_orders(step: dict[str, Any], num_layers: int) -> list[list[int]]:
    orders = step.get("local_kv_head_ids_by_layer")
    if isinstance(orders, list):
        return [[int(head) for head in order] for order in orders]
    global_order = step.get("local_kv_head_ids")
    if not isinstance(global_order, list):
        raise TypeError("Transfer step is missing local KV-head IDs")
    return [[int(head) for head in global_order] for _ in range(num_layers)]


def load_step_samples(
    paths: Iterable[str], skip_steps: int
) -> tuple[list[StepSample], int, int, int]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    byte_samples: list[float] = []
    profile_ranks: dict[str, set[int]] = {}
    profile_tp_sizes: dict[str, int] = {}
    num_layers = 0
    max_head = -1

    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        profile = _profile_key(path)
        steps = payload.get("steps")
        if not isinstance(steps, list):
            raise TypeError(f"{path} does not contain a steps list")
        for step_pos, step in enumerate(steps):
            if step_pos < skip_steps:
                continue
            masks = step.get("layer_prefetch_head_masks")
            if not isinstance(masks, list):
                raise TypeError(
                    f"{path} has no per-head masks; record with DITTO_RECORD_HEAD_MASKS=1"
                )
            rank = int(step.get("attn_tp_rank", 0))
            tp_size = int(step.get("attn_tp_size", 1))
            profile_ranks.setdefault(profile, set()).add(rank)
            old_tp_size = profile_tp_sizes.setdefault(profile, tp_size)
            if old_tp_size != tp_size:
                raise ValueError(f"Mismatched TP size in profile {profile}")
            num_layers = max(num_layers, len(masks))
            orders = _layer_orders(step, len(masks))
            for order in orders:
                if order:
                    max_head = max(max_head, max(order))

            prefetch_k = max(int(step.get("prefetch_k", 0)), 0)
            layer_h2d = step.get("layer_h2d_bytes", [])
            layer_prefetch = step.get("layer_prefetch_heads", [])
            layer_d2h = step.get("layer_d2h_bytes", [])
            layer_offloaded = step.get("layer_offloaded_heads", [])
            for layer_idx in range(len(masks)):
                prefetch_heads = int(layer_prefetch[layer_idx])
                if prefetch_k > 0 and prefetch_heads > 0:
                    byte_samples.append(
                        int(layer_h2d[layer_idx]) / (prefetch_heads * prefetch_k)
                    )
                offloaded_heads = int(layer_offloaded[layer_idx])
                if offloaded_heads > 0:
                    byte_samples.append(int(layer_d2h[layer_idx]) / offloaded_heads)

            step_idx = int(step.get("step", step_pos + 1))
            key = (profile, step_idx)
            record = records.setdefault(
                key,
                {
                    "seq_len": int(step.get("seq_len", 0)),
                    "prefetch_k": prefetch_k,
                    "layers": {},
                },
            )
            if record["prefetch_k"] != prefetch_k:
                raise ValueError(f"Mismatched prefetch_k for profile step {key}")
            for layer_idx, (layer_mask, layer_order) in enumerate(zip(masks, orders)):
                if len(layer_mask) != len(layer_order):
                    raise ValueError(
                        f"Mask/order size mismatch at {path}, step {step_idx}, layer {layer_idx}"
                    )
                head_values = record["layers"].setdefault(layer_idx, {})
                for miss, head in zip(layer_mask, layer_order):
                    miss_value = int(bool(miss))
                    old_value = head_values.setdefault(int(head), miss_value)
                    if old_value != miss_value:
                        raise ValueError(
                            f"Conflicting mask for profile step {key}, "
                            f"layer {layer_idx}, head {head}"
                        )

    if num_layers <= 0 or max_head < 0:
        raise ValueError("No per-head transfer samples were found")
    for profile, tp_size in profile_tp_sizes.items():
        expected_ranks = set(range(tp_size))
        if profile_ranks[profile] != expected_ranks:
            raise ValueError(
                f"Incomplete TP profile {profile}; found ranks "
                f"{sorted(profile_ranks[profile])}, expected {sorted(expected_ranks)}"
            )
    num_heads = max_head + 1
    if num_heads % 2:
        raise ValueError(f"TP2 requires an even number of global KV heads, got {num_heads}")
    if not byte_samples:
        raise ValueError("Unable to infer bytes per KV token/head from transfer profiles")
    per_token_head_bytes = round(statistics.median(byte_samples))
    if any(abs(value - per_token_head_bytes) > 0.5 for value in byte_samples):
        raise ValueError("Inconsistent per-token KV head byte sizes in transfer profiles")

    samples: list[StepSample] = []
    expected_heads = set(range(num_heads))
    for (profile, step_idx), record in sorted(records.items()):
        layers: list[tuple[int, ...]] = []
        for layer_idx in range(num_layers):
            head_values = record["layers"].get(layer_idx, {})
            if set(head_values) != expected_heads:
                missing = sorted(expected_heads - set(head_values))
                raise ValueError(
                    f"Incomplete TP profile {profile}, step {step_idx}, layer {layer_idx}; "
                    f"missing heads {missing}"
                )
            layers.append(tuple(head_values[head] for head in range(num_heads)))
        samples.append(
            StepSample(
                profile=profile,
                step=step_idx,
                seq_len=int(record["seq_len"]),
                prefetch_k=int(record["prefetch_k"]),
                masks=tuple(layers),
            )
        )
    return samples, num_layers, num_heads, per_token_head_bytes


def load_resident_heads(path: str | None, num_layers: int) -> list[set[int]]:
    if path is None:
        return [set() for _ in range(num_layers)]
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_layers = payload.get("resident_heads")
    if not isinstance(raw_layers, list) or len(raw_layers) != num_layers:
        raise ValueError(
            f"Resident placement must contain exactly {num_layers} layer entries"
        )
    return [{int(head) for head in layer} for layer in raw_layers]


def load_reference_orders(
    path: str | None, num_layers: int, num_heads: int
) -> list[list[int]]:
    if path is None:
        linear = list(range(num_heads))
        return [linear[:] for _ in range(num_layers)]
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    orders = payload.get("orders")
    if not isinstance(orders, list) or len(orders) != num_layers:
        raise ValueError(f"Reference mapping must contain exactly {num_layers} orders")
    expected = list(range(num_heads))
    result = []
    for layer_idx, order in enumerate(orders):
        parsed = [int(head) for head in order]
        if sorted(parsed) != expected:
            raise ValueError(f"Reference layer {layer_idx} is not a head permutation")
        result.append(parsed)
    return result


def _rank_cost_us(
    sample: StepSample,
    layer_idx: int,
    heads: tuple[int, ...],
    resident: set[int],
    rank: int,
    per_token_head_bytes: int,
    config: CostConfig,
) -> float:
    offloaded = tuple(head for head in heads if head not in resident)
    active_heads = sum(sample.masks[layer_idx][head] for head in offloaded)
    h2d_bytes = active_heads * sample.prefetch_k * per_token_head_bytes
    d2h_bytes = len(offloaded) * per_token_head_bytes
    cost = h2d_bytes / (config.h2d_gbps[rank] * 1000.0)
    cost += d2h_bytes / (config.d2h_gbps[rank] * 1000.0)
    if active_heads:
        cost += config.h2d_launch_us + active_heads * config.active_head_us
    if offloaded:
        cost += config.d2h_launch_us
    return cost


def score_partition(
    layer_idx: int,
    rank_heads: tuple[tuple[int, ...], tuple[int, ...]],
    samples: list[StepSample],
    resident: set[int],
    per_token_head_bytes: int,
    config: CostConfig,
) -> dict[str, Any]:
    by_profile: dict[str, list[tuple[float, float]]] = {}
    for sample in samples:
        costs = tuple(
            _rank_cost_us(
                sample,
                layer_idx,
                rank_heads[rank],
                resident,
                rank,
                per_token_head_bytes,
                config,
            )
            for rank in range(2)
        )
        by_profile.setdefault(sample.profile, []).append(costs)

    profile_details: dict[str, Any] = {}
    profile_scores = []
    all_rank_costs: list[tuple[float, float]] = []
    for profile, costs in sorted(by_profile.items()):
        all_rank_costs.extend(costs)
        critical = [max(pair) for pair in costs]
        imbalance = [abs(pair[0] - pair[1]) for pair in costs]
        rank_means = [statistics.mean(pair[rank] for pair in costs) for rank in range(2)]
        empirical = statistics.mean(critical)
        aggregate = max(rank_means)
        blended = (1.0 - config.shrinkage) * empirical + config.shrinkage * aggregate
        tail = _percentile(critical, config.tail_quantile)
        score = blended
        score += config.tail_weight * max(tail - empirical, 0.0)
        score += config.imbalance_weight * statistics.mean(imbalance)
        profile_scores.append(score)
        profile_details[profile] = {
            "samples": len(costs),
            "mean_critical_us": empirical,
            "aggregate_critical_us": aggregate,
            "tail_critical_us": tail,
            "mean_rank_us": rank_means,
            "score_us": score,
        }

    mean_score = statistics.mean(profile_scores)
    worst_score = max(profile_scores)
    objective = mean_score + config.worst_profile_weight * max(worst_score - mean_score, 0.0)
    mean_rank = [
        statistics.mean(pair[rank] for pair in all_rank_costs) for rank in range(2)
    ]
    return {
        "objective_us": objective,
        "mean_profile_score_us": mean_score,
        "worst_profile_score_us": worst_score,
        "mean_rank_us": mean_rank,
        "profiles": profile_details,
    }


def _partition_distance(
    rank0: tuple[int, ...], reference_order: list[int], local_heads: int
) -> int:
    reference_rank0 = set(reference_order[:local_heads])
    return len(set(rank0) ^ reference_rank0)


def choose_partition(
    layer_idx: int,
    samples: list[StepSample],
    resident: set[int],
    per_token_head_bytes: int,
    config: CostConfig,
    reference_order: list[int],
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    num_heads = len(samples[0].masks[layer_idx])
    local_heads = num_heads // 2
    all_heads = tuple(range(num_heads))
    best: tuple[Any, ...] | None = None
    for rank0 in itertools.combinations(all_heads, local_heads):
        rank1 = tuple(head for head in all_heads if head not in rank0)
        metrics = score_partition(
            layer_idx,
            (rank0, rank1),
            samples,
            resident,
            per_token_head_bytes,
            config,
        )
        key = (
            round(float(metrics["objective_us"]), 9),
            round(float(metrics["worst_profile_score_us"]), 9),
            _partition_distance(rank0, reference_order, local_heads),
            rank0,
        )
        candidate = (key, rank0, rank1, metrics)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError(f"No TP2 partition found for layer {layer_idx}")
    return best[1], best[2], best[3]


def build_mapping(
    samples: list[StepSample],
    num_layers: int,
    num_heads: int,
    per_token_head_bytes: int,
    residents: list[set[int]],
    references: list[list[int]],
    skip_layers: int,
    config: CostConfig,
) -> tuple[list[list[int]], list[dict[str, Any]], list[float]]:
    local_heads = num_heads // 2
    orders: list[list[int]] = []
    details: list[dict[str, Any]] = []
    cumulative_rank_us = [0.0, 0.0]
    symmetric_links = (
        config.h2d_gbps[0] == config.h2d_gbps[1]
        and config.d2h_gbps[0] == config.d2h_gbps[1]
    )

    for layer_idx in range(num_layers):
        reference = references[layer_idx]
        reference_groups = (
            tuple(reference[:local_heads]),
            tuple(reference[local_heads:]),
        )
        reference_metrics = score_partition(
            layer_idx,
            reference_groups,
            samples,
            residents[layer_idx],
            per_token_head_bytes,
            config,
        )
        if layer_idx < skip_layers:
            rank0, rank1 = reference_groups
            metrics = reference_metrics
        else:
            rank0, rank1, metrics = choose_partition(
                layer_idx,
                samples,
                residents[layer_idx],
                per_token_head_bytes,
                config,
                reference,
            )

        if symmetric_links:
            direct = abs(
                cumulative_rank_us[0]
                + metrics["mean_rank_us"][0]
                - cumulative_rank_us[1]
                - metrics["mean_rank_us"][1]
            )
            swapped = abs(
                cumulative_rank_us[0]
                + metrics["mean_rank_us"][1]
                - cumulative_rank_us[1]
                - metrics["mean_rank_us"][0]
            )
            if swapped < direct:
                rank0, rank1 = rank1, rank0
                metrics = score_partition(
                    layer_idx,
                    (rank0, rank1),
                    samples,
                    residents[layer_idx],
                    per_token_head_bytes,
                    config,
                )

        cumulative_rank_us[0] += metrics["mean_rank_us"][0]
        cumulative_rank_us[1] += metrics["mean_rank_us"][1]
        orders.append(list(rank0 + rank1))
        details.append(
            {
                "layer_idx": layer_idx,
                "rank_heads": [list(rank0), list(rank1)],
                "resident_heads": sorted(residents[layer_idx]),
                "predicted": metrics,
                "reference_objective_us": reference_metrics["objective_us"],
                "predicted_objective_improvement_pct": (
                    100.0
                    * (reference_metrics["objective_us"] - metrics["objective_us"])
                    / reference_metrics["objective_us"]
                    if reference_metrics["objective_us"] > 0.0
                    else 0.0
                ),
            }
        )
    return orders, details, cumulative_rank_us


def main() -> None:
    args = parse_args()
    _validate_fraction("shrinkage", args.shrinkage)
    _validate_fraction("tail_quantile", args.tail_quantile)
    if args.skip_steps < 0 or args.skip_layers < 0:
        raise ValueError("skip counts must be non-negative")
    h2d = (
        args.rank0_h2d_gbps
        if args.rank0_h2d_gbps is not None
        else args.h2d_gbps,
        args.rank1_h2d_gbps
        if args.rank1_h2d_gbps is not None
        else args.h2d_gbps,
    )
    d2h = (
        args.rank0_d2h_gbps
        if args.rank0_d2h_gbps is not None
        else args.d2h_gbps,
        args.rank1_d2h_gbps
        if args.rank1_d2h_gbps is not None
        else args.d2h_gbps,
    )
    if any(value <= 0.0 for value in (*h2d, *d2h)):
        raise ValueError(f"Bandwidths must be positive: h2d={h2d}, d2h={d2h}")
    penalty_values = {
        "h2d_launch_us": args.h2d_launch_us,
        "d2h_launch_us": args.d2h_launch_us,
        "active_head_us": args.active_head_us,
        "tail_weight": args.tail_weight,
        "imbalance_weight": args.imbalance_weight,
        "worst_profile_weight": args.worst_profile_weight,
    }
    if any(value < 0.0 for value in penalty_values.values()):
        raise ValueError(f"Cost weights must be non-negative: {penalty_values}")
    config = CostConfig(
        h2d_gbps=h2d,
        d2h_gbps=d2h,
        h2d_launch_us=args.h2d_launch_us,
        d2h_launch_us=args.d2h_launch_us,
        active_head_us=args.active_head_us,
        shrinkage=args.shrinkage,
        tail_quantile=args.tail_quantile,
        tail_weight=args.tail_weight,
        imbalance_weight=args.imbalance_weight,
        worst_profile_weight=args.worst_profile_weight,
    )
    samples, num_layers, num_heads, per_token_head_bytes = load_step_samples(
        args.transfer_json, args.skip_steps
    )
    residents = load_resident_heads(args.resident_heads_file, num_layers)
    references = load_reference_orders(args.reference_mapping, num_layers, num_heads)
    orders, layers, cumulative_rank_us = build_mapping(
        samples,
        num_layers,
        num_heads,
        per_token_head_bytes,
        residents,
        references,
        args.skip_layers,
        config,
    )

    profiles = sorted({sample.profile for sample in samples})
    output = {
        "orders": orders,
        "metadata": {
            "objective": "byte-aware TP2 critical-rank transfer cost",
            "scope": "data-path cost only; predicted gains are not end-to-end latency",
            "transfer_json": args.transfer_json,
            "resident_heads_file": args.resident_heads_file,
            "reference_mapping": args.reference_mapping,
            "profiles": profiles,
            "profile_count": len(profiles),
            "profile_steps": len(samples),
            "skip_steps": args.skip_steps,
            "skip_layers": args.skip_layers,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "per_token_head_bytes": per_token_head_bytes,
            "cost_config": {
                "h2d_gbps": h2d,
                "d2h_gbps": d2h,
                "h2d_launch_us": args.h2d_launch_us,
                "d2h_launch_us": args.d2h_launch_us,
                "active_head_us": args.active_head_us,
                "shrinkage": args.shrinkage,
                "tail_quantile": args.tail_quantile,
                "tail_weight": args.tail_weight,
                "imbalance_weight": args.imbalance_weight,
                "worst_profile_weight": args.worst_profile_weight,
            },
            "cumulative_mean_rank_us": cumulative_rank_us,
        },
        "layers": layers,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {num_layers} TP2 layer mappings from {len(profiles)} profiles "
        f"({len(samples)} post-skip steps) to {output_path}"
    )


if __name__ == "__main__":
    main()
