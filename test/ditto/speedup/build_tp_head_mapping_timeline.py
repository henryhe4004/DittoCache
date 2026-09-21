#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from build_tp_head_mapping_cost_model import (
    StepSample,
    load_reference_orders,
    load_resident_heads,
    load_step_samples,
)


@dataclass(frozen=True)
class TimelineConfig:
    h2d_gbps: tuple[float, float]
    d2h_gbps: tuple[float, float]
    h2d_launch_us: float
    d2h_launch_us: float
    active_head_us: float
    tail_quantile: float
    tail_weight: float
    worst_profile_weight: float
    beam_width: int
    local_passes: int


@dataclass(frozen=True)
class TimelineTiming:
    pre_wait_us: np.ndarray
    post_launch_us: np.ndarray
    tp_sync_us: np.ndarray
    resolved_profiles: dict[str, dict[str, list[float]]]
    source: str | None


@dataclass(frozen=True)
class TransferCosts:
    partitions: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    h2d_us: np.ndarray
    d2h_us: np.ndarray


@dataclass(frozen=True)
class TimelineState:
    layer_end_us: np.ndarray
    engine_ready_us: np.ndarray
    prefetch_launch_us: np.ndarray
    path: tuple[int, ...]
    objective: float
    worst_profile_regret: float
    distance: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a TP2 KV-head mapping by jointly optimizing the complete "
            "layer timeline. Profiles must contain per-head prefetch masks."
        )
    )
    parser.add_argument("--transfer-json", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--per-profile-output-dir",
        help="Also optimize one mapping from each independently collected profile.",
    )
    parser.add_argument("--resident-heads-file")
    parser.add_argument("--reference-mapping")
    parser.add_argument("--timing-json")
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
        "--pre-wait-us",
        type=float,
        default=0.0,
        help="Default compute time before a layer consumes prefetched KV.",
    )
    parser.add_argument(
        "--post-launch-us",
        type=float,
        default=0.0,
        help="Default compute time after launching the next-layer prefetch.",
    )
    parser.add_argument(
        "--tp-sync-us",
        type=float,
        default=0.0,
        help="Default fixed TP synchronization time at each layer boundary.",
    )
    parser.add_argument("--tail-quantile", type=float, default=0.95)
    parser.add_argument("--tail-weight", type=float, default=0.10)
    parser.add_argument("--worst-profile-weight", type=float, default=0.25)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--local-passes", type=int, default=1)
    return parser.parse_args()


def _validate_fraction(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _profile_label(profile: str) -> str:
    match = re.search(r"seq([0-9]+(?:\.[0-9]+)?[KMG]?)", profile, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return Path(profile).stem


def _expand_layer_values(value: Any, num_layers: int, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        result = [float(value)] * num_layers
    elif isinstance(value, list) and len(value) == num_layers:
        result = [float(item) for item in value]
    else:
        raise ValueError(f"{name} must be a number or a {num_layers}-element list")
    if any(item < 0.0 or not math.isfinite(item) for item in result):
        raise ValueError(f"{name} values must be finite and non-negative")
    return result


def _merge_timing_entry(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(base)
    for key in ("pre_wait_us", "post_launch_us", "tp_sync_us"):
        if key in override:
            merged[key] = override[key]
    return merged


def load_timeline_timing(
    path: str | None,
    samples: list[StepSample],
    num_layers: int,
    pre_wait_us: float,
    post_launch_us: float,
    tp_sync_us: float,
) -> TimelineTiming:
    defaults: dict[str, Any] = {
        "pre_wait_us": pre_wait_us,
        "post_launch_us": post_launch_us,
        "tp_sync_us": tp_sync_us,
    }
    profile_overrides: dict[str, Any] = {}
    if path is not None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Timing calibration must be a JSON object")
        raw_default = payload.get("default", {})
        raw_profiles = payload.get("profiles", {})
        if not isinstance(raw_default, dict) or not isinstance(raw_profiles, dict):
            raise TypeError("Timing calibration default/profiles must be objects")
        defaults = _merge_timing_entry(defaults, raw_default)
        profile_overrides = raw_profiles

    resolved: dict[str, dict[str, list[float]]] = {}
    for profile in sorted({sample.profile for sample in samples}):
        basename = Path(profile).name
        label = _profile_label(profile)
        entry = defaults
        for key in (label, f"seq{label}", basename, profile):
            override = profile_overrides.get(key)
            if override is not None:
                if not isinstance(override, dict):
                    raise TypeError(f"Timing profile {key!r} must be an object")
                entry = _merge_timing_entry(entry, override)
        resolved[profile] = {
            field: _expand_layer_values(entry[field], num_layers, f"{profile}:{field}")
            for field in ("pre_wait_us", "post_launch_us", "tp_sync_us")
        }

    return TimelineTiming(
        pre_wait_us=np.asarray(
            [resolved[sample.profile]["pre_wait_us"] for sample in samples],
            dtype=np.float64,
        ),
        post_launch_us=np.asarray(
            [resolved[sample.profile]["post_launch_us"] for sample in samples],
            dtype=np.float64,
        ),
        tp_sync_us=np.asarray(
            [resolved[sample.profile]["tp_sync_us"] for sample in samples],
            dtype=np.float64,
        ),
        resolved_profiles=resolved,
        source=str(Path(path).resolve()) if path is not None else None,
    )


def enumerate_partitions(
    num_heads: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    local_heads = num_heads // 2
    all_heads = tuple(range(num_heads))
    return tuple(
        (
            rank0,
            tuple(head for head in all_heads if head not in rank0),
        )
        for rank0 in itertools.combinations(all_heads, local_heads)
    )


def build_transfer_costs(
    samples: list[StepSample],
    num_layers: int,
    num_heads: int,
    per_token_head_bytes: int,
    residents: list[set[int]],
    config: TimelineConfig,
) -> TransferCosts:
    partitions = enumerate_partitions(num_heads)
    masks = np.asarray([sample.masks for sample in samples], dtype=np.int8)
    prefetch_k = np.asarray([sample.prefetch_k for sample in samples], dtype=np.float64)
    h2d = np.zeros(
        (num_layers, len(partitions), len(samples), 2), dtype=np.float64
    )
    d2h = np.zeros_like(h2d)

    for layer_idx in range(num_layers):
        resident = residents[layer_idx]
        for candidate_idx, rank_heads in enumerate(partitions):
            for rank, heads in enumerate(rank_heads):
                offloaded = tuple(head for head in heads if head not in resident)
                if not offloaded:
                    continue
                active = masks[:, layer_idx, offloaded].sum(axis=1, dtype=np.int64)
                transfer = (
                    active.astype(np.float64)
                    * prefetch_k
                    * per_token_head_bytes
                    / (config.h2d_gbps[rank] * 1000.0)
                )
                transfer += np.where(
                    active > 0,
                    config.h2d_launch_us + active * config.active_head_us,
                    0.0,
                )
                h2d[layer_idx, candidate_idx, :, rank] = transfer
                d2h[layer_idx, candidate_idx, :, rank] = (
                    len(offloaded)
                    * per_token_head_bytes
                    / (config.d2h_gbps[rank] * 1000.0)
                    + config.d2h_launch_us
                )
    return TransferCosts(partitions=partitions, h2d_us=h2d, d2h_us=d2h)


def _initial_state(num_samples: int) -> TimelineState:
    return TimelineState(
        layer_end_us=np.zeros(num_samples, dtype=np.float64),
        engine_ready_us=np.zeros((num_samples, 2), dtype=np.float64),
        prefetch_launch_us=np.zeros((num_samples, 2), dtype=np.float64),
        path=(),
        objective=0.0,
        worst_profile_regret=0.0,
        distance=0,
    )


def _transition_batch(
    state: TimelineState,
    h2d_us: np.ndarray,
    d2h_us: np.ndarray,
    pre_wait_us: np.ndarray,
    post_launch_us: np.ndarray,
    tp_sync_us: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if h2d_us.ndim == 2:
        h2d_us = h2d_us[None, ...]
        d2h_us = d2h_us[None, ...]
    engine_ready = state.engine_ready_us[None, ...]
    launch = state.prefetch_launch_us[None, ...]
    has_h2d = h2d_us > 0.0
    h2d_done = np.where(
        has_h2d,
        np.maximum(engine_ready, launch) + h2d_us,
        engine_ready,
    )

    consumer_arrival = (
        state.layer_end_us[None, :, None] + pre_wait_us[None, :, None]
    )
    data_ready = np.where(has_h2d, h2d_done, consumer_arrival)
    wait_us = np.maximum(data_ready - consumer_arrival, 0.0)
    ready = consumer_arrival + wait_us

    has_d2h = d2h_us > 0.0
    d2h_done = np.where(
        has_d2h,
        np.maximum(h2d_done, ready) + d2h_us,
        h2d_done,
    )
    next_launch = np.maximum(ready, d2h_done)
    rank_finish = next_launch + post_launch_us[None, :, None]
    next_end = rank_finish.max(axis=2) + tp_sync_us[None, :]
    return next_end, d2h_done, next_launch, wait_us


def _profile_indices(samples: list[StepSample]) -> dict[str, np.ndarray]:
    profiles: dict[str, list[int]] = {}
    for sample_idx, sample in enumerate(samples):
        profiles.setdefault(sample.profile, []).append(sample_idx)
    return {
        profile: np.asarray(indices, dtype=np.int64)
        for profile, indices in sorted(profiles.items())
    }


def _score_end_batch(
    layer_end_us: np.ndarray,
    reference_end_us: np.ndarray,
    normalizer_us: np.ndarray,
    profile_indices: dict[str, np.ndarray],
    config: TimelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if layer_end_us.ndim == 1:
        layer_end_us = layer_end_us[None, :]
    regret = (layer_end_us - reference_end_us[None, :]) / np.maximum(
        normalizer_us[None, :], 1e-9
    )
    profile_scores = []
    for indices in profile_indices.values():
        values = regret[:, indices]
        mean = values.mean(axis=1)
        ordered = np.sort(values, axis=1)
        tail_pos = min(
            max(math.ceil(config.tail_quantile * len(indices)) - 1, 0),
            len(indices) - 1,
        )
        tail = ordered[:, tail_pos]
        profile_scores.append(mean + config.tail_weight * np.maximum(tail - mean, 0.0))
    scores = np.stack(profile_scores, axis=1)
    mean_score = scores.mean(axis=1)
    worst_score = scores.max(axis=1)
    objective = mean_score + config.worst_profile_weight * np.maximum(
        worst_score - mean_score, 0.0
    )
    return objective, worst_score


def _reference_candidate_ids(
    partitions: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    references: list[list[int]],
) -> list[int]:
    by_rank0 = {
        tuple(sorted(rank_heads[0])): candidate_idx
        for candidate_idx, rank_heads in enumerate(partitions)
    }
    local_heads = len(references[0]) // 2
    return [by_rank0[tuple(sorted(order[:local_heads]))] for order in references]


def _simulate_path(
    path: tuple[int, ...] | list[int],
    costs: TransferCosts,
    timing: TimelineTiming,
) -> tuple[TimelineState, np.ndarray]:
    state = _initial_state(costs.h2d_us.shape[2])
    waits = []
    for layer_idx, candidate_idx in enumerate(path):
        next_end, engine_ready, next_launch, wait_us = _transition_batch(
            state,
            costs.h2d_us[layer_idx, candidate_idx],
            costs.d2h_us[layer_idx, candidate_idx],
            timing.pre_wait_us[:, layer_idx],
            timing.post_launch_us[:, layer_idx],
            timing.tp_sync_us[:, layer_idx],
        )
        state = TimelineState(
            layer_end_us=next_end[0],
            engine_ready_us=engine_ready[0],
            prefetch_launch_us=next_launch[0],
            path=tuple(path[: layer_idx + 1]),
            objective=0.0,
            worst_profile_regret=0.0,
            distance=0,
        )
        waits.append(wait_us[0])
    return state, np.stack(waits, axis=1)


def _reference_prefixes(
    reference_path: list[int],
    costs: TransferCosts,
    timing: TimelineTiming,
) -> tuple[list[np.ndarray], np.ndarray]:
    state = _initial_state(costs.h2d_us.shape[2])
    prefixes = []
    for layer_idx, candidate_idx in enumerate(reference_path):
        next_end, engine_ready, next_launch, _ = _transition_batch(
            state,
            costs.h2d_us[layer_idx, candidate_idx],
            costs.d2h_us[layer_idx, candidate_idx],
            timing.pre_wait_us[:, layer_idx],
            timing.post_launch_us[:, layer_idx],
            timing.tp_sync_us[:, layer_idx],
        )
        state = TimelineState(
            layer_end_us=next_end[0],
            engine_ready_us=engine_ready[0],
            prefetch_launch_us=next_launch[0],
            path=tuple(reference_path[: layer_idx + 1]),
            objective=0.0,
            worst_profile_regret=0.0,
            distance=0,
        )
        prefixes.append(state.layer_end_us.copy())
    return prefixes, prefixes[-1]


def _partition_distance(
    partition: tuple[tuple[int, ...], tuple[int, ...]], reference_order: list[int]
) -> int:
    local_heads = len(reference_order) // 2
    return len(set(partition[0]) ^ set(reference_order[:local_heads]))


def _score_path(
    path: tuple[int, ...] | list[int],
    costs: TransferCosts,
    timing: TimelineTiming,
    reference_end_us: np.ndarray,
    profile_indices: dict[str, np.ndarray],
    config: TimelineConfig,
) -> tuple[float, float, TimelineState, np.ndarray]:
    state, waits = _simulate_path(path, costs, timing)
    objective, worst = _score_end_batch(
        state.layer_end_us,
        reference_end_us,
        reference_end_us,
        profile_indices,
        config,
    )
    return float(objective[0]), float(worst[0]), state, waits


def optimize_timeline(
    costs: TransferCosts,
    timing: TimelineTiming,
    samples: list[StepSample],
    references: list[list[int]],
    skip_layers: int,
    config: TimelineConfig,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    num_layers, num_candidates, num_samples, _ = costs.h2d_us.shape
    reference_path = _reference_candidate_ids(costs.partitions, references)
    reference_prefixes, reference_end_us = _reference_prefixes(
        reference_path, costs, timing
    )
    groups = _profile_indices(samples)
    candidate_distances = np.asarray(
        [
            [
                _partition_distance(partition, references[layer_idx])
                for partition in costs.partitions
            ]
            for layer_idx in range(num_layers)
        ],
        dtype=np.int64,
    )
    beam = [_initial_state(num_samples)]

    for layer_idx in range(num_layers):
        if layer_idx < skip_layers:
            allowed = np.asarray([reference_path[layer_idx]], dtype=np.int64)
        else:
            allowed = np.arange(num_candidates, dtype=np.int64)
        expansion_keys: list[tuple[float, float, int, tuple[int, ...], int, int]] = []
        for parent_idx, state in enumerate(beam):
            next_end, _, _, _ = _transition_batch(
                state,
                costs.h2d_us[layer_idx, allowed],
                costs.d2h_us[layer_idx, allowed],
                timing.pre_wait_us[:, layer_idx],
                timing.post_launch_us[:, layer_idx],
                timing.tp_sync_us[:, layer_idx],
            )
            objectives, worst = _score_end_batch(
                next_end,
                reference_prefixes[layer_idx],
                reference_end_us,
                groups,
                config,
            )
            for position, candidate_idx in enumerate(allowed):
                candidate = int(candidate_idx)
                path = state.path + (candidate,)
                distance = state.distance + int(
                    candidate_distances[layer_idx, candidate]
                )
                expansion_keys.append(
                    (
                        round(float(objectives[position]), 12),
                        round(float(worst[position]), 12),
                        distance,
                        path,
                        parent_idx,
                        candidate,
                    )
                )

        expansion_keys.sort(key=lambda item: item[:4])
        next_beam = []
        for objective, worst, distance, path, parent_idx, candidate in expansion_keys[
            : config.beam_width
        ]:
            parent = beam[parent_idx]
            next_end, engine_ready, next_launch, _ = _transition_batch(
                parent,
                costs.h2d_us[layer_idx, candidate],
                costs.d2h_us[layer_idx, candidate],
                timing.pre_wait_us[:, layer_idx],
                timing.post_launch_us[:, layer_idx],
                timing.tp_sync_us[:, layer_idx],
            )
            next_beam.append(
                TimelineState(
                    layer_end_us=next_end[0],
                    engine_ready_us=engine_ready[0],
                    prefetch_launch_us=next_launch[0],
                    path=path,
                    objective=objective,
                    worst_profile_regret=worst,
                    distance=distance,
                )
            )
        beam = next_beam

    best = min(
        beam,
        key=lambda state: (
            state.objective,
            state.worst_profile_regret,
            state.distance,
            state.path,
        ),
    )
    path = list(best.path)
    objective, worst, _, _ = _score_path(
        path, costs, timing, reference_end_us, groups, config
    )

    completed_passes = 0
    completed_moves = 0
    for _ in range(config.local_passes):
        pass_improved = False
        for layer_idx in range(skip_layers, num_layers):
            old_candidate = path[layer_idx]
            best_move = (
                round(objective, 12),
                round(worst, 12),
                old_candidate,
            )
            for candidate in range(num_candidates):
                if candidate == old_candidate:
                    continue
                trial = path.copy()
                trial[layer_idx] = candidate
                trial_objective, trial_worst, _, _ = _score_path(
                    trial, costs, timing, reference_end_us, groups, config
                )
                key = (
                    round(trial_objective, 12),
                    round(trial_worst, 12),
                    candidate,
                )
                if key < best_move:
                    best_move = key
            if best_move[0] < round(objective, 12):
                objective, worst, candidate = best_move
                path[layer_idx] = candidate
                completed_moves += 1
                pass_improved = True
        if not pass_improved:
            break
        completed_passes += 1

    objective, worst, final_state, waits = _score_path(
        path, costs, timing, reference_end_us, groups, config
    )
    if objective > 1e-12:
        path = reference_path
        objective, worst, final_state, waits = _score_path(
            path, costs, timing, reference_end_us, groups, config
        )

    profile_metrics = {}
    regret = (final_state.layer_end_us - reference_end_us) / np.maximum(
        reference_end_us, 1e-9
    )
    for profile, indices in groups.items():
        profile_wait = waits[indices]
        profile_metrics[profile] = {
            "label": _profile_label(profile),
            "samples": int(len(indices)),
            "reference_mean_timeline_us": float(reference_end_us[indices].mean()),
            "optimized_mean_timeline_us": float(
                final_state.layer_end_us[indices].mean()
            ),
            "mean_regret": float(regret[indices].mean()),
            "mean_critical_exposed_wait_us": float(
                profile_wait.max(axis=2).sum(axis=1).mean()
            ),
            "mean_rank_exposed_wait_us": [
                float(profile_wait[:, :, rank].sum(axis=1).mean())
                for rank in range(2)
            ],
        }
    return tuple(path), {
        "objective": objective,
        "worst_profile_regret": worst,
        "reference_path": reference_path,
        "reference_mean_timeline_us": float(reference_end_us.mean()),
        "optimized_mean_timeline_us": float(final_state.layer_end_us.mean()),
        "local_improvement_passes": completed_passes,
        "local_improvement_moves": completed_moves,
        "profiles": profile_metrics,
    }


def _build_output(
    path: tuple[int, ...],
    metrics: dict[str, Any],
    costs: TransferCosts,
    samples: list[StepSample],
    residents: list[set[int]],
    references: list[list[int]],
    timing: TimelineTiming,
    config: TimelineConfig,
    args: argparse.Namespace,
    per_token_head_bytes: int,
) -> dict[str, Any]:
    orders = [
        list(costs.partitions[candidate][0] + costs.partitions[candidate][1])
        for candidate in path
    ]
    layers = []
    local_heads = len(orders[0]) // 2
    for layer_idx, (candidate, order) in enumerate(zip(path, orders)):
        rank_heads = costs.partitions[candidate]
        layers.append(
            {
                "layer_idx": layer_idx,
                "rank_heads": [list(rank_heads[0]), list(rank_heads[1])],
                "resident_heads": sorted(residents[layer_idx]),
                "reference_rank_heads": [
                    references[layer_idx][:local_heads],
                    references[layer_idx][local_heads:],
                ],
                "changed_from_reference": set(order[:local_heads])
                != set(references[layer_idx][:local_heads]),
                "mean_h2d_us": [
                    float(costs.h2d_us[layer_idx, candidate, :, rank].mean())
                    for rank in range(2)
                ],
                "mean_d2h_us": [
                    float(costs.d2h_us[layer_idx, candidate, :, rank].mean())
                    for rank in range(2)
                ],
            }
        )
    return {
        "orders": orders,
        "metadata": {
            "objective": "normalized full-layer TP2 timeline regret",
            "scope": (
                "transfer timeline proxy; end-to-end prediction requires calibrated "
                "per-profile timing"
            ),
            "transfer_json": args.transfer_json,
            "resident_heads_file": args.resident_heads_file,
            "reference_mapping": args.reference_mapping,
            "timing_json": timing.source,
            "profiles": sorted({sample.profile for sample in samples}),
            "profile_count": len({sample.profile for sample in samples}),
            "profile_steps": len(samples),
            "skip_steps": args.skip_steps,
            "skip_layers": args.skip_layers,
            "num_layers": len(path),
            "num_heads": len(orders[0]),
            "per_token_head_bytes": per_token_head_bytes,
            "search": {
                "algorithm": "layerwise beam search plus full-timeline coordinate refinement",
                "beam_width": config.beam_width,
                "local_passes": config.local_passes,
            },
            "cost_config": {
                "h2d_gbps": config.h2d_gbps,
                "d2h_gbps": config.d2h_gbps,
                "h2d_launch_us": config.h2d_launch_us,
                "d2h_launch_us": config.d2h_launch_us,
                "active_head_us": config.active_head_us,
                "tail_quantile": config.tail_quantile,
                "tail_weight": config.tail_weight,
                "worst_profile_weight": config.worst_profile_weight,
            },
            "resolved_timing": timing.resolved_profiles,
            "timeline_metrics": metrics,
        },
        "layers": layers,
    }


def _write_output(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _slice_timing(
    timing: TimelineTiming,
    samples: list[StepSample],
    indices: list[int],
) -> TimelineTiming:
    index = np.asarray(indices, dtype=np.int64)
    selected = {samples[sample_idx].profile for sample_idx in indices}
    return TimelineTiming(
        pre_wait_us=timing.pre_wait_us[index],
        post_launch_us=timing.post_launch_us[index],
        tp_sync_us=timing.tp_sync_us[index],
        resolved_profiles={
            profile: values
            for profile, values in timing.resolved_profiles.items()
            if profile in selected
        },
        source=timing.source,
    )


def _slice_costs(costs: TransferCosts, indices: list[int]) -> TransferCosts:
    index = np.asarray(indices, dtype=np.int64)
    return TransferCosts(
        partitions=costs.partitions,
        h2d_us=costs.h2d_us[:, :, index, :],
        d2h_us=costs.d2h_us[:, :, index, :],
    )


def main() -> None:
    args = parse_args()
    _validate_fraction("tail_quantile", args.tail_quantile)
    if args.skip_steps < 0 or args.skip_layers < 0:
        raise ValueError("skip counts must be non-negative")
    if args.beam_width <= 0 or args.local_passes < 0:
        raise ValueError("beam width must be positive and local passes non-negative")

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
    numeric_values = (
        *h2d,
        *d2h,
        args.h2d_launch_us,
        args.d2h_launch_us,
        args.active_head_us,
        args.pre_wait_us,
        args.post_launch_us,
        args.tp_sync_us,
        args.tail_weight,
        args.worst_profile_weight,
    )
    if any(value < 0.0 or not math.isfinite(value) for value in numeric_values):
        raise ValueError("Costs and weights must be finite and non-negative")
    if any(value <= 0.0 for value in (*h2d, *d2h)):
        raise ValueError(f"Bandwidths must be positive: h2d={h2d}, d2h={d2h}")

    config = TimelineConfig(
        h2d_gbps=h2d,
        d2h_gbps=d2h,
        h2d_launch_us=args.h2d_launch_us,
        d2h_launch_us=args.d2h_launch_us,
        active_head_us=args.active_head_us,
        tail_quantile=args.tail_quantile,
        tail_weight=args.tail_weight,
        worst_profile_weight=args.worst_profile_weight,
        beam_width=args.beam_width,
        local_passes=args.local_passes,
    )
    samples, num_layers, num_heads, per_token_head_bytes = load_step_samples(
        args.transfer_json, args.skip_steps
    )
    residents = load_resident_heads(args.resident_heads_file, num_layers)
    references = load_reference_orders(args.reference_mapping, num_layers, num_heads)
    timing = load_timeline_timing(
        args.timing_json,
        samples,
        num_layers,
        args.pre_wait_us,
        args.post_launch_us,
        args.tp_sync_us,
    )
    costs = build_transfer_costs(
        samples,
        num_layers,
        num_heads,
        per_token_head_bytes,
        residents,
        config,
    )

    path, metrics = optimize_timeline(
        costs, timing, samples, references, args.skip_layers, config
    )
    payload = _build_output(
        path,
        metrics,
        costs,
        samples,
        residents,
        references,
        timing,
        config,
        args,
        per_token_head_bytes,
    )
    _write_output(args.output, payload)
    print(
        f"Wrote joint {num_layers}-layer mapping from "
        f"{len({sample.profile for sample in samples})} profiles to {args.output}; "
        f"normalized regret={metrics['objective']:.6%}"
    )

    if args.per_profile_output_dir:
        output_dir = Path(args.per_profile_output_dir)
        by_profile = _profile_indices(samples)
        for profile, sample_indices in by_profile.items():
            indices = [int(index) for index in sample_indices]
            profile_samples = [samples[index] for index in indices]
            profile_costs = _slice_costs(costs, indices)
            profile_timing = _slice_timing(timing, samples, indices)
            profile_path, profile_metrics = optimize_timeline(
                profile_costs,
                profile_timing,
                profile_samples,
                references,
                args.skip_layers,
                config,
            )
            profile_payload = _build_output(
                profile_path,
                profile_metrics,
                profile_costs,
                profile_samples,
                residents,
                references,
                profile_timing,
                config,
                args,
                per_token_head_bytes,
            )
            profile_output = output_dir / f"timeline-{_profile_label(profile)}.json"
            _write_output(profile_output, profile_payload)
            print(
                f"Wrote {_profile_label(profile)} mapping to {profile_output}; "
                f"normalized regret={profile_metrics['objective']:.6%}"
            )


if __name__ == "__main__":
    main()
