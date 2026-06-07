#!/usr/bin/env python3
import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path


STALL_RE = re.compile(
    r"\[DittoStall\].*?stage=(?P<stage>\S+)\s+layer=(?P<layer>\d+)(?:\s+|$)(?P<rest>.*)"
)
MS_RE = re.compile(r"(?:^|\s)ms=(?P<ms>[0-9.]+)")


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * pct)))
    return values[idx]


def summarize(values):
    return {
        "count": len(values),
        "avg": statistics.fmean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize Ditto decode step timing from server logs."
    )
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--last-steps", type=int, default=0)
    args = parser.parse_args()

    per_stage = defaultdict(list)
    per_stage_layer = defaultdict(list)
    per_decode_step = defaultdict(lambda: defaultdict(float))
    decode_step = -1

    for line in args.log_file.open(encoding="utf-8", errors="replace"):
        match = STALL_RE.search(line)
        if not match:
            continue
        stage = match.group("stage")
        layer = int(match.group("layer"))
        rest = match.group("rest")

        if stage == "decode_enter" and layer == 0:
            decode_step += 1
            continue

        if not stage.endswith("_exit"):
            continue
        ms_match = MS_RE.search(rest)
        if not ms_match:
            continue

        ms = float(ms_match.group("ms"))
        stage_name = stage.removesuffix("_exit")
        per_stage[stage_name].append(ms)
        per_stage_layer[(stage_name, layer)].append(ms)
        if decode_step >= 0:
            per_decode_step[decode_step][stage_name] += ms

    step_items = sorted(per_decode_step.items())
    if args.last_steps > 0:
        step_items = step_items[-args.last_steps :]

    print("== per decode-token summed stages ==")
    if not step_items:
        print("no decode steps found")
    else:
        stages = sorted({s for _, row in step_items for s in row})
        print("decode_step " + " ".join(stages) + " total_ms")
        for step, row in step_items:
            total = sum(row.values())
            vals = " ".join(f"{row.get(stage, 0.0):.3f}" for stage in stages)
            print(f"{step:>11} {vals} {total:.3f}")

    print("\n== per stage across all layers/steps ==")
    for stage in sorted(per_stage):
        stats = summarize(per_stage[stage])
        print(
            f"{stage:>24} count={stats['count']:<6} avg={stats['avg']:.3f} "
            f"p50={stats['p50']:.3f} p90={stats['p90']:.3f} "
            f"p99={stats['p99']:.3f} max={stats['max']:.3f}"
        )

    print("\n== per stage per layer ==")
    for (stage, layer), values in sorted(per_stage_layer.items()):
        stats = summarize(values)
        print(
            f"{stage:>24} layer={layer:<3} count={stats['count']:<5} "
            f"avg={stats['avg']:.3f} p90={stats['p90']:.3f} max={stats['max']:.3f}"
        )


if __name__ == "__main__":
    main()
