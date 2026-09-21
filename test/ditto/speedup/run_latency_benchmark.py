#!/usr/bin/env python3
"""Run burst-concurrency TTFT/TPOT measurements against one SGLang server."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
DEFAULT_CONCURRENCIES = [1, 2, 4, 8, 16, 24, 32, 48, 64]
DEFAULT_WARMUP_CAP = {"full": 16, "ditto": 64}
METRICS = [
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "request_throughput",
    "output_throughput",
    "duration",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send exactly N requests at once for each concurrency point. "
            "TTFT therefore includes any queueing inside the server."
        )
    )
    parser.add_argument("--method", choices=("full", "ditto"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--model-path",
        default="/jhe/Llama-3-8B-Instruct-Gradient-1048k",
    )
    parser.add_argument(
        "--dataset-path",
        default="/jhe/vllm/ShareGPT_V3_unfiltered_cleaned_split.json",
    )
    parser.add_argument(
        "--concurrencies",
        nargs="+",
        type=int,
        default=DEFAULT_CONCURRENCIES,
    )
    parser.add_argument("--input-len", type=int, default=8192)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=None,
        help=(
            "Warmup request count; defaults to the target concurrency capped at "
            "the method's active batch capacity."
        ),
    )
    parser.add_argument("--python-bin", default="/opt/conda/bin/python3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_DIR.parent / "results" / "latency",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    paths = [
        Path(args.python_bin),
        Path(args.model_path) / "config.json",
        Path(args.dataset_path),
        REPO_ROOT / "python" / "sglang" / "bench_serving.py",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(missing))
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.warmup_requests is not None and args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be >= 0")
    if any(value < 1 for value in args.concurrencies):
        raise ValueError("all concurrency values must be >= 1")


def run_once(
    args: argparse.Namespace,
    concurrency: int,
    repeat: int,
    raw_dir: Path,
    log_dir: Path,
) -> dict:
    stem = f"{args.method}_c{concurrency:02d}_r{repeat:02d}"
    result_path = raw_dir / f"{stem}.jsonl"
    log_path = log_dir / f"{stem}.log"
    result_path.unlink(missing_ok=True)

    warmup_requests = (
        min(concurrency, DEFAULT_WARMUP_CAP[args.method])
        if args.warmup_requests is None
        else args.warmup_requests
    )
    command = [
        args.python_bin,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--base-url",
        args.base_url.rstrip("/"),
        "--model",
        args.model_path,
        "--tokenizer",
        args.model_path,
        "--dataset-name",
        "random",
        "--dataset-path",
        args.dataset_path,
        "--num-prompts",
        str(concurrency),
        "--random-input-len",
        str(args.input_len),
        "--random-output-len",
        str(args.output_len),
        "--random-range-ratio",
        "1.0",
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(concurrency),
        "--warmup-requests",
        str(warmup_requests),
        "--ready-check-timeout-sec",
        "10",
        "--disable-tqdm",
        "--tokenize-prompt",
        "--seed",
        str(1000 + repeat),
        "--tag",
        stem,
        "--output-file",
        str(result_path),
    ]
    env = os.environ.copy()
    repo_python = str(REPO_ROOT / "python")
    env["PYTHONPATH"] = repo_python + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )

    print(
        f"[RUN] method={args.method} concurrency={concurrency} "
        f"repeat={repeat}/{args.repeats} warmup={warmup_requests}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-60:])
        raise RuntimeError(
            f"benchmark failed ({stem}, exit={completed.returncode}); "
            f"log={log_path}\n{tail}"
        )
    rows = [
        json.loads(line)
        for line in result_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one result in {result_path}, found {len(rows)}")
    result = rows[0]
    if int(result.get("completed", -1)) != concurrency:
        raise RuntimeError(
            f"{stem}: completed={result.get('completed')} but expected {concurrency}"
        )
    result.update(
        method=args.method,
        target_concurrency=concurrency,
        repeat=repeat,
        input_len=args.input_len,
        output_len=args.output_len,
    )
    print(
        f"[OK] c={concurrency} r={repeat} "
        f"TTFT={result['mean_ttft_ms']:.2f} ms "
        f"TPOT={result['mean_tpot_ms']:.2f} ms",
        flush=True,
    )
    return result


def load_existing_result(
    args: argparse.Namespace,
    concurrency: int,
    repeat: int,
    raw_dir: Path,
) -> dict | None:
    stem = f"{args.method}_c{concurrency:02d}_r{repeat:02d}"
    result_path = raw_dir / f"{stem}.jsonl"
    if not result_path.exists():
        return None

    try:
        rows = [
            json.loads(line)
            for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != 1:
            return None
        result = rows[0]
        expected = {
            "completed": concurrency,
            "total_input_tokens": concurrency * args.input_len,
            "total_output_tokens": concurrency * args.output_len,
        }
        if any(int(result.get(key, -1)) != value for key, value in expected.items()):
            print(f"[STALE] ignoring invalid result {result_path}", flush=True)
            return None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        print(f"[STALE] ignoring unreadable result {result_path}", flush=True)
        return None

    result.update(
        method=args.method,
        target_concurrency=concurrency,
        repeat=repeat,
        input_len=args.input_len,
        output_len=args.output_len,
    )
    print(
        f"[RESUME] method={args.method} concurrency={concurrency} repeat={repeat}",
        flush=True,
    )
    return result


def write_summary(args: argparse.Namespace, results: list[dict], output_dir: Path) -> None:
    records_path = output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=True) + "\n")

    summary_rows = []
    for concurrency in args.concurrencies:
        group = [
            row for row in results if row["target_concurrency"] == concurrency
        ]
        summary = {
            "method": args.method,
            "concurrency": concurrency,
            "repeats": len(group),
            "completed_per_repeat": concurrency,
            "input_len": args.input_len,
            "output_len": args.output_len,
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            summary[metric] = statistics.fmean(values)
        summary["repeat_std_mean_ttft_ms"] = (
            statistics.stdev(row["mean_ttft_ms"] for row in group)
            if len(group) > 1
            else 0.0
        )
        summary["repeat_std_mean_tpot_ms"] = (
            statistics.stdev(row["mean_tpot_ms"] for row in group)
            if len(group) > 1
            else 0.0
        )
        summary_rows.append(summary)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[DONE] records={records_path}", flush=True)
    print(f"[DONE] summary={summary_path}", flush=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.resolve() / args.method
    raw_dir = output_dir / "raw"
    log_dir = output_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for concurrency in args.concurrencies:
        for repeat in range(1, args.repeats + 1):
            result = load_existing_result(args, concurrency, repeat, raw_dir)
            if result is None:
                result = run_once(args, concurrency, repeat, raw_dir, log_dir)
            results.append(result)
    write_summary(args, results, output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise
