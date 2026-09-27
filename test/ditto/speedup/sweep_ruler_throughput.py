#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CLIENT = THIS_DIR / "launch_client.py"
DEFAULT_SERVER_BASE_URL = "http://127.0.0.1:30000"


CSV_COLUMNS = [
    "concurrency",
    "total_requests",
    "ok_requests",
    "failed_requests",
    "returncode",
    "service_window_s",
    "subprocess_wall_s",
    "request_per_s",
    "input_tok_per_s",
    "output_tok_per_s",
    "total_tok_per_s",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "latency_avg_ms",
    "latency_p50_ms",
    "latency_p90_ms",
    "latency_p99_ms",
    "raw_jsonl",
    "client_log",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep online Ditto throughput on RULER and plot token/s."
    )
    parser.add_argument("--concurrencies", default="1,2,4,8,16")
    parser.add_argument("--ruler-len", default="8K")
    parser.add_argument("--ruler-task", default="niah_single_3")
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--model-path", default="/models/Llama-3-8B-Instruct")
    parser.add_argument("--server-base-url", default=DEFAULT_SERVER_BASE_URL)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="Hold each concurrency point for this many seconds by continuously "
        "refilling requests. When set, --total-requests is treated as an optional cap; "
        "leave it unset to run unbounded for the duration.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help="Per-request client truncation cap. For DITTO_MAX_BATCH_SIZE=32 "
        "and DITTO_MAX_TOKENS=262144, keep this at 8192.",
    )
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument(
        "--requests-per-concurrency",
        type=int,
        default=4,
        help="Used when --total-requests is not set. Total = concurrency * this.",
    )
    parser.add_argument(
        "--total-requests",
        type=int,
        default=None,
        help="Use the same total request count for every concurrency.",
    )
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rid-prefix", default="llama-ruler8k-sweep")
    parser.add_argument("--launch-client", type=Path, default=DEFAULT_CLIENT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--send-text", action="store_true")
    parser.add_argument("--settle-sec", type=float, default=0.0)
    parser.add_argument(
        "--idle-timeout-sec",
        type=float,
        default=30.0,
        help="Wait for server running/waiting reqs to drop to zero after each stage.",
    )
    parser.add_argument(
        "--idle-poll-sec",
        type=float,
        default=0.5,
        help="Polling interval while waiting for the server to become idle.",
    )
    parser.add_argument(
        "--force-output-len",
        dest="force_output_len",
        action="store_true",
        default=True,
        help="Pass --ignore-eos and --min-new-tokens=max_new_tokens.",
    )
    parser.add_argument(
        "--no-force-output-len",
        dest="force_output_len",
        action="store_false",
        help="Allow requests to stop on EOS. Useful for accuracy, noisy for throughput.",
    )
    return parser.parse_args()


def parse_concurrencies(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.replace(",", " ").split():
        value = int(part)
        if value <= 0:
            raise ValueError(f"concurrency must be positive: {value}")
        values.append(value)
    if not values:
        raise ValueError("empty --concurrencies")
    return values


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * pct / 100.0
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return data[int(pos)]
    return data[low] * (high - pos) + data[high] * (pos - low)


def response_obj(row: dict[str, Any]) -> dict[str, Any]:
    text = row.get("response_text")
    if not text:
        return {}
    try:
        obj = json.loads(text)
    except Exception:
        return {}
    if isinstance(obj, list):
        obj = obj[0] if obj else {}
    return obj if isinstance(obj, dict) else {}


def to_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def extract_token_counts(row: dict[str, Any]) -> tuple[int, int]:
    obj = response_obj(row)
    meta = obj.get("meta_info") if isinstance(obj.get("meta_info"), dict) else {}
    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}

    prompt_tokens = to_int(
        meta.get("prompt_tokens"),
        to_int(usage.get("prompt_tokens"), to_int(row.get("prompt_tokens"))),
    )

    completion_tokens = to_int(meta.get("completion_tokens"), -1)
    if completion_tokens < 0:
        completion_tokens = to_int(usage.get("completion_tokens"), -1)
    if completion_tokens < 0 and isinstance(obj.get("output_ids"), list):
        completion_tokens = len(obj["output_ids"])
    if completion_tokens < 0:
        completion_tokens = 0
    return prompt_tokens, completion_tokens


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def fetch_json(url: str, timeout_sec: float = 5.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            return json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def get_server_loads(base_url: str) -> dict[str, Any] | None:
    loads_url = f"{base_url.rstrip('/')}/v1/loads?include=queues"
    return fetch_json(loads_url)


def wait_for_server_idle(
    base_url: str,
    timeout_sec: float,
    poll_sec: float,
) -> bool:
    deadline = time.time() + max(timeout_sec, 0.0)
    last_total: int | None = None
    while time.time() <= deadline:
        data = get_server_loads(base_url)
        if isinstance(data, dict):
            aggregate = data.get("aggregate")
            if isinstance(aggregate, dict):
                running = to_int(aggregate.get("total_running_reqs"), -1)
                waiting = to_int(aggregate.get("total_waiting_reqs"), -1)
                total = to_int(aggregate.get("total_reqs"), running + waiting)
                if running == 0 and waiting == 0 and total == 0:
                    return True
                last_total = total
        time.sleep(max(poll_sec, 0.05))
    if last_total is not None:
        print(
            f"[idle-wait-timeout] server still has total_reqs={last_total} "
            f"after {timeout_sec:.1f}s"
        )
    else:
        print(
            f"[idle-wait-timeout] could not query {base_url}/v1/loads "
            f"within {timeout_sec:.1f}s"
        )
    return False


def build_client_cmd(args: argparse.Namespace, concurrency: int, total: int) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(args.launch_client),
        "--server",
        args.server_base_url.rstrip("/"),
        "--dataset",
        "ruler",
        "--ruler-len",
        args.ruler_len,
        "--ruler-task",
        args.ruler_task,
        "--total-requests",
        str(total),
        "--concurrency",
        str(concurrency),
        "--ordered",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-seq-len",
        str(args.max_seq_len),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--max-samples",
        str(max(args.max_samples, total)),
        "--model-path",
        args.model_path,
        "--seed",
        str(args.seed),
        "--rid-prefix",
        f"{args.rid_prefix}-c{concurrency}",
    ]
    if args.duration_sec > 0:
        cmd += ["--duration-sec", str(args.duration_sec)]
    if args.data_file is not None:
        cmd += ["--data-file", str(args.data_file)]
    if args.send_text:
        cmd.append("--send-text")
    if args.force_output_len:
        cmd += ["--ignore-eos", "--min-new-tokens", str(args.max_new_tokens)]
    return cmd


def summarize_run(
    concurrency: int,
    total: int,
    returncode: int,
    subprocess_wall_s: float,
    raw_jsonl: Path,
    client_log: Path,
) -> dict[str, Any]:
    rows = read_jsonl(raw_jsonl)
    ok_rows = [
        r for r in rows if bool(r.get("ok")) and to_int(r.get("status")) == 200
    ]
    latencies = [float(r.get("latency_ms", 0.0)) for r in ok_rows]

    send_times = [
        float(r["send_time_unix"])
        for r in rows
        if isinstance(r.get("send_time_unix"), (int, float))
    ]
    done_times = [
        float(r["done_time_unix"])
        for r in rows
        if isinstance(r.get("done_time_unix"), (int, float))
    ]
    if send_times and done_times:
        service_window_s = max(done_times) - min(send_times)
    elif latencies:
        service_window_s = max(latencies) / 1000.0
    else:
        service_window_s = subprocess_wall_s
    service_window_s = max(service_window_s, 1e-9)

    prompt_tokens = 0
    completion_tokens = 0
    for row in ok_rows:
        p_tokens, c_tokens = extract_token_counts(row)
        prompt_tokens += p_tokens
        completion_tokens += c_tokens

    total_tokens = prompt_tokens + completion_tokens
    # Duration mode uses total=0 as an unlimited request sentinel.
    total = max(total, len(rows))
    return {
        "concurrency": concurrency,
        "total_requests": total,
        "ok_requests": len(ok_rows),
        "failed_requests": max(total - len(ok_rows), 0),
        "returncode": returncode,
        "service_window_s": round(service_window_s, 4),
        "subprocess_wall_s": round(subprocess_wall_s, 4),
        "request_per_s": round(len(ok_rows) / service_window_s, 4),
        "input_tok_per_s": round(prompt_tokens / service_window_s, 2),
        "output_tok_per_s": round(completion_tokens / service_window_s, 2),
        "total_tok_per_s": round(total_tokens / service_window_s, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_avg_ms": round(sum(latencies) / len(latencies), 2)
        if latencies
        else 0.0,
        "latency_p50_ms": round(percentile(latencies, 50), 2),
        "latency_p90_ms": round(percentile(latencies, 90), 2),
        "latency_p99_ms": round(percentile(latencies, 99), 2),
        "raw_jsonl": str(raw_jsonl),
        "client_log": str(client_log),
    }


def should_echo_client_line(line: str) -> bool:
    prefixes = (
        "[launch]",
        "[progress]",
        "[dispatch]",
        "[accepted]",
        "[send]",      # legacy client logs
        "[fail]",
        "[idle-wait-timeout]",
        "Traceback",
        "KeyboardInterrupt",
        "asyncio.exceptions.",
    )
    return line.startswith(prefixes)


def run_client_with_tee(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    terminal_tag: str,
) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            log_file.write(raw_line)
            log_file.flush()
            line = raw_line.rstrip("\n")
            if should_echo_client_line(line):
                print(f"[{terminal_tag}] {line}")
        return proc.wait()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_COLUMNS})


def plot_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: int(r["concurrency"]))
    xs = [int(r["concurrency"]) for r in rows]
    total_tps = [float(r["total_tok_per_s"]) for r in rows]
    input_tps = [float(r["input_tok_per_s"]) for r in rows]
    output_tps = [float(r["output_tok_per_s"]) for r in rows]
    rps = [float(r["request_per_s"]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(xs, total_tps, marker="o", label="total tok/s")
    axes[0].plot(xs, input_tps, marker="o", label="input tok/s")
    axes[0].plot(xs, output_tps, marker="o", label="output tok/s")
    axes[0].set_ylabel("tokens / second")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(xs, rps, marker="o", color="tab:green", label="requests/s")
    axes[1].set_xlabel("concurrency")
    axes[1].set_ylabel("requests / second")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Ditto online throughput on RULER")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\nconcurrency  ok/total  req/s    input tok/s  output tok/s  total tok/s  p50 ms")
    print("-" * 82)
    for row in sorted(rows, key=lambda r: int(r["concurrency"])):
        print(
            f"{int(row['concurrency']):>11}  "
            f"{int(row['ok_requests']):>2}/{int(row['total_requests']):<5}  "
            f"{float(row['request_per_s']):>7.3f}  "
            f"{float(row['input_tok_per_s']):>11.2f}  "
            f"{float(row['output_tok_per_s']):>12.2f}  "
            f"{float(row['total_tok_per_s']):>11.2f}  "
            f"{float(row['latency_p50_ms']):>8.2f}"
        )
    good = [r for r in rows if int(r.get("ok_requests", 0)) > 0]
    if good:
        best_total = max(good, key=lambda r: float(r["total_tok_per_s"]))
        best_output = max(good, key=lambda r: float(r["output_tok_per_s"]))
        print(
            "\nbest total tok/s: "
            f"c={best_total['concurrency']} total_tok/s={best_total['total_tok_per_s']}"
        )
        print(
            "best output tok/s: "
            f"c={best_output['concurrency']} output_tok/s={best_output['output_tok_per_s']}"
        )


def main() -> None:
    args = parse_args()
    concurrencies = parse_concurrencies(args.concurrencies)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = THIS_DIR / "throughput_results" / f"ruler_{args.ruler_len}_{timestamp}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"output_dir={out_dir}")
    print(f"concurrencies={concurrencies}")
    print(
        "fixed_output_len="
        f"{args.force_output_len} max_new_tokens={args.max_new_tokens}"
    )
    print(
        f"server_base_url={args.server_base_url} warmup_requests={args.warmup_requests} "
        f"idle_timeout_sec={args.idle_timeout_sec} duration_sec={args.duration_sec}"
    )

    if args.warmup_requests > 0:
        warmup_cmd = build_client_cmd(args, 1, args.warmup_requests)
        warmup_cmd += ["--log-file", str(out_dir / "warmup.jsonl")]
        warmup_log = out_dir / "warmup.log"
        print(f"\n[warmup] {' '.join(warmup_cmd)}")
        if not args.dry_run:
            run_client_with_tee(
                warmup_cmd,
                cwd=THIS_DIR,
                log_path=warmup_log,
                terminal_tag="warmup",
            )
            wait_for_server_idle(
                args.server_base_url,
                timeout_sec=args.idle_timeout_sec,
                poll_sec=args.idle_poll_sec,
            )

    rows: list[dict[str, Any]] = []
    for concurrency in concurrencies:
        total = (
            args.total_requests
            if args.total_requests is not None
            else concurrency * args.requests_per_concurrency
        )
        if args.duration_sec > 0 and args.total_requests is None:
            total = 0
        else:
            total = max(total, concurrency)
        cmd = build_client_cmd(args, concurrency, total)
        raw_jsonl = out_dir / f"raw_c{concurrency}.jsonl"
        client_log = out_dir / f"client_c{concurrency}.log"
        cmd += ["--log-file", str(raw_jsonl)]
        print(f"\n[run] concurrency={concurrency} total_requests={total}")
        print(" ".join(cmd))
        if args.dry_run:
            continue

        start = time.perf_counter()
        returncode = run_client_with_tee(
            cmd,
            cwd=THIS_DIR,
            log_path=client_log,
            terminal_tag=f"c{concurrency}",
        )
        subprocess_wall_s = time.perf_counter() - start
        row = summarize_run(
            concurrency,
            total,
            returncode,
            subprocess_wall_s,
            raw_jsonl,
            client_log,
        )
        rows.append(row)
        print(
            f"[done] c={concurrency} ok={row['ok_requests']}/{row['total_requests']} "
            f"req/s={row['request_per_s']} total_tok/s={row['total_tok_per_s']} "
            f"output_tok/s={row['output_tok_per_s']}"
        )
        wait_for_server_idle(
            args.server_base_url,
            timeout_sec=args.idle_timeout_sec,
            poll_sec=args.idle_poll_sec,
        )
        if args.settle_sec > 0:
            time.sleep(args.settle_sec)

    if args.dry_run:
        return

    csv_path = out_dir / "ruler_throughput.csv"
    write_csv(csv_path, rows)
    print_summary(rows)
    print(f"\ncsv: {csv_path}")

    if not args.no_plot:
        png_path = out_dir / "ruler_throughput.png"
        plot_rows(png_path, rows)
        print(f"plot: {png_path}")

    if any(row["returncode"] != 0 or row["failed_requests"] > 0
           or row["ok_requests"] == 0 for row in rows):
        raise SystemExit("Throughput sweep failed; inspect CSV and client logs.")


if __name__ == "__main__":
    main()
