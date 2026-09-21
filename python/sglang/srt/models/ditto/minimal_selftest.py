from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from transformers import AutoConfig

from sglang import Engine


def _utc_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"


def _child_process_snapshot(parent_pid: int) -> list[str]:
    children_path = Path(f"/proc/{parent_pid}/task/{parent_pid}/children")
    if not children_path.exists():
        return []

    try:
        child_pids = [int(x) for x in children_path.read_text().strip().split() if x.strip()]
    except Exception:
        return []

    snapshot: list[str] = []
    for pid in child_pids:
        status_path = Path(f"/proc/{pid}/status")
        if not status_path.exists():
            snapshot.append(f"{pid}:exited")
            continue
        try:
            lines = status_path.read_text().splitlines()
            name = "unknown"
            state = "unknown"
            for line in lines:
                if line.startswith("Name:"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("State:"):
                    state = line.split(":", 1)[1].strip()
            snapshot.append(f"{pid}:{name}:{state}")
        except Exception:
            snapshot.append(f"{pid}:status_read_error")
    return snapshot


def _start_generate_heartbeat(interval_sec: float) -> tuple[threading.Event, threading.Thread | None]:
    stop_event = threading.Event()
    if interval_sec <= 0:
        return stop_event, None

    parent_pid = os.getpid()
    start_time = time.perf_counter()

    def _run():
        while not stop_event.wait(interval_sec):
            elapsed = time.perf_counter() - start_time
            children = _child_process_snapshot(parent_pid)
            print(
                "[ditto] generate heartbeat "
                f"ts={_utc_ts()} elapsed_s={elapsed:.2f} "
                f"parent_pid={parent_pid} children={children}",
                flush=True,
            )

    thread = threading.Thread(target=_run, name="ditto-generate-heartbeat", daemon=True)
    thread.start()
    return stop_event, thread


def build_args():
    parser = argparse.ArgumentParser(description="Ditto minimal smoke test.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Hello Ditto")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument(
        "--variant",
        type=str,
        default="offloading",
        choices=["fullattn", "offloading", "loki", "hash", "infinigen", "quest"],
    )
    parser.add_argument(
        "--offloading-method",
        type=str,
        default="hash",
        choices=["hash", "loki", "infinigen", "quest"],
        help="Only used when --variant offloading.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-budget", type=float, default=16.0)
    parser.add_argument("--token-budget", type=float, default=0.2)
    parser.add_argument("--sink-budget", type=int, default=4)
    parser.add_argument("--recent-budget", type=int, default=128)
    parser.add_argument("--selective-start-len", type=int, default=0)
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument("--rbits", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--aux-data-path", type=str, default=None)
    parser.add_argument("--attn-pattern-path", type=str, default="")
    parser.add_argument("--num-omp-threads", type=int, default=4)
    parser.add_argument("--num-skip-layers", type=int, default=0)
    parser.add_argument("--num-overlapped-heads", type=int, default=0)
    parser.add_argument(
        "--max-reuse-count",
        type=int,
        default=10,
        help="Force gather for a KV head after this many consecutive reuses.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--attention-backend", type=str, default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=0.92)
    parser.add_argument("--max-total-tokens", type=int, default=512)
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument(
        "--generate-heartbeat-sec",
        type=float,
        default=10.0,
        help="Heartbeat interval while waiting in engine.generate; <=0 disables heartbeat.",
    )
    parser.add_argument(
        "--disable-cuda-graph",
        dest="disable_cuda_graph",
        action="store_true",
        default=True,
        help="Disable CUDA graph capture (recommended for Ditto bring-up debug).",
    )
    parser.add_argument(
        "--enable-cuda-graph",
        dest="disable_cuda_graph",
        action="store_false",
        help="Enable CUDA graph capture (may improve perf after bring-up is stable).",
    )
    parser.add_argument(
        "--ditto-disable-cuda-graph",
        dest="ditto_enable_cuda_graph",
        action="store_false",
        default=False,
        help="Disable Ditto internal CUDA graph path in custom_config (default).",
    )
    parser.add_argument(
        "--ditto-enable-cuda-graph",
        dest="ditto_enable_cuda_graph",
        action="store_true",
        help="Enable Ditto internal CUDA graph path in custom_config.",
    )
    parser.add_argument(
        "--ditto-custom-config-path",
        type=str,
        default=None,
        help=(
            "Optional path to a JSON file that directly provides model override args "
            "(e.g. minimal_custom_config_template.json). If omitted, this script "
            "builds a Ditto override from CLI flags."
        ),
    )
    parser.add_argument(
        "--disable-dual-chunk-config",
        action="store_true",
        default=False,
        help=(
            "Deprecated compatibility flag. Dual-chunk override injection is now "
            "disabled by default and this flag is ignored."
        ),
    )
    return parser.parse_args()


def resolve_ditto_architecture(model_path: str) -> str:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = str(getattr(cfg, "model_type", "")).lower()
    archs = [str(x).lower() for x in (getattr(cfg, "architectures", None) or [])]

    if "qwen2" in model_type or any("qwen2" in a for a in archs):
        return "DittoQwen2ForCausalLM"
    if "llama" in model_type or any("llama" in a for a in archs):
        return "DittoLlamaForCausalLM"
    return "DittoLlamaForCausalLM"


def build_ditto_override(args) -> dict:
    architecture = resolve_ditto_architecture(args.model_path)
    override = {
        "architectures": [architecture],
        "ditto_variant": args.variant,
        "custom_config": {
            "enable_cuda_graph": bool(args.ditto_enable_cuda_graph),
            "new_config": True,
            "is_profiling": False,
            "num_channels": args.num_channels,
            "rbits": args.rbits,
            "block_size": args.block_size,
            "aux_data_path": args.aux_data_path,
            "kvcache_manager_config": {
                "max_tokens": args.max_tokens,
                "max_batch_size": args.max_batch_size,
                "gpu_memory_budget": args.gpu_memory_budget,
            },
            "sparse_attention_config": {
                "token_budget": args.token_budget,
                "sink_budget": args.sink_budget,
                "recent_budget": args.recent_budget,
                "selective_start_len": int(args.selective_start_len),
            },
            "offload_config": {
                "attn_pattern_path": args.attn_pattern_path,
                "reuse_threshold_upper": 0.95,
                "reuse_threshold_lower": 0.7,
                "decay_p": 2.0,
                "cosine_padding": 0.02,
                "max_reuse_count": int(args.max_reuse_count),
                "num_skip_layers": int(args.num_skip_layers),
                "num_overlapped_heads": int(args.num_overlapped_heads),
                "num_omp_threads": args.num_omp_threads,
            },
        },
    }
    if args.variant == "offloading":
        override["offloading_method"] = args.offloading_method
        override["custom_config"]["offloading_method"] = args.offloading_method
    return override


def main():
    args = build_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")
    if args.prompt_file is not None:
        args.prompt = args.prompt_file.read_text()
    torch = __import__("torch")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in current environment. "
            "Run this script inside a GPU-enabled container/session "
            "(e.g. a GPU-enabled docker image with NVIDIA runtime)."
        )

    if args.ditto_custom_config_path:
        cfg_path = Path(args.ditto_custom_config_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Ditto custom config file not found: {cfg_path}")
        model_override = json.loads(cfg_path.read_text())
    else:
        model_override = build_ditto_override(args)

    if args.disable_dual_chunk_config:
        print(
            "[ditto] --disable-dual-chunk-config is deprecated and ignored. "
            "No dual_chunk_attention_config override is injected.",
            flush=True,
        )

    engine = Engine(
        model_path=args.model_path,
        model_impl="auto",
        trust_remote_code=True,
        log_level="info",
        device=args.device,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        attention_backend=args.attention_backend,
        json_model_override_args=json.dumps(model_override),
        # Minimize SGLang runtime features for bring-up.
        chunked_prefill_size=-1,
        disable_radix_cache=True,
        enable_mixed_chunk=False,
        schedule_policy="fcfs",
        # Minimize SGLang pre-allocated pools to avoid OOM during bring-up.
        mem_fraction_static=args.mem_fraction_static,
        max_total_tokens=args.max_total_tokens,
        max_running_requests=args.max_running_requests,
        page_size=args.page_size,
        disable_cuda_graph=args.disable_cuda_graph,
    )

    print(
        "[ditto] active override: "
        f"architectures={model_override.get('architectures')}, "
        f"variant={model_override.get('ditto_variant', args.variant)}, "
        f"offloading_method={model_override.get('offloading_method', model_override.get('custom_config', {}).get('offloading_method'))}, "
        f"num_skip_layers={model_override.get('custom_config', {}).get('offload_config', {}).get('num_skip_layers')}, "
        f"num_overlapped_heads={model_override.get('custom_config', {}).get('offload_config', {}).get('num_overlapped_heads')}, "
        f"max_reuse_count={model_override.get('custom_config', {}).get('offload_config', {}).get('max_reuse_count')}, "
        f"ditto_enable_cuda_graph={model_override.get('custom_config', {}).get('enable_cuda_graph')}, "
        f"engine_disable_cuda_graph={args.disable_cuda_graph}, "
        f"tp_size={args.tp_size}, "
        f"pp_size={args.pp_size}",
        flush=True,
    )

    sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
    }

    print(
        "[ditto] entering engine.generate "
        f"ts={_utc_ts()} prompt_chars={len(args.prompt)} "
        f"max_new_tokens={args.max_new_tokens}",
        flush=True,
    )

    stop_event, heartbeat_thread = _start_generate_heartbeat(args.generate_heartbeat_sec)
    t0 = time.perf_counter()
    try:
        outputs = [
            engine.generate(prompt=args.prompt, sampling_params=sampling_params)
            for _ in range(args.repeat)
        ]
        out = outputs[0] if args.repeat == 1 else {"runs": outputs}
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(
            "[ditto] engine.generate raised "
            f"ts={_utc_ts()} elapsed_s={elapsed:.2f} err={repr(exc)}",
            flush=True,
        )
        raise
    finally:
        stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)
        engine.shutdown()

    elapsed = time.perf_counter() - t0
    print(
        f"[ditto] engine.generate returned ts={_utc_ts()} elapsed_s={elapsed:.2f}",
        flush=True,
    )
    print(out, flush=True)
    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
