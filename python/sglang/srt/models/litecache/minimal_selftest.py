from __future__ import annotations

import argparse
import json

from sglang import Engine


def build_args():
    parser = argparse.ArgumentParser(description="LiteCache minimal smoke test.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Hello LiteCache")
    parser.add_argument("--variant", type=str, default="loki", choices=["loki", "hash", "infinigen", "quest"])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-budget", type=float, default=16.0)
    parser.add_argument("--token-budget", type=float, default=0.2)
    parser.add_argument("--sink-budget", type=int, default=4)
    parser.add_argument("--recent-budget", type=int, default=128)
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument("--rbits", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--aux-data-path", type=str, default=None)
    parser.add_argument("--attn-pattern-path", type=str, default="")
    parser.add_argument("--num-omp-threads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attention-backend", type=str, default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=0.92)
    parser.add_argument("--max-total-tokens", type=int, default=512)
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument(
        "--disable-dual-chunk-config",
        action="store_true",
        default=True,
        help="Override model config and disable dual_chunk_attention_config.",
    )
    return parser.parse_args()


def main():
    args = build_args()
    torch = __import__("torch")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in current environment. "
            "Run this script inside a GPU-enabled container/session "
            "(e.g. your `jhe_sglang_lite` docker with NVIDIA runtime)."
        )

    model_override = {}
    if args.disable_dual_chunk_config:
        # Keep it as a dict (not None), because ModelConfig verifier
        # unconditionally writes keys into this field when it exists.
        model_override["dual_chunk_attention_config"] = {}

    engine = Engine(
        model_path=args.model_path,
        model_impl="auto",
        trust_remote_code=True,
        log_level="info",
        device=args.device,
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
    )

    # Variant selection is read from model config (`litecache_variant`) by
    # LiteCacheLlamaForCausalLM. Keep this argument here for CLI symmetry.
    print(
        f"[litecache] requested variant={args.variant}. "
        "Set `litecache_variant` in model config.json to take effect."
    )

    sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": args.max_new_tokens,
    }
    out = engine.generate(prompt=args.prompt, sampling_params=sampling_params)
    print(out)


if __name__ == "__main__":
    main()

