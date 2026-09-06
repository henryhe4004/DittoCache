#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any

_ablation_bootstrap = argparse.ArgumentParser(add_help=False)
_ablation_bootstrap.add_argument("--ablation-profile")
_ablation_bootstrap_args, _ = _ablation_bootstrap.parse_known_args()
if _ablation_bootstrap_args.ablation_profile:
    os.environ["USE_INTRA_GQA_AGGREGATION"] = "0"

import yaml
from transformers import AutoConfig, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
SGLANG_PY_ROOT = (THIS_DIR.parent.parent.parent / "python").resolve()
if str(THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(THIS_DIR.parent))
from ablation_profiles import (
    ABLATION_PROFILES,
    apply_ablation_profile,
    summarize_ablation_config,
)
if SGLANG_PY_ROOT.is_dir() and str(SGLANG_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(SGLANG_PY_ROOT))


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def iter_asset_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for raw_root in (
        os.environ.get("DITTO_ROOT"),
        str(THIS_DIR.parent),
    ):
        if not raw_root:
            continue
        root = Path(raw_root).resolve()
        if root in seen:
            continue
        seen.add(root)
        roots.append(root)
    return roots


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Real):
        return float(value)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SGLang Ditto speed benchmark migrated from internal prototype/speedup."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument(
        "--data",
        type=str,
        default="data/RULER-Qwen2.5-14B-Instruct-1M-8K.jsonl",
    )
    parser.add_argument("--num_decode_steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=131072)
    parser.add_argument("--method", type=str, default="offloading")
    parser.add_argument("--topk", type=float, default=None)
    parser.add_argument("--offloading-method", type=str, default="hash")
    parser.add_argument(
        "--ablation-profile",
        type=str.upper,
        choices=ABLATION_PROFILES,
        default=None,
        help="Apply a cumulative Ditto speedup ablation profile (B0-B6).",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help=(
            "Physical GPU id(s) for CUDA_VISIBLE_DEVICES. "
            "If omitted, keep current environment value."
        ),
    )
    parser.add_argument("--attention-backend", type=str, default=None)
    parser.add_argument(
        "--sglang-log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="SGLang internal logger level.",
    )
    parser.add_argument(
        "--disable-sglang-batch-log",
        action="store_true",
        help="Suppress SGLang scheduler batch logs (e.g. Prefill batch / Decode batch).",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="auto",
        choices=["auto", "fp8_e5m2", "fp8_e4m3", "bf16", "bfloat16", "fp4_e2m1"],
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.92)
    parser.add_argument("--max-total-tokens", type=int, default=None)
    parser.add_argument("--max-running-requests", type=int, default=None)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--decode-log-interval", type=int, default=40)
    parser.add_argument("--watchdog-timeout", type=float, default=300.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--disable-cuda-graph",
        dest="disable_cuda_graph",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--enable-cuda-graph",
        dest="disable_cuda_graph",
        action="store_false",
    )
    parser.add_argument("--chunked-prefill-size", type=int, default=None)
    parser.add_argument(
        "--allow-auto-truncate",
        action="store_true",
        help="Allow SGLang to auto-truncate overlong prompts.",
    )
    parser.add_argument(
        "--ditto-enable-cuda-graph",
        dest="ditto_enable_cuda_graph",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--ditto-disable-cuda-graph",
        dest="ditto_enable_cuda_graph",
        action="store_false",
    )
    parser.add_argument("--record-transfer-stats", action="store_true")

    parser.add_argument("--gpu-memory-budget", type=float, default=16.0)
    parser.add_argument("--token-budget", type=float, default=0.2)
    parser.add_argument("--sink-budget", type=int, default=4)
    parser.add_argument("--recent-budget", type=int, default=128)
    parser.add_argument("--num-omp-threads", type=int, default=4)
    parser.add_argument("--num-skip-layers", type=int, default=0)
    parser.add_argument("--num-overlapped-heads", type=int, default=0)
    parser.add_argument("--profile-reserve-ratio", type=float, default=0.85)
    parser.add_argument("--aux-data-path", type=str, default=None)
    parser.add_argument("--attn-pattern-path", type=str, default="")
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument("--rbits", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)

    parser.add_argument(
        "--skip-invalid-batch",
        dest="skip_invalid_batch",
        action="store_true",
        default=False,
        help="Skip run when Ditto method is used with batch_size != 1.",
    )
    parser.add_argument(
        "--strict-invalid-batch",
        dest="skip_invalid_batch",
        action="store_false",
        help="Raise error instead of skipping when Ditto method gets batch_size != 1.",
    )
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument("--result-json", type=str, default=None)
    return parser.parse_args()


def _parse_offloading_method_name(name: str | None) -> str | None:
    if not name:
        return None
    raw = str(name).strip().lower()
    if raw == "offloading":
        return None

    if raw.startswith("offloading-"):
        method = raw[len("offloading-") :]
    elif raw.endswith("-offloading"):
        method = raw[: -len("-offloading")]
    else:
        return None

    if method in {"hash", "loki", "infinigen", "quest"}:
        return method
    return None


def is_ditto_method(method: str) -> bool:
    m = method.lower()
    return (
        ("offloading" in m)
        or ("hash" in m)
        or ("loki" in m)
        or ("infinigen" in m)
        or ("quest" in m)
    )


def method_to_variant(method: str, resolved_cfg_key: str | None = None) -> str:
    candidates: list[str] = []
    if resolved_cfg_key:
        candidates.append(resolved_cfg_key.lower())
    candidates.append(method.lower())

    for candidate in candidates:
        if candidate == "offloading" or _parse_offloading_method_name(candidate) is not None:
            return "offloading"
        if candidate == "hash" or "hash" in candidate:
            return "hash"
        if candidate == "loki" or "loki" in candidate:
            return "loki"
        if candidate == "infinigen" or "infinigen" in candidate:
            return "infinigen"
        if candidate == "quest" or "quest" in candidate:
            return "quest"

    return "offloading"


def resolve_path(path_value: str | None, config_file: str | None) -> str | None:
    if not path_value:
        return None
    if os.path.isabs(path_value):
        return path_value

    candidates: list[str] = []
    if config_file:
        config_dir = os.path.dirname(os.path.abspath(config_file))
        candidates.append(os.path.normpath(os.path.join(config_dir, path_value)))

    for asset_root in iter_asset_roots():
        if path_value.startswith("../"):
            candidates.append(
                os.path.normpath(os.path.join(asset_root, path_value[3:]))
            )
        candidates.append(os.path.normpath(os.path.join(asset_root, path_value)))

    for cand in candidates:
        if os.path.exists(cand):
            return cand

    return candidates[0] if candidates else path_value


def resolve_cli_path(raw_path: str, extra_roots: list[Path] | None = None) -> str:
    p = Path(raw_path)
    if p.is_absolute():
        return str(p)

    candidates: list[Path] = [Path.cwd() / p, THIS_DIR / p]
    if extra_roots:
        for root in extra_roots:
            candidates.append(root / p)
    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())
    return str((Path.cwd() / p).resolve())


def load_config_file(config_file: str) -> Any:
    with open(config_file, "r", encoding="utf-8") as f:
        text = f.read()

    suffix = Path(config_file).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    if suffix == ".json":
        return json.loads(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def yaml_cfg_to_runtime(raw_cfg: dict[str, Any], config_file: str) -> dict[str, Any]:
    km = raw_cfg.get("kvcache_manager", {}) if isinstance(raw_cfg, dict) else {}
    sparse = raw_cfg.get("sparse_attention", {}) if isinstance(raw_cfg, dict) else {}
    offload = raw_cfg.get("offload", {}) if isinstance(raw_cfg, dict) else {}
    method_cfg = sparse.get("method_config", {}) if isinstance(sparse, dict) else {}

    cfg: dict[str, Any] = {}

    def set_if(key: str, value: Any) -> None:
        if value is not None:
            cfg[key] = value

    set_if("max_num_tokens", km.get("max_tokens"))
    set_if("max_batch_size", km.get("max_batch_size"))
    set_if("max_gpu_memory_size", km.get("gpu_memory_budget"))
    set_if("topk", sparse.get("token_budget"))
    set_if("sink_budget", sparse.get("sink_budget"))
    set_if("recent_budget", sparse.get("recent_budget"))
    set_if("reuse_threshold_lower", offload.get("reuse_threshold_lower"))
    set_if("reuse_threshold_upper", offload.get("reuse_threshold_upper"))
    set_if("decay_p", offload.get("decay_p"))
    set_if("cosine_padding", offload.get("cosine_padding"))
    set_if("enable_similarity", offload.get("enable_similarity"))
    set_if("use_adaptive_threshold", offload.get("use_adaptive_threshold"))
    set_if("enable_resident_cache", offload.get("enable_resident_cache"))
    set_if("enable_layer_prefetch", offload.get("enable_layer_prefetch"))
    set_if("transfer_backend", offload.get("transfer_backend"))
    set_if("num_omp_threads", offload.get("num_omp_threads"))
    set_if("num_overlapped_heads", offload.get("num_overlapped_heads"))
    set_if("num_skip_layers", offload.get("num_skip_layers"))
    set_if("chunk_prefill_size", raw_cfg.get("chunk_prefill_size"))
    set_if("_yaml_enable_cuda_graph", raw_cfg.get("enable_cuda_graph"))
    set_if("_yaml_sparse_method", sparse.get("method"))
    set_if("rbits", method_cfg.get("rbit", method_cfg.get("rbits")))
    set_if("num_channels", method_cfg.get("num_channels"))
    set_if("block_size", method_cfg.get("block_size"))

    cfg["attn_pattern_path"] = resolve_path(offload.get("attn_pattern_path"), config_file)
    cfg["aux_data_path"] = resolve_path(method_cfg.get("aux_data_path"), config_file)
    return cfg


def load_method_cfg(
    method: str,
    config_file: str | None,
    topk_override: float | None,
    ablation_profile: str | None,
) -> tuple[dict[str, Any], str | None]:
    if not config_file:
        return {}, None

    raw_cfg = load_config_file(config_file)
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"config_file={config_file} should contain a mapping object.")

    if "kvcache_manager" in raw_cfg or Path(config_file).suffix.lower() in {".yaml", ".yml"}:
        cfg = yaml_cfg_to_runtime(raw_cfg, config_file)
        if topk_override is not None:
            cfg["topk"] = topk_override
        cfg = apply_ablation_profile(cfg, ablation_profile)
        return cfg, "yaml"

    key = method.lower()
    candidates = [key]
    parsed = _parse_offloading_method_name(key)
    if parsed is not None:
        candidates.extend(
            [
                f"offloading-{parsed}",
                f"{parsed}-offloading",
                parsed,
            ]
        )
    elif key == "offloading":
        candidates.extend(
            [
                "offloading-hash",
                "hash-offloading",
                "offloading-loki",
                "loki-offloading",
                "offloading-quest",
                "quest-offloading",
                "offloading-infinigen",
                "infinigen-offloading",
            ]
        )
    elif key.endswith("-offloading"):
        base = key[: -len("-offloading")]
        candidates.extend([f"offloading-{base}", base])

    method_cfg = None
    resolved_key = None
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        maybe = raw_cfg.get(candidate)
        if maybe is not None:
            method_cfg = dict(maybe)
            resolved_key = candidate
            break

    if method_cfg is None:
        available = ", ".join(sorted(raw_cfg.keys()))
        raise ValueError(
            f"method={method} not found in config_file={config_file}. Available keys: {available}"
        )

    if topk_override is not None:
        method_cfg["topk"] = topk_override
    method_cfg["attn_pattern_path"] = resolve_path(method_cfg.get("attn_pattern_path"), config_file)
    method_cfg["aux_data_path"] = resolve_path(method_cfg.get("aux_data_path"), config_file)
    method_cfg = apply_ablation_profile(method_cfg, ablation_profile)
    return method_cfg, resolved_key


def resolve_ditto_architecture(model_path: str) -> str:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = str(getattr(cfg, "model_type", "")).lower()
    archs = [str(x).lower() for x in (getattr(cfg, "architectures", None) or [])]
    if "qwen2" in model_type or any("qwen2" in a for a in archs):
        return "DittoQwen2ForCausalLM"
    if "llama" in model_type or any("llama" in a for a in archs):
        return "DittoLlamaForCausalLM"
    return "DittoLlamaForCausalLM"


def infer_offloading_method_from_cfg(cfg: dict[str, Any]) -> str | None:
    sparse_method = str(cfg.get("_yaml_sparse_method", "")).lower()
    if "hash" in sparse_method:
        return "hash"
    if "loki" in sparse_method:
        return "loki"
    if "infinigen" in sparse_method or "infini" in sparse_method:
        return "infinigen"
    if "quest" in sparse_method:
        return "quest"
    return None


def resolve_offloading_method(
    method: str,
    resolved_cfg_key: str | None,
    cfg: dict[str, Any],
    default_method: str,
) -> str:
    for candidate in (resolved_cfg_key, method):
        parsed = _parse_offloading_method_name(candidate)
        if parsed is not None:
            return parsed

    from_cfg = infer_offloading_method_from_cfg(cfg)
    if from_cfg is not None:
        return from_cfg
    return default_method


def load_first_prompt(data_file: str) -> str:
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("input") or obj.get("prompt") or obj.get("text")
            if prompt:
                return str(prompt)
    raise ValueError(f"No usable prompt found in data file: {data_file}")


def _read_prompts(data_file: Path, limit: int) -> list[str]:
    prompts: list[str] = []
    with data_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("input") or obj.get("prompt") or obj.get("text")
            if prompt:
                prompts.append(str(prompt))
                if len(prompts) >= limit:
                    break
    return prompts


def load_prompt_batch(data_path: str, batch_size: int) -> list[str]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    if path.is_file():
        prompt = load_first_prompt(str(path))
        return [prompt] * batch_size

    files = sorted(path.rglob("*.jsonl"))
    if not files:
        raise ValueError(f"No JSONL files found under data directory: {data_path}")
    per_file_limit = (batch_size + len(files) - 1) // len(files)
    buckets = [_read_prompts(file, per_file_limit) for file in files]
    prompt_batch: list[str] = []
    for row_idx in range(per_file_limit):
        for bucket in buckets:
            if row_idx < len(bucket):
                prompt_batch.append(bucket[row_idx])
                if len(prompt_batch) == batch_size:
                    return prompt_batch
    raise ValueError(
        f"Requested {batch_size} prompts but found only {len(prompt_batch)} "
        f"under {data_path}"
    )


def fit_prompts_to_token_budget(
    prompts: list[str], tokenizer: Any, max_prompt_tokens: int
) -> tuple[list[str], list[int], int]:
    if max_prompt_tokens <= 0:
        raise ValueError(
            f"max_prompt_tokens must be > 0, got {max_prompt_tokens}."
        )

    fitted_prompts: list[str] = []
    token_counts: list[int] = []
    truncated_count = 0
    for prompt in prompts:
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(token_ids) > max_prompt_tokens:
            truncated_count += 1
            keep_tokens = max_prompt_tokens
            while True:
                fitted_prompt = tokenizer.decode(
                    token_ids[:keep_tokens],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                fitted_ids = tokenizer(
                    fitted_prompt, add_special_tokens=False
                )["input_ids"]
                if len(fitted_ids) <= max_prompt_tokens:
                    prompt = fitted_prompt
                    token_ids = fitted_ids
                    break
                keep_tokens -= max(len(fitted_ids) - max_prompt_tokens, 1)
                if keep_tokens <= 0:
                    raise RuntimeError(
                        "Tokenizer round-trip could not fit prompt within "
                        f"{max_prompt_tokens} tokens."
                    )
        fitted_prompts.append(prompt)
        token_counts.append(len(token_ids))

    return fitted_prompts, token_counts, truncated_count


def extract_completion_tokens(output: Any) -> int:
    def _one(item: Any) -> int:
        if not isinstance(item, dict):
            return 0
        meta = item.get("meta_info")
        if not isinstance(meta, dict):
            return 0
        for key in (
            "completion_tokens",
            "output_tokens",
            "generated_tokens",
            "num_output_tokens",
        ):
            value = meta.get(key)
            parsed = _as_int(value)
            if parsed is not None:
                return parsed
        return 0

    if isinstance(output, list):
        return sum(_one(x) for x in output)
    return _one(output)


def _iter_output_items(output: Any) -> list[dict[str, Any]]:
    if isinstance(output, list):
        return [x for x in output if isinstance(x, dict)]
    if isinstance(output, dict):
        return [output]
    return []


def _extract_internal_forward_timing(
    items: list[dict[str, Any]], batch_size: int, completion_tokens: int
) -> tuple[float, float, float, float, dict[str, Any]]:
    if not items:
        return 0.0, 0.0, 0.0, 0.0, {"source": "empty_items"}

    prefill_candidates: list[float] = []
    decode_forward_latency_candidates: list[float] = []
    decode_forward_steps_candidates: list[int] = []
    decode_forward_latency_drop20_candidates: list[float] = []
    decode_forward_steps_drop20_candidates: list[int] = []
    sample_meta_keys: list[str] | None = None
    sample_meta_values: dict[str, Any] | None = None

    for item in items:
        meta = item.get("meta_info")
        if not isinstance(meta, dict):
            continue
        if sample_meta_keys is None:
            sample_meta_keys = sorted(str(k) for k in meta.keys())
            inspect_keys = [
                "completion_tokens",
                "prefill_forward_latency",
                "prefill_launch_latency",
                "prefill_forward_steps",
                "prefill_forward_latency_ms_per_step",
                "decode_forward_latency",
                "decode_forward_steps",
                "decode_forward_latency_ms_per_step",
                "decode_forward_latency_drop_first_20",
                "decode_forward_steps_drop_first_20",
                "decode_forward_latency_ms_per_step_drop_first_20",
                "decode_throughput",
            ]
            sample_meta_values = {k: meta.get(k) for k in inspect_keys}

        prefill_value = meta.get(
            "prefill_forward_latency", meta.get("prefill_launch_latency")
        )
        parsed_prefill = _as_float(prefill_value)
        if parsed_prefill is not None and parsed_prefill >= 0.0:
            prefill_candidates.append(parsed_prefill)
        else:
            fe = meta.get("forward_entry_time")
            pf = meta.get("prefill_finished_time")
            parsed_fe = _as_float(fe)
            parsed_pf = _as_float(pf)
            if parsed_fe is not None and parsed_pf is not None:
                delta = parsed_pf - parsed_fe
                if delta >= 0.0:
                    prefill_candidates.append(delta)

        decode_forward_latency = meta.get("decode_forward_latency")
        parsed_decode_latency = _as_float(decode_forward_latency)
        if parsed_decode_latency is not None and parsed_decode_latency > 0.0:
            decode_forward_latency_candidates.append(parsed_decode_latency)

        decode_steps = meta.get("decode_forward_steps")
        parsed_decode_steps = _as_int(decode_steps)
        if parsed_decode_steps is not None and parsed_decode_steps > 0:
            decode_forward_steps_candidates.append(parsed_decode_steps)

        decode_forward_latency_drop20 = meta.get("decode_forward_latency_drop_first_20")
        parsed_decode_latency_drop20 = _as_float(decode_forward_latency_drop20)
        if parsed_decode_latency_drop20 is not None and parsed_decode_latency_drop20 > 0.0:
            decode_forward_latency_drop20_candidates.append(parsed_decode_latency_drop20)

        decode_steps_drop20 = meta.get("decode_forward_steps_drop_first_20")
        parsed_decode_steps_drop20 = _as_int(decode_steps_drop20)
        if parsed_decode_steps_drop20 is not None and parsed_decode_steps_drop20 > 0:
            decode_forward_steps_drop20_candidates.append(parsed_decode_steps_drop20)

    prefill_latency = max(prefill_candidates) if prefill_candidates else 0.0
    if decode_forward_latency_drop20_candidates and decode_forward_steps_drop20_candidates:
        decode_forward_latency_sum = max(decode_forward_latency_drop20_candidates)
        decode_forward_steps = max(decode_forward_steps_drop20_candidates)
        timing_source = "meta_drop_first_20"
    else:
        decode_forward_latency_sum = (
            max(decode_forward_latency_candidates)
            if decode_forward_latency_candidates
            else 0.0
        )
        decode_forward_steps = (
            max(decode_forward_steps_candidates) if decode_forward_steps_candidates else 0
        )
        timing_source = "meta_full_decode" if decode_forward_latency_candidates else "missing"

    # In both internal prototype and this benchmark setup, the first generated token belongs
    # to prefill. Decode stage starts from token #2 for each request.
    if decode_forward_steps > 0:
        decode_tokens = batch_size * decode_forward_steps
    else:
        decode_tokens = max(completion_tokens - batch_size, 0)

    if completion_tokens > 0 and prefill_latency <= 0.0:
        log(
            "[WARN] Internal prefill timing missing in meta_info. "
            "Prefill latency is reported as 0. "
            f"meta_info keys={sample_meta_keys} sample_values={sample_meta_values}"
        )
    if decode_tokens > 0 and decode_forward_latency_sum <= 0.0:
        log(
            "[WARN] Internal decode timing missing in meta_info. "
            "Decode latency/throughput may be inaccurate. "
            f"meta_info keys={sample_meta_keys} sample_values={sample_meta_values}"
        )

    decode_latency_steps = decode_forward_steps
    if decode_latency_steps <= 0 and decode_tokens > 0:
        decode_latency_steps = max(decode_tokens // max(batch_size, 1), 1)

    decode_latency_ms_per_step = (
        decode_forward_latency_sum / decode_latency_steps * 1000.0
        if decode_latency_steps > 0 and decode_forward_latency_sum > 0.0
        else 0.0
    )
    decode_throughput = (
        decode_tokens / decode_forward_latency_sum
        if decode_tokens > 0 and decode_forward_latency_sum > 0.0
        else 0.0
    )
    timing_debug = {
        "source": timing_source,
        "prefill_candidates": len(prefill_candidates),
        "decode_latency_candidates": len(decode_forward_latency_candidates),
        "decode_steps_candidates": len(decode_forward_steps_candidates),
        "decode_latency_drop20_candidates": len(decode_forward_latency_drop20_candidates),
        "decode_steps_drop20_candidates": len(decode_forward_steps_drop20_candidates),
        "decode_forward_latency_sum_s": decode_forward_latency_sum,
        "decode_forward_steps": decode_forward_steps,
        "decode_latency_ms_per_step": decode_latency_ms_per_step,
        "sample_meta_keys": sample_meta_keys,
        "sample_meta_values": sample_meta_values,
    }
    return (
        prefill_latency,
        decode_forward_latency_sum,
        decode_latency_ms_per_step,
        decode_throughput,
        timing_debug,
    )


def extract_transfer_stats_from_meta(outputs_for_meta: list[dict[str, Any]]) -> dict[str, Any] | None:
    best_stats = None
    best_steps = -1
    for item in outputs_for_meta:
        meta = item.get("meta_info") if isinstance(item, dict) else None
        if not isinstance(meta, dict):
            continue
        transfer_stats = meta.get("ditto_decode_transfer_stats")
        if not isinstance(transfer_stats, dict):
            continue
        steps = int(transfer_stats.get("decode_steps", -1))
        if steps >= best_steps:
            best_stats = transfer_stats
            best_steps = steps
    return best_stats


def load_transfer_stats_file(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def generate_with_internal_forward_timing(
    engine: Engine,
    prompt: str | list[str],
    sampling_params: dict[str, Any],
    batch_size: int,
    transfer_stats_path: str | None = None,
) -> tuple[Any, float, int, float, float, float, dict | None, dict[str, Any]]:
    outputs_for_meta: list[dict[str, Any]] = []
    last_chunk: Any = None
    t0 = time.perf_counter()
    first_token_ts: float | None = None
    last_ts = t0
    max_completion_tokens = 0
    token_timestamps: list[float] = []
    for chunk in engine.generate(prompt=prompt, sampling_params=sampling_params, stream=True):
        last_chunk = chunk
        now = time.perf_counter()
        last_ts = now
        completion_tokens_chunk = extract_completion_tokens(chunk)
        if completion_tokens_chunk > max_completion_tokens:
            max_completion_tokens = completion_tokens_chunk
            token_timestamps.append(now)
        if first_token_ts is None and completion_tokens_chunk > 0:
            first_token_ts = now
        outputs_for_meta.extend(_iter_output_items(chunk))
    if last_chunk is None:
        t0 = time.perf_counter()
        last_chunk = engine.generate(prompt=prompt, sampling_params=sampling_params, stream=False)
        last_ts = time.perf_counter()
        outputs_for_meta = _iter_output_items(last_chunk)
        max_completion_tokens = extract_completion_tokens(last_chunk)
        first_token_ts = last_ts

    completion_tokens = extract_completion_tokens(last_chunk)
    if completion_tokens <= 0:
        completion_tokens = max(
            (extract_completion_tokens(item) for item in outputs_for_meta),
            default=0,
        )
    if completion_tokens <= 0:
        sample_meta = None
        for item in outputs_for_meta:
            meta = item.get("meta_info") if isinstance(item, dict) else None
            if isinstance(meta, dict):
                sample_meta = {k: type(v).__name__ for k, v in meta.items()}
                break
        log(
            "[WARN] completion_tokens is 0 from streaming chunks. "
            f"sample_meta_value_types={sample_meta}"
        )
    (
        prefill_latency,
        decode_forward_latency_sum,
        decode_latency_ms_per_step,
        decode_throughput,
        timing_debug,
    ) = _extract_internal_forward_timing(
        outputs_for_meta,
        batch_size=batch_size,
        completion_tokens=completion_tokens,
    )

    # Internal forward fields can be unavailable in some runtime paths.
    # Fall back to stream timeline so benchmark still remains usable.
    if (
        completion_tokens > 0
        and prefill_latency <= 0.0
        and decode_forward_latency_sum <= 0.0
    ):
        if first_token_ts is None:
            first_token_ts = last_ts
        prefill_latency = max(first_token_ts - t0, 0.0)
        decode_elapsed = max(last_ts - first_token_ts, 0.0)
        decode_tokens = max(max_completion_tokens - batch_size, 0)
        decode_forward_latency_sum = decode_elapsed

        # Align with internal prototype timer semantics:
        # use per-step decode forward latencies and drop first 20 decode steps.
        decode_step_latencies_ms: list[float] = []
        if len(token_timestamps) >= 2:
            for i in range(1, len(token_timestamps)):
                decode_step_latencies_ms.append(
                    (token_timestamps[i] - token_timestamps[i - 1]) * 1000.0
                )
        # Very defensive fallback when per-token timestamps are unavailable.
        if not decode_step_latencies_ms and decode_tokens > 0 and decode_elapsed > 0.0:
            decode_step_latencies_ms = [decode_elapsed / decode_tokens * 1000.0] * decode_tokens

        dropped = 20
        if len(decode_step_latencies_ms) > dropped:
            used_lat_ms = decode_step_latencies_ms[dropped:]
        else:
            used_lat_ms = decode_step_latencies_ms

        if used_lat_ms:
            total_used_ms = sum(used_lat_ms)
            decode_latency_ms_per_step = total_used_ms / len(used_lat_ms)
            decode_throughput = (
                batch_size * len(used_lat_ms) / total_used_ms * 1000.0
                if total_used_ms > 0
                else 0.0
            )
        else:
            decode_latency_ms_per_step = 0.0
            decode_throughput = 0.0
        timing_debug = dict(timing_debug)
        timing_debug.update({
            "source": "stream_fallback_drop_first_20",
            "stream_decode_step_count": len(decode_step_latencies_ms),
            "stream_used_step_count": len(used_lat_ms),
            "stream_decode_elapsed_s": decode_elapsed,
        })
        log(
            "[WARN] Internal forward timing is unavailable; "
            "falling back to stream-based timing for this run."
        )

    internal_elapsed = max(prefill_latency + decode_forward_latency_sum, 0.0)
    if os.environ.get("DITTO_TIMING_DEBUG", "0") == "1":
        log(f"[TIMING_DEBUG] {json.dumps(timing_debug, sort_keys=True, default=str)}")

    transfer_stats = load_transfer_stats_file(transfer_stats_path)
    if transfer_stats is None:
        transfer_stats = extract_transfer_stats_from_meta(outputs_for_meta)
    return (
        last_chunk,
        internal_elapsed,
        completion_tokens,
        prefill_latency,
        decode_latency_ms_per_step,
        decode_throughput,
        transfer_stats,
        timing_debug,
    )


def build_engine_for_bench(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    cfg_key: str | None,
) -> tuple[Engine, dict[str, Any]]:
    method = args.method.lower()
    ditto_enabled = is_ditto_method(method)
    intra_gqa_aggregation_value: bool | None = None
    if ditto_enabled and "enable_intra_gqa_aggregation" in cfg:
        intra_gqa_aggregation_value = bool(cfg["enable_intra_gqa_aggregation"])
        os.environ["USE_INTRA_GQA_AGGREGATION"] = (
            "1" if intra_gqa_aggregation_value else "0"
        )

    # Import after applying process-wide Ditto feature flags. The offloading
    # module reads USE_INTRA_GQA_AGGREGATION when it is first imported.
    from sglang import Engine

    if args.max_total_tokens is not None:
        max_total_tokens = int(args.max_total_tokens)
        log(f"[Engine] max_total_tokens={max_total_tokens} (from --max-total-tokens)")
    else:
        cfg_max_tokens = _as_int(cfg.get("max_num_tokens"))
        if cfg_max_tokens is not None and cfg_max_tokens > 0:
            max_total_tokens = cfg_max_tokens
            log(
                f"[Engine] max_total_tokens={max_total_tokens} "
                "(from config kvcache_manager.max_tokens)"
            )
        else:
            max_total_tokens = None
            log("[Engine] max_total_tokens=<auto-profiled by available GPU memory>")

    chunked_prefill_size_value = (
        args.chunked_prefill_size
        if args.chunked_prefill_size is not None
        else cfg.get("chunk_prefill_size", 8192)
    )
    chunked_prefill_size = int(chunked_prefill_size_value)
    max_running_requests = (
        int(args.max_running_requests)
        if args.max_running_requests is not None
        else max(1, int(args.batch_size))
    )
    sglang_log_level = str(args.sglang_log_level).lower()
    decode_log_interval = int(args.decode_log_interval)
    if args.disable_sglang_batch_log:
        if sglang_log_level in {"debug", "info"}:
            sglang_log_level = "warning"
        decode_log_interval = max(decode_log_interval, 10**9)

    model_override = None
    variant = None
    offloading_method = None
    ditto_enable_cuda_graph_value: bool | None = None
    if ditto_enabled:
        variant = method_to_variant(method, cfg_key)
        offloading_method = (
            resolve_offloading_method(method, cfg_key, cfg, args.offloading_method)
            if variant == "offloading"
            else None
        )

        ditto_cfg_max_tokens = int(cfg.get("max_num_tokens", args.max_seq_len))
        if max_total_tokens is None:
            ditto_kvcache_max_tokens = ditto_cfg_max_tokens
        else:
            ditto_kvcache_max_tokens = min(
                ditto_cfg_max_tokens,
                max_total_tokens,
            )
        if ditto_kvcache_max_tokens <= 0:
            raise ValueError(
                "ditto_kvcache_max_tokens must be positive. "
                f"Got {ditto_kvcache_max_tokens}."
            )

        decay_p = float(cfg.get("decay_p", cfg.get("deacy_p", 2.0)))
        profile_graph = bool(
            cfg.get(
                "enable_ditto_cuda_graph",
                cfg.get("_yaml_enable_cuda_graph", False),
            )
        )
        if args.ablation_profile:
            ditto_enable_cuda_graph = profile_graph
        elif args.ditto_enable_cuda_graph is None:
            ditto_enable_cuda_graph = profile_graph
        else:
            ditto_enable_cuda_graph = bool(args.ditto_enable_cuda_graph)
        ditto_enable_cuda_graph_value = ditto_enable_cuda_graph

        architecture = resolve_ditto_architecture(args.model)
        model_override = {
            "architectures": [architecture],
            "ditto_variant": variant,
            "offloading_method": offloading_method,
            "custom_config": {
                "enable_cuda_graph": ditto_enable_cuda_graph,
                "new_config": True,
                "is_profiling": False,
                "profile_reserve_ratio": float(
                    cfg.get("profile_reserve_ratio", args.profile_reserve_ratio)
                ),
                "offloading_method": offloading_method,
                "num_channels": int(cfg.get("num_channels", args.num_channels)),
                "rbits": int(cfg.get("rbits", args.rbits)),
                "block_size": int(cfg.get("block_size", args.block_size)),
                "aux_data_path": cfg.get("aux_data_path") or args.aux_data_path,
                "kvcache_manager_config": {
                    "max_tokens": int(ditto_kvcache_max_tokens),
                    "max_batch_size": int(args.batch_size),
                    "gpu_memory_budget": float(
                        cfg.get("max_gpu_memory_size", args.gpu_memory_budget)
                    ),
                },
                "sparse_attention_config": {
                    "token_budget": float(cfg.get("topk", args.token_budget)),
                    "sink_budget": int(cfg.get("sink_budget", args.sink_budget)),
                    "recent_budget": int(cfg.get("recent_budget", args.recent_budget)),
                },
                "offload_config": {
                    "attn_pattern_path": cfg.get("attn_pattern_path") or args.attn_pattern_path,
                    "reuse_threshold_upper": float(
                        cfg.get("reuse_threshold_upper", 0.95)
                    ),
                    "reuse_threshold_lower": float(
                        cfg.get("reuse_threshold_lower", 0.7)
                    ),
                    "decay_p": decay_p,
                    "cosine_padding": float(cfg.get("cosine_padding", 0.02)),
                    "enable_similarity": bool(cfg.get("enable_similarity", True)),
                    "use_adaptive_threshold": bool(
                        cfg.get("use_adaptive_threshold", True)
                    ),
                    "enable_resident_cache": bool(
                        cfg.get("enable_resident_cache", True)
                    ),
                    "enable_layer_prefetch": bool(
                        cfg.get("enable_layer_prefetch", True)
                    ),
                    "transfer_backend": str(cfg.get("transfer_backend", "gdrcopy")),
                    "num_skip_layers": int(cfg.get("num_skip_layers", args.num_skip_layers)),
                    "num_overlapped_heads": int(
                        cfg.get("num_overlapped_heads", args.num_overlapped_heads)
                    ),
                    "num_omp_threads": int(cfg.get("num_omp_threads", args.num_omp_threads)),
                },
            },
        }

        log(
            f"[Ditto] method={args.method} resolved_cfg_key={cfg_key} "
            f"variant={variant} offloading_method={offloading_method} "
            f"token_budget={model_override['custom_config']['sparse_attention_config']['token_budget']} "
            f"kvcache_max_tokens={model_override['custom_config']['kvcache_manager_config']['max_tokens']} "
            f"ditto_enable_cuda_graph={ditto_enable_cuda_graph_value}"
        )

        if args.ablation_profile:
            log(f"[Ditto ablation] {summarize_ablation_config(cfg)}")
    engine_kwargs = {
        "model_path": args.model,
        "model_impl": "auto",
        "trust_remote_code": True,
        "log_level": sglang_log_level,
        "device": args.device,
        "attention_backend": args.attention_backend,
        "kv_cache_dtype": args.kv_cache_dtype,
        "chunked_prefill_size": chunked_prefill_size,
        "disable_radix_cache": True,
        "enable_mixed_chunk": False,
        "schedule_policy": "fcfs",
        "mem_fraction_static": args.mem_fraction_static,
        "max_total_tokens": max_total_tokens,
        "max_running_requests": max_running_requests,
        "page_size": args.page_size,
        "disable_cuda_graph": args.disable_cuda_graph,
        "disable_piecewise_cuda_graph": True,
        "allow_auto_truncate": args.allow_auto_truncate,
        "decode_log_interval": decode_log_interval,
        "watchdog_timeout": args.watchdog_timeout,
        "random_seed": args.random_seed,
        "enable_metrics": True,
        "tp_size": int(args.tp_size),
        "pp_size": int(args.pp_size),
    }
    if model_override is not None:
        engine_kwargs["json_model_override_args"] = json.dumps(model_override)

    engine = Engine(**engine_kwargs)
    runtime_meta = {
        "ditto_enabled": ditto_enabled,
        "variant": variant,
        "offloading_method": offloading_method,
        "max_total_tokens": max_total_tokens,
        "sglang_cuda_graph_enabled": not bool(args.disable_cuda_graph),
        "ditto_cuda_graph_enabled": ditto_enable_cuda_graph_value,
        "ablation_profile": args.ablation_profile,
        "transfer_stats_enabled": bool(args.record_transfer_stats),
        "intra_gqa_aggregation_enabled": intra_gqa_aggregation_value,
        "random_seed": args.random_seed,
        "tp_size": int(args.tp_size),
        "pp_size": int(args.pp_size),
    }
    return engine, runtime_meta


def main() -> int:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        log(f"[CUDA] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {args.batch_size}.")
    if args.epoch <= 0:
        raise ValueError(f"epoch must be > 0, got {args.epoch}.")
    if args.warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {args.warmup}.")

    args.data = resolve_cli_path(args.data)
    if args.config_file:
        args.config_file = resolve_cli_path(args.config_file, extra_roots=[THIS_DIR.parent])

    ditto_enabled = is_ditto_method(args.method)
    if ditto_enabled and args.batch_size != 1 and args.skip_invalid_batch:
        result = {
            "status": "skipped",
            "reason": "ditto_batch_size_only_one",
            "method": args.method,
            "ablation_profile": args.ablation_profile,
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "data": args.data,
        }
        log(
            "[SKIP] Ditto SGLang bridge currently supports batch_size=1 only. "
            f"Got batch_size={args.batch_size}."
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.result_json:
            out_path = Path(resolve_cli_path(args.result_json))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0

    cfg, cfg_key = load_method_cfg(
        args.method, args.config_file, args.topk, args.ablation_profile
    )

    prompts = load_prompt_batch(args.data, args.batch_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    original_input_token_counts = [
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in prompts
    ]
    max_prompt_tokens = args.max_seq_len - (args.num_decode_steps + 1)
    prompts, input_token_counts, truncated_count = fit_prompts_to_token_budget(
        prompts, tokenizer, max_prompt_tokens
    )
    input_tokens = max(input_token_counts)
    min_input_tokens = min(input_token_counts)
    total_input_tokens = sum(input_token_counts)
    expected_total_tokens = input_tokens + args.num_decode_steps + 1
    log(
        f"[DATA] prompts={len(prompts)} prompt_tokens_min={min_input_tokens} "
        f"prompt_tokens_max={input_tokens} total_prompt_tokens={total_input_tokens} "
        f"original_prompt_tokens_max={max(original_input_token_counts)} "
        f"truncated_prompts={truncated_count} prompt_token_budget={max_prompt_tokens} "
        f"expected_max_total_tokens={expected_total_tokens} "
        f"max_seq_len={args.max_seq_len}"
    )
    if expected_total_tokens > args.max_seq_len:
        raise RuntimeError(
            f"expected_max_total_tokens={expected_total_tokens} exceeds "
            f"max_seq_len={args.max_seq_len} after prompt fitting."
        )

    prompt_batch: str | list[str]
    if args.batch_size == 1:
        prompt_batch = prompts[0]
    else:
        prompt_batch = prompts

    log(
        f"[RUN] method={args.method} warmup={args.warmup} epoch={args.epoch} "
        f"batch_size={args.batch_size} config_file={args.config_file}"
    )
    transfer_stats_path = None
    if args.record_transfer_stats:
        transfer_stats_path = os.environ.get("DITTO_TRANSFER_STATS_FILE")
        if not transfer_stats_path:
            transfer_stats_path = f"/tmp/ditto_transfer_stats_{os.getpid()}.json"
        Path(transfer_stats_path).unlink(missing_ok=True)
    os.environ["DITTO_RECORD_TRANSFER_STATS"] = "1" if args.record_transfer_stats else "0"
    if transfer_stats_path is not None:
        os.environ["DITTO_TRANSFER_STATS_FILE"] = transfer_stats_path
    else:
        os.environ.pop("DITTO_TRANSFER_STATS_FILE", None)
    engine = None
    runtime_meta: dict[str, Any] = {}
    epoch_latencies: list[float] = []
    epoch_tps: list[float] = []
    epoch_tokens: list[int] = []
    epoch_prefill_latencies: list[float] = []
    epoch_decode_latencies_ms: list[float] = []
    epoch_decode_tps: list[float] = []
    epoch_transfer_stats: list[dict] = []
    epoch_timing_debug: list[dict[str, Any]] = []
    warmup_timing_debug: list[dict[str, Any]] = []
    sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": args.num_decode_steps + 1,
        "min_new_tokens": args.num_decode_steps + 1,
    }

    total_iters = args.warmup + args.epoch
    try:
        engine, runtime_meta = build_engine_for_bench(args, cfg, cfg_key)
        for i in range(total_iters):
            phase = "warmup" if i < args.warmup else "epoch"
            phase_idx = i + 1 if phase == "warmup" else i - args.warmup + 1
            phase_total = args.warmup if phase == "warmup" else args.epoch
            (
                out,
                elapsed,
                actual_completion_tokens,
                prefill_latency,
                decode_latency_ms_per_step,
                decode_throughput,
                transfer_stats,
                timing_debug,
            ) = generate_with_internal_forward_timing(
                engine=engine,
                prompt=prompt_batch,
                sampling_params=sampling_params,
                batch_size=args.batch_size,
                transfer_stats_path=transfer_stats_path,
            )
            expected_completion_tokens = (args.num_decode_steps + 1) * args.batch_size
            completion_tokens = (
                actual_completion_tokens
                if actual_completion_tokens > 0
                else expected_completion_tokens
            )
            tps = completion_tokens / elapsed if elapsed > 0 else float("inf")
            print(f"Prefilling latency: {prefill_latency:.3f} s", flush=True)
            print(
                f"Decoding latency: {decode_latency_ms_per_step:.3f} ms/step, "
                f"Throughput: {decode_throughput:.3f} tokens/s",
                flush=True,
            )
            log(
                f"[BENCH] phase={phase} iter={phase_idx}/{phase_total} "
                f"elapsed_s={elapsed:.4f} completion_tokens={completion_tokens} tok_per_s={tps:.2f}"
            )
            if args.record_transfer_stats and transfer_stats is not None:
                transfer_log = (
                    f"[TRANSFER] phase={phase} iter={phase_idx}/{phase_total} "
                    f"decode_steps={transfer_stats['decode_steps']} "
                    f"total_bytes={transfer_stats['total_bytes']} "
                    f"avg_bytes_per_step={transfer_stats['avg_bytes_per_step']:.1f}"
                )
                if "overall_hit_rate" in transfer_stats:
                    transfer_log += (
                        f" selected_tokens={int(transfer_stats.get('total_selected_tokens', 0))} "
                        f"recalled_tokens={int(transfer_stats.get('total_recalled_tokens', 0))} "
                        f"hit_rate={float(transfer_stats.get('overall_hit_rate', 0.0)):.4f}"
                    )
                log(transfer_log)
            if phase == "warmup":
                warmup_timing_debug.append(timing_debug)
                print("Warmup end", flush=True)
            if args.print_output:
                print(out, flush=True)
            if phase == "epoch":
                epoch_latencies.append(elapsed)
                epoch_tps.append(tps)
                epoch_tokens.append(completion_tokens)
                epoch_prefill_latencies.append(prefill_latency)
                epoch_decode_latencies_ms.append(decode_latency_ms_per_step)
                epoch_decode_tps.append(decode_throughput)
                epoch_timing_debug.append(timing_debug)
                if transfer_stats is not None:
                    epoch_transfer_stats.append(transfer_stats)
    finally:
        if engine is not None:
            engine.shutdown()

    if not epoch_latencies:
        raise RuntimeError("No epoch runs completed. Please set --epoch >= 1.")

    avg_latency = statistics.mean(epoch_latencies)
    p50_latency = statistics.median(epoch_latencies)
    avg_tps = statistics.mean(epoch_tps)
    total_tokens = sum(epoch_tokens)
    total_time = sum(epoch_latencies)
    overall_tps = total_tokens / total_time if total_time > 0 else float("inf")
    avg_prefill_latency = statistics.mean(epoch_prefill_latencies)
    avg_decode_latency_ms = statistics.mean(epoch_decode_latencies_ms)
    avg_decode_tps = statistics.mean(epoch_decode_tps)

    result = {
        "status": "ok",
        "method": args.method,
        "ablation_profile": args.ablation_profile,
        "config_file": args.config_file,
        "input_tokens_min": min_input_tokens,
        "input_tokens_max": input_tokens,
        "total_input_tokens": total_input_tokens,
        "data": args.data,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "warmup": args.warmup,
        "epoch": args.epoch,
        "input_tokens": input_tokens,
        "avg_elapsed_s": avg_latency,
        "p50_elapsed_s": p50_latency,
        "avg_tokens_per_s": avg_tps,
        "overall_tokens_per_s": overall_tps,
        "epoch_elapsed_s": epoch_latencies,
        "epoch_tokens_per_s": epoch_tps,
        "avg_prefill_latency_s": avg_prefill_latency,
        "avg_decode_latency_ms_per_step": avg_decode_latency_ms,
        "avg_decode_tokens_per_s": avg_decode_tps,
        "epoch_prefill_latency_s": epoch_prefill_latencies,
        "epoch_decode_latency_ms_per_step": epoch_decode_latencies_ms,
        "epoch_decode_tokens_per_s": epoch_decode_tps,
        "warmup_timing_debug": warmup_timing_debug,
        "epoch_timing_debug": epoch_timing_debug,
        "runtime_meta": runtime_meta,
    }
    if args.record_transfer_stats:
        result["epoch_decode_transfer_stats"] = epoch_transfer_stats
        if epoch_transfer_stats:
            total_selected_tokens = int(
                sum(int(ts.get("total_selected_tokens", 0)) for ts in epoch_transfer_stats)
            )
            total_recalled_tokens = int(
                sum(int(ts.get("total_recalled_tokens", 0)) for ts in epoch_transfer_stats)
            )
            total_hit_tokens = int(
                sum(int(ts.get("total_hit_tokens", 0)) for ts in epoch_transfer_stats)
            )
            result["transfer_stats_summary"] = {
                "epochs": int(len(epoch_transfer_stats)),
                "total_selected_tokens": total_selected_tokens,
                "total_recalled_tokens": total_recalled_tokens,
                "total_hit_tokens": total_hit_tokens,
                "overall_hit_rate": float(
                    (total_hit_tokens / total_selected_tokens)
                    if total_selected_tokens > 0
                    else 0.0
                ),
            }

    log(
        f"[SUMMARY] method={args.method} batch_size={args.batch_size} seq={args.max_seq_len} "
        f"avg_elapsed_s={avg_latency:.4f} p50_elapsed_s={p50_latency:.4f} "
        f"avg_tok_per_s={avg_tps:.2f} overall_tok_per_s={overall_tps:.2f}"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)

    if args.result_json:
        out_path = Path(resolve_cli_path(args.result_json))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"[WRITE] result_json={out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
