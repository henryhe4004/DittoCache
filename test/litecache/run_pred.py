import argparse
import json
import os
import random
import sys
import threading
import time
from datetime import datetime
from functools import partial

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SGLANG_PY_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "python"))
if os.path.isdir(SGLANG_PY_ROOT) and SGLANG_PY_ROOT not in sys.path:
    sys.path.insert(0, SGLANG_PY_ROOT)

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from sglang import Engine

from dataloader import (
    ARCManager,
    HumanEvalManager,
    InfiniteBenchManager,
    LongBenchManager,
    LongBenchV2Manager,
    MathManager,
    NIAHManager,
    RULERManager,
)


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def apply_chat_template(prompt, tok):
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tok, "apply_chat_template"):
            prompt_text = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            encoded = tok(prompt_text)
        else:
            encoded = tok(prompt)
        return encoded

    return tokenizer, apply_chat_template


def get_dataset(args):
    if args.dataset_name == "longbench":
        dataset_manager = LongBenchManager(
            args.dataset_path,
            args.dataset_path,
            "test",
            args.e,
        )
        tasks = (
            dataset_manager.get_dataset_names(with_e=args.e)
            if args.tasks is None
            else args.tasks.split(",")
        )
    elif args.dataset_name == "infinitebench":
        dataset_manager = InfiniteBenchManager(args.dataset_path, args.dataset_path)
        tasks = dataset_manager.get_dataset_names() if args.tasks is None else args.tasks.split(",")
    elif args.dataset_name == "niah":
        dataset_manager = NIAHManager(args.dataset_path, args.dataset_path)
        tasks = dataset_manager.get_dataset_names() if args.tasks is None else args.tasks.split(",")
    elif args.dataset_name == "ruler":
        dataset_manager = RULERManager(args.dataset_path, args.dataset_path)
        tasks = dataset_manager.get_dataset_names() if args.tasks is None else args.tasks.split(",")
    elif args.dataset_name == "longbench-v2":
        dataset_manager = LongBenchV2Manager(args.dataset_path, args.dataset_path)
        tasks = dataset_manager.get_dataset_names() if args.tasks is None else args.tasks.split(",")
    elif args.dataset_name == "math":
        dataset_manager = MathManager(args.dataset_path, args.dataset_path)
        tasks = dataset_manager.get_dataset_names() if args.tasks is None else args.tasks.split(",")
    elif args.dataset_name == "humaneval":
        dataset_manager = HumanEvalManager(args.dataset_path, args.dataset_path)
        tasks = dataset_manager.get_dataset_names()
    elif args.dataset_name == "arc":
        dataset_manager = ARCManager(args.dataset_path, args.dataset_path)
        tasks = dataset_manager.get_dataset_names() if args.tasks is None else args.tasks.split(",")
    else:
        raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")

    log(f"Datasets: {tasks}")
    return dataset_manager, tasks


def resolve_litecache_architecture(model_path: str) -> str:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = str(getattr(cfg, "model_type", "")).lower()
    archs = [str(x).lower() for x in (getattr(cfg, "architectures", None) or [])]

    if "qwen2" in model_type or any("qwen2" in a for a in archs):
        return "LiteCacheQwen2ForCausalLM"
    if "llama" in model_type or any("llama" in a for a in archs):
        return "LiteCacheLlamaForCausalLM"

    log(
        "[WARN] Cannot infer LiteCache architecture from model config "
        f"(model_type={model_type}, archs={archs}). "
        "Fallback to LiteCacheLlamaForCausalLM."
    )
    return "LiteCacheLlamaForCausalLM"


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


def resolve_offloading_method(method: str, resolved_cfg_key: str | None = None) -> str:
    for candidate in (resolved_cfg_key, method):
        parsed = _parse_offloading_method_name(candidate)
        if parsed is not None:
            return parsed
    return "hash"


def method_to_variant(method: str, resolved_cfg_key: str | None = None) -> str:
    """
    Resolve runtime variant from (in priority order):
    1) resolved config key (if any)
    2) method string

    Any method name shaped like `offloading-*` or `*-offloading` should run
    the non-duohead offloading framework.
    """
    candidates = []
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

    log(
        f"[WARN] Cannot map method={method} resolved_cfg_key={resolved_cfg_key} "
        "to a LiteCache variant. Fallback to offloading."
    )
    return "offloading"


def resolve_path(path_value: str | None, config_file: str) -> str | None:
    if not path_value:
        return None
    if os.path.isabs(path_value):
        return path_value

    config_dir = os.path.dirname(os.path.abspath(config_file))
    candidates = [os.path.normpath(os.path.join(config_dir, path_value))]

    my_root = os.environ.get("MYTRANSFORMER_ROOT", "/jhe/myTransformer")
    if path_value.startswith("../"):
        candidates.append(os.path.normpath(os.path.join(my_root, path_value[3:])))
    candidates.append(os.path.normpath(os.path.join(my_root, path_value)))

    for cand in candidates:
        if os.path.exists(cand):
            return cand

    return candidates[0]


def load_method_cfg(args):
    if not args.config_file:
        return {}, None

    with open(args.config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    key = args.method.lower()

    candidates = [key]
    parsed = _parse_offloading_method_name(key)
    if parsed is not None:
        candidates.extend([
            f"offloading-{parsed}",
            f"{parsed}-offloading",
            parsed,
        ])
    elif key == "offloading":
        candidates.extend([
            "offloading-hash",
            "hash-offloading",
            "offloading-loki",
            "loki-offloading",
            "offloading-quest",
            "quest-offloading",
            "offloading-infinigen",
            "infinigen-offloading",
        ])
    elif key.endswith("-offloading"):
        base = key[: -len("-offloading")]
        candidates.extend([f"offloading-{base}", base])

    method_cfg = None
    resolved_key = None
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        method_cfg = cfg.get(candidate)
        if method_cfg is not None:
            resolved_key = candidate
            break
    if method_cfg is None:
        available = ", ".join(sorted(cfg.keys()))
        raise ValueError(
            f"method={args.method} not found in config_file={args.config_file}. "
            f"Available keys: {available}"
        )

    method_cfg = dict(method_cfg)
    if args.topk is not None:
        method_cfg["topk"] = args.topk

    method_cfg["attn_pattern_path"] = resolve_path(
        method_cfg.get("attn_pattern_path"), args.config_file
    )
    method_cfg["aux_data_path"] = resolve_path(
        method_cfg.get("aux_data_path"), args.config_file
    )

    return method_cfg, resolved_key


def build_engine(args):
    method = args.method.lower()
    litecache_enabled = (
        ("offloading" in method)
        or ("hash" in method)
        or ("loki" in method)
        or ("infinigen" in method)
        or ("quest" in method)
    )

    model_override = None
    cfg = {}
    cfg_key = None
    if litecache_enabled:
        cfg, cfg_key = load_method_cfg(args)

    if args.max_total_tokens is not None:
        max_total_tokens = int(args.max_total_tokens)
        log(f"[Engine] max_total_tokens={max_total_tokens} (from --max-total-tokens)")
    else:
        max_total_tokens = None
        log("[Engine] max_total_tokens=<auto-profiled by available GPU memory>")

    # LiteCache cache tensors are sized by per-request context length.
    # Keep it bounded by max_seq_len, and by max_total_tokens if the user explicitly sets it.
    if max_total_tokens is None:
        litecache_kvcache_max_tokens = int(args.max_seq_len)
    else:
        litecache_kvcache_max_tokens = min(int(args.max_seq_len), max_total_tokens)
    if litecache_kvcache_max_tokens <= 0:
        raise ValueError(
            "litecache_kvcache_max_tokens must be positive. "
            f"Got {litecache_kvcache_max_tokens}."
        )

    if litecache_enabled:
        variant = method_to_variant(method, cfg_key)
        offloading_method = (
            resolve_offloading_method(method, cfg_key)
            if variant == "offloading"
            else None
        )
        decay_p = cfg.get("decay_p", cfg.get("deacy_p", 2.0))
        log(
            f"[LiteCache] method={args.method} resolved_cfg_key={cfg_key} "
            f"variant={variant} "
            f"offloading_method={offloading_method}"
        )

        architecture = resolve_litecache_architecture(args.model)
        model_override = {
            "architectures": [architecture],
            "litecache_variant": variant,
            "custom_config": {
                "enable_cuda_graph": False,
                "new_config": True,
                "is_profiling": False,
                "profile_reserve_ratio": float(cfg.get("profile_reserve_ratio", 0.85)),
                "offloading_method": offloading_method,
                "num_channels": int(cfg.get("num_channels", 32)),
                "rbits": int(cfg.get("rbits", 32)),
                "block_size": int(cfg.get("block_size", 64)),
                "aux_data_path": cfg.get("aux_data_path"),
                "kvcache_manager_config": {
                    "max_tokens": litecache_kvcache_max_tokens,
                    "max_batch_size": int(args.batch_size),
                    "gpu_memory_budget": float(cfg.get("max_gpu_memory_size", 16.0)),
                },
                "sparse_attention_config": {
                    "token_budget": float(cfg.get("topk", 0.2)),
                    "sink_budget": int(cfg.get("sink_budget", 4)),
                    "recent_budget": int(cfg.get("recent_budget", 128)),
                },
                "offload_config": {
                    "attn_pattern_path": cfg.get("attn_pattern_path") or "",
                    "reuse_threshold_upper": float(cfg.get("reuse_threshold_upper", 0.95)),
                    "reuse_threshold_lower": float(cfg.get("reuse_threshold_lower", 0.7)),
                    "decay_p": float(decay_p),
                    "cosine_padding": float(cfg.get("cosine_padding", 0.02)),
                    "num_skip_layers": int(cfg.get("num_skip_layers", 0)),
                    "num_overlapped_heads": int(cfg.get("num_overlapped_heads", 0)),
                    "num_omp_threads": int(cfg.get("num_omp_threads", 4)),
                },
            },
        }

        log(
            "[LiteCache] "
            f"architecture={model_override['architectures'][0]} "
            f"variant={variant} "
            f"attn_pattern_path={model_override['custom_config']['offload_config']['attn_pattern_path']} "
            f"token_budget={model_override['custom_config']['sparse_attention_config']['token_budget']} "
            f"kvcache_max_tokens={model_override['custom_config']['kvcache_manager_config']['max_tokens']} "
            f"profile_reserve_ratio={model_override['custom_config']['profile_reserve_ratio']}"
        )

    if litecache_enabled and args.batch_size != 1:
        raise ValueError(
            "LiteCache SGLang bridge currently supports batch_size=1. "
            f"Got batch_size={args.batch_size}."
        )

    if args.mp_num != 1:
        log(
            f"[WARN] mp_num={args.mp_num} is ignored in sglang Engine mode; using single-process inference."
        )
    engine_kwargs = {
        "model_path": args.model,
        "model_impl": "auto",
        "trust_remote_code": True,
        "log_level": "info",
        "device": args.device,
        "attention_backend": args.attention_backend,
        "chunked_prefill_size": 8192,
        "disable_radix_cache": True,
        "enable_mixed_chunk": False,
        "schedule_policy": "fcfs",
        "mem_fraction_static": args.mem_fraction_static,
        "max_total_tokens": max_total_tokens,
        "max_running_requests": args.max_running_requests,
        "page_size": args.page_size,
        "disable_cuda_graph": args.disable_cuda_graph,
        "disable_piecewise_cuda_graph": True,
        "decode_log_interval": args.decode_log_interval,
    }
    if model_override is not None:
        engine_kwargs["json_model_override_args"] = json.dumps(model_override)

    engine = Engine(**engine_kwargs)

    return engine



def _heartbeat_loop(stop_event, dataset_name, sample_id, input_len, heartbeat_sec, start_t):
    while not stop_event.wait(timeout=heartbeat_sec):
        elapsed = time.time() - start_t
        log(
            f"[HEARTBEAT] dataset={dataset_name} sample={sample_id} "
            f"waiting_engine_generate elapsed={elapsed:.1f}s input_tokens={input_len}"
        )


def run_dataset(args, dataset_manager, tokenizer, apply_chat_template, engine, dataset_name):
    raw_data = dataset_manager.get_data(dataset_name)
    if args.dataset_limit > 0:
        raw_data = raw_data.select(range(min(args.dataset_limit, len(raw_data))))

    _, dataset_maxlen, _ = dataset_manager.get_dataset_info(dataset_name)
    log(
        f"[DATASET] {dataset_name} samples={len(raw_data)} "
        f"max_new_tokens={dataset_maxlen} max_seq_len={args.max_seq_len}"
    )

    process_fn = partial(
        dataset_manager.process_raw_data,
        tokenizer=tokenizer,
        apply_chat_template=apply_chat_template,
        task=dataset_name,
        max_length=args.max_seq_len - dataset_maxlen,
        truncate_from_middle=True,
    )

    remove_columns = []
    for key in raw_data[0]:
        if key not in [
            "length",
            "all_classes",
            "answers",
            "depth_percent",
            "difficulty",
            "domain",
            "sub_domain",
            "answer",
            "canonical_solution",
            "test",
            "entry_point",
            "answerKey",
        ]:
            remove_columns.append(key)

    map_t0 = time.time()
    encoded_data = raw_data.map(
        process_fn,
        batched=True,
        num_proc=1,
        batch_size=10,
        with_indices=True,
        remove_columns=remove_columns,
    )
    log(f"[DATASET] {dataset_name} tokenization done in {time.time() - map_t0:.1f}s")

    out_file = os.path.join(args.output_dir, f"{dataset_name}.jsonl")
    if os.path.exists(out_file):
        os.remove(out_file)

    sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": dataset_maxlen,
    }

    task_name = dataset_name[:-2] if dataset_name.endswith("_e") else dataset_name
    is_code_completion_task = task_name in {"lcc", "repobench-p"}

    limit = args.dataset_limit if args.dataset_limit > 0 else len(encoded_data)
    for i in tqdm(range(min(limit, len(encoded_data))), desc=f"Run {dataset_name}"):
        row = encoded_data[i]
        input_ids = row["input_ids"]
        sample_id = i
        input_len = len(input_ids)

        log(
            f"[SAMPLE-START] dataset={dataset_name} sample={sample_id} "
            f"input_tokens={input_len} max_new_tokens={dataset_maxlen}"
        )

        start_t = time.time()
        stop_event = threading.Event()
        hb_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(
                stop_event,
                dataset_name,
                sample_id,
                input_len,
                max(1, int(args.heartbeat_sec)),
                start_t,
            ),
            daemon=True,
        )
        hb_thread.start()

        try:
            out = engine.generate(input_ids=input_ids, sampling_params=sampling_params)
        finally:
            stop_event.set()
            hb_thread.join(timeout=1)

        pred = out["text"] if isinstance(out, dict) else ""
        meta_info = out.get("meta_info", {}) if isinstance(out, dict) else {}
        completion_tokens = meta_info.get("completion_tokens", None)

        retry_reason = None
        if is_code_completion_task:
            if not str(pred).strip():
                retry_reason = "empty_pred"
            elif completion_tokens is not None and int(completion_tokens) <= 1:
                retry_reason = f"completion_tokens={completion_tokens}"

        if retry_reason is not None:
            retry_sampling_params = dict(sampling_params)
            retry_sampling_params["min_new_tokens"] = min(16, dataset_maxlen)
            retry_sampling_params["ignore_eos"] = True
            log(
                f"[RETRY] dataset={dataset_name} sample={sample_id} "
                f"reason={retry_reason} sampling={retry_sampling_params}"
            )
            out_retry = engine.generate(
                input_ids=input_ids,
                sampling_params=retry_sampling_params,
            )
            retry_pred = out_retry["text"] if isinstance(out_retry, dict) else ""
            if str(retry_pred).strip():
                out = out_retry
                pred = retry_pred
                meta_info = out_retry.get("meta_info", {}) if isinstance(out_retry, dict) else {}
                completion_tokens = meta_info.get("completion_tokens", None)

        elapsed = time.time() - start_t
        finish_reason = meta_info.get("finish_reason", None)
        pred_preview = str(pred).replace("\n", "\\n")[:120]
        log(
            f"[SAMPLE-END] dataset={dataset_name} sample={sample_id} "
            f"elapsed={elapsed:.2f}s completion_tokens={completion_tokens} "
            f"finish_reason={finish_reason} pred_chars={len(str(pred))} "
            f"pred_preview='{pred_preview}'"
        )

        out_info = {}
        for k, v in row.items():
            if k in ["input_ids", "attention_mask"]:
                continue
            out_info[k] = [v]

        dataset_manager.write_one_result_v3(
            args.output_dir,
            pred,
            0,
            out_info,
            dataset_name,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="offloading")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--write_in_time", action="store_true")
    parser.add_argument("--mp_num", default=1, type=int)
    parser.add_argument("--pp_num", default=1, type=int)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=131072)
    parser.add_argument("--topk", type=float, default=None)
    parser.add_argument("--dataset_limit", type=int, default=0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attention-backend", type=str, default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=0.92)
    parser.add_argument("--max-total-tokens", type=int, default=None)
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--decode-log-interval", type=int, default=40)
    parser.add_argument("--heartbeat-sec", type=int, default=30)
    parser.add_argument(
        "--disable-cuda-graph",
        dest="disable_cuda_graph",
        action="store_true",
        default=True,
        help="Disable CUDA graph capture (recommended for LiteCache debug).",
    )
    parser.add_argument(
        "--enable-cuda-graph",
        dest="disable_cuda_graph",
        action="store_false",
        help="Enable CUDA graph capture.",
    )

    args = parser.parse_args()
    log(f"Args: {args}")

    seed_everything(args.seed)

    dataset_manager, tasks = get_dataset(args)
    tokenizer, apply_chat_template = load_tokenizer(args.model)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    engine = build_engine(args)
    try:
        for dataset_name in tasks:
            run_dataset(
                args,
                dataset_manager,
                tokenizer,
                apply_chat_template,
                engine,
                dataset_name,
            )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
