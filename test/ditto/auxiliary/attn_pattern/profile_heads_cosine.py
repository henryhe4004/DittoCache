import argparse
import csv
import json
import os
import sys
import time
from functools import partial
from types import SimpleNamespace

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import pandas as pd
import torch
import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
import transformers.activations as hf_activations

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DITTO_TEST_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if DITTO_TEST_DIR not in sys.path:
    sys.path.insert(0, DITTO_TEST_DIR)

from dataloader import LongBenchManager

DEBUG_LOG_PATH = "/tmp/ditto_debug.log"
DEBUG_SESSION_ID = "9e2373"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception as e:
        print(f"[DEBUG-LOG-ERROR] {e}", file=sys.stderr, flush=True)


def ensure_awq_transformers_compat():
    """
    autoawq may import PytorchGELUTanh from transformers.activations.
    Newer transformers versions removed this symbol.
    """
    if hasattr(hf_activations, "PytorchGELUTanh"):
        return

    class PytorchGELUTanh(torch.nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.gelu(input, approximate="tanh")

    hf_activations.PytorchGELUTanh = PytorchGELUTanh


def set_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model",
        type=str,
        default="/nfs/shared_LLM_model/Qwen/Qwen2.5-14B-Instruct-1M")
    parser.add_argument("--dataset_path",
                        type=str,
                        default="/nfs/shared_LLM_dataset/LongBench")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Start sample index in the selected task dataset.",
    )
    parser.add_argument("--max_context_length", type=int, default=65536)
    parser.add_argument("--pp_num", type=int, default=1)
    parser.add_argument(
        "--trace_layers",
        type=str,
        default="",
        help="Comma-separated decoder layer ids to trace q similarity lines.",
    )
    parser.add_argument(
        "--trace_kv_heads",
        type=str,
        default="",
        help="Comma-separated kv head ids to trace. Empty means all kv heads.",
    )
    parser.add_argument(
        "--trace_task",
        type=str,
        default="",
        help="If set, only trace this LongBench task.",
    )


def infer_model_arch(model_path: str):
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = str(getattr(cfg, "model_type", "")).lower()
    if "qwen2" in model_type:
        return "qwen2"
    if "llama" in model_type:
        return "llama"
    raise ValueError(f"Unsupported model_type={model_type} for {model_path}")


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              use_fast=True)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def apply_chat_template(prompt, tok):
        if hasattr(tok, "apply_chat_template"):
            text = tok.apply_chat_template([{
                "role": "user",
                "content": prompt
            }],
                                           add_generation_prompt=True,
                                           tokenize=False)
            return tok(text)
        return tok(prompt)

    return tokenizer, apply_chat_template


def parse_indices(raw: str):
    s = (raw or "").strip()
    if not s:
        return None
    return sorted({int(x.strip()) for x in s.split(",") if x.strip()})


def plot_q_similarity_lines(df: pd.DataFrame, save_path: str):
    if df.empty or plt is None:
        return
    plt.figure(figsize=(11, 6))
    for (layer, kv_head), grp in df.groupby(["layer", "kv_head"]):
        grp = grp.sort_values("step")
        plt.plot(
            grp["step"].to_numpy(),
            grp["cosine_similarity"].to_numpy(),
            linewidth=1.3,
            label=f"L{layer}-H{kv_head}",
        )
    plt.xlabel("Decoding Step")
    plt.ylabel("Cosine Similarity")
    plt.title("Ditto Q Similarity by Decode Step")
    if len(df.groupby(["layer", "kv_head"])) <= 15:
        plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_args(parser)
    args = parser.parse_args()
    # region agent log
    print(
        f"[AGENT-DEBUG] profile_heads_cosine loaded file={__file__} marker=debug-v3",
        flush=True,
    )
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H4",
        location="profile_heads_cosine.py:startup",
        message="Script startup marker",
        data={"file": __file__, "marker": "debug-v3"},
    )
    # endregion

    model_arch = infer_model_arch(args.model)
    if model_arch != "qwen2":
        raise ValueError(
            f"This local profile script currently supports qwen2 only, got {model_arch}."
        )

    tokenizer, apply_chat_template = load_tokenizer(args.model)
    ensure_awq_transformers_compat()
    model_config = AutoConfig.from_pretrained(args.model,
                                              trust_remote_code=True)
    model_config._attn_implementation = "flash_attention_2"
    model_config.torch_dtype = torch.float16
    # region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H3",
        location="profile_heads_cosine.py:model_config",
        message="Configured model dtype and attention implementation",
        data={
            "model": args.model,
            "attn_impl": str(model_config._attn_implementation),
            "torch_dtype": str(model_config.torch_dtype),
            "max_context_length": int(args.max_context_length),
        },
    )
    # endregion

    from modeling_qwen2_fa_profile import CustomQwen2ForCausalLM
    model = CustomQwen2ForCausalLM.from_pretrained(args.model,
                                                   config=model_config)
    # Keep Ditto cache dtype aligned with profiling dtype.
    # Some non-quantized checkpoints keep text_config.torch_dtype=float32,
    # which doubles KV cache memory and can trip memory-budget checks.
    model.config.torch_dtype = torch.float16
    if hasattr(model.config, "get_text_config"):
        text_cfg = model.config.get_text_config()
        text_cfg.torch_dtype = torch.float16

    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model = model.to(torch.float16).eval()

    device_ids = [i for i in range(args.pp_num)]
    if len(device_ids) > 1:
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils import get_balanced_memory

        max_memory = {}
        for did in device_ids:
            max_memory[did] = torch.cuda.mem_get_info(did)[0]
        map_kwargs = {"max_memory": get_balanced_memory(model, max_memory)}
        device_map = infer_auto_device_map(model,
                                           no_split_module_classes=[
                                               "CustomQwen2DecoderLayer",
                                           ],
                                           **map_kwargs)
        model = dispatch_model(model, device_map=device_map)
    else:
        model = model.to(device_ids[0])

    dataset_manager = LongBenchManager(
        args.dataset_path,
        args.dataset_path,
        "test",
        False,
    )
    tasks = [args.trace_task] if args.trace_task else ["gov_report", "lcc", "lsht"]
    trace_layers = parse_indices(args.trace_layers)
    trace_kv_heads = parse_indices(args.trace_kv_heads)
    trace_records = []

    save_model_name = os.path.normpath(args.model).split("/")[-1]
    q_importance_path = os.path.join(save_model_name, "q_heads_importance.tsv")
    q_importance_candidates = [
        q_importance_path,
        os.path.join(THIS_DIR, save_model_name, "q_heads_importance.tsv"),
        os.path.join(os.path.dirname(os.path.normpath(args.model)), "q_heads_importance.tsv"),
    ]
    # Common fallback for Qwen AWQ profiling when user runs script directly.
    if "qwen2.5" in save_model_name.lower() and "awq" in save_model_name.lower():
        q_importance_candidates.append(
            os.path.join(THIS_DIR, "Qwen2.5-14B-Instruct-1M", "q_heads_importance.tsv")
        )
    resolved_q_importance_path = None
    for cand in q_importance_candidates:
        if os.path.exists(cand):
            resolved_q_importance_path = cand
            break
    q_head_importance_arr = None
    # region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H1",
        location="profile_heads_cosine.py:setup_paths",
        message="Computed q importance primary path and cwd",
        data={
            "cwd": os.getcwd(),
            "model": args.model,
            "save_model_name": save_model_name,
            "q_importance_path": q_importance_path,
            "exists_primary": os.path.exists(q_importance_path),
            "resolved_q_importance_path": resolved_q_importance_path,
        },
    )
    # endregion
    if resolved_q_importance_path is not None:
        q_head_importance_arr = pd.read_csv(
            resolved_q_importance_path, sep="\t", header=None
        ).to_numpy()
        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H2",
            location="profile_heads_cosine.py:load_q_importance_early",
            message="Loaded q_heads_importance early",
            data={
                "shape": list(q_head_importance_arr.shape),
                "path": resolved_q_importance_path,
            },
        )
        # endregion
    else:
        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H2",
            location="profile_heads_cosine.py:load_q_importance_early",
            message="Early q_heads_importance missing",
            data={"missing_paths": q_importance_candidates},
        )
        # endregion
    for task in tasks:
        print(f"Profile on {task} dataset from LongBench")
        _, dataset_maxlen, _ = dataset_manager.get_dataset_info(task)
        raw_data = dataset_manager.get_data(task)

        process_fn = partial(
            dataset_manager.process_raw_data,
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
            task=task,
            max_length=args.max_context_length -
            dataset_manager.get_dataset_info(task)[1],
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
        encoded_data = raw_data.map(
            process_fn,
            batched=True,
            num_proc=4,
            batch_size=10,
            with_indices=True,
            remove_columns=remove_columns,
        )

        generation_kwargs = {
            "max_gpu_cache_memory": 18 * 1024 * 1024 * 1024,
        }
        generation_config = GenerationConfig(**generation_kwargs)
        # Pin Ditto capacity for profiling to avoid undersized defaults (e.g. 512).
        generation_config.custom_config = SimpleNamespace(
            enable_cuda_graph=False,
            kvcache_manager_config=SimpleNamespace(
                max_tokens=int(args.max_context_length),
                max_batch_size=1,
                gpu_memory_budget=18.0,
            ),
        )
        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H4",
            location="profile_heads_cosine.py:generation_config",
            message="Prepared generation cache config",
            data={
                "task": task,
                "max_tokens": int(generation_config.custom_config.kvcache_manager_config.max_tokens),
                "gpu_memory_budget": float(generation_config.custom_config.kvcache_manager_config.gpu_memory_budget),
                "max_gpu_cache_memory_bytes": int(generation_kwargs["max_gpu_cache_memory"]),
            },
        )
        # endregion

        num_layers = model_config.num_hidden_layers
        if hasattr(model_config, "num_key_value_heads"):
            num_key_value_heads = model_config.num_key_value_heads
        elif hasattr(model_config, "multi_query_group_num"):
            num_key_value_heads = model_config.multi_query_group_num
        else:
            num_key_value_heads = model_config.num_attention_heads

        sample_start = max(int(args.sample_idx), 0)
        sample_end = min(sample_start + int(args.num_samples), len(encoded_data))
        if sample_start >= len(encoded_data):
            raise ValueError(
                f"sample_idx={sample_start} out of range for task={task}, "
                f"dataset_size={len(encoded_data)}"
            )

        for sample_i in tqdm.tqdm(
            range(sample_start, sample_end),
            desc=f"profiling cosine similarity on {task}...",
        ):
            for l in range(num_layers):
                attn = model.model.layers[l].self_attn
                should_trace = (trace_layers is None) or (l in trace_layers)
                attn.trace_q_similarity = should_trace
                attn.trace_selected_kv_heads = set(trace_kv_heads) if trace_kv_heads else None
                attn.trace_records = []
                attn.decode_step = 0
                if q_head_importance_arr is not None:
                    attn.q_head_importance = torch.from_numpy(
                        q_head_importance_arr[l]
                    ).to(device_ids[0]).to(torch.float32)
                else:
                    attn.q_head_importance = None

            model_input = {
                "input_ids":
                torch.Tensor([encoded_data["input_ids"][sample_i]]).long().cuda(),
                "attention_mask":
                torch.Tensor([encoded_data["attention_mask"][sample_i]]).long().cuda(),
            }

            model.generate(**model_input,
                           do_sample=False,
                           max_new_tokens=dataset_maxlen,
                           generation_config=generation_config)
            for l in range(num_layers):
                attn = model.model.layers[l].self_attn
                if not attn.trace_q_similarity:
                    continue
                for rec in attn.trace_records:
                    rec["task"] = task
                    rec["sample_idx"] = sample_i
                    trace_records.append(rec)

    q_importance_path_late = resolved_q_importance_path
    # region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H3",
        location="profile_heads_cosine.py:load_q_importance_late",
        message="About to load q_heads_importance late",
        data={
            "late_path": q_importance_path_late,
            "exists_late": bool(q_importance_path_late and os.path.exists(q_importance_path_late)),
            "early_loaded": q_head_importance_arr is not None,
        },
    )
    # endregion
    if q_head_importance_arr is None:
        q_head_importance_arr = torch.ones(
            (num_layers, model_config.num_attention_heads), dtype=torch.float32
        ).numpy()
        print(
            "[WARN] q_heads_importance.tsv not found; fallback to uniform q-head importance.",
            flush=True,
        )
    q_head_importance = q_head_importance_arr

    cosine_list = []
    for l in range(num_layers):
        cosine = model.model.layers[l].self_attn.cosine_similarity.cpu()
        iters = model.model.layers[l].self_attn.num_iters
        cosine = cosine / iters
        importance = torch.from_numpy(q_head_importance[l])
        importance = torch.clamp(importance * 1.2, 0.5, 1.0)
        cosine = torch.cos(importance * torch.acos(cosine))
        cosine = cosine.view(num_key_value_heads, -1).min(dim=-1).values
        cosine_list.append(cosine.unsqueeze(0))
    cosine_list = torch.cat(cosine_list, dim=0)
    cosine_list = cosine_list.numpy()

    sample_tag = f"sample{int(args.sample_idx)}"
    save_path = os.path.join(save_model_name, f"q_heads_cosine_similarity_{sample_tag}.csv")
    os.makedirs(save_model_name, exist_ok=True)
    with open(save_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in cosine_list:
            csv_writer.writerow(row.tolist())

    if trace_records:
        trace_df = pd.DataFrame(trace_records)
        trace_csv = os.path.join(save_model_name, f"q_similarity_trace_{sample_tag}.csv")
        trace_df.to_csv(trace_csv, index=False)
        trace_plot = os.path.join(save_model_name, f"q_similarity_trace_{sample_tag}.png")
        plot_q_similarity_lines(trace_df, trace_plot)
        print(f"[INFO] trace csv saved to {trace_csv}")
        print(f"[INFO] trace plot saved to {trace_plot}")
