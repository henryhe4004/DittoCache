import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from fastchat.model import get_conversation_template
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.activations as hf_activations
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LITECACHE_TEST_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if LITECACHE_TEST_DIR not in sys.path:
    sys.path.insert(0, LITECACHE_TEST_DIR)

from dataloader import datasets_prompt

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        default="/nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--max_context_length", type=int, default=131072)
    parser.add_argument("--save_path",
                        type=str,
                        default="/mnt/ramdisk/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--longbench_path",
                        type=str,
                        default="/nfs/shared_LLM_dataset/LongBench/data")
    parser.add_argument("--longbench_v2_path",
                        type=str,
                        default="/nfs/shared_LLM_dataset/LongBench-v2")
    parser.add_argument("--pp_num", type=int, default=1)
    parser.add_argument("--apply_template", default=False, action="store_true")
    parser.add_argument("--pos_sample_ratio", type=float, default=0.1)
    parser.add_argument(
        "--max_prompt_chars",
        type=int,
        default=0,
        help="Optional pre-tokenization character budget. 0 means auto by context length.",
    )


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def infer_model_arch(model_path: str):
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = str(getattr(cfg, "model_type", "")).lower()
    if "qwen2" in model_type:
        return "qwen2"
    if "llama" in model_type:
        return "llama"
    if "glm" in model_type:
        return "glm"
    raise ValueError(f"Unsupported model_type={model_type} for {model_path}")


def apply_template(prompt, model_arch, tokenizer):
    # llama3 & glm
    if model_arch in ["llama", "glm"]:
        messages = [{"role": "user", "content": f"{prompt}"}]
        prompt = tokenizer.apply_chat_template(messages,
                                               add_generation_prompt=True,
                                               tokenize=False)
    # llama2-instruct
    elif model_arch == "llama2":
        prompt = f"[INST] {prompt} [/INST]"
    # longchat
    elif model_arch == "longchat":
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    return prompt


def load_longbench_dataset(path, data_name):
    fin = open(os.path.join(path, f"{data_name}.jsonl"), "r", encoding="utf-8")
    lines = fin.readlines()
    fin.close()
    ret = []
    for line in lines:
        eg = json.loads(line)
        ret.append(eg)
    return ret


def load_longbench(path, code_num=3, eng_text_num=5, chn_text_num=3):
    print("LongBench")

    # chinese text
    lsht = load_longbench_dataset(path, "lsht")
    print(f"#Total lsht data: {len(lsht)}, pick {chn_text_num}")
    random.shuffle(lsht)
    lsht = lsht[:chn_text_num]
    for i, item in enumerate(lsht):
        lsht[i] = datasets_prompt["lsht"].format(input=item["input"],
                                                 context=item["context"])

    # english text
    gov_report_e = load_longbench_dataset(path, "qasper_e")
    print(f"#Total qasper_e data: {len(gov_report_e)}, pick {eng_text_num}")
    random.shuffle(gov_report_e)
    gov_report_e = gov_report_e[:eng_text_num]
    for i, item in enumerate(gov_report_e):
        gov_report_e[i] = datasets_prompt["qasper"].format(
            input=item["input"], context=item["context"])

    # code text
    repobenchp_e = load_longbench_dataset(path, "repobench-p_e")
    print(f"#Total repobench-p_e data: {len(repobenchp_e)}, pick {code_num}")
    random.shuffle(repobenchp_e)
    repobenchp_e = repobenchp_e[:code_num]
    for i, item in enumerate(repobenchp_e):
        repobenchp_e[i] = datasets_prompt["repobench-p"].format(
            input=item["input"], context=item["context"])

    return gov_report_e + lsht + repobenchp_e


def load_longbench_v2(path, code_num=1, text_num=1):
    # _id, domain, sub_domain, difficulty, length, question, choice_A, choice_B, choice_C, choice_D, answer, context
    data_file = os.path.join(path, "data.json")
    if not os.path.exists(data_file):
        print(f"[WARN] LongBench-v2 not found at {data_file}, skip this dataset")
        return []
    data = json.load(
        open(data_file, 'r', encoding='utf-8'))
    code_data = []
    text_data = []
    for item in data:
        if item["length"] != "long":
            continue
        domain = item["domain"]
        if domain == "Code Repository Understanding":
            code_data.append(item)
        elif domain in ["Multi-Document QA", "Single-Document QA"]:
            text_data.append(item)
    print("LongBench-v2")
    print(f"#Total long code data: {len(code_data)}, pick {code_num}")
    print(f"#Total long Doc QA data: {len(text_data)}, pick {text_num}")
    random.shuffle(code_data)
    random.shuffle(text_data)
    ret = code_data[:code_num] + text_data[:text_num]
    prompt_template = datasets_prompt["longbench-v2"]
    for i, item in enumerate(ret):
        ret[i] = prompt_template.format(
            context=item["context"],
            question=item["question"],
            C_A=item["choice_A"],
            C_B=item["choice_B"],
            C_C=item["choice_C"],
            C_D=item["choice_D"],
        )
    return ret


def maybe_pretruncate_prompt(prompt: str, max_context_length: int, max_prompt_chars: int):
    if max_prompt_chars is None or int(max_prompt_chars) <= 0:
        # Heuristic: keep enough characters for long-context tokenization while
        # avoiding pathological multi-million-token prompts.
        max_prompt_chars = int(max_context_length) * 8
    if len(prompt) <= max_prompt_chars:
        return prompt
    half = max_prompt_chars // 2
    return prompt[:half] + prompt[-(max_prompt_chars - half):]


class LayerChunkWriter:
    def __init__(self, save_path, num_layers, num_skip_layers, num_heads,
                 num_kv_heads, head_dim, buffer_size):
        self.save_path = save_path
        self.num_layers = num_layers
        self.num_skip_layers = num_skip_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.buffer_size = buffer_size
        self.layer_saved_chunk_num = [0 for _ in range(num_layers)]
        self.layer_q = {}
        self.layer_k = {}
        self.layer_s = {}
        self.layer_filled = {}

    def _ensure_layer_buffer(self, layer_idx):
        if layer_idx in self.layer_q:
            return
        self.layer_q[layer_idx] = torch.empty(
            (self.num_heads, self.buffer_size, self.head_dim),
            dtype=torch.bfloat16,
            device="cpu",
        )
        self.layer_k[layer_idx] = torch.empty(
            (self.num_kv_heads, self.buffer_size, self.head_dim),
            dtype=torch.bfloat16,
            device="cpu",
        )
        self.layer_s[layer_idx] = torch.empty(
            (self.num_heads, self.buffer_size),
            dtype=torch.bfloat16,
            device="cpu",
        )
        self.layer_filled[layer_idx] = 0

    def _save_chunk(self, layer_idx, q, k, s):
        layer_dir = os.path.join(self.save_path, f"layer{layer_idx:02d}")
        os.makedirs(layer_dir, exist_ok=True)
        chunk_id = self.layer_saved_chunk_num[layer_idx]
        torch.save(q, os.path.join(layer_dir, f"chunk{chunk_id:03d}_q.pt"))
        torch.save(k, os.path.join(layer_dir, f"chunk{chunk_id:03d}_k.pt"))
        torch.save(s, os.path.join(layer_dir, f"chunk{chunk_id:03d}_s.pt"))
        self.layer_saved_chunk_num[layer_idx] += 1

    def append(self, layer_idx, q, k, s):
        if layer_idx < self.num_skip_layers:
            return
        self._ensure_layer_buffer(layer_idx)
        total = int(s.shape[1])
        offset = 0
        while total > 0:
            filled = self.layer_filled[layer_idx]
            can_fill = min(self.buffer_size - filled, total)
            self.layer_q[layer_idx][:, filled:filled + can_fill, :] = q[:,
                                                                       offset:offset + can_fill, :]
            self.layer_k[layer_idx][:, filled:filled + can_fill, :] = k[:,
                                                                       offset:offset + can_fill, :]
            self.layer_s[layer_idx][:, filled:filled + can_fill] = s[:,
                                                                     offset:offset + can_fill]
            filled += can_fill
            total -= can_fill
            offset += can_fill
            self.layer_filled[layer_idx] = filled
            if filled == self.buffer_size:
                self._save_chunk(
                    layer_idx,
                    self.layer_q[layer_idx].clone(),
                    self.layer_k[layer_idx].clone(),
                    self.layer_s[layer_idx].clone(),
                )
                self.layer_filled[layer_idx] = 0

    def flush(self):
        for layer_idx in list(self.layer_q.keys()):
            filled = self.layer_filled[layer_idx]
            if filled > 0:
                self._save_chunk(
                    layer_idx,
                    self.layer_q[layer_idx][:, :filled, :].clone(),
                    self.layer_k[layer_idx][:, :filled, :].clone(),
                    self.layer_s[layer_idx][:, :filled].clone(),
                )
                self.layer_filled[layer_idx] = 0


class StreamQKCollector:
    def __init__(self, model, writer, num_heads, num_kv_heads, head_dim):
        self.model = model
        self.writer = writer
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gqa_size = num_heads // num_kv_heads
        self.query_idx = 0
        self.pos_sample_ratio = 0.1
        self._handles = []

    def _hook(self, module, args, kwargs):
        hidden_states = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
        position_embeddings = kwargs.get("position_embeddings", None)
        if hidden_states is None or position_embeddings is None:
            return

        if hidden_states.dim() != 3:
            return
        bsz, q_len, _ = hidden_states.shape
        if bsz != 1 or q_len <= 1:
            return

        # Align sampled query to current sequence length.
        query_idx = min(int(self.query_idx), q_len - 1)
        hidden_shape = (bsz, q_len, -1, self.head_dim)
        q = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Shapes:
        # q: [1, num_heads, seq, head_dim]
        # k: [1, num_kv_heads, seq, head_dim]
        q_sel = q[0, :, query_idx, :]  # [num_heads, head_dim]
        k_ctx = k[0, :, :query_idx + 1, :]  # [num_kv_heads, ctx, head_dim]
        ctx_len = int(k_ctx.shape[1])
        if ctx_len <= 0:
            return

        k_for_q = k_ctx.repeat_interleave(self.gqa_size, dim=0)  # [num_heads, ctx, head_dim]
        score = torch.einsum("hd,hcd->hc", q_sel, k_for_q) / np.sqrt(self.head_dim)
        topk = max(1, int(ctx_len * float(self.pos_sample_ratio)))
        topk_indices = torch.topk(score, k=topk, dim=-1, largest=True).indices
        score.fill_(-1.0)
        topk_scores = torch.linspace(
            20.0,
            1.0,
            steps=topk,
            dtype=score.dtype,
            device=score.device,
        ).unsqueeze(0).expand(self.num_heads, -1)
        score.scatter_(dim=-1, index=topk_indices, src=topk_scores)

        q_rep = q_sel.unsqueeze(1).expand(self.num_heads, ctx_len, self.head_dim).to(torch.bfloat16).cpu()
        k_save = k_ctx.to(torch.bfloat16).cpu()
        s_save = score.to(torch.bfloat16).cpu()
        self.writer.append(int(module.layer_idx), q_rep, k_save, s_save)

    def register(self):
        for layer in self.model.model.layers:
            h = layer.self_attn.register_forward_pre_hook(self._hook, with_kwargs=True)
            self._handles.append(h)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_args(parser)
    args = parser.parse_args()
    seed_everything(42)

    model_arch = infer_model_arch(args.model)
    if model_arch != "qwen2":
        raise ValueError(
            f"build_dataset.py currently supports qwen2 only in sglang auxiliary, got {model_arch}"
        )
    ensure_awq_transformers_compat()
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True,
                                              use_fast=True)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.torch_dtype = torch.bfloat16
    config._attn_implementation = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        config=config,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

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
                                               "CustomLlamaDecoderLayer",
                                               "CustomGlmDecoderLayer",
                                               "CustomGLMBlock",
                                               "CustomQwen2DecoderLayer"
                                           ],
                                           **map_kwargs)
        model = dispatch_model(model, device_map=device_map)
        print(device_map)
    else:
        model = model.to(device_ids[0])

    model.eval()

    num_layers = int(config.num_hidden_layers)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // num_heads))
    writer = LayerChunkWriter(
        save_path=args.save_path,
        num_layers=num_layers,
        num_skip_layers=0,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        buffer_size=32768,
    )
    collector = StreamQKCollector(model, writer, num_heads, num_kv_heads, head_dim)
    collector.register()

    longbench_v2_dataset = load_longbench_v2(args.longbench_v2_path)
    longbench_dataset = load_longbench(args.longbench_path)
    dataset = longbench_v2_dataset + longbench_dataset
    num_items = len(dataset)

    total_len = 0
    total_sample_len = 0
    for it, prompt in enumerate(dataset, start=1):
        if args.apply_template:
            prompt = apply_template(prompt, model_arch, tokenizer)
        prompt = maybe_pretruncate_prompt(prompt, args.max_context_length, args.max_prompt_chars)
        encoded = tokenizer(prompt, return_tensors="pt")

        seq_len = encoded.input_ids.shape[1]
        if seq_len > args.max_context_length:
            input_ids = torch.cat(
                [
                    encoded.input_ids[:, :args.max_context_length // 2],
                    encoded.input_ids[:, -(args.max_context_length -
                                           args.max_context_length // 2):],
                ],
                dim=-1,
            ).to(model.device)
            attention_mask = torch.cat(
                [
                    encoded.attention_mask[:, :args.max_context_length // 2],
                    encoded.attention_mask[:,
                                           -(args.max_context_length -
                                             args.max_context_length // 2):],
                ],
                dim=-1,
            ).to(model.device)
        else:
            input_ids = encoded.input_ids.to(model.device)
            attention_mask = encoded.attention_mask.to(model.device)
        seq_len = input_ids.shape[1]

        query_idx = random.randint(seq_len // 2, max(seq_len - 1, seq_len // 2))
        print(f"Sample {it}, sequence length {seq_len} sample {query_idx + 1} qks")
        total_len += seq_len
        total_sample_len += query_idx + 1

        # Only run the backbone model to collect hidden states.
        # Avoid CausalLM lm_head logits allocation, which can OOM on long context.
        collector.query_idx = query_idx
        collector.pos_sample_ratio = args.pos_sample_ratio
        with torch.no_grad():
            _ = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )

    collector.remove()
    writer.flush()

    print(f"Total sequence length {total_len} sample {total_sample_len} qks")
