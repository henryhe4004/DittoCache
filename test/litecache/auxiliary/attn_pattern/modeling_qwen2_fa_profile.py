import os
import sys
import json
import time
from types import SimpleNamespace
from typing import Dict, List
import torch
import torch.nn as nn
import transformers
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2MLP,
    Qwen2Model,
    Qwen2RMSNorm,
)
from transformers.utils import logging
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast

SGLANG_PY_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python")
)
if SGLANG_PY_ROOT not in sys.path:
    sys.path.insert(0, SGLANG_PY_ROOT)

from sglang.litecache.kvcache_full_attn import (
    CustomStaticCache,
    prepare_cache_for_generation as _litecache_prepare_cache_for_generation,
)
import flash_attn
import flashinfer
from transformers.modeling_flash_attention_utils import _flash_attention_forward

try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2FlashAttention2 as _Qwen2BaseAttention
except ImportError:
    # transformers versions such as 4.57 expose only Qwen2Attention.
    _Qwen2BaseAttention = Qwen2Attention

logger = logging.get_logger(__name__)

CHUNK_SIZE = 8192
DEBUG_LOG_PATH = "/jhe/.cursor/debug-9e2373.log"
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
    except Exception:
        pass


def _prepare_cache_for_generation_compat(
    self,
    generation_config,
    model_kwargs,
    assistant_model=None,
    batch_size=None,
    max_cache_length=None,
    device=None,
    **kwargs,
):
    """
    Transformers changed _prepare_cache_for_generation signature across versions.
    Normalize arguments and forward to LiteCache helper.
    """
    del kwargs  # unused

    if batch_size is None:
        if model_kwargs.get("input_ids", None) is not None:
            batch_size = int(model_kwargs["input_ids"].shape[0])
        elif model_kwargs.get("inputs_embeds", None) is not None:
            batch_size = int(model_kwargs["inputs_embeds"].shape[0])
        else:
            batch_size = 1

    if device is None:
        if model_kwargs.get("inputs_embeds", None) is not None:
            device = model_kwargs["inputs_embeds"].device
        elif model_kwargs.get("input_ids", None) is not None:
            device = model_kwargs["input_ids"].device
        else:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if max_cache_length is None:
        max_cache_length = getattr(generation_config, "max_length", 0)

    max_gpu_cache_memory = getattr(generation_config, "max_gpu_cache_memory", None)
    if max_gpu_cache_memory is None:
        gpu_memory_budget = 20.0
    else:
        gpu_memory_budget = float(max_gpu_cache_memory) / float(1024**3)
        gpu_memory_budget = max(gpu_memory_budget, 1.0)

    input_len = 0
    if model_kwargs.get("input_ids", None) is not None:
        input_len = int(model_kwargs["input_ids"].shape[1])
    elif model_kwargs.get("inputs_embeds", None) is not None:
        input_len = int(model_kwargs["inputs_embeds"].shape[1])
    max_new_tokens = int(getattr(generation_config, "max_new_tokens", 0) or 0)
    required_tokens = input_len + max(max_new_tokens, 256)
    if required_tokens <= 0:
        required_tokens = int(max_cache_length) if int(max_cache_length) > 0 else 32768

    # sglang litecache helper expects generation_config.custom_config.
    # Build or update a minimal fallback config for profiling scripts.
    if not hasattr(generation_config, "custom_config") or generation_config.custom_config is None:
        custom_config = SimpleNamespace(
            enable_cuda_graph=False,
            kvcache_manager_config=SimpleNamespace(
                max_tokens=required_tokens,
                max_batch_size=max(int(batch_size), 1),
                gpu_memory_budget=gpu_memory_budget,
            ),
        )
        generation_config.custom_config = custom_config
    else:
        km = generation_config.custom_config.kvcache_manager_config
        km.max_tokens = max(int(getattr(km, "max_tokens", 0) or 0), required_tokens)
        km.max_batch_size = max(int(getattr(km, "max_batch_size", 1) or 1), int(batch_size))
        km.gpu_memory_budget = float(getattr(km, "gpu_memory_budget", gpu_memory_budget) or gpu_memory_budget)

    # If cache was built with smaller capacity, rebuild it.
    if hasattr(self, "_cache"):
        try:
            current_tokens = int(self._cache.config.kvcache_manager_config.max_tokens)
        except Exception:
            current_tokens = 0
        if current_tokens < required_tokens:
            delattr(self, "_cache")

    return _litecache_prepare_cache_for_generation(
        self,
        generation_config,
        model_kwargs,
        assistant_model,
        batch_size,
        max_cache_length,
        device,
    )


class CustomerQwen2MLP(Qwen2MLP):

    def __init__(self, config):
        super().__init__(config)
        self.torch_dtype = config.torch_dtype
        self.hidden_act = config.hidden_act
        self.converted = False
        self.use_fallback_forward = False
        self._agent_logged_matmul_dtype = False
        assert self.hidden_act in ["silu"]
        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H5",
            location="modeling_qwen2_fa_profile.py:CustomerQwen2MLP.__init__",
            message="Initialized Custom MLP dtype",
            data={"torch_dtype": str(self.torch_dtype)},
        )
        # endregion

    def convert_fusion_exec(self):
        if not self.converted:
            # AWQ linear modules (e.g., WQLinear_GEMM) do not expose .weight.
            # In that case, keep original Qwen2MLP forward path.
            if not (
                hasattr(self.down_proj, "weight")
                and hasattr(self.gate_proj, "weight")
                and hasattr(self.up_proj, "weight")
            ):
                self.use_fallback_forward = True
                self.converted = True
                return

            device = self.down_proj.weight.device
            fuse_dtype = self.down_proj.weight.dtype
            self.gate_up_proj = nn.Linear(self.hidden_size,
                                          self.intermediate_size * 2,
                                          bias=False,
                                          dtype=fuse_dtype,
                                          device=device)
            # region agent log
            _debug_log(
                run_id="pre-fix",
                hypothesis_id="H6",
                location="modeling_qwen2_fa_profile.py:CustomerQwen2MLP.convert_fusion_exec",
                message="Created fused gate_up_proj",
                data={
                    "requested_dtype": str(self.torch_dtype),
                    "fuse_dtype_from_down_proj": str(fuse_dtype),
                    "created_weight_dtype": str(self.gate_up_proj.weight.dtype),
                    "down_proj_weight_dtype": str(self.down_proj.weight.dtype),
                    "gate_proj_weight_dtype": str(self.gate_proj.weight.dtype),
                    "up_proj_weight_dtype": str(self.up_proj.weight.dtype),
                },
            )
            # endregion
            self.gate_up_proj.weight.data[:self.
                                          intermediate_size, :] = self.gate_proj.weight.data
            self.gate_up_proj.weight.data[
                self.intermediate_size:, :] = self.up_proj.weight.data
            self.fn = flashinfer.activation.silu_and_mul

            del self.gate_proj
            del self.up_proj
            torch.cuda.empty_cache()
            self.converted = True

    def forward(self, x):
        self.convert_fusion_exec()
        if self.use_fallback_forward:
            return super().forward(x)
        bsz = x.shape[0]
        chunk_size = 32768
        if bsz > CHUNK_SIZE:
            num_chunks = (bsz + CHUNK_SIZE - 1) // CHUNK_SIZE
            chunk_intermediate1 = torch.zeros(
                (CHUNK_SIZE, self.intermediate_size * 2),
                dtype=x.dtype,
                device=x.device)
            chunk_intermediate2 = torch.zeros(
                (CHUNK_SIZE, self.intermediate_size),
                dtype=x.dtype,
                device=x.device)
            for i in range(num_chunks):
                start = i * CHUNK_SIZE
                end = min(start + CHUNK_SIZE, bsz)
                if not self._agent_logged_matmul_dtype:
                    # region agent log
                    _debug_log(
                        run_id="pre-fix",
                        hypothesis_id="H7",
                        location="modeling_qwen2_fa_profile.py:CustomerQwen2MLP.forward",
                        message="Before matmul in chunked MLP",
                        data={
                            "x_dtype": str(x[start:end, ...].dtype),
                            "gate_up_proj_weight_dtype": str(self.gate_up_proj.weight.dtype),
                            "chunk_intermediate1_dtype": str(chunk_intermediate1.dtype),
                        },
                    )
                    # endregion
                    self._agent_logged_matmul_dtype = True
                torch.matmul(x[start:end, ...],
                             self.gate_up_proj.weight.T.data,
                             out=chunk_intermediate1[:end - start])
                self.fn(chunk_intermediate1[:end - start],
                        out=chunk_intermediate2[:end - start])
                torch.matmul(chunk_intermediate2[:end - start],
                             self.down_proj.weight.T.data,
                             out=x[start:end, ...])
            del chunk_intermediate1, chunk_intermediate2
            torch.cuda.empty_cache()
        else:
            x = self.gate_up_proj(x)
            x = self.fn(x)
            x = self.down_proj(x)
        return x


class CustomQwen2RotaryEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config is not None
        if "rope_scaling" in config and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        assert self.rope_type in ["default", "llama3", "linear"]

        self.fn = None
        self.fn_kwargs = {}

        if self.rope_type == "linear":
            self.fn_kwargs['interleave'] = False
            self.fn_kwargs['rope_scale'] = config.rope_scaling["factor"]
            self.fn_kwargs['rope_theta'] = config.rope_theta
            self.fn = flashinfer.apply_rope_inplace

        elif self.rope_type == "llama3":
            self.fn_kwargs['interleave'] = False
            self.fn_kwargs['high_freq_factor'] = config.rope_scaling[
                'high_freq_factor']
            self.fn_kwargs['low_freq_factor'] = config.rope_scaling[
                'low_freq_factor']
            self.fn_kwargs['rope_theta'] = config.rope_theta
            self.fn_kwargs['rope_scale'] = config.rope_scaling['factor']
            self.fn_kwargs['old_context_len'] = config.rope_scaling[
                'original_max_position_embeddings']
            self.fn = flashinfer.apply_llama31_rope_inplace

        elif self.rope_type == "default":
            self.fn_kwargs['interleave'] = False
            self.fn_kwargs['rope_scale'] = 1
            self.fn_kwargs['rope_theta'] = config.rope_theta
            self.fn = flashinfer.apply_rope_inplace

    def forward(self, query_states, key_states, past_key_values):
        torch.cuda.nvtx.range_push("get_rope_metadata")
        indptr, offsets = past_key_values.get_rope_metadata(
            query_states.device)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("rope_fn")
        self.fn(query_states, key_states, indptr, offsets, **self.fn_kwargs)
        torch.cuda.nvtx.range_pop()
        return query_states, key_states


class CustomQwen2Attention(_Qwen2BaseAttention):

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.rotary_emb = CustomQwen2RotaryEmbedding(config)
        # Compatibility for transformers versions that do not define this flag.
        if not hasattr(self, "_flash_attn_uses_top_left_mask"):
            self._flash_attn_uses_top_left_mask = False
        # Compatibility for transformers versions that rename attention fields.
        if not hasattr(self, "num_heads"):
            self.num_heads = getattr(
                self, "num_attention_heads", getattr(config, "num_attention_heads", None)
            )
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = getattr(
                config, "num_key_value_heads", self.num_heads
            )
        if not hasattr(self, "head_dim"):
            self.head_dim = getattr(
                config,
                "head_dim",
                config.hidden_size // max(int(self.num_heads), 1),
            )
        if not hasattr(self, "is_causal"):
            self.is_causal = True
        self.trace_q_similarity = False
        self.trace_selected_kv_heads = None
        self.trace_records: List[Dict[str, float]] = []
        self.decode_step = 0
        self.q_head_importance = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[CustomStaticCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor,
                  torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        if not hasattr(self, "cosine_similarity"):
            self.cosine_similarity = torch.zeros((self.num_heads, ),
                                                 dtype=torch.float32,
                                                 device=hidden_states.device)
            self.prev_query_states = None
            self.num_iters = 0

        batch_size = past_key_value.curr_batch_size
        q_len = past_key_value.get_cur_q_len()
        is_prefill = q_len > 1
        token_num, hidden_size = hidden_states.size()
        num_chunks = (token_num + CHUNK_SIZE - 1) // CHUNK_SIZE

        if is_prefill:
            self.prev_query_states = None

        torch.cuda.nvtx.range_push("kv_proj")
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        torch.cuda.nvtx.range_pop()
        if is_prefill and token_num > CHUNK_SIZE:
            torch.cuda.nvtx.range_push("chunked_q_proj")
            for i in range(num_chunks):
                start = i * CHUNK_SIZE
                end = min(start + CHUNK_SIZE, token_num)
                hidden_states[start:end] = self.q_proj(
                    hidden_states[start:end])
            query_states = hidden_states
        else:
            torch.cuda.nvtx.range_push("q_proj")
            query_states = self.q_proj(hidden_states)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("rope")
        query_states = query_states.view(-1, self.num_heads, self.head_dim)
        key_states = key_states.view(-1, self.num_key_value_heads,
                                     self.head_dim)
        query_states, key_states = self.rotary_emb(query_states, key_states,
                                                   past_key_value)
        query_states = query_states.view(batch_size, -1, self.num_heads,
                                         self.head_dim)
        key_states = key_states.view(batch_size, -1, self.num_key_value_heads,
                                     self.head_dim)
        value_states = value_states.view(batch_size, -1,
                                         self.num_key_value_heads,
                                         self.head_dim)
        torch.cuda.nvtx.range_pop()

        if not is_prefill:
            if self.prev_query_states is not None:
                # (b, 1, h)
                cosine_similarity = torch.cosine_similarity(
                    query_states, self.prev_query_states, dim=-1)
                cosine_similarity = cosine_similarity.mean(dim=0).view(
                    self.num_heads)
                # cosine_similarity = cosine_similarity.min(dim=-1).values
                self.cosine_similarity += cosine_similarity
                self.num_iters += 1
                if self.trace_q_similarity:
                    group_size = self.num_heads // self.num_key_value_heads
                    kv_cos = cosine_similarity.view(self.num_key_value_heads,
                                                    group_size)
                    use_intra_gqa = os.environ.get(
                        "USE_INTRA_GQA_AGGREGATION", "1") != "0"
                    if use_intra_gqa and self.q_head_importance is not None:
                        q_importance = self.q_head_importance.to(
                            cosine_similarity.device).view(
                            self.num_key_value_heads, group_size)
                        q_importance = torch.clamp(q_importance, min=0.0)
                        q_importance_sum = q_importance.sum(
                            dim=-1, keepdim=True).clamp(min=1e-8)
                        q_importance = q_importance / q_importance_sum
                        kv_cos = 1.0 / torch.sum(
                            q_importance / kv_cos.clamp(min=1e-8), dim=-1)
                    else:
                        kv_cos = kv_cos.min(dim=-1).values

                    for kv_head_idx in range(self.num_key_value_heads):
                        if (
                            self.trace_selected_kv_heads is not None
                            and kv_head_idx not in self.trace_selected_kv_heads
                        ):
                            continue
                        self.trace_records.append({
                            "layer": int(self.layer_idx),
                            "step": int(self.decode_step),
                            "kv_head": int(kv_head_idx),
                            "cosine_similarity": float(kv_cos[kv_head_idx].item()),
                        })
                    self.decode_step += 1
            self.prev_query_states = query_states.clone()

        torch.cuda.nvtx.range_push("kvcache append")
        if hasattr(past_key_value, "append"):
            key_states = past_key_value.append(key_states,
                                               self.layer_idx,
                                               type="key",
                                               inc_seq_len=False)
            value_states = past_key_value.append(value_states,
                                                 self.layer_idx,
                                                 type="value",
                                                 inc_seq_len=True)
        elif hasattr(past_key_value, "append_prefill") and hasattr(
                past_key_value, "append_decode"):
            if is_prefill:
                key_states, value_states = past_key_value.append_prefill(
                    key_states, value_states, self.layer_idx)
            else:
                key_states, value_states = past_key_value.append_decode(
                    key_states, value_states, self.layer_idx)
        else:
            raise AttributeError(
                "Unsupported cache interface: missing append/append_prefill APIs."
            )
        torch.cuda.nvtx.range_pop()

        if is_prefill and token_num > CHUNK_SIZE:
            attn_chunk_size = CHUNK_SIZE // batch_size
            attn_num_chunks = (q_len + attn_chunk_size - 1) // attn_chunk_size
            for i in range(attn_num_chunks):
                start = i * attn_chunk_size
                end = min(start + attn_chunk_size, q_len)
                chunk_k = key_states[:, :end, ...]
                chunk_v = value_states[:, :end, ...]
                chunk_q = query_states[:, start:end, ...]
                chunk_attn_out = flash_attn.flash_attn_with_kvcache(
                    chunk_q,
                    k_cache=chunk_k,
                    v_cache=chunk_v,
                    causal=self.is_causal)
                query_states[:, start:end, ...] = chunk_attn_out
            attn_output = query_states
        else:
            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                position_ids=position_ids,
                dropout=0,
                sliding_window=getattr(self, "sliding_window", None),
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
                is_causal=self.is_causal,
            )

        attn_output = attn_output.view(-1, hidden_size)
        if is_prefill and token_num > CHUNK_SIZE:
            hidden_states = hidden_states.view(-1, hidden_size)
            for i in range(num_chunks):
                start = i * CHUNK_SIZE
                end = min(start + CHUNK_SIZE, token_num)
                hidden_states[start:end] = self.o_proj(attn_output[start:end])
            attn_output = hidden_states
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class CustomQwen2DecoderLayer(Qwen2DecoderLayer):

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = CustomQwen2Attention(config, layer_idx)
        self.input_layernorm = CustomQwen2RMSNorm(config.hidden_size,
                                                  eps=config.rms_norm_eps)
        self.post_attention_layernorm = CustomQwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = CustomerQwen2MLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[CustomStaticCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor,
                  torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:

        torch.cuda.nvtx.range_push("Decode Layer")
        if hidden_states.device.index != torch.cuda.current_device():
            torch.cuda.set_device(hidden_states.device)

        residual = hidden_states.clone()
        torch.cuda.nvtx.range_push("input_layernorm")
        hidden_states = self.input_layernorm(hidden_states)
        torch.cuda.nvtx.range_pop()

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            residual=residual,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        torch.add(residual, hidden_states, out=hidden_states)

        # Fully Connected
        residual.copy_(hidden_states)
        torch.cuda.nvtx.range_push("post_layernorm")
        hidden_states = self.post_attention_layernorm(hidden_states)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("ffn")
        hidden_states = self.mlp(hidden_states)
        torch.cuda.nvtx.range_pop()
        torch.add(residual, hidden_states, out=hidden_states)

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )
        torch.cuda.nvtx.range_pop()

        return outputs


class CustomQwen2RMSNorm(Qwen2RMSNorm):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        bsz = hidden_states.shape[0]
        if bsz > CHUNK_SIZE:
            num_chunks = (bsz + CHUNK_SIZE - 1) // CHUNK_SIZE
            for i in range(num_chunks):
                start = i * CHUNK_SIZE
                end = min(start + CHUNK_SIZE, bsz)
                hidden_states[start:end] = flashinfer.norm.rmsnorm(
                    hidden_states[start:end], self.weight,
                    self.variance_epsilon)
        else:
            hidden_states = flashinfer.norm.rmsnorm(hidden_states, self.weight,
                                                    self.variance_epsilon)
        return hidden_states


class CustomQwen2Model(Qwen2Model):

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([
            CustomQwen2DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = CustomQwen2RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.rotary_emb = None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[CustomStaticCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        bsz, seq_len = input_ids.shape

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length(
            ) if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens,
                                          past_seen_tokens +
                                          inputs_embeds.shape[1],
                                          device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # transformers internal causal-mask helpers changed across versions.
        # LiteCache attention path is causal by design, so fallback to None.
        if hasattr(self, "_update_causal_mask"):
            causal_mask = self._update_causal_mask(attention_mask, inputs_embeds,
                                                   cache_position,
                                                   past_key_values,
                                                   output_attentions)
        else:
            causal_mask = None
        hidden_states = inputs_embeds

        # all the layers share the same allocation/update plan
        if hasattr(past_key_values, "alloc"):
            past_key_values.alloc(seq_len)
        elif hasattr(past_key_values, "update_metadata"):
            past_key_values.update_metadata(
                seq_len, is_prefill=(seq_len > 1), layer_idx=0
            )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        kwargs = {}

        hidden_states = hidden_states.view(bsz * seq_len, -1)
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=None,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[
                    2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1], )

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(bsz, seq_len, -1)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CustomQwen2ForCausalLM(Qwen2ForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = CustomQwen2Model(config)
        transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = (
            _prepare_cache_for_generation_compat
        )
