from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.models.ditto.common_utils import (
    _dense_ffn_decode_forward,
    _dense_ffn_prefill_forward,
    _ffn_prepare_cuda_graph_metadata,
    _fuse_gate_up_proj,
    _layernorm_decode_forward,
    _layernorm_prefill_forward,
    _layernorm_prepare_cuda_graph_metadata,
)


def apply_ditto_tp_attention_layout(module: nn.Module, config) -> None:
    module.hidden_size = int(config.hidden_size)
    total_num_heads = int(config.num_attention_heads)
    total_num_kv_heads = int(config.num_key_value_heads)
    attn_tp_rank = int(get_attention_tp_rank())
    attn_tp_size = int(get_attention_tp_size())

    if total_num_heads <= 0 or total_num_kv_heads <= 0:
        raise ValueError(
            "Invalid Ditto attention head config: "
            f"num_attention_heads={total_num_heads}, "
            f"num_key_value_heads={total_num_kv_heads}."
        )

    module.total_num_heads = total_num_heads
    module.total_num_key_value_heads = total_num_kv_heads
    module.attn_tp_rank = attn_tp_rank
    module.attn_tp_size = attn_tp_size

    if attn_tp_size <= 0:
        raise ValueError(f"Invalid attn_tp_size={attn_tp_size}")
    if total_num_heads % attn_tp_size != 0:
        raise ValueError(
            f"Ditto TP requires num_attention_heads={total_num_heads} to be "
            f"divisible by attn_tp_size={attn_tp_size}"
        )

    module.num_heads = total_num_heads // attn_tp_size
    module.query_head_start = attn_tp_rank * module.num_heads

    if attn_tp_size <= 1:
        module.num_key_value_heads = total_num_kv_heads
        module.kv_head_start = 0
        module.kv_head_replicas = 1
    elif total_num_kv_heads >= attn_tp_size:
        if total_num_kv_heads % attn_tp_size != 0:
            raise ValueError(
                f"num_key_value_heads={total_num_kv_heads} is not divisible by attn_tp_size={attn_tp_size}"
            )
        module.num_key_value_heads = total_num_kv_heads // attn_tp_size
        module.kv_head_start = attn_tp_rank * module.num_key_value_heads
        module.kv_head_replicas = 1
    else:
        if attn_tp_size % total_num_kv_heads != 0:
            raise ValueError(
                f"attn_tp_size={attn_tp_size} is not divisible by num_key_value_heads={total_num_kv_heads}"
            )
        module.num_key_value_heads = 1
        module.kv_head_replicas = attn_tp_size // total_num_kv_heads
        module.kv_head_start = attn_tp_rank // module.kv_head_replicas

    if module.num_heads % module.num_key_value_heads != 0:
        raise ValueError(
            f"Ditto TP requires local num_heads={module.num_heads} to be "
            f"divisible by local num_key_value_heads={module.num_key_value_heads}"
        )

    module.local_attn_hidden_size = module.num_heads * module.head_dim


class CustomerQwen2MLP(Qwen2MLP):
    def __init__(self, config):
        super().__init__(config)
        self.torch_dtype = config.torch_dtype
        self.hidden_act = config.hidden_act
        assert self.hidden_act in ["silu"]

        self.converted = False
        self.use_unfused_awq = False
        self._graph_buffers = {}

    def forward(self, x, is_prefill=False):
        if not self.converted:
            _fuse_gate_up_proj(self)
        if is_prefill:
            return _dense_ffn_prefill_forward(self, x)

        bsz = x.shape[0]
        if bsz not in self._graph_buffers:
            _ffn_prepare_cuda_graph_metadata(
                self,
                bsz,
                x.dtype,
                x.device,
            )
        return _dense_ffn_decode_forward(self, x)


class CustomQwen2RMSNorm(Qwen2RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self._graph_buffers = {}

    def forward(self, hidden_states, is_prefill=False):
        if is_prefill:
            return _layernorm_prefill_forward(self, hidden_states)

        bsz = hidden_states.shape[0]
        if bsz not in self._graph_buffers:
            _layernorm_prepare_cuda_graph_metadata(
                self,
                bsz,
                hidden_states.shape[-1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        return _layernorm_decode_forward(self, hidden_states)


class CustomQwen2RotaryEmbedding(nn.Module):
    """RoPE adapter matching internal prototype's cache metadata API."""

    def __init__(self, config):
        super().__init__()
        import flashinfer

        if getattr(config, "rope_scaling", None) is not None:
            rope_scaling = config.rope_scaling
            self.rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        assert self.rope_type in ["default", "llama3", "linear"]

        self.fn = None
        self.fn_kwargs = {}
        if self.rope_type == "linear":
            self.fn_kwargs["interleave"] = False
            self.fn_kwargs["rope_scale"] = config.rope_scaling["factor"]
            self.fn_kwargs["rope_theta"] = config.rope_theta
            self.fn = flashinfer.apply_rope_inplace
        elif self.rope_type == "llama3":
            self.fn_kwargs["interleave"] = False
            self.fn_kwargs["high_freq_factor"] = config.rope_scaling["high_freq_factor"]
            self.fn_kwargs["low_freq_factor"] = config.rope_scaling["low_freq_factor"]
            self.fn_kwargs["rope_theta"] = config.rope_theta
            self.fn_kwargs["rope_scale"] = config.rope_scaling["factor"]
            self.fn_kwargs["old_context_len"] = config.rope_scaling["original_max_position_embeddings"]
            self.fn = flashinfer.apply_llama31_rope_inplace
        else:
            self.fn_kwargs["interleave"] = False
            self.fn_kwargs["rope_scale"] = 1
            self.fn_kwargs["rope_theta"] = config.rope_theta
            self.fn = flashinfer.apply_rope_inplace

    def forward(self, query_states, key_states, past_key_values):
        indptr, offsets = past_key_values.get_rope_metadata(query_states.device)
        self.fn(query_states, key_states, indptr, offsets, **self.fn_kwargs)
        return query_states, key_states
