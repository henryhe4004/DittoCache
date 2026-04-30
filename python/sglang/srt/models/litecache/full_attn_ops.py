from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

from sglang.jit_kernel.triton_kernels.attention import (
    decode_attention_fwd_grouped,
    decode_attention_fwd_grouped_split,
)
from sglang.litecache.kvcache_full_attn import CustomStaticCache
from sglang.srt.models.litecache.common_utils import _custom_linear_forward_wrapper
from sglang.srt.models.litecache.offloading_ops import (
    _flash_attn_with_kvcache,
    _get_split_num,
)


def _local_attn_hidden_size(self) -> int:
    return int(getattr(self, "local_attn_hidden_size", self.num_heads * self.head_dim))


def attention_prepare_cuda_graph_metadata(
    self,
    bsz,
    dtype=torch.bfloat16,
    device="cuda",
):
    local_attn_hidden_size = _local_attn_hidden_size(self)
    self._graph_metadata[bsz] = {
        "max_split_num": 128,
        "split_num": _get_split_num(bsz, self.num_key_value_heads, 128),
    }
    self._graph_buffers[bsz] = {
        "query_states": torch.zeros(
            (bsz, 1, self.num_heads, self.head_dim),
            dtype=dtype,
            device=device,
        ),
        "key_states": torch.zeros(
            (bsz, self.num_key_value_heads * self.head_dim),
            dtype=dtype,
            device=device,
        ),
        "value_states": torch.zeros(
            (bsz, self.num_key_value_heads * self.head_dim),
            dtype=dtype,
            device=device,
        ),
        "attn_output": torch.zeros(
            (bsz, 1, self.num_heads, self.head_dim),
            dtype=dtype,
            device=device,
        ),
        "output_hidden_states": torch.zeros(
            (bsz, self.hidden_size),
            dtype=dtype,
            device=device,
        ),
    }
    if self._graph_metadata[bsz]["split_num"] > 1:
        self._graph_buffers[bsz]["split_num"] = torch.full(
            (1,),
            self._graph_metadata[bsz]["split_num"],
            dtype=torch.int32,
            device=device,
        )
        self._graph_buffers[bsz]["intermediate_attn_logits"] = torch.zeros(
            (bsz, self.num_heads, self._graph_metadata[bsz]["max_split_num"], self.head_dim),
            dtype=dtype,
            device=device,
        )
        self._graph_buffers[bsz]["intermediate_attn_lse"] = torch.zeros(
            (bsz, self.num_heads, self._graph_metadata[bsz]["max_split_num"]),
            dtype=dtype,
            device=device,
        )


def full_attention_prefill_forward(
    self,
    hidden_states: torch.Tensor,
    past_key_value: Optional[CustomStaticCache] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    batch_size = past_key_value.get_cur_batch_size()
    hidden_size = hidden_states.shape[-1]
    local_attn_hidden_size = _local_attn_hidden_size(self)
    prefix_len = past_key_value.get_seq_length(self.layer_idx)

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(-1, self.num_heads, self.head_dim)
    key_states = key_states.view(-1, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(-1, self.num_key_value_heads, self.head_dim)

    query_states, key_states = self.rotary_emb(query_states, key_states, past_key_value)

    key_states, value_states = past_key_value.append_prefill(
        key_states,
        value_states,
        self.layer_idx,
    )
    query_states = query_states.view(batch_size, -1, self.num_heads, self.head_dim)

    attn_output = _flash_attn_with_kvcache(
        query_states,
        k_cache=key_states,
        v_cache=value_states,
        causal=True,
        q_start_idx=prefix_len,
    )
    attn_output = attn_output.view(-1, local_attn_hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output


def full_attention_decode_forward(
    self,
    hidden_input_buffer: torch.Tensor,
    past_key_value: Optional[CustomStaticCache] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, hidden_size = hidden_input_buffer.shape
    local_attn_hidden_size = _local_attn_hidden_size(self)
    assert bsz == past_key_value.get_cur_batch_size()

    _custom_linear_forward_wrapper(
        hidden_input_buffer,
        self.q_proj,
        out=self._graph_buffers[bsz]["query_states"].view(-1, local_attn_hidden_size),
    )
    _custom_linear_forward_wrapper(
        hidden_input_buffer,
        self.k_proj,
        out=self._graph_buffers[bsz]["key_states"].view(
            -1, self.num_key_value_heads * self.head_dim
        ),
    )
    _custom_linear_forward_wrapper(
        hidden_input_buffer,
        self.v_proj,
        out=self._graph_buffers[bsz]["value_states"].view(
            -1, self.num_key_value_heads * self.head_dim
        ),
    )
    query_states = self._graph_buffers[bsz]["query_states"].view(
        -1, self.num_heads, self.head_dim
    )
    key_states = self._graph_buffers[bsz]["key_states"].view(
        -1, self.num_key_value_heads, self.head_dim
    )
    value_states = self._graph_buffers[bsz]["value_states"].view(
        -1, self.num_key_value_heads, self.head_dim
    )
    query_states, key_states = self.rotary_emb(query_states, key_states, past_key_value)
    key_states, value_states = past_key_value.append_decode(
        key_states,
        value_states,
        self.layer_idx,
    )
    query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)

    if self._graph_metadata[bsz]["split_num"] > 1:
        decode_attention_fwd_grouped_split(
            query_states,
            key_states,
            value_states,
            self._graph_buffers[bsz]["attn_output"],
            self._graph_buffers[bsz]["intermediate_attn_logits"],
            self._graph_buffers[bsz]["intermediate_attn_lse"],
            past_key_value.get_seqlen_tensor(self.layer_idx),
            self._graph_buffers[bsz]["split_num"],
            self._graph_metadata[bsz]["max_split_num"],
            self._scale,
        )
    else:
        decode_attention_fwd_grouped(
            query_states,
            key_states,
            value_states,
            self._graph_buffers[bsz]["attn_output"],
            past_key_value.get_seqlen_tensor(self.layer_idx),
            self._scale,
        )

    attn_output = self._graph_buffers[bsz]["attn_output"].view(-1, local_attn_hidden_size)
    output_buffer = self._graph_buffers[bsz]["output_hidden_states"]
    _custom_linear_forward_wrapper(
        attn_output,
        self.o_proj,
        out=output_buffer,
    )
    return output_buffer


def transformer_layer_forward_prefill(
    self,
    hidden_states,
    past_key_value=None,
):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states, is_prefill=True)
    hidden_states = self.self_attn(
        hidden_states=hidden_states,
        past_key_value=past_key_value,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states, is_prefill=True)
    hidden_states = self.mlp(hidden_states, is_prefill=True)
    hidden_states = residual + hidden_states
    return hidden_states


def transformer_layer_forward_decode(
    self,
    hidden_input_buffer,
    past_key_value=None,
):
    hidden_prenorm_buffer = self.input_layernorm(hidden_input_buffer, is_prefill=False)
    hidden_attn_buffer = self.self_attn(
        hidden_prenorm_buffer,
        past_key_value=past_key_value,
    )
    hidden_input_buffer.add_(hidden_attn_buffer)

    hidden_postnorm_buffer = self.post_attention_layernorm(
        hidden_input_buffer,
        is_prefill=False,
    )
    hidden_mlp_buffer = self.mlp(hidden_postnorm_buffer, is_prefill=False)
    hidden_input_buffer.add_(hidden_mlp_buffer)
    return hidden_input_buffer


def llm_prepare_cuda_graph_metadata(
    self,
    bsz,
    hidden_size,
    dtype=torch.bfloat16,
    device="cuda",
):
    self._graph_buffers[bsz] = {
        "input_hidden_states": torch.zeros(
            (bsz, hidden_size),
            device=device,
            dtype=dtype,
        ),
        "output_hidden_states": torch.zeros(
            (bsz, hidden_size),
            device=device,
            dtype=dtype,
        ),
    }


def llm_prefill_forward(
    self,
    input_ids: torch.LongTensor = None,
    past_key_values: Optional[CustomStaticCache] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    chunk_size = getattr(past_key_values.config, "chunk_prefill_size", 0)
    chunk_size = input_ids.shape[1] if chunk_size <= 0 else chunk_size
    for chunk_start in range(0, input_ids.shape[1], chunk_size):
        chunk_input_ids = input_ids[:, chunk_start : chunk_start + chunk_size]
        hidden_states = self.embed_tokens(chunk_input_ids)
        bsz, q_len, _ = hidden_states.shape
        past_key_values.update_metadata(q_len)
        hidden_states = hidden_states.view(bsz * q_len, -1)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                past_key_value=past_key_values,
            )

    hidden_states = hidden_states.view(bsz, q_len, -1)[:, -1, :].view(bsz, -1).contiguous()
    hidden_states = self.norm(hidden_states, is_prefill=True)
    hidden_states = hidden_states.view(bsz, 1, -1)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=None,
        attentions=None,
    )


def llm_decode_forward(
    self,
    input_ids: torch.LongTensor = None,
    past_key_values: Optional[CustomStaticCache] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    hidden_states = self.embed_tokens(input_ids)
    bsz, q_len, _ = hidden_states.shape
    assert q_len == 1, "Only support decode with q_len == 1"
    past_key_values.update_metadata(q_len)
    hidden_states = hidden_states.view(bsz * q_len, -1)

    if getattr(past_key_values.config, "enable_cuda_graph", False):
        if bsz not in self._graph_buffers:
            llm_prepare_cuda_graph_metadata(
                self,
                bsz,
                hidden_states.shape[-1],
                hidden_states.dtype,
                hidden_states.device,
            )
            self._graph_buffers[bsz]["input_hidden_states"].copy_(hidden_states)
            input_buffer = self._graph_buffers[bsz]["input_hidden_states"]
            for decoder_layer in self.layers:
                input_buffer = decoder_layer(
                    input_buffer,
                    past_key_value=past_key_values,
                )
            hidden_states = self.norm(input_buffer, is_prefill=False)
            self._graph_buffers[bsz]["output_hidden_states"].copy_(hidden_states)
        elif bsz not in self._graphs:
            self._graphs[bsz] = torch.cuda.CUDAGraph()
            self._graph_buffers[bsz]["input_hidden_states"].copy_(hidden_states)
            with torch.cuda.graph(self._graphs[bsz]):
                input_buffer = self._graph_buffers[bsz]["input_hidden_states"]
                for decoder_layer in self.layers:
                    input_buffer = decoder_layer(
                        input_buffer,
                        past_key_value=past_key_values,
                    )
                hidden_states = self.norm(input_buffer, is_prefill=False)
                self._graph_buffers[bsz]["output_hidden_states"].copy_(hidden_states)
        else:
            self._graph_buffers[bsz]["input_hidden_states"].copy_(hidden_states)
            self._graphs[bsz].replay()
        hidden_states = self._graph_buffers[bsz]["output_hidden_states"]
    else:
        if bsz not in self._graph_buffers:
            llm_prepare_cuda_graph_metadata(
                self,
                bsz,
                hidden_states.shape[-1],
                hidden_states.dtype,
                hidden_states.device,
            )
        self._graph_buffers[bsz]["input_hidden_states"].copy_(hidden_states)
        input_buffer = self._graph_buffers[bsz]["input_hidden_states"]
        for decoder_layer in self.layers:
            input_buffer = decoder_layer(
                input_buffer,
                past_key_value=past_key_values,
            )
        hidden_states = self.norm(input_buffer, is_prefill=False)
        self._graph_buffers[bsz]["output_hidden_states"].copy_(hidden_states)
        hidden_states = self._graph_buffers[bsz]["output_hidden_states"]

    hidden_states = hidden_states.view(bsz, 1, -1)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=None,
        attentions=None,
    )
