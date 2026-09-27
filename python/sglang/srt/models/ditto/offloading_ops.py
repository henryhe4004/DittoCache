from __future__ import annotations

import logging
import os
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast

from sglang.jit_kernel.triton_kernels.attention import (
    decode_attention_fwd_grouped,
    decode_attention_fwd_grouped_split,
    decode_mixed_attention_fwd_grouped,
    decode_mixed_split_attention_fwd_grouped,
)
from sglang.ditto.kvcache_offloading import OffloadingCache
from sglang.srt.models.ditto.awq_linear import ditto_linear_forward

try:
    import flash_attn as _flash_attn
except ImportError:
    _flash_attn = None

logger = logging.getLogger(__name__)
DITTO_CUDA_GRAPH_DEBUG = os.getenv("DITTO_CUDA_GRAPH_DEBUG", "0") == "1"


def custom_linear_forward_wrapper(x, linear, out):
    return ditto_linear_forward(x, linear, out=out)


def _local_attn_hidden_size(self) -> int:
    return int(getattr(self, "local_attn_hidden_size", self.num_heads * self.head_dim))


def _flash_attn_with_kvcache(
    query_states: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    causal: bool = True,
    q_start_idx: int = 0,
) -> torch.Tensor:
    flash_attn_can_use_end_aligned_causal = (
        not causal or q_start_idx == k_cache.shape[1] - query_states.shape[1]
    )
    if _flash_attn is not None and flash_attn_can_use_end_aligned_causal:
        return _flash_attn.flash_attn_with_kvcache(
            query_states,
            k_cache=k_cache,
            v_cache=v_cache,
            causal=causal,
        )

    # Fallback for environments without flash_attn Python package.
    if k_cache.shape[2] != query_states.shape[2]:
        if query_states.shape[2] % k_cache.shape[2] != 0:
            raise ValueError(
                f"num_heads={query_states.shape[2]} is not divisible by "
                f"num_kv_heads={k_cache.shape[2]}"
            )
        repeat = query_states.shape[2] // k_cache.shape[2]
        k_cache = k_cache.repeat_interleave(repeat, dim=2)
        v_cache = v_cache.repeat_interleave(repeat, dim=2)

    q = query_states.transpose(1, 2)
    k = k_cache.transpose(1, 2)
    v = v_cache.transpose(1, 2)
    # PyTorch SDPA causal mask assumes query positions are 0..q_len-1.
    # For chunked prefill with K-cache prefix (q_start_idx > 0), we must
    # build an offset-aware mask so query i can attend keys <= q_start_idx + i.
    if causal:
        q_len = q.shape[2]
        k_len = k.shape[2]
        if q_start_idx == 0 and q_len == k_len:
            attn_mask = None
            is_causal = True
        else:
            q_pos = torch.arange(
                q_start_idx,
                q_start_idx + q_len,
                device=q.device,
            ).unsqueeze(1)
            k_pos = torch.arange(k_len, device=q.device).unsqueeze(0)
            attn_mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
            is_causal = False
    else:
        attn_mask = None
        is_causal = False

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    return out.transpose(1, 2).contiguous()


def _get_split_num(batch_size, num_kv_heads, max_splits=128):
    device = torch.cuda.current_device()
    device_props = torch.cuda.get_device_properties(device)

    num_sms = device_props.multi_processor_count
    num_blocks = batch_size * num_kv_heads
    if num_blocks >= 2 * num_sms:
        return 1

    max_eff = 0.0
    effs = []

    for num_splits in range(1, max_splits + 1):
        num_total_blocks = num_blocks * num_splits
        num_waves = num_total_blocks / num_sms
        num_waves_ceil = (num_total_blocks + num_sms - 1) // num_sms
        eff = num_waves / num_waves_ceil
        if eff > max_eff:
            max_eff = eff
        effs.append(eff)

    for num_splits in range(1, max_splits + 1):
        if effs[num_splits - 1] >= 0.85 * max_eff:
            return num_splits

    return 1


def _get_topk_split_num(batch_size, num_kv_heads, max_splits):
    device = torch.cuda.current_device()
    device_props = torch.cuda.get_device_properties(device)

    num_sms = device_props.multi_processor_count
    num_blocks = batch_size * num_kv_heads
    if num_blocks >= 2 * num_sms:
        return 1

    max_eff = 0.0
    effs = []

    for num_splits in range(1, max_splits + 1):
        total_blocks = num_blocks * num_splits
        n_waves = total_blocks / num_sms
        ceil_n_waves = (total_blocks + num_sms - 1) // num_sms
        eff = n_waves / ceil_n_waves
        if eff > max_eff:
            max_eff = eff
        effs.append(eff)

    for num_splits in range(1, max_splits + 1):
        if effs[num_splits - 1] >= 0.85 * max_eff:
            return num_splits

    return 1


def attention_prepare_cuda_graph_metadata(
    self,
    bsz,
    dtype=torch.bfloat16,
    device="cuda",
):
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


def attention_sparse_offloading_prepare_cuda_graph_metadata(
    self,
    bsz,
    dtype=torch.bfloat16,
    device="cuda",
    topk=None,
):
    attention_prepare_cuda_graph_metadata(
        self,
        bsz,
        dtype=dtype,
        device=device,
    )

    max_splits = 1
    if topk is not None:
        max_splits = max(1, int(topk) // 256)

    self._graph_metadata[bsz]["topk_split_num"] = _get_topk_split_num(
        bsz,
        self.num_key_value_heads,
        max_splits,
    )

    if self._graph_metadata[bsz]["topk_split_num"] > 1:
        self._graph_buffers[bsz]["topk_split_num"] = torch.full(
            (1,),
            self._graph_metadata[bsz]["topk_split_num"],
            dtype=torch.int32,
            device=device,
        )
        if "intermediate_attn_logits" not in self._graph_buffers[bsz]:
            self._graph_buffers[bsz]["intermediate_attn_logits"] = torch.zeros(
                (bsz, self.num_heads, self._graph_metadata[bsz]["max_split_num"], self.head_dim),
                dtype=dtype,
                device=device,
            )
        if "intermediate_attn_lse" not in self._graph_buffers[bsz]:
            self._graph_buffers[bsz]["intermediate_attn_lse"] = torch.zeros(
                (bsz, self.num_heads, self._graph_metadata[bsz]["max_split_num"]),
                dtype=dtype,
                device=device,
            )

    self._graph_buffers[bsz]["prefetch_query_states"] = torch.zeros(
        (bsz, 1, self.num_heads, self.head_dim),
        dtype=dtype,
        device=device,
    )


def attention_sparse_offloading_prefill_forward(
    self,
    hidden_states: torch.Tensor,
    past_key_value: Optional[OffloadingCache] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    chunk_size = 8192
    batch_size = past_key_value.get_cur_batch_size()
    token_num, hidden_size = hidden_states.shape
    local_attn_hidden_size = _local_attn_hidden_size(self)
    seq_len = token_num // batch_size

    past_key_value.sync_offload_prefill()

    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    reuse_hidden_query_buffer = local_attn_hidden_size == hidden_size
    if reuse_hidden_query_buffer:
        query_states_2d = hidden_states
    else:
        query_states_2d = torch.empty(
            (token_num, local_attn_hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    num_chunks = (token_num + chunk_size - 1) // chunk_size
    if num_chunks > 1:
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, token_num)
            if reuse_hidden_query_buffer:
                query_states_2d[start:end] = self.q_proj(hidden_states[start:end])
            else:
                custom_linear_forward_wrapper(
                    hidden_states[start:end],
                    self.q_proj,
                    out=query_states_2d[start:end],
                )
    else:
        if reuse_hidden_query_buffer:
            hidden_states[:] = self.q_proj(hidden_states)
        else:
            custom_linear_forward_wrapper(
                hidden_states,
                self.q_proj,
                out=query_states_2d,
            )
    query_states = query_states_2d

    query_states = query_states.view(-1, self.num_heads, self.head_dim)
    key_states = key_states.view(-1, self.num_key_value_heads, self.head_dim)
    query_states, key_states = self.rotary_emb(query_states, key_states, past_key_value)
    query_states = query_states.view(batch_size, -1, self.num_heads, self.head_dim)
    key_states = key_states.view(batch_size, -1, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(batch_size, -1, self.num_key_value_heads, self.head_dim)
    extend_seq_lens = getattr(past_key_value, "_current_extend_seq_lens", None)
    if extend_seq_lens is None:
        row_lens = [int(seq_len)] * batch_size
    else:
        row_lens = [int(x) for x in extend_seq_lens]

    past_key_value.append_topk_cache_prefill(
        query_states,
        key_states,
        value_states,
        self.layer_idx,
    )
    key_states, value_states = past_key_value.append_prefill(
        key_states,
        value_states,
        self.layer_idx,
    )

    old_lengths_by_layer = getattr(past_key_value, "_last_prefill_old_lengths", {})
    prefill_old_lengths = old_lengths_by_layer.get(self.layer_idx)
    ragged_prefill = (
        prefill_old_lengths is not None
        and (
            len(set(int(x) for x in prefill_old_lengths)) > 1
            or len(set(int(x) for x in row_lens)) > 1
            or any(int(x) != query_states.shape[1] for x in row_lens)
        )
    )

    if ragged_prefill:
        attn_output = torch.zeros_like(query_states)
        for row, (old_len, row_len) in enumerate(zip(prefill_old_lengths, row_lens)):
            if int(row_len) <= 0:
                continue
            k_len = int(old_len) + int(row_len)
            attn_output[row:row + 1, :row_len] = _flash_attn_with_kvcache(
                query_states[row:row + 1, :row_len],
                k_cache=key_states[row:row + 1, :k_len, ...],
                v_cache=value_states[row:row + 1, :k_len, ...],
                causal=self.is_causal,
                q_start_idx=int(old_len),
            )
        query_states[:] = attn_output
    elif num_chunks <= 1:
        q_start_idx = max(int(key_states.shape[1]) - int(seq_len), 0)
        attn_output = _flash_attn_with_kvcache(
            query_states,
            k_cache=key_states,
            v_cache=value_states,
            causal=self.is_causal,
            q_start_idx=q_start_idx,
        )
        query_states[:] = attn_output
    else:
        q_start_base = max(int(key_states.shape[1]) - int(seq_len), 0)
        attn_chunk_size = chunk_size // batch_size
        attn_num_chunks = (seq_len + attn_chunk_size - 1) // attn_chunk_size
        for i in range(attn_num_chunks):
            start = i * attn_chunk_size
            end = min(start + attn_chunk_size, seq_len)
            chunk_k = key_states[:, :q_start_base + end, ...]
            chunk_v = value_states[:, :q_start_base + end, ...]
            chunk_q = query_states[:, start:end, ...]
            chunk_attn_out = _flash_attn_with_kvcache(
                chunk_q,
                k_cache=chunk_k,
                v_cache=chunk_v,
                causal=self.is_causal,
                q_start_idx=q_start_base + start,
            )
            query_states[:, start:end, ...] = chunk_attn_out
    attn_output = query_states.view(-1, local_attn_hidden_size)

    if num_chunks > 1:
        output_hidden_states = hidden_states.view(-1, hidden_size)
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, token_num)
            if reuse_hidden_query_buffer:
                output_hidden_states[start:end] = self.o_proj(attn_output[start:end])
            else:
                custom_linear_forward_wrapper(
                    attn_output[start:end],
                    self.o_proj,
                    out=output_hidden_states[start:end],
                )
        attn_output = output_hidden_states
    else:
        attn_output = self.o_proj(attn_output)

    return attn_output


def _compute_prefetch_query(
    self,
    hidden_input_buffer: torch.Tensor,
    past_key_value: OffloadingCache,
):
    bsz, hidden_size = hidden_input_buffer.shape
    local_attn_hidden_size = _local_attn_hidden_size(self)
    hidden_normed = self.next_input_layernorm(hidden_input_buffer, is_prefill=False)
    custom_linear_forward_wrapper(
        hidden_normed,
        self.next_q_proj,
        out=self._graph_buffers[bsz]["prefetch_query_states"].view(
            -1,
            local_attn_hidden_size,
        ),
    )
    query_states = self._graph_buffers[bsz]["prefetch_query_states"].view(
        -1,
        self.num_heads,
        self.head_dim,
    )
    query_states, _ = self.next_rotary_emb(query_states, query_states, past_key_value)
    query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)

    return query_states


def attention_sparse_offloading_decode_forward(
    self,
    hidden_input_buffer: torch.Tensor,
    residual: torch.Tensor,
    past_key_value: Optional[OffloadingCache] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, hidden_size = hidden_input_buffer.shape
    local_attn_hidden_size = _local_attn_hidden_size(self)
    assert bsz == past_key_value.get_cur_batch_size()

    torch.cuda.nvtx.range_push("qkv")
    custom_linear_forward_wrapper(
        hidden_input_buffer,
        self.q_proj,
        out=self._graph_buffers[bsz]["query_states"].view(
            -1,
            local_attn_hidden_size,
        ),
    )
    custom_linear_forward_wrapper(
        hidden_input_buffer,
        self.k_proj,
        out=self._graph_buffers[bsz]["key_states"].view(
            -1,
            self.num_key_value_heads * self.head_dim,
        ),
    )
    custom_linear_forward_wrapper(
        hidden_input_buffer,
        self.v_proj,
        out=self._graph_buffers[bsz]["value_states"].view(
            -1,
            self.num_key_value_heads * self.head_dim,
        ),
    )
    query_states = self._graph_buffers[bsz]["query_states"].view(-1, self.num_heads, self.head_dim)
    key_states = self._graph_buffers[bsz]["key_states"].view(-1, self.num_key_value_heads, self.head_dim)
    value_states = self._graph_buffers[bsz]["value_states"].view(-1, self.num_key_value_heads, self.head_dim)
    query_states, key_states = self.rotary_emb(query_states, key_states, past_key_value)
    query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)
    torch.cuda.nvtx.range_pop()

    prefetch_query_states = None
    if past_key_value.enable_layer_prefetch:
        torch.cuda.nvtx.range_push("simq")
        prefetch_query_states = _compute_prefetch_query(self, residual, past_key_value)
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("decode append")
    past_key_value.append_decode(
        key_states,
        value_states,
        self.layer_idx,
        prefetch_query_states,
        query_states,
    )
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("attn")
    (
        k_cache,
        v_cache,
        topk_index,
        topk_index_count,
        k_buffer,
        v_buffer,
        buffer_count,
        mask,
        mixed_head_ids,
    ) = past_key_value.get_attention_data(
        self.layer_idx,
        query_states.device.index,
    )
    if self._graph_metadata[bsz]["topk_split_num"] > 1:
        decode_mixed_split_attention_fwd_grouped(
            query_states,
            self._graph_buffers[bsz]["attn_output"],
            self._graph_buffers[bsz]["intermediate_attn_logits"],
            self._graph_buffers[bsz]["intermediate_attn_lse"],
            k_cache,
            v_cache,
            topk_index,
            topk_index_count,
            k_buffer,
            v_buffer,
            buffer_count,
            mask,
            mixed_head_ids,
            self._graph_buffers[bsz]["topk_split_num"],
            self._graph_metadata[bsz]["max_split_num"],
            self._scale,
        )
    else:
        decode_mixed_attention_fwd_grouped(
            query_states,
            self._graph_buffers[bsz]["attn_output"],
            k_cache,
            v_cache,
            topk_index,
            topk_index_count,
            k_buffer,
            v_buffer,
            buffer_count,
            mask,
            mixed_head_ids,
            self._scale,
        )
    torch.cuda.nvtx.range_pop()

    attn_output = self._graph_buffers[bsz]["attn_output"].view(
        -1,
        local_attn_hidden_size,
    )
    output_buffer = self._graph_buffers[bsz]["output_hidden_states"]
    torch.cuda.nvtx.range_push("o_proj")
    custom_linear_forward_wrapper(
        attn_output,
        self.o_proj,
        out=output_buffer,
    )
    torch.cuda.nvtx.range_pop()
    return output_buffer


def transformer_layer_sparse_offloading_forward_prefill(
    self,
    hidden_states,
    past_key_value: Optional[OffloadingCache] = None,
):
    residual = hidden_states.clone()
    hidden_states = self.input_layernorm(hidden_states, is_prefill=True)

    hidden_states = self.self_attn(
        hidden_states=hidden_states,
        past_key_value=past_key_value,
    )
    torch.add(residual, hidden_states, out=hidden_states)

    residual.copy_(hidden_states)
    hidden_states = self.post_attention_layernorm(hidden_states, is_prefill=True)
    hidden_states = self.mlp(hidden_states, is_prefill=True)
    torch.add(residual, hidden_states, out=hidden_states)

    return hidden_states


def transformer_layer_sparse_offloading_forward_decode(
    self,
    hidden_input_buffer,
    past_key_value: Optional[OffloadingCache] = None,
):
    torch.cuda.nvtx.range_push("layer forward")
    hidden_prenorm_buffer = self.input_layernorm(hidden_input_buffer, is_prefill=False)
    kwargs = {
        "residual": hidden_input_buffer,
    }
    torch.cuda.nvtx.range_push("self_attn")
    hidden_attn_buffer = self.self_attn(
        hidden_prenorm_buffer,
        past_key_value=past_key_value,
        **kwargs,
    )
    torch.cuda.nvtx.range_pop()
    hidden_input_buffer.add_(hidden_attn_buffer)

    hidden_postnorm_buffer = self.post_attention_layernorm(
        hidden_input_buffer,
        is_prefill=False,
    )
    torch.cuda.nvtx.range_push("ffn")
    hidden_mlp_buffer = self.mlp(hidden_postnorm_buffer, is_prefill=False)
    torch.cuda.nvtx.range_pop()
    hidden_input_buffer.add_(hidden_mlp_buffer)
    torch.cuda.nvtx.range_pop()

    return hidden_input_buffer


def register_prefetch_module(self):
    num_layers = len(self.layers)
    for i in range(num_layers):
        next_id = (i + 1) % num_layers
        self.layers[i].self_attn.next_input_layernorm = self.layers[next_id].input_layernorm
        self.layers[i].self_attn.next_q_proj = self.layers[next_id].self_attn.q_proj
        self.layers[i].self_attn.next_rotary_emb = self.layers[next_id].self_attn.rotary_emb


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


def llm_sparse_offloading_prepare_cuda_graph_metadata(
    self,
    bsz,
    hidden_size,
    dtype=torch.bfloat16,
    device="cuda",
):
    llm_prepare_cuda_graph_metadata(self, bsz, hidden_size, dtype, device)
    register_prefetch_module(self)


def llm_sparse_offloading_prefill_forward(
    self,
    input_ids: torch.LongTensor = None,
    past_key_values: Optional[OffloadingCache] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    hidden_states = self.embed_tokens(input_ids)

    bsz, q_len, _ = hidden_states.shape
    past_key_values.update_metadata(q_len, is_prefill=True)
    hidden_states = hidden_states.view(bsz * q_len, -1)
    for decoder_layer in self.layers:
        hidden_states = decoder_layer(
            hidden_states,
            past_key_value=past_key_values,
        )
    # Align with internal prototype: only keep the last real prefill token as decode
    # input. Online batching can pad ragged extend rows to a common q_len, so
    # the final physical column is not always a real token.
    hidden_states = hidden_states.view(bsz, q_len, -1)
    extend_seq_lens = getattr(past_key_values, "_current_extend_seq_lens", None)
    if extend_seq_lens is None:
        hidden_states = hidden_states[:, -1, :]
    else:
        row_ids = torch.arange(bsz, dtype=torch.long, device=hidden_states.device)
        token_ids = torch.tensor(
            [int(x) - 1 for x in extend_seq_lens],
            dtype=torch.long,
            device=hidden_states.device,
        )
        if int(token_ids.min().item()) < 0 or int(token_ids.max().item()) >= q_len:
            raise RuntimeError(
                "Ditto prefill extend_seq_lens out of range: "
                f"extend_seq_lens={extend_seq_lens}, q_len={q_len}"
            )
        hidden_states = hidden_states[row_ids, token_ids, :]
    hidden_states = hidden_states.view(bsz, -1).contiguous()
    hidden_states = self.norm(hidden_states, is_prefill=True)
    hidden_states = hidden_states.view(bsz, 1, -1)

    past_key_values.sync_offload_prefill()

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=None,
        attentions=None,
    )


def llm_sparse_offloading_decode_forward(
    self,
    input_ids: torch.LongTensor = None,
    past_key_values: Optional[OffloadingCache] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    hidden_states = self.embed_tokens(input_ids)
    bsz, q_len, _ = hidden_states.shape
    assert q_len == 1, "Only support decode with q_len == 1"
    past_key_values.update_metadata(q_len)
    hidden_states = hidden_states.view(bsz * q_len, -1)
    graph_path = "eager"

    in_outer_cuda_graph_capture = False
    try:
        in_outer_cuda_graph_capture = torch.cuda.is_current_stream_capturing()
    except Exception:
        in_outer_cuda_graph_capture = False

    if past_key_values.config.enable_cuda_graph and not in_outer_cuda_graph_capture:
        if bsz not in self._graph_buffers:
            graph_path = "warmup"
            llm_sparse_offloading_prepare_cuda_graph_metadata(
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
            graph_path = "capture"
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
            graph_path = "replay"
            self._graph_buffers[bsz]["input_hidden_states"].copy_(hidden_states)
            self._graphs[bsz].replay()

        hidden_states = self._graph_buffers[bsz]["output_hidden_states"]

    else:
        graph_path = "fallback_eager"
        if bsz not in self._graph_buffers:
            llm_sparse_offloading_prepare_cuda_graph_metadata(
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
    if DITTO_CUDA_GRAPH_DEBUG:
        logger.info(
            "Ditto decode graph path=%s, enable_cuda_graph=%s, in_outer_capture=%s, bsz=%d",
            graph_path,
            bool(past_key_values.config.enable_cuda_graph),
            bool(in_outer_cuda_graph_capture),
            int(bsz),
        )

    past_key_values.record_decode_transfer_step()

    hidden_states = hidden_states.view(bsz, 1, -1)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=None,
        attentions=None,
    )
