from __future__ import annotations

import gc
import math
import time
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2Model,
)
from transformers.utils import logging

try:
    # transformers<=4.47
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2FlashAttention2 as _Qwen2FlashAttention2,
    )
except ImportError:
    # transformers>=4.57 removed Qwen2FlashAttention2 symbol. Keep a thin wrapper
    # so downstream custom attention can continue to subclass the same semantic base.
    class _Qwen2FlashAttention2(Qwen2Attention):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)


import sgl_kernel.kvlib as KVLib
from sglang.ditto.kvcache_offloading_duohead_base import OffloadingCache
from sglang.srt.models.ditto.qwen2_utils import (
    CustomerQwen2MLP,
    CustomQwen2RMSNorm,
    CustomQwen2RotaryEmbedding,
    apply_ditto_tp_attention_layout,
)

try:
    import flash_attn as _flash_attn
except ImportError:
    _flash_attn = None

logger = logging.get_logger(__name__)

CHUNK_SIZE = 8192


def _local_attn_hidden_size(self) -> int:
    return int(getattr(self, "local_attn_hidden_size", self.num_heads * self.head_dim))

def _replace_backbone_model(parent: nn.Module, model_cls, config) -> None:
    """
    Avoid a temporary 2x memory peak:
    LlamaForCausalLM/Qwen2ForCausalLM creates a default backbone in super().__init__.
    Free it before instantiating our custom backbone.
    """
    old_model = parent.model
    parent.model = None
    del old_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    parent.model = model_cls(config)


def _flash_attn_with_kvcache(
    query_states: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    causal: bool = True,
    cache_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if _flash_attn is not None:
        return _flash_attn.flash_attn_with_kvcache(
            query_states,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            causal=causal,
        )

    # Bring-up fallback when flash_attn Python package is unavailable.
    # Keeps correctness for integration debugging (lower performance).
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
    if cache_seqlens is None:
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
        )
    else:
        out_rows = []
        for row, seq_len in enumerate(cache_seqlens.detach().cpu().tolist()):
            out_rows.append(
                F.scaled_dot_product_attention(
                    q[row:row + 1],
                    k[row:row + 1, :, : int(seq_len), :],
                    v[row:row + 1, :, : int(seq_len), :],
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=causal,
                )
            )
        out = torch.cat(out_rows, dim=0)
    return out.transpose(1, 2).contiguous()


class CustomQwen2Attention(_Qwen2FlashAttention2):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        apply_ditto_tp_attention_layout(self, config)
        self.rotary_emb = CustomQwen2RotaryEmbedding(config)
        self.scale = 1 / math.sqrt(self.head_dim)
        self.next_input_layernorm = None
        self.next_q_proj = None
        self.next_rotary_emb = None

    def compute_sim_query(
        self,
        residual: torch.Tensor,
        past_key_value: OffloadingCache,
    ):
        batch_size = past_key_value.curr_batch_size
        next_sim_query = self.next_input_layernorm(residual)
        next_sim_query = self.next_q_proj(next_sim_query)
        next_sim_query = next_sim_query.view(-1, self.num_heads, self.head_dim)
        next_sim_query, _ = self.next_rotary_emb(
            next_sim_query, next_sim_query, past_key_value
        )
        next_sim_query = next_sim_query.view(
            batch_size, -1, self.num_heads, self.head_dim
        )
        return next_sim_query

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[OffloadingCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        del attention_mask, position_ids, output_attentions, use_cache, cache_position, position_embeddings, kwargs

        batch_size = past_key_value.curr_batch_size
        q_len = past_key_value.get_cur_q_len()
        is_prefill = q_len > 1
        token_num, hidden_size = hidden_states.size()
        local_attn_hidden_size = _local_attn_hidden_size(self)
        num_chunks = (token_num + CHUNK_SIZE - 1) // CHUNK_SIZE

        if is_prefill:
            past_key_value.prefill_sync()
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            if local_attn_hidden_size == hidden_size:
                query_states_2d = hidden_states
            else:
                query_states_2d = torch.empty(
                    (token_num, local_attn_hidden_size),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
            if token_num > CHUNK_SIZE:
                for i in range(num_chunks):
                    start = i * CHUNK_SIZE
                    end = min(start + CHUNK_SIZE, token_num)
                    query_states_2d[start:end] = self.q_proj(hidden_states[start:end])
                query_states = query_states_2d
            else:
                query_states = self.q_proj(hidden_states)
        else:
            if past_key_value.is_first_decode_step() and self.layer_idx == 0:
                past_key_value.prefill_sync()
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(-1, self.num_heads, self.head_dim)
        key_states = key_states.view(-1, self.num_key_value_heads, self.head_dim)
        query_states, key_states = self.rotary_emb(query_states, key_states, past_key_value)
        query_states = query_states.view(batch_size, -1, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, -1, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, -1, self.num_key_value_heads, self.head_dim)

        if is_prefill:
            past_key_value.prefill_append(query_states, key_states, value_states, self.layer_idx)
            if token_num < CHUNK_SIZE:
                attn_output = _flash_attn_with_kvcache(
                    query_states,
                    k_cache=key_states,
                    v_cache=value_states,
                    causal=self.is_causal,
                )
            else:
                attn_chunk_size = CHUNK_SIZE // batch_size
                attn_num_chunks = (q_len + attn_chunk_size - 1) // attn_chunk_size
                for i in range(attn_num_chunks):
                    start = i * attn_chunk_size
                    end = min(start + attn_chunk_size, q_len)
                    chunk_k = key_states[:, :end, ...]
                    chunk_v = value_states[:, :end, ...]
                    chunk_q = query_states[:, start:end, ...]
                    chunk_attn_out = _flash_attn_with_kvcache(
                        chunk_q,
                        k_cache=chunk_k,
                        v_cache=chunk_v,
                        causal=self.is_causal,
                    )
                    query_states[:, start:end, ...] = chunk_attn_out
                attn_output = query_states
        else:
            if past_key_value.need_prefetch(self.layer_idx + 1):
                prefetch_query_states = self.compute_sim_query(residual, past_key_value)
            else:
                prefetch_query_states = None

            past_key_value.decode_append(
                key_states, value_states, self.layer_idx, prefetch_query_states, query_states
            )

            if past_key_value.need_prefetch(self.layer_idx):
                if past_key_value.has_gpu_heads(self.layer_idx):
                    (
                        key_cache,
                        value_cache,
                        topk_index,
                        key_buffer,
                        value_buffer,
                        mask,
                        hindex,
                        buffer_len,
                    ) = past_key_value.decode_get_attn_data_mixed(self.layer_idx)
                    attn_output, _ = KVLib.flash_mixed_decode(
                        query_states,
                        key_cache,
                        value_cache,
                        topk_index,
                        key_buffer,
                        value_buffer,
                        mask,
                        hindex,
                        buffer_len,
                        self.scale,
                    )
                else:
                    key_states, value_states, cache_seqlens = past_key_value.decode_get_attn_data_full_cpu(
                        self.layer_idx
                    )
                    attn_output = _flash_attn_with_kvcache(
                        query_states,
                        k_cache=key_states,
                        v_cache=value_states,
                        cache_seqlens=cache_seqlens,
                    )
            else:
                key_states, value_states, topk_indices, topk_count = past_key_value.decode_get_attn_data_full_gpu(
                    self.layer_idx
                )
                attn_output, _ = KVLib.flash_index_decode(
                    query_states,
                    key_states,
                    value_states,
                    topk_indices,
                    topk_count,
                    self.scale,
                )

        attn_output = attn_output.view(-1, local_attn_hidden_size)
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
        self.self_attn = None
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.mlp = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.self_attn = CustomQwen2Attention(config, layer_idx)
        self.input_layernorm = CustomQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CustomQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = CustomerQwen2MLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[OffloadingCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if hidden_states.device.index != torch.cuda.current_device():
            torch.cuda.set_device(hidden_states.device)
        residual = hidden_states.clone()
        hidden_states = self.input_layernorm(hidden_states)
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
        residual.copy_(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        torch.add(residual, hidden_states, out=hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class CustomQwen2Model(Qwen2Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = None
        self.norm = None
        self.rotary_emb = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.layers = nn.ModuleList(
            [CustomQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = CustomQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        for i in range(config.num_hidden_layers):
            next_id = (i + 1) % config.num_hidden_layers
            self.layers[i].self_attn.next_input_layernorm = self.layers[next_id].input_layernorm
            self.layers[i].self_attn.next_q_proj = self.layers[next_id].self_attn.q_proj
            self.layers[i].self_attn.next_rotary_emb = self.layers[next_id].self_attn.rotary_emb

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[OffloadingCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one")
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # In transformers>=4.57, Qwen2Model no longer exposes `_update_causal_mask`.
        # Ditto attention path does not consume this mask, so keep behavior by using None.
        causal_mask = None
        hidden_states = inputs_embeds
        bsz, seq_len, _ = hidden_states.shape

        past_key_values.alloc(seq_len)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        hidden_states = hidden_states.view(bsz * seq_len, -1)

        if hasattr(self, "profiling"):
            torch.cuda.synchronize()
            tic = time.time()

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=None,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if hasattr(self, "profiling"):
            torch.cuda.synchronize()
            toc = time.time()
            if seq_len == 1:
                self.decoding_time_list.append(toc - tic)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(bsz, seq_len, -1)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class HashQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        _replace_backbone_model(self, CustomQwen2Model, config)
        from sglang.ditto.kvcache_offloading_hash import prepare_cache_for_generation

        transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = (
            prepare_cache_for_generation
        )


class LokiQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        _replace_backbone_model(self, CustomQwen2Model, config)
        from sglang.ditto.kvcache_offloading_loki import prepare_cache_for_generation

        transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = (
            prepare_cache_for_generation
        )


class InfiniGenQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        _replace_backbone_model(self, CustomQwen2Model, config)
        from sglang.ditto.kvcache_offloading_infinigen import prepare_cache_for_generation

        transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = (
            prepare_cache_for_generation
        )


class QuestQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        _replace_backbone_model(self, CustomQwen2Model, config)
        from sglang.ditto.kvcache_offloading_quest import prepare_cache_for_generation

        transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = (
            prepare_cache_for_generation
        )
