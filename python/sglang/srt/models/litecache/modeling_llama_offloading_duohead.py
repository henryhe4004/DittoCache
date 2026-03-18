import math
import time
from typing import Optional, Tuple, Union

import flash_attn
import sgl_kernel.kvlib as KVLib
import torch
import torch.nn as nn
import transformers
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaFlashAttention2,
    LlamaForCausalLM,
    LlamaModel,
)
from transformers.utils import logging

from sglang.litecache.kvcache_offloading_duohead_base import OffloadingCache
from sglang.srt.models.litecache.llama_utils import (
    CustomerLlamaMLP,
    CustomLlamaRMSNorm,
    CustomLlamaRotaryEmbedding,
)

logger = logging.get_logger(__name__)

CHUNK_SIZE = 8192


class CustomLlamaAttention(LlamaFlashAttention2):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.rotary_emb = CustomLlamaRotaryEmbedding(config)
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
        num_chunks = (token_num + CHUNK_SIZE - 1) // CHUNK_SIZE

        if is_prefill:
            past_key_value.prefill_sync()
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            if token_num > CHUNK_SIZE:
                for i in range(num_chunks):
                    start = i * CHUNK_SIZE
                    end = min(start + CHUNK_SIZE, token_num)
                    hidden_states[start:end] = self.q_proj(hidden_states[start:end])
                query_states = hidden_states
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
                attn_output = flash_attn.flash_attn_with_kvcache(
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
                    chunk_attn_out = flash_attn.flash_attn_with_kvcache(
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
                    key_states, value_states = past_key_value.decode_get_attn_data_full_cpu(
                        self.layer_idx
                    )
                    attn_output = flash_attn.flash_attn_with_kvcache(
                        query_states, k_cache=key_states, v_cache=value_states
                    )
            else:
                key_states, value_states, topk_indices = past_key_value.decode_get_attn_data_full_gpu(
                    self.layer_idx
                )
                attn_output, _ = KVLib.flash_index_decode(
                    query_states, key_states, value_states, topk_indices, self.scale
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


class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = CustomLlamaAttention(config, layer_idx)
        self.input_layernorm = CustomLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CustomLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = CustomerLlamaMLP(config=config)

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


class CustomLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [CustomLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = CustomLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = None

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

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
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


class HashLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlamaModel(config)
        from sglang.litecache.kvcache_offloading_hash import prepare_cache_for_generation

        transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = (
            prepare_cache_for_generation
        )


class LokiLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlamaModel(config)
        from sglang.litecache.kvcache_offloading_loki import prepare_cache_for_generation

        transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = (
            prepare_cache_for_generation
        )


class InfiniGenLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlamaModel(config)
        from sglang.litecache.kvcache_offloading_infinigen import (
            prepare_cache_for_generation,
        )

        transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = (
            prepare_cache_for_generation
        )


class QuestLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlamaModel(config)
        from sglang.litecache.kvcache_offloading_quest import prepare_cache_for_generation

        transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = (
            prepare_cache_for_generation
        )

