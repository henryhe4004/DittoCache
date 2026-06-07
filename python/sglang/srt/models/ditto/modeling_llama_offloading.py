from __future__ import annotations

import gc
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaFlashAttention2,
    LlamaForCausalLM,
    LlamaModel,
)
from transformers.utils import logging

from sglang.srt.models.ditto.llama_utils import (
    CustomerLlamaMLP,
    CustomLlamaRMSNorm,
    CustomLlamaRotaryEmbedding,
    apply_ditto_tp_attention_layout,
)
from sglang.srt.models.ditto.offloading_ops import (
    attention_sparse_offloading_decode_forward,
    attention_sparse_offloading_prefill_forward,
    attention_sparse_offloading_prepare_cuda_graph_metadata,
    llm_sparse_offloading_decode_forward,
    llm_sparse_offloading_prefill_forward,
    transformer_layer_sparse_offloading_forward_decode,
    transformer_layer_sparse_offloading_forward_prefill,
)

logger = logging.get_logger(__name__)


def _replace_backbone_model(parent: nn.Module, model_cls, config) -> None:
    """
    Avoid a temporary 2x memory peak:
    LlamaForCausalLM creates a default backbone in super().__init__.
    Free it before instantiating our custom backbone.
    """
    old_model = parent.model
    parent.model = None
    del old_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    parent.model = model_cls(config)


class CustomLlamaAttention(LlamaFlashAttention2):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        apply_ditto_tp_attention_layout(self, config)
        self.rotary_emb = CustomLlamaRotaryEmbedding(config)

        self._graph_metadata = {}
        self._graph_buffers = {}
        self._scale = 1 / math.sqrt(self.head_dim)

        self.next_input_layernorm = None
        self.next_q_proj = None
        self.next_rotary_emb = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        _ = attention_mask, position_ids, output_attentions, use_cache, cache_position, position_embeddings

        q_len = hidden_states.shape[0] // past_key_value.curr_batch_size
        if q_len > 1:
            return attention_sparse_offloading_prefill_forward(
                self,
                hidden_states,
                past_key_value,
            )

        bsz = past_key_value.get_cur_batch_size()
        if bsz not in self._graph_buffers:
            attention_sparse_offloading_prepare_cuda_graph_metadata(
                self,
                bsz,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
                topk=past_key_value.max_buffer_len,
            )

        return attention_sparse_offloading_decode_forward(
            self,
            hidden_states,
            kwargs["residual"],
            past_key_value,
        )


class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = CustomLlamaAttention(config, layer_idx)
        self.input_layernorm = CustomLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CustomLlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.mlp = CustomerLlamaMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        _ = attention_mask, position_ids, output_attentions, use_cache, cache_position, position_embeddings, kwargs

        q_len = hidden_states.shape[0] // past_key_value.curr_batch_size
        if q_len > 1:
            return transformer_layer_sparse_offloading_forward_prefill(
                self,
                hidden_states,
                past_key_value=past_key_value,
            )
        return transformer_layer_sparse_offloading_forward_decode(
            self,
            hidden_states,
            past_key_value=past_key_value,
        )


class CustomLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = None
        self.norm = None
        self.rotary_emb = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.layers = nn.ModuleList(
            [CustomLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = CustomLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = None

        self._graphs = {}
        self._graph_buffers = {}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        _ = attention_mask, position_ids, inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict, cache_position

        seq_len = input_ids.shape[1]
        if seq_len > 1:
            return llm_sparse_offloading_prefill_forward(
                self,
                input_ids=input_ids,
                past_key_values=past_key_values,
            )

        return llm_sparse_offloading_decode_forward(
            self,
            input_ids=input_ids,
            past_key_values=past_key_values,
        )


class OffloadingLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        _replace_backbone_model(self, CustomLlamaModel, config)
        from sglang.ditto.kvcache_hash import prepare_cache_for_generation

        transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = (
            prepare_cache_for_generation
        )
