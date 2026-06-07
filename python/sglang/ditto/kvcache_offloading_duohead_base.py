from __future__ import annotations

from typing import Optional, Tuple

import torch

from .kvcache_offloading import OffloadingCache as _BaseOffloadingCache


class OffloadingCache(_BaseOffloadingCache):
    """
    Compatibility adapter for internal prototype duohead model framework.

    The existing SGLang Ditto implementation uses slightly different method
    names/signatures than internal prototype's duohead model file. This adapter
    bridges those APIs so we can port model/framework code with minimal edits.
    """

    def alloc(self, q_len: int):
        # internal prototype API: alloc() prepares rope metadata for this step.
        # SGLang Ditto API already does this in update_metadata().
        self.update_metadata(q_len=q_len, is_prefill=(q_len > 1), layer_idx=0)

    def prefill_sync(self):
        # internal prototype API name
        return self.sync_offload_prefill()

    def is_first_decode_step(self):
        # internal prototype API name.
        return self.first_decode_layer_step

    def need_prefetch(self, layer_idx: int):
        # internal prototype API name.
        layer_idx = layer_idx % self.num_layers
        return not self.layers_full_gpu_mask[layer_idx]

    def has_gpu_heads(self, layer_idx: int):
        # internal prototype API name.
        layer_idx = layer_idx % self.num_layers
        return self.layers_num_gpu_buffer_heads[layer_idx] < self.num_key_value_heads

    def prefill_append(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        # Keep internal prototype semantics: prefill updates both KV cache and topk data.
        key_states, value_states = self.append_prefill(key_states, value_states, layer_idx)
        self.append_topk_cache_prefill(query_states, key_states, value_states, layer_idx)
        return key_states, value_states

    def decode_append(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        prefetch_query_states: Optional[torch.Tensor] = None,
        current_query_states: Optional[torch.Tensor] = None,
    ):
        # internal prototype API name.
        return self.append_decode(
            key_states,
            value_states,
            layer_idx,
            prefetch_query_states,
            current_query_states,
        )

    def decode_get_attn_data_full_gpu(self, layer_idx: int):
        device_idx = self.layer_devices[layer_idx]
        (
            key_cache,
            value_cache,
            topk_index,
            topk_count,
            _key_buffer,
            _value_buffer,
            _buffer_count,
            _mask,
            _mixed_head_ids,
        ) = self.get_attention_data(layer_idx, device_idx)
        return key_cache, value_cache, topk_index, topk_count[: self.curr_batch_size]

    def decode_get_attn_data_full_cpu(self, layer_idx: int):
        device_idx = self.layer_devices[layer_idx]
        (
            _key_cache,
            _value_cache,
            _topk_index,
            _topk_count,
            key_buffer,
            value_buffer,
            buffer_count,
            _mask,
            _mixed_head_ids,
        ) = self.get_attention_data(layer_idx, device_idx)
        if buffer_count is None:
            return key_buffer, value_buffer, None
        active_count = buffer_count[: self.curr_batch_size]
        buffer_len = int(active_count.max().item())
        key_states = key_buffer[:, :buffer_len]
        value_states = value_buffer[:, :buffer_len]
        return key_states, value_states, active_count

    def decode_get_attn_data_mixed(self, layer_idx: int):
        device_idx = self.layer_devices[layer_idx]
        (
            key_cache,
            value_cache,
            topk_index,
            _topk_count,
            key_buffer,
            value_buffer,
            buffer_count,
            mask,
            mixed_head_ids,
        ) = self.get_attention_data(layer_idx, device_idx)
        buffer_len = None if buffer_count is None else buffer_count[: self.curr_batch_size]
        return (
            key_cache,
            value_cache,
            topk_index,
            key_buffer,
            value_buffer,
            mask,
            mixed_head_ids,
            buffer_len,
        )
