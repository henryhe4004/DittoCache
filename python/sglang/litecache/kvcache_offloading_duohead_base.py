from __future__ import annotations

from typing import Optional, Tuple

import torch

from .kvcache_offloading import OffloadingCache as _BaseOffloadingCache


class OffloadingCache(_BaseOffloadingCache):
    """
    Compatibility adapter for myTransformer duohead model framework.

    The existing SGLang LiteCache implementation uses slightly different method
    names/signatures than myTransformer's duohead model file. This adapter
    bridges those APIs so we can port model/framework code with minimal edits.
    """

    def alloc(self, q_len: int):
        # myTransformer API: alloc() prepares rope metadata for this step.
        # SGLang LiteCache API already does this in update_metadata().
        self.update_metadata(q_len=q_len, is_prefill=(q_len > 1), layer_idx=0)

    def prefill_sync(self):
        # myTransformer API name
        return self.sync_offload_prefill()

    def prefill_append(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        # query_states is used by append_topk_cache_prefill in subclasses.
        return self.append_prefill(key_states, value_states, layer_idx, query_states)

    def decode_get_attn_data_full_gpu(self, layer_idx: int):
        device_idx = self.layer_devices[layer_idx]
        (
            key_cache,
            value_cache,
            topk_index,
            _topk_count,
            _key_buffer,
            _value_buffer,
            _buffer_count,
            _mask,
            _mixed_head_ids,
        ) = self.get_attention_data(layer_idx, device_idx)
        return key_cache, value_cache, topk_index

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
            return key_buffer, value_buffer
        buffer_len = int(buffer_count.item())
        key_states = key_buffer[:, :buffer_len]
        value_states = value_buffer[:, :buffer_len]
        return key_states, value_states

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
        buffer_len = 0 if buffer_count is None else int(buffer_count.item())
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

