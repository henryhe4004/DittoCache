from __future__ import annotations

import os
from typing import Dict, Optional, Union

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig

import sgl_kernel.kvlib as KVLib
from sglang.jit_kernel.legacy_triton_cache_kernels.triton_loki_kernels import (
    decode_loki_encode_k,
    decode_loki_encode_qk,
    decode_loki_encode_qqk,
    loki_score,
    prefill_loki_encode,
)

from .kvcache_offloading_duohead_base import OffloadingCache
from .config_utils import ensure_ditto_custom_config


class LokiOffloadingCache(OffloadingCache):
    def __init__(
        self,
        config: PretrainedConfig,
        custom_config,
        device: torch.device = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        self.num_channels = int(getattr(custom_config, "num_channels", 32))
        self.aux_data_path = getattr(custom_config, "aux_data_path", None)
        super().__init__(config, custom_config, device, layer_device_map)

    def _plan_topk_used_gpu_memory(self):
        gpu_partial_key_numel = (
            self.config.kvcache_manager_config.max_batch_size
            * self.config.kvcache_manager_config.max_tokens
            * self.num_key_value_heads
            * self.num_channels
        )
        gpu_need_mem = self.num_layers * gpu_partial_key_numel * self.dtype.itemsize
        assert gpu_need_mem <= self.mem_budget, (
            f"Loki partial-key cache needs {gpu_need_mem / 1024**3:.2f} GB, "
            f"but only {self.mem_budget / 1024**3:.2f} GB is available."
        )
        self.mem_budget -= gpu_need_mem

    def _create_topk_tensors(self):
        max_batch = self.config.kvcache_manager_config.max_batch_size
        max_seq = self.config.kvcache_manager_config.max_tokens
        gpu_partial_key_numel = max_batch * max_seq * self.num_key_value_heads * self.num_channels

        self.layers_partial_key_cache = [None for _ in range(self.num_layers)]
        self.layers_partial_key_cache_data = [None for _ in range(self.num_layers)]
        self.layers_gpu_partial_key_cache_length = [0 for _ in range(self.num_layers)]
        self.layers_pca_matrix = [None for _ in range(self.num_layers)]

        for layer in range(self.num_layers):
            self.layers_partial_key_cache_data[layer] = torch.zeros(
                (gpu_partial_key_numel,),
                dtype=self.dtype,
                device=self.layer_devices[layer],
            )

            if self.aux_data_path is None:
                pca = torch.randn(
                    (self.num_key_value_heads, self.head_dim, self.head_dim),
                    dtype=self.dtype,
                    device=self.layer_devices[layer],
                )
            else:
                pca = torch.load(
                    os.path.join(self.aux_data_path, f"pca_components/pca_components_layer_{layer:02d}.pt"),
                    weights_only=True,
                )
                pca = pca.view(-1, self.head_dim, self.head_dim)
                pca = self._slice_local_kv_head_tensor(
                    pca,
                    tensor_name=f"pca_components_layer_{layer:02d}",
                    head_dim=0,
                )
                pca = pca.transpose(-1, -2).contiguous().to(self.dtype).to(self.layer_devices[layer])
            self.layers_pca_matrix[layer] = pca

    def _reset_topk_tensors(self, batch_size):
        max_seq = self.max_seq_len
        gpu_partial_key_numel = batch_size * max_seq * self.num_key_value_heads * self.num_channels
        for layer in range(self.num_layers):
            self.layers_partial_key_cache[layer] = self.layers_partial_key_cache_data[layer][
                :gpu_partial_key_numel
            ].view(batch_size, max_seq, self.num_key_value_heads, self.num_channels)
            self.layers_gpu_partial_key_cache_length[layer] = 0

    def append_topk_cache_prefill(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        values_states: torch.Tensor,
        layer_idx: int,
    ):
        del query_states, values_states
        sequence_length = key_states.shape[1]
        prefill_loki_encode(
            key_states,
            self.layers_pca_matrix[layer_idx],
            self.layers_partial_key_cache[layer_idx],
            self.num_channels,
        )
        self.layers_gpu_partial_key_cache_length[layer_idx] += sequence_length

    def _decode_append_loki_qk(self, query_states, key_states, query_layer_idx, key_layer_idx):
        query_out = torch.zeros(
            (self.curr_batch_size, 1, self.num_heads, self.num_channels),
            device=query_states.device,
            dtype=self.dtype,
        )
        decode_loki_encode_qk(
            key_states,
            self.layers_partial_key_cache[key_layer_idx],
            self.layers_pca_matrix[key_layer_idx],
            query_states,
            query_out,
            self.layers_pca_matrix[query_layer_idx % self.num_layers],
            self.num_channels,
            self.layers_gpu_partial_key_cache_length[key_layer_idx],
        )
        self.layers_gpu_partial_key_cache_length[key_layer_idx] += 1
        return query_out

    def _decode_append_loki_qqk(
        self,
        query_states,
        query_states2,
        key_states,
        query_layer_idx,
        query_layer_idx2,
        key_layer_idx,
    ):
        query_out1 = torch.zeros(
            (self.curr_batch_size, 1, self.num_heads, self.num_channels),
            device=query_states.device,
            dtype=self.dtype,
        )
        query_out2 = torch.zeros(
            (self.curr_batch_size, 1, self.num_heads, self.num_channels),
            device=query_states.device,
            dtype=self.dtype,
        )
        decode_loki_encode_qqk(
            key_states,
            self.layers_partial_key_cache[key_layer_idx],
            self.layers_pca_matrix[key_layer_idx],
            query_states,
            query_out1,
            self.layers_pca_matrix[query_layer_idx % self.num_layers],
            query_states2,
            query_out2,
            self.layers_pca_matrix[query_layer_idx2 % self.num_layers],
            self.num_channels,
            self.layers_gpu_partial_key_cache_length[key_layer_idx],
        )
        self.layers_gpu_partial_key_cache_length[key_layer_idx] += 1
        return query_out1, query_out2

    def _decode_append_loki_k(self, key_states, layer_idx):
        decode_loki_encode_k(
            key_states,
            self.layers_partial_key_cache[layer_idx],
            self.layers_pca_matrix[layer_idx],
            self.num_channels,
            self.layers_gpu_partial_key_cache_length[layer_idx],
        )
        self.layers_gpu_partial_key_cache_length[layer_idx] += 1

    def append_topk_cache_decode(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        prefetch_query_states: Optional[torch.Tensor] = None,
        current_query_states: Optional[torch.Tensor] = None,
    ):
        del value_states
        next_layer_idx = (layer_idx + 1) % self.num_layers
        encode_current_query = self.needs_current_retrieval_query(layer_idx)
        encode_prefetch_query = not self.layers_full_gpu_mask[next_layer_idx]
        if encode_current_query and encode_prefetch_query:
            prefetch_query_code, current_query_code = self._decode_append_loki_qqk(
                prefetch_query_states,
                current_query_states,
                key_states,
                layer_idx + 1,
                layer_idx,
                layer_idx,
            )
        elif encode_current_query:
            current_query_code = self._decode_append_loki_qk(
                current_query_states, key_states, layer_idx, layer_idx
            )
            prefetch_query_code = None
        elif encode_prefetch_query:
            prefetch_query_code = self._decode_append_loki_qk(
                prefetch_query_states, key_states, layer_idx + 1, layer_idx
            )
            current_query_code = None
        else:
            self._decode_append_loki_k(key_states, layer_idx)
            prefetch_query_code = None
            current_query_code = None
        return prefetch_query_code, current_query_code

    def compute_topk(self, query, layer_idx, mask, is_prefetch=False):
        if is_prefetch:
            cache_length = self.layers_gpu_partial_key_cache_length[layer_idx]
            if self.topk_ratio < 1:
                fetch_num = int((cache_length + 1) * self.topk_ratio)
                fetch_num = min(
                    fetch_num,
                    cache_length
                    - self.config.sparse_attention_config.sink_budget
                    - self.config.sparse_attention_config.recent_budget,
                )
            else:
                fetch_num = min(
                    int(self.topk_ratio),
                    cache_length
                    - self.config.sparse_attention_config.sink_budget
                    - self.config.sparse_attention_config.recent_budget,
                )
            fetch_num = max(fetch_num, 0)
            valid_seq_len = (
                cache_length - self.config.sparse_attention_config.recent_budget
            )
            if fetch_num == 0 or valid_seq_len <= 0:
                return torch.empty(
                    (query.shape[0] * self.num_key_value_heads, 0),
                    dtype=torch.int32,
                    device=query.device,
                )

            score = loki_score(
                query,
                self.layers_partial_key_cache[layer_idx],
                valid_seq_len,
                head_mask=mask,
            )
            score[..., : self.config.sparse_attention_config.sink_budget] = torch.finfo(score.dtype).min
            topk_indices = self._batch_topk_masked_compat(score, mask, fetch_num, True).view(-1, fetch_num)
            topk_indices = topk_indices - self.config.sparse_attention_config.sink_budget
            return topk_indices

        cache_length = self.layers_gpu_partial_key_cache_length[layer_idx]
        if self.layers_full_gpu_mask[layer_idx]:
            if self.topk_ratio < 1:
                fetch_num = int(cache_length * self.topk_ratio) + self.config.sparse_attention_config.recent_budget + self.config.sparse_attention_config.sink_budget
                fetch_num = min(fetch_num, cache_length)
            else:
                fetch_num = min(
                    int(self.topk_ratio) + self.config.sparse_attention_config.recent_budget + self.config.sparse_attention_config.sink_budget,
                    cache_length,
                )
        else:
            fetch_num = int(self.metadata_tensors[f"topk_current_k_{query.device.index}"][0].item())

        if mask is not None:
            score = loki_score(
                query, self.layers_partial_key_cache[layer_idx], cache_length, head_mask=mask
            )
            score[..., : self.config.sparse_attention_config.sink_budget] = torch.finfo(score.dtype).max
            if self.config.sparse_attention_config.recent_budget > 0:
                score[..., -self.config.sparse_attention_config.recent_budget :] = torch.finfo(score.dtype).max
            topk_indices = self._batch_topk_masked_compat(score, mask, fetch_num, True).view(-1, fetch_num)
        else:
            score = loki_score(query, self.layers_partial_key_cache[layer_idx], cache_length)
            score[..., : self.config.sparse_attention_config.sink_budget] = torch.finfo(score.dtype).max
            if self.config.sparse_attention_config.recent_budget > 0:
                score[..., -self.config.sparse_attention_config.recent_budget :] = torch.finfo(score.dtype).max
            topk_indices = KVLib.batch_topk(score, fetch_num, True).view(-1, fetch_num)
        return topk_indices


def prepare_cache_for_generation(
    self,
    generation_config: GenerationConfig,
    model_kwargs: Dict,
    assistant_model,
    batch_size: int,
    max_cache_length: int,
    device: torch.device,
) -> bool:
    del assistant_model, max_cache_length
    generation_config.custom_config = ensure_ditto_custom_config(
        getattr(generation_config, "custom_config", None),
        self.config.get_text_config(),
    )
    if not hasattr(self, "_cache") or getattr(generation_config, "new_config", False):
        if hasattr(self, "_cache"):
            del self._cache

        def get_layer_device_map(execution_device_map: Optional[dict] = None):
            if execution_device_map is None or len(execution_device_map) <= 1:
                return None
            layer_device_map = {}
            for layer in execution_device_map:
                for idx in range(self.config.num_hidden_layers):
                    if f".{idx}." in f"{layer}.":
                        layer_device_map[idx] = execution_device_map[layer]
                        break
            for idx in range(self.config.num_hidden_layers):
                if idx not in layer_device_map:
                    raise RuntimeError(f"layer {idx} has not been mapped to a device.")
            return layer_device_map

        execution_device_map = None
        if hasattr(self, "hf_device_map"):
            main_device = [d for d in self.hf_device_map.values() if d not in ["cpu", "disk"]][0]
            execution_device_map = {
                name: main_device if dev in ["cpu", "disk"] else dev
                for name, dev in self.hf_device_map.items()
            }
        layer_device_map = get_layer_device_map(execution_device_map)
        self._cache = LokiOffloadingCache(
            config=self.config.get_text_config(),
            custom_config=generation_config.custom_config,
            device=device,
            layer_device_map=layer_device_map,
        )
        self._cache.build_cache()

    self._cache.reset(batch_size)
    model_kwargs["past_key_values"] = self._cache
    return True
