from __future__ import annotations

import os
from typing import Dict, Optional, Union

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig

import sgl_kernel.kvlib as KVLib
from sglang.jit_kernel.legacy_triton_cache_kernels.triton_loki_kernels import loki_score

from .kvcache_offloading_duohead_base import OffloadingCache
from .config_utils import ensure_litecache_custom_config


class InfiniGenOffloadingCache(OffloadingCache):
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
        max_batch = self.config.kvcache_manager_config.max_batch_size
        max_seq = self.config.kvcache_manager_config.max_tokens
        gpu_partial_key_numel = max_batch * max_seq * self.num_key_value_heads * self.num_channels
        gpu_need_mem = self.num_layers * gpu_partial_key_numel * self.dtype.itemsize
        assert gpu_need_mem <= self.mem_budget, (
            f"InfiniGen partial-key cache needs {gpu_need_mem / 1024**3:.2f} GB, "
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
        self.layers_skewing_matrix = [None for _ in range(self.num_layers)]
        self.partial_idx = [None for _ in range(self.num_layers)]

        for layer in range(self.num_layers):
            self.layers_partial_key_cache_data[layer] = torch.zeros(
                (gpu_partial_key_numel,),
                dtype=self.dtype,
                device=self.layer_devices[layer],
            )
            if self.aux_data_path is None:
                skewing_matrix = torch.eye(
                    self.head_dim,
                    dtype=self.dtype,
                    device=self.layer_devices[layer],
                ).repeat(self.num_key_value_heads, 1, 1)
            else:
                skewing_matrix = torch.load(
                    os.path.join(self.aux_data_path, f"skewing_martix_{layer:02d}.pt"),
                    weights_only=True,
                )
                skewing_matrix = skewing_matrix.view(
                    self.num_key_value_heads, self.head_dim, self.head_dim
                ).to(self.dtype).to(self.layer_devices[layer])
            self.layers_skewing_matrix[layer] = skewing_matrix

    def _reset_topk_tensors(self, batch_size):
        gpu_partial_key_numel = batch_size * self.max_seq_len * self.num_key_value_heads * self.num_channels
        self.partial_idx = [None for _ in range(self.num_layers)]
        for layer in range(self.num_layers):
            self.layers_partial_key_cache[layer] = self.layers_partial_key_cache_data[layer][
                :gpu_partial_key_numel
            ].view(batch_size, self.max_seq_len, self.num_key_value_heads, self.num_channels)
            self.layers_gpu_partial_key_cache_length[layer] = 0

    def _skewing(self, states: torch.Tensor, layer_idx: int):
        b, s, h, d = states.shape
        states = states.view(b, s, self.num_key_value_heads, -1, d)
        skewing_matrix = self.layers_skewing_matrix[layer_idx]
        states = torch.einsum("bshgd,hdd->bshgd", states, skewing_matrix)
        return states.view(b, s, h, d)

    def _extract(self, states: torch.Tensor, layer_idx: int):
        b, s, h, d = states.shape
        states = states.view(b, s, self.num_key_value_heads, -1, d)
        idx = self.partial_idx[layer_idx]
        g = states.shape[-2]
        idx = idx.expand(b, s, self.num_key_value_heads, g, self.num_channels)
        states = torch.gather(states, dim=-1, index=idx)
        return states.reshape(b, s, h, self.num_channels)

    def append_topk_cache_prefill(self, query_states, key_states, values_states, layer_idx):
        del values_states
        sequence_length = key_states.shape[1]
        self.layers_gpu_partial_key_cache_length[layer_idx] += sequence_length
        query_states = self._skewing(query_states, layer_idx)
        key_states = self._skewing(key_states, layer_idx)
        query_states = query_states.abs().sum(dim=1, keepdim=True)
        query_states = query_states.view(self.curr_batch_size, 1, self.num_key_value_heads, -1, self.head_dim)
        query_states = query_states.sum(dim=3, keepdim=True)
        partial_idx = torch.topk(query_states, k=self.num_channels, dim=-1, largest=True).indices
        self.partial_idx[layer_idx] = partial_idx
        partial_key_states = self._extract(key_states, layer_idx)
        self.layers_partial_key_cache[layer_idx][:, :sequence_length] = partial_key_states

    def append_topk_cache_decode(self, key_states, value_states, layer_idx, prefetch_query_states=None, current_query_states=None):
        del value_states
        next_layer_idx = (layer_idx + 1) % self.num_layers
        encode_current_query = self.layers_gpu_head_ids[layer_idx].numel() > 0
        encode_prefetch_query = not self.layers_full_gpu_mask[next_layer_idx]

        prefetch_query_code = None
        current_query_code = None

        if encode_prefetch_query:
            prefetch_query_code = self._extract(self._skewing(prefetch_query_states, next_layer_idx), next_layer_idx)
        if encode_current_query:
            current_query_code = self._extract(self._skewing(current_query_states, layer_idx), layer_idx)

        key_code = self._extract(self._skewing(key_states, layer_idx), layer_idx)
        length = self.layers_gpu_partial_key_cache_length[layer_idx]
        self.layers_partial_key_cache[layer_idx][:, length : length + 1] = key_code
        self.layers_gpu_partial_key_cache_length[layer_idx] = length + 1
        return prefetch_query_code, current_query_code

    def compute_topk(self, query, layer_idx, mask, is_prefetch=False):
        if is_prefetch:
            cache_length = self.layers_gpu_partial_key_cache_length[layer_idx]
            if self.topk_ratio < 1:
                fetch_num = int((cache_length + 1) * self.topk_ratio)
                fetch_num = min(fetch_num, cache_length - self.config.sparse_attention_config.sink_budget - self.config.sparse_attention_config.recent_budget)
            else:
                fetch_num = min(int(self.topk_ratio), cache_length - self.config.sparse_attention_config.sink_budget - self.config.sparse_attention_config.recent_budget)
            fetch_num = max(fetch_num, 0)
            score = loki_score(
                query,
                self.layers_partial_key_cache[layer_idx],
                cache_length - self.config.sparse_attention_config.recent_budget,
                head_mask=mask,
            )
            score[..., : self.config.sparse_attention_config.sink_budget] = torch.finfo(score.dtype).min
            topk_indices = KVLib.batch_topk_masked(score, mask, fetch_num, True).view(-1, fetch_num)
            return topk_indices - self.config.sparse_attention_config.sink_budget

        cache_length = self.layers_gpu_partial_key_cache_length[layer_idx]
        if self.layers_full_gpu_mask[layer_idx]:
            if self.topk_ratio < 1:
                fetch_num = int(cache_length * self.topk_ratio) + self.config.sparse_attention_config.recent_budget + self.config.sparse_attention_config.sink_budget
                fetch_num = min(fetch_num, cache_length)
            else:
                fetch_num = min(int(self.topk_ratio) + self.config.sparse_attention_config.recent_budget + self.config.sparse_attention_config.sink_budget, cache_length)
        else:
            fetch_num = int(self.metadata_tensors[f"topk_current_k_{query.device.index}"][0].item())

        if mask is not None:
            score = loki_score(query, self.layers_partial_key_cache[layer_idx], cache_length, head_mask=mask)
            score[..., : self.config.sparse_attention_config.sink_budget] = torch.finfo(score.dtype).max
            if self.config.sparse_attention_config.recent_budget > 0:
                score[..., -self.config.sparse_attention_config.recent_budget :] = torch.finfo(score.dtype).max
            return KVLib.batch_topk_masked(score, mask, fetch_num, True).view(-1, fetch_num)

        score = loki_score(query, self.layers_partial_key_cache[layer_idx], cache_length)
        score[..., : self.config.sparse_attention_config.sink_budget] = torch.finfo(score.dtype).max
        if self.config.sparse_attention_config.recent_budget > 0:
            score[..., -self.config.sparse_attention_config.recent_budget :] = torch.finfo(score.dtype).max
        return KVLib.batch_topk(score, fetch_num, True).view(-1, fetch_num)


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
    generation_config.custom_config = ensure_litecache_custom_config(
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
        self._cache = InfiniGenOffloadingCache(
            config=self.config.get_text_config(),
            custom_config=generation_config.custom_config,
            device=device,
            layer_device_map=layer_device_map,
        )
        self._cache.build_cache()

    self._cache.reset(batch_size)
    model_kwargs["past_key_values"] = self._cache
    return True

