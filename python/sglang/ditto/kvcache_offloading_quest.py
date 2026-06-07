from __future__ import annotations

from typing import Dict, Optional, Union

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig

import sgl_kernel.kvlib as KVLib
from sglang.jit_kernel.legacy_triton_cache_kernels.triton_quest_kernels import quest_score

from .kvcache_offloading_duohead_base import OffloadingCache
from .config_utils import ensure_ditto_custom_config


class QuestOffloadingCache(OffloadingCache):
    def __init__(
        self,
        config: PretrainedConfig,
        custom_config,
        device: torch.device = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        self.block_size = int(getattr(custom_config, "block_size", 64))
        super().__init__(config, custom_config, device, layer_device_map)

    def _plan_topk_used_gpu_memory(self):
        max_batch = self.config.kvcache_manager_config.max_batch_size
        max_seq = self.config.kvcache_manager_config.max_tokens
        max_block_num = (max_seq + self.block_size - 1) // self.block_size
        gpu_block_key_numel = max_batch * max_block_num * self.num_key_value_heads * self.head_dim
        gpu_recent_buffer_numel = max_batch * self.block_size * self.num_key_value_heads * self.head_dim
        gpu_need_mem = self.num_layers * (2 * gpu_block_key_numel + gpu_recent_buffer_numel) * self.dtype.itemsize
        assert gpu_need_mem <= self.mem_budget, (
            f"Quest block cache needs {gpu_need_mem / 1024**3:.2f} GB, "
            f"but only {self.mem_budget / 1024**3:.2f} GB is available."
        )
        self.mem_budget -= gpu_need_mem

    def _create_topk_tensors(self):
        max_batch = self.config.kvcache_manager_config.max_batch_size
        max_seq = self.config.kvcache_manager_config.max_tokens
        self.max_block_num = (max_seq + self.block_size - 1) // self.block_size

        gpu_block_key_numel = max_batch * self.max_block_num * self.num_key_value_heads * self.head_dim
        gpu_recent_buffer_numel = max_batch * self.block_size * self.num_key_value_heads * self.head_dim

        self.layers_block_max_cache = [None for _ in range(self.num_layers)]
        self.layers_block_max_cache_data = [None for _ in range(self.num_layers)]
        self.layers_block_min_cache = [None for _ in range(self.num_layers)]
        self.layers_block_min_cache_data = [None for _ in range(self.num_layers)]
        self.layers_recent_key_buffer = [None for _ in range(self.num_layers)]
        self.layers_recent_key_buffer_data = [None for _ in range(self.num_layers)]
        self.layers_gpu_block_num = [0 for _ in range(self.num_layers)]
        self.layers_gpu_recent_buffer_length = [0 for _ in range(self.num_layers)]
        self.layers_gpu_topk_total_length = [0 for _ in range(self.num_layers)]

        for layer in range(self.num_layers):
            self.layers_block_max_cache_data[layer] = torch.zeros(
                (gpu_block_key_numel,),
                dtype=self.dtype,
                device=self.layer_devices[layer],
            )
            self.layers_block_min_cache_data[layer] = torch.zeros(
                (gpu_block_key_numel,),
                dtype=self.dtype,
                device=self.layer_devices[layer],
            )
            self.layers_recent_key_buffer_data[layer] = torch.zeros(
                (gpu_recent_buffer_numel,),
                dtype=self.dtype,
                device=self.layer_devices[layer],
            )

    def _reset_topk_tensors(self, batch_size):
        gpu_block_key_numel = batch_size * self.max_block_num * self.num_key_value_heads * self.head_dim
        gpu_recent_buffer_numel = batch_size * self.block_size * self.num_key_value_heads * self.head_dim
        for layer in range(self.num_layers):
            self.layers_block_max_cache[layer] = self.layers_block_max_cache_data[layer][
                :gpu_block_key_numel
            ].view(batch_size, self.max_block_num, self.num_key_value_heads, self.head_dim)
            self.layers_block_min_cache[layer] = self.layers_block_min_cache_data[layer][
                :gpu_block_key_numel
            ].view(batch_size, self.max_block_num, self.num_key_value_heads, self.head_dim)
            self.layers_recent_key_buffer[layer] = self.layers_recent_key_buffer_data[layer][
                :gpu_recent_buffer_numel
            ].view(batch_size, self.block_size, self.num_key_value_heads, self.head_dim)
            self.layers_gpu_block_num[layer] = 0
            self.layers_gpu_recent_buffer_length[layer] = 0
            self.layers_gpu_topk_total_length[layer] = 0

    def append_topk_cache_prefill(self, query_states, key_states, values_states, layer_idx):
        del query_states, values_states
        sequence_length = key_states.shape[1] - self.config.sparse_attention_config.sink_budget
        key_states = key_states[:, self.config.sparse_attention_config.sink_budget :, ...]
        block_num = sequence_length // self.block_size
        left_num = sequence_length - block_num * self.block_size

        if left_num > 0:
            self.layers_recent_key_buffer[layer_idx][:, :left_num, ...] = key_states[:, -left_num:, ...]
            key_states = key_states[:, :-left_num, ...]

        key_states = key_states.view(
            self.curr_batch_size,
            block_num,
            self.block_size,
            self.num_key_value_heads,
            self.head_dim,
        )
        self.layers_block_max_cache[layer_idx][:, :block_num, ...] = key_states.max(dim=2).values
        self.layers_block_min_cache[layer_idx][:, :block_num, ...] = key_states.min(dim=2).values
        self.layers_gpu_topk_total_length[layer_idx] += sequence_length
        self.layers_gpu_block_num[layer_idx] += block_num
        self.layers_gpu_recent_buffer_length[layer_idx] += left_num

    def _decode_append_quest_k(self, key_states: torch.Tensor, layer_idx: int):
        recent_buffer_size = self.layers_gpu_recent_buffer_length[layer_idx]
        self.layers_recent_key_buffer[layer_idx][:, recent_buffer_size : recent_buffer_size + 1, ...] = key_states
        recent_buffer_size += 1
        if recent_buffer_size >= self.block_size:
            block_num = self.layers_gpu_block_num[layer_idx]
            block_data = self.layers_recent_key_buffer[layer_idx][:, : self.block_size, ...].unsqueeze(1)
            self.layers_block_max_cache[layer_idx][:, block_num : block_num + 1, :, :] = block_data.max(dim=2).values
            self.layers_block_min_cache[layer_idx][:, block_num : block_num + 1, :, :] = block_data.min(dim=2).values
            self.layers_gpu_block_num[layer_idx] = block_num + 1
            recent_buffer_size = 0
        self.layers_gpu_recent_buffer_length[layer_idx] = recent_buffer_size
        self.layers_gpu_topk_total_length[layer_idx] += 1

    def append_topk_cache_decode(self, key_states, value_states, layer_idx, prefetch_query_states=None, current_query_states=None):
        del value_states
        self._decode_append_quest_k(key_states, layer_idx)
        return prefetch_query_states, current_query_states

    def compute_topk(self, query, layer_idx, mask, is_prefetch=False):
        cache_length = self.layers_gpu_topk_total_length[layer_idx]
        if is_prefetch:
            valid_block_num = (cache_length - self.config.sparse_attention_config.recent_budget) // self.block_size
        else:
            valid_block_num = (cache_length - self.config.sparse_attention_config.recent_budget - 1) // self.block_size

        if self.topk_ratio < 1:
            fetch_block_num = (int(cache_length * self.topk_ratio) + self.block_size - 1) // self.block_size
        else:
            fetch_block_num = min(int(self.topk_ratio) // self.block_size, valid_block_num)

        # Triton quest kernels require both dot operands to share dtype.
        target_dtype = self.layers_block_max_cache[layer_idx].dtype
        if query.dtype != target_dtype:
            query = query.to(target_dtype)

        score = quest_score(
            query,
            self.layers_block_max_cache[layer_idx],
            self.layers_block_min_cache[layer_idx],
            valid_block_num,
            head_mask=mask,
        )

        if is_prefetch:
            topk_indices = self._batch_topk_masked_compat(score, mask, fetch_block_num, True)
            topk_indices = KVLib.block_id_to_token_id_head_mask(
                topk_indices,
                self.block_size,
                0,
                0,
                self.layers_gpu_topk_total_length[layer_idx],
                mask,
            )
            fetch_num = topk_indices.shape[-1]
            return topk_indices.view(-1, fetch_num)

        if mask is not None:
            topk_indices = self._batch_topk_masked_compat(score, mask, fetch_block_num, True)
            recent_budget = int(
                self.metadata_tensors[f"topk_current_k_{query.device.index}"][0].item()
                - self.config.sparse_attention_config.sink_budget
                - fetch_block_num * self.block_size
            )
        else:
            topk_indices = KVLib.batch_topk(score, fetch_block_num, True)
            recent_budget = self.config.sparse_attention_config.recent_budget
        topk_indices = KVLib.block_id_to_token_id(
            topk_indices,
            self.block_size,
            self.config.sparse_attention_config.sink_budget,
            recent_budget,
            self.layers_gpu_topk_total_length[layer_idx] + self.config.sparse_attention_config.sink_budget,
        )
        fetch_num = topk_indices.shape[-1]
        return topk_indices.view(-1, fetch_num)


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
        self._cache = QuestOffloadingCache(
            config=self.config.get_text_config(),
            custom_config=generation_config.custom_config,
            device=device,
            layer_device_map=layer_device_map,
        )
        self._cache.build_cache()

    self._cache.reset(batch_size)
    model_kwargs["past_key_values"] = self._cache
    return True

