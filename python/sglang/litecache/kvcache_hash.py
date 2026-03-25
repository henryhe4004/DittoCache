from typing import Dict, Optional, Union, Any
import os
import csv
import math
import torch
import numpy as np
import pandas as pd
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig
from .kvcache_offloading import OffloadingCache
from .config_utils import ensure_litecache_custom_config
from sglang.jit_kernel.triton_kernels.hash.prefill_encode import (
    hash_encode_append_prefill,
)
from sglang.jit_kernel.triton_kernels.hash.decode_encode import (
    hash_encode_append_decode_k,
    hash_encode_append_decode_qk,
    hash_encode_append_decode_qqk,
)
import sgl_kernel.kvlib as KVLib


class HashOffloadingCache(OffloadingCache):

    def __init__(
        self,
        config: PretrainedConfig,
        custom_config: Any,
        device: torch.device = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device,
                                                   int]]] = None,
    ) -> None:
        method_cfg = getattr(getattr(config, "sparse_attention_config", None), "method_config", None)
        default_rbits = getattr(method_cfg, "rbit", 32) if method_cfg is not None else 32
        self.rbits = int(getattr(custom_config, "rbits", default_rbits))
        default_aux_data_path = getattr(method_cfg, "aux_data_path", None) if method_cfg is not None else None
        self.aux_data_path = getattr(custom_config, "aux_data_path", default_aux_data_path)

        super().__init__(
            config,
            custom_config,
            device,
            layer_device_map,
        )

    def _plan_topk_used_gpu_memory(self):
        numel_one_layer = (self.config.kvcache_manager_config.max_tokens *
                          self.num_key_value_heads * self.rbits // 32)
        gpu_need_mem = self.num_layers * numel_one_layer * torch.int32.itemsize
        assert gpu_need_mem <= self.mem_budget, \
            f"max_tokens = {self.config.kvcache_manager_config.max_tokens}, " \
            f"hash code cache requires {gpu_need_mem / 1024**3:.2f} GB GPU memory.\n" \
            f"However, only {self.mem_budget / 1024**3:.2f} GB left!"
        self.mem_budget -= gpu_need_mem
        print(
            f"Hash code cache consumed GPU memory: " \
            f"{gpu_need_mem / 1024**3:.2f} GB. " \
            f"{self.mem_budget / 1024**3:.2f} GB budget left.")

    def _create_metadata_tensors(self):
        super()._create_metadata_tensors()

        self.hash_dim = self.rbits // 32

        for device_idx in self.unique_devices:
            packbit_aux_tensor = torch.pow(
                2, torch.arange(0, 32, 1, dtype=torch.int32, device=device_idx))
            self.metadata_tensors[
                f'packbit_aux_tensor_{device_idx}'] = packbit_aux_tensor

        self.metadata_tensors['hash_weights'] = [None for _ in range(self.num_layers)]
        aux_data_path = self.aux_data_path

        for l in range(self.num_layers):
            layer_device = self.layer_devices[l]
            if aux_data_path is None:
                self.metadata_tensors['hash_weights'][l] = torch.randn(
                    (self.num_key_value_heads, self.head_dim, self.rbits),
                    dtype=self.dtype,
                    device=layer_device)
            else:
                self.metadata_tensors['hash_weights'][l] = torch.load(os.path.join(aux_data_path,
                                 f"hash_weight_layer_{l:02d}.pt"), weights_only=True).to(
                    layer_device)

        for device_idx in self.unique_devices:
            self.metadata_tensors[f'curr_query_code_{device_idx}'] = torch.zeros(
                (self.config.kvcache_manager_config.max_batch_size, 1, self.num_heads, self.hash_dim),
                device=device_idx,
                dtype=torch.int32,
            )
            self.metadata_tensors[f'prefetch_query_code_{device_idx}'] = torch.zeros(
                (self.config.kvcache_manager_config.max_batch_size, 1, self.num_heads, self.hash_dim),
                device=device_idx,
                dtype=torch.int32,
            )

    def _create_topk_tensors(self):
        numel_one_layer = (self.config.kvcache_manager_config.max_tokens * self.num_key_value_heads *
                           self.hash_dim)
        self.cache_tensors['topk_code_data'] = [None for l in range(self.num_layers)]
        self.cache_tensors['topk_code_length'] = [None for l in range(self.num_layers)]
        for l in range(self.num_layers):
            layer_device = self.layer_devices[l]
            code_data = torch.zeros((numel_one_layer, ),
                                     dtype=torch.int32,
                                     device=layer_device)
            code_length = torch.zeros((1, ),
                                      dtype=torch.int32,
                                      device=layer_device)
            self.cache_tensors['topk_code_data'][l] = code_data
            self.cache_tensors['topk_code_length'][l] = code_length
        self.topk_codes = [None for _ in range(self.num_layers)]

    def _reset_topk_tensors(self, batch_size):
        for l in range(self.num_layers):
            code_data = self.cache_tensors['topk_code_data'][l]
            self.topk_codes[l] = code_data[:batch_size * self.max_seq_len *
                                           self.num_key_value_heads *
                                           self.hash_dim].view(
                                               batch_size, self.max_seq_len,
                                               self.num_key_value_heads,
                                               self.hash_dim)
            code_length = self.cache_tensors['topk_code_length'][l]
            code_length.zero_()

    def append_topk_cache_prefill(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        values_states: torch.Tensor,
        layer_idx: int,
    ):
        key_states = key_states.view(self.curr_batch_size, -1, self.num_key_value_heads, self.head_dim)
        torch.cuda.nvtx.range_push("append hash")
        hash_encode_append_prefill(
            key_states,
            self.topk_codes[layer_idx],
            self.metadata_tensors['hash_weights'][layer_idx],
            self.cache_tensors['topk_code_length'][layer_idx],
            self.metadata_tensors[f'packbit_aux_tensor_{key_states.device.index}'],
        )
        seqlen_tensor = self.cache_tensors['topk_code_length'][layer_idx]
        seqlen_tensor[0] = seqlen_tensor[0].item() + key_states.shape[1]
        torch.cuda.nvtx.range_pop()

    def _decode_append_hash_qk(self, query_states: torch.Tensor,
                               key_states: torch.Tensor, query_layer_idx: int,
                               key_layer_idx: int, is_prefetch=False):
        if is_prefetch:
            query_out = self.metadata_tensors[f'prefetch_query_code_{query_states.device.index}']
        else:
            query_out = self.metadata_tensors[f'curr_query_code_{query_states.device.index}']
        query_out = query_out[:self.curr_batch_size]
        hash_encode_append_decode_qk(
            key_states,
            self.topk_codes[key_layer_idx],
            self.metadata_tensors['hash_weights'][key_layer_idx],
            query_states,
            query_out,
            self.metadata_tensors['hash_weights'][query_layer_idx],
            self.metadata_tensors[f'packbit_aux_tensor_{key_states.device.index}'],
            self.cache_tensors['topk_code_length'][key_layer_idx],
        )
        return query_out

    def _decode_append_hash_qqk(self, query_states: torch.Tensor,
                                query_states2: torch.Tensor,
                                key_states: torch.Tensor, query_layer_idx: int,
                                query_layer_idx2: int, key_layer_idx: int):
        query_out1 = self.metadata_tensors[f'prefetch_query_code_{query_states.device.index}'][:self.curr_batch_size]
        query_out2 = self.metadata_tensors[f'curr_query_code_{query_states.device.index}'][:self.curr_batch_size]
        
        hash_encode_append_decode_qqk(
            key_states,
            self.topk_codes[key_layer_idx],
            self.metadata_tensors['hash_weights'][key_layer_idx],
            query_states,
            query_out1,
            self.metadata_tensors['hash_weights'][query_layer_idx],
            query_states2,
            query_out2,
            self.metadata_tensors['hash_weights'][query_layer_idx2],
            self.metadata_tensors[f'packbit_aux_tensor_{key_states.device.index}'],
            self.cache_tensors['topk_code_length'][key_layer_idx],
        )

        return query_out1, query_out2

    def _decode_append_hash_k(self, key_states: torch.Tensor, layer_idx: int):
        hash_encode_append_decode_k(
            key_states,
            self.topk_codes[layer_idx],
            self.metadata_tensors['hash_weights'][layer_idx],
            self.metadata_tensors[f'packbit_aux_tensor_{key_states.device.index}'],
            self.cache_tensors['topk_code_length'][layer_idx],
        )

    def append_topk_cache_decode(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        prefetch_query_states: Optional[torch.Tensor] = None,
        current_query_states: Optional[torch.Tensor] = None,
    ):
        torch.cuda.nvtx.range_push("append hash")

        next_layer_idx = (layer_idx + 1) % self.num_layers
        encode_current_query = self.layers_gpu_head_ids[layer_idx].numel() > 0
        encode_prefetch_query = not self.layers_full_gpu_mask[next_layer_idx]

        if encode_current_query and encode_prefetch_query:
            prefetch_query_code, current_query_code = self._decode_append_hash_qqk(
                prefetch_query_states, current_query_states, key_states,
                next_layer_idx, layer_idx, layer_idx)

        elif encode_current_query:
            current_query_code = self._decode_append_hash_qk(
                current_query_states, key_states, layer_idx, layer_idx, is_prefetch=False)
            prefetch_query_code = None

        elif encode_prefetch_query:
            prefetch_query_code = self._decode_append_hash_qk(
                prefetch_query_states, key_states, next_layer_idx, layer_idx, is_prefetch=True)
            current_query_code = None

        else:
            self._decode_append_hash_k(key_states, layer_idx)
            prefetch_query_code = None
            current_query_code = None

        self.cache_tensors['topk_code_length'][layer_idx] += 1

        torch.cuda.nvtx.range_pop()

        return prefetch_query_code, current_query_code

    def compute_topk(self, query, layer_idx, mask, is_prefetch=False):
        device_idx = query.device.index
        if is_prefetch:
            include_sink = 0
            include_recent = 0
            exclude_sink = self.config.sparse_attention_config.sink_budget
            exclude_recent = self.config.sparse_attention_config.recent_budget
            k = self.metadata_tensors[f'topk_prefetch_k_{device_idx}']
        else:
            include_sink = self.config.sparse_attention_config.sink_budget
            include_recent = self.config.sparse_attention_config.recent_budget
            exclude_sink = 0
            exclude_recent = 0
            k = self.metadata_tensors[f'topk_current_k_{device_idx}']
        # checked
        KVLib.static_hamming_score_mask(
            self.topk_codes[layer_idx],
            query,
            mask,
            self.metadata_tensors[f'gpu_topk_scores_{device_idx}'],
            self.cache_tensors['topk_code_length'][layer_idx],
            self.rbits,
            torch.finfo(torch.float16).max,
            0.0,
            include_sink,
            include_recent,
            exclude_sink,
            exclude_recent,
        )
        # checked
        KVLib.batch_topk_masked(
            self.metadata_tensors[f'gpu_topk_scores_{device_idx}'],
            mask,
            self.metadata_tensors[f'gpu_topk_indices_{device_idx}'],
            self.metadata_tensors[f'gpu_topk_values_{device_idx}'],
            self.cache_tensors['topk_code_length'][layer_idx],
            k,
            False,
        )
        # index = self.metadata_tensors[f'gpu_topk_indices_{device_idx}'][..., :k[0].item()]
        # print(index.min(), index.max())
        # non_valid_index = self.metadata_tensors[f'gpu_topk_indices_{device_idx}'][..., k[0].item():]
        # print(non_valid_index.min(), non_valid_index.max())
        # print(self.cache_tensors['topk_code_length'][layer_idx])
        return self.metadata_tensors[f'gpu_topk_indices_{device_idx}']

"""
===================================================
Hugging Face api reload
===================================================
"""



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
        self._cache = HashOffloadingCache(
            config=self.config.get_text_config(),
            custom_config=generation_config.custom_config,
            device=device,
            layer_device_map=layer_device_map,
        )
        self._cache.build_cache()

    self._cache.reset(batch_size)
    model_kwargs["past_key_values"] = self._cache
    return True
