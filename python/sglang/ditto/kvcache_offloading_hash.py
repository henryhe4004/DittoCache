from __future__ import annotations

import os
from typing import Dict, Optional, Union

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig

import sgl_kernel.kvlib as KVLib
from sglang.jit_kernel.legacy_triton_cache_kernels.triton_hash_encode_new import (
    decode_multi_hash_encode_k,
    decode_multi_hash_encode_qk,
    decode_multi_hash_encode_qqk,
    prefill_multi_hash_encode,
)

from .kvcache_offloading_duohead_base import OffloadingCache
from .config_utils import ensure_ditto_custom_config


class HashOffloadingCache(OffloadingCache):
    def __init__(
        self,
        config: PretrainedConfig,
        custom_config,
        device: torch.device = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        self.rbits = int(getattr(custom_config, "rbits", 32))
        self.aux_data_path = getattr(custom_config, "aux_data_path", None)
        super().__init__(config, custom_config, device, layer_device_map)

    def _plan_topk_used_gpu_memory(self):
        max_batch = self.config.kvcache_manager_config.max_batch_size
        max_seq = self.config.kvcache_manager_config.max_tokens
        gpu_hash_numel = max_batch * max_seq * self.num_key_value_heads * self.rbits // 32
        gpu_need_mem = self.num_layers * gpu_hash_numel * torch.int32.itemsize
        assert gpu_need_mem <= self.mem_budget, (
            f"Hash code cache needs {gpu_need_mem / 1024**3:.2f} GB, "
            f"but only {self.mem_budget / 1024**3:.2f} GB is available."
        )
        self.mem_budget -= gpu_need_mem

    def _create_topk_tensors(self):
        max_batch = self.config.kvcache_manager_config.max_batch_size
        max_seq = self.config.kvcache_manager_config.max_tokens
        gpu_hash_numel = max_batch * max_seq * self.num_key_value_heads * self.rbits // 32

        self.layers_hash_cache = [None for _ in range(self.num_layers)]
        self.layers_hash_cache_data = [None for _ in range(self.num_layers)]
        self.layers_hash_cache_length_tensor = [None for _ in range(self.num_layers)]
        self.layers_gpu_hash_cache_length = [
            [0] * max_batch for _ in range(self.num_layers)
        ]
        self.layers_hash_weight = [None for _ in range(self.num_layers)]

        for layer in range(self.num_layers):
            self.layers_hash_cache_data[layer] = torch.zeros(
                (gpu_hash_numel,),
                dtype=torch.int32,
                device=self.layer_devices[layer],
            )
            if self.aux_data_path is None:
                self.layers_hash_weight[layer] = torch.randn(
                    (self.num_key_value_heads, self.head_dim, self.rbits),
                    dtype=self.dtype,
                    device=self.layer_devices[layer],
                )
            else:
                hash_weight = torch.load(
                    os.path.join(self.aux_data_path, f"hash_weight_layer_{layer:02d}.pt"),
                    weights_only=True,
                )
                hash_weight = self._slice_local_kv_head_tensor(
                    hash_weight,
                    layer_idx=layer,
                    tensor_name=f"hash_weight_layer_{layer:02d}",
                    head_dim=0,
                )
                self.layers_hash_weight[layer] = hash_weight.to(self.layer_devices[layer]).to(self.dtype)
            self.layers_hash_cache_length_tensor[layer] = torch.zeros(
                (max_batch,),
                dtype=torch.int32,
                device=self.layer_devices[layer],
            )

        self.hash_packbit_aux_tensors = {}
        for device in self.unique_devices:
            self.hash_packbit_aux_tensors[device] = torch.pow(
                2, torch.arange(0, 32, 1, dtype=torch.int32, device=device)
            )

    def _reset_topk_tensors(self, batch_size):
        gpu_hash_numel = batch_size * self.max_seq_len * self.num_key_value_heads * self.rbits // 32
        for layer in range(self.num_layers):
            self.layers_hash_cache[layer] = self.layers_hash_cache_data[layer][
                :gpu_hash_numel
            ].view(batch_size, self.max_seq_len, self.num_key_value_heads, self.rbits // 32)
            self.layers_gpu_hash_cache_length[layer] = [
                0
            ] * self.config.kvcache_manager_config.max_batch_size
            self.layers_hash_cache_length_tensor[layer].zero_()

    def _active_hash_lengths(self, layer_idx: int, device: torch.device) -> torch.Tensor:
        return self.layers_hash_cache_length_tensor[layer_idx][
            :self.curr_batch_size
        ].to(device=device, non_blocking=True)

    def _set_active_hash_lengths(self, layer_idx: int, value: int) -> None:
        self.layers_hash_cache_length_tensor[layer_idx][
            :self.curr_batch_size
        ].fill_(value)
        for idx in range(self.curr_batch_size):
            self.layers_gpu_hash_cache_length[layer_idx][idx] = int(value)

    def _increment_active_hash_lengths(self, layer_idx: int) -> None:
        self.layers_hash_cache_length_tensor[layer_idx][
            :self.curr_batch_size
        ].add_(1)
        for idx in range(self.curr_batch_size):
            self.layers_gpu_hash_cache_length[layer_idx][idx] += 1

    def reset_batch_rows(self, row_indices: list[int]) -> None:
        super().reset_batch_rows(row_indices)
        if not row_indices:
            return
        rows = torch.tensor(row_indices, dtype=torch.long)
        for layer_idx in range(self.num_layers):
            layer_rows = rows.to(
                device=self.layers_hash_cache_length_tensor[layer_idx].device,
                non_blocking=True,
            )
            self.layers_hash_cache_length_tensor[layer_idx][layer_rows] = 0
            for row in row_indices:
                self.layers_gpu_hash_cache_length[layer_idx][row] = 0

    def move_batch_rows(self, old_to_new_rows: dict[int, int]) -> None:
        super().move_batch_rows(old_to_new_rows)
        normalized = {
            int(old): int(new)
            for old, new in old_to_new_rows.items()
            if int(old) != int(new)
        }
        if not normalized:
            return

        def _index_copy_rows(tensor: torch.Tensor | None, dim: int) -> None:
            if tensor is None:
                return
            old_rows = torch.tensor(
                list(normalized.keys()), dtype=torch.long, device=tensor.device
            )
            new_rows = torch.tensor(
                list(normalized.values()), dtype=torch.long, device=tensor.device
            )
            src = tensor.index_select(dim, old_rows).clone()
            tensor.index_copy_(dim, new_rows, src)

        def _copy_hash_rows(
            tensor: torch.Tensor | None,
            row_lengths: dict[int, int],
        ) -> None:
            if tensor is None:
                return
            snapshots = []
            seq_cap = int(tensor.size(1))
            for old, new in normalized.items():
                valid_len = max(0, min(int(row_lengths.get(old, 0)), seq_cap))
                if valid_len == 0:
                    continue
                snapshots.append(
                    (new, valid_len, tensor[old : old + 1, :valid_len].clone())
                )
            for new, valid_len, src in snapshots:
                tensor[new : new + 1, :valid_len].copy_(src)

        for layer_idx in range(self.num_layers):
            row_lengths = {
                old: self.layers_gpu_hash_cache_length[layer_idx][old]
                for old in normalized.keys()
            }
            _index_copy_rows(self.layers_hash_cache_length_tensor[layer_idx], 0)
            _copy_hash_rows(self.layers_hash_cache[layer_idx], row_lengths)
            hash_lengths = [
                self.layers_gpu_hash_cache_length[layer_idx][old]
                for old in normalized.keys()
            ]
            for new, value in zip(normalized.values(), hash_lengths):
                self.layers_gpu_hash_cache_length[layer_idx][new] = int(value)

    def trim_prefill_padding(self, extend_seq_lens: list[int], padded_q_len: int) -> None:
        super().trim_prefill_padding(extend_seq_lens, padded_q_len)
        if getattr(self, "_prefill_row_aware", False):
            return
        if not extend_seq_lens or all(int(x) == int(padded_q_len) for x in extend_seq_lens):
            return
        trims = torch.tensor(
            [int(padded_q_len) - int(x) for x in extend_seq_lens],
            dtype=torch.int32,
        )
        for layer_idx in range(self.num_layers):
            active = self.layers_hash_cache_length_tensor[layer_idx][
                :self.curr_batch_size
            ]
            active.sub_(trims.to(device=active.device, non_blocking=True))
            for row, trim in enumerate(trims.tolist()):
                self.layers_gpu_hash_cache_length[layer_idx][row] -= int(trim)

    def append_topk_cache_prefill(self, query_states, key_states, values_states, layer_idx):
        del query_states, values_states
        sequence_length = key_states.shape[1]
        extend_seq_lens = getattr(self, "_current_extend_seq_lens", None)
        if extend_seq_lens is None:
            row_lens = [int(sequence_length)] * self.curr_batch_size
        else:
            if len(extend_seq_lens) != self.curr_batch_size:
                raise RuntimeError(
                    "Ditto hash extend length mismatch: "
                    f"extend_seq_lens={extend_seq_lens}, "
                    f"curr_batch_size={self.curr_batch_size}."
                )
            row_lens = [int(x) for x in extend_seq_lens]
        active_lengths = self.layers_gpu_hash_cache_length[layer_idx][
            :self.curr_batch_size
        ]
        new_lengths = [int(x) + int(row_len) for x, row_len in zip(active_lengths, row_lens)]
        max_new_seq_len = max(new_lengths) if new_lengths else 0
        assert max_new_seq_len <= self.max_seq_len, \
            f"hash cache append exceeds max_seq_len at layer={layer_idx}: " \
            f"max({active_lengths}) + max_row_len={max(row_lens) if row_lens else 0} > {self.max_seq_len}"
        for row, seq_offset in enumerate(active_lengths):
            seq_offset = int(seq_offset)
            row_len = int(row_lens[row])
            if row_len <= 0:
                continue
            new_seq_len = seq_offset + row_len
            prefill_multi_hash_encode(
                key_states[row:row + 1, :row_len],
                self.layers_hash_weight[layer_idx],
                self.layers_hash_cache[layer_idx][
                    row:row + 1, seq_offset:new_seq_len, :, :
                ],
                self.hash_packbit_aux_tensors[key_states.device.index],
            )
        tensor = self.layers_hash_cache_length_tensor[layer_idx][
            :self.curr_batch_size
        ]
        tensor.copy_(torch.tensor(new_lengths, dtype=tensor.dtype, device=tensor.device))
        for idx, value in enumerate(new_lengths):
            self.layers_gpu_hash_cache_length[layer_idx][idx] = int(value)

    def _decode_append_hash_qk(self, query_states, key_states, query_layer_idx, key_layer_idx):
        query_out = torch.zeros(
            (self.curr_batch_size, 1, self.num_heads, self.rbits // 32),
            device=query_states.device,
            dtype=torch.int32,
        )
        decode_multi_hash_encode_qk(
            key_states,
            self.layers_hash_cache[key_layer_idx],
            self.layers_hash_weight[key_layer_idx],
            query_states,
            query_out,
            self.layers_hash_weight[query_layer_idx % self.num_layers],
            self.hash_packbit_aux_tensors[key_states.device.index],
            self._active_hash_lengths(key_layer_idx, key_states.device),
        )
        self._increment_active_hash_lengths(key_layer_idx)
        return query_out

    def _decode_append_hash_qqk(self, query_states, query_states2, key_states, query_layer_idx, query_layer_idx2, key_layer_idx):
        query_out1 = torch.zeros(
            (self.curr_batch_size, 1, self.num_heads, self.rbits // 32),
            device=query_states.device,
            dtype=torch.int32,
        )
        query_out2 = torch.zeros(
            (self.curr_batch_size, 1, self.num_heads, self.rbits // 32),
            device=query_states.device,
            dtype=torch.int32,
        )
        decode_multi_hash_encode_qqk(
            key_states,
            self.layers_hash_cache[key_layer_idx],
            self.layers_hash_weight[key_layer_idx],
            query_states,
            query_out1,
            self.layers_hash_weight[query_layer_idx % self.num_layers],
            query_states2,
            query_out2,
            self.layers_hash_weight[query_layer_idx2 % self.num_layers],
            self.hash_packbit_aux_tensors[key_states.device.index],
            self._active_hash_lengths(key_layer_idx, key_states.device),
        )
        self._increment_active_hash_lengths(key_layer_idx)
        return query_out1, query_out2

    def _decode_append_hash_k(self, key_states, layer_idx):
        decode_multi_hash_encode_k(
            key_states,
            self.layers_hash_cache[layer_idx],
            self.layers_hash_weight[layer_idx],
            self.hash_packbit_aux_tensors[key_states.device.index],
            self._active_hash_lengths(layer_idx, key_states.device),
        )
        self._increment_active_hash_lengths(layer_idx)

    def append_topk_cache_decode(self, key_states, value_states, layer_idx, prefetch_query_states=None, current_query_states=None):
        del value_states
        next_layer_idx = (layer_idx + 1) % self.num_layers
        encode_current_query = self.layers_gpu_head_ids[layer_idx].numel() > 0
        encode_prefetch_query = not self.layers_full_gpu_mask[next_layer_idx]
        if encode_current_query and encode_prefetch_query:
            return self._decode_append_hash_qqk(
                prefetch_query_states,
                current_query_states,
                key_states,
                layer_idx + 1,
                layer_idx,
                layer_idx,
            )
        if encode_current_query:
            return None, self._decode_append_hash_qk(current_query_states, key_states, layer_idx, layer_idx)
        if encode_prefetch_query:
            return self._decode_append_hash_qk(prefetch_query_states, key_states, layer_idx + 1, layer_idx), None
        self._decode_append_hash_k(key_states, layer_idx)
        return None, None

    def compute_topk(self, query, layer_idx, mask, is_prefetch=False):
        """
        Align hash top-k flow with internal prototype kvcache_hash:
        static_hamming_score_mask -> batch_topk_masked using preallocated buffers.
        """
        device_idx = query.device.index

        if is_prefetch:
            include_sink = 0
            include_recent = 0
            exclude_sink = self.config.sparse_attention_config.sink_budget
            exclude_recent = self.config.sparse_attention_config.recent_budget
            k_tensor = self.metadata_tensors[f"topk_prefetch_k_{device_idx}"]
        else:
            include_sink = self.config.sparse_attention_config.sink_budget
            include_recent = self.config.sparse_attention_config.recent_budget
            exclude_sink = 0
            exclude_recent = 0
            k_tensor = self.metadata_tensors[f"topk_current_k_{device_idx}"]

        k = int(self.topk_prefetch_k_host if is_prefetch else self.topk_current_k_host)
        if k <= 0:
            return torch.empty(
                (self.curr_batch_size, self.num_key_value_heads, 0),
                dtype=torch.int32,
                device=query.device,
            )

        seq_len_tensor = self._active_hash_lengths(layer_idx, query.device)
        score_buf = self.metadata_tensors[f"gpu_topk_scores_{device_idx}"][
            : self.curr_batch_size
        ]
        index_buf = self.metadata_tensors[f"gpu_topk_indices_{device_idx}"][
            : self.curr_batch_size
        ]
        value_buf = self.metadata_tensors[f"gpu_topk_values_{device_idx}"][
            : self.curr_batch_size
        ]

        KVLib.static_hamming_score_mask(
            self.layers_hash_cache[layer_idx][: self.curr_batch_size],
            query,
            mask,
            score_buf,
            seq_len_tensor,
            self.rbits,
            float(torch.finfo(torch.float16).max),
            0.0,
            include_sink,
            include_recent,
            exclude_sink,
            exclude_recent,
        )

        KVLib.batch_topk_masked(
            score_buf,
            mask,
            index_buf,
            value_buf,
            seq_len_tensor.repeat_interleave(self.num_key_value_heads),
            k_tensor[:self.curr_batch_size].repeat_interleave(
                self.num_key_value_heads
            ),
            False,
        )
        topk_indices = index_buf
        active_indices = topk_indices.reshape(
            self.curr_batch_size * self.num_key_value_heads,
            self.max_buffer_len,
        )
        per_head_seq_lens = seq_len_tensor.repeat_interleave(
            self.num_key_value_heads
        ).to(device=topk_indices.device, dtype=topk_indices.dtype)
        # Dynamic online batching can mix rows with different existing KV
        # lengths. Clamp per row/head; using the batch max can leave invalid
        # indices for shorter rows and later trigger a CUDA IndexKernel assert.
        # Keep this graph-safe: no tensor .item() inside decode CUDA graph capture.
        row_max_indices = (per_head_seq_lens - 1).clamp_min(0).view(-1, 1)
        active_indices.clamp_min_(0)
        active_indices.copy_(torch.minimum(active_indices, row_max_indices))
        active_indices.masked_fill_((per_head_seq_lens <= 0).view(-1, 1), 0)

        return topk_indices[:, :, :k].contiguous()


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
