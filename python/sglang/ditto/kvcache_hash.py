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
from .config_utils import ensure_ditto_custom_config
from sglang.jit_kernel.triton_kernels.hash.prefill_encode import (
    hash_encode_append_prefill,
)
from sglang.jit_kernel.triton_kernels.hash.decode_encode import (
    hash_encode_append_decode_k,
    hash_encode_append_decode_qk,
    hash_encode_append_decode_qqk,
)
import sgl_kernel.kvlib as KVLib


def _compute_overlap_metrics(
    real_topk_np: np.ndarray,
    prefetch_topk_np: np.ndarray,
) -> dict[str, Any]:
    if real_topk_np.shape != prefetch_topk_np.shape:
        raise ValueError(
            f"Overlap topk shape mismatch: {real_topk_np.shape} vs {prefetch_topk_np.shape}"
        )

    if real_topk_np.size == 0:
        return {
            "active_heads": int(real_topk_np.shape[0]),
            "prefetch_k": int(real_topk_np.shape[1]) if real_topk_np.ndim == 2 else 0,
            "mean_recall": 0.0,
            "mean_precision": 0.0,
            "mean_jaccard": 0.0,
            "union_recall": 0.0,
            "union_precision": 0.0,
            "union_jaccard": 0.0,
            "mean_intersection": 0.0,
            "union_intersection_count": 0,
            "union_real_count": 0,
            "union_prefetch_count": 0,
            "sum_intersection_count": 0,
            "sum_real_count": 0,
            "sum_prefetch_count": 0,
            "per_head_recall": [],
            "per_head_precision": [],
            "per_head_jaccard": [],
            "per_head_intersection_count": [],
            "per_head_real_count": [],
            "per_head_prefetch_count": [],
        }

    per_head_recall = []
    per_head_precision = []
    per_head_jaccard = []
    per_head_intersection = []
    per_head_real_count = []
    per_head_prefetch_count = []

    for head_idx in range(real_topk_np.shape[0]):
        real_set = np.unique(real_topk_np[head_idx])
        prefetch_set = np.unique(prefetch_topk_np[head_idx])
        intersection = np.intersect1d(real_set, prefetch_set, assume_unique=True)
        union = np.union1d(real_set, prefetch_set)
        per_head_intersection.append(int(intersection.size))
        per_head_real_count.append(int(real_set.size))
        per_head_prefetch_count.append(int(prefetch_set.size))
        per_head_recall.append(float(intersection.size / max(real_set.size, 1)))
        per_head_precision.append(float(intersection.size / max(prefetch_set.size, 1)))
        per_head_jaccard.append(float(intersection.size / max(union.size, 1)))

    union_real = np.unique(real_topk_np.reshape(-1))
    union_prefetch = np.unique(prefetch_topk_np.reshape(-1))
    union_intersection = np.intersect1d(union_real, union_prefetch, assume_unique=True)
    union_all = np.union1d(union_real, union_prefetch)
    sum_intersection = int(sum(per_head_intersection))
    sum_real = int(sum(int(np.unique(real_topk_np[head_idx]).size) for head_idx in range(real_topk_np.shape[0])))
    sum_prefetch = int(
        sum(int(np.unique(prefetch_topk_np[head_idx]).size) for head_idx in range(prefetch_topk_np.shape[0]))
    )

    return {
        "active_heads": int(real_topk_np.shape[0]),
        "prefetch_k": int(real_topk_np.shape[1]),
        "mean_recall": float(np.mean(per_head_recall)),
        "mean_precision": float(np.mean(per_head_precision)),
        "mean_jaccard": float(np.mean(per_head_jaccard)),
        "union_recall": float(union_intersection.size / max(union_real.size, 1)),
        "union_precision": float(union_intersection.size / max(union_prefetch.size, 1)),
        "union_jaccard": float(union_intersection.size / max(union_all.size, 1)),
        "mean_intersection": float(np.mean(per_head_intersection)),
        "union_intersection_count": int(union_intersection.size),
        "union_real_count": int(union_real.size),
        "union_prefetch_count": int(union_prefetch.size),
        "sum_intersection_count": int(sum_intersection),
        "sum_real_count": int(sum_real),
        "sum_prefetch_count": int(sum_prefetch),
        "per_head_recall": [float(v) for v in per_head_recall],
        "per_head_precision": [float(v) for v in per_head_precision],
        "per_head_jaccard": [float(v) for v in per_head_jaccard],
        "per_head_intersection_count": [int(v) for v in per_head_intersection],
        "per_head_real_count": [int(v) for v in per_head_real_count],
        "per_head_prefetch_count": [int(v) for v in per_head_prefetch_count],
    }


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
                hash_weight = torch.load(os.path.join(aux_data_path,
                                 f"hash_weight_layer_{l:02d}.pt"), weights_only=True)
                hash_weight = self._slice_local_kv_head_tensor(
                    hash_weight,
                    layer_idx=l,
                    tensor_name=f"hash_weight_layer_{l:02d}",
                    head_dim=0,
                )
                self.metadata_tensors['hash_weights'][l] = hash_weight.to(
                    device=layer_device,
                    dtype=self.dtype,
                )

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
        max_batch = self.config.kvcache_manager_config.max_batch_size
        for l in range(self.num_layers):
            layer_device = self.layer_devices[l]
            code_data = torch.zeros((numel_one_layer, ),
                                     dtype=torch.int32,
                                     device=layer_device)
            code_length = torch.zeros((max_batch, ),
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

    def reset_batch_rows(self, row_indices: list[int]) -> None:
        super().reset_batch_rows(row_indices)
        if not row_indices:
            return
        rows = torch.tensor(row_indices, dtype=torch.long)
        for layer_idx in range(self.num_layers):
            layer_rows = rows.to(
                device=self.cache_tensors['topk_code_length'][layer_idx].device,
                non_blocking=True,
            )
            self.cache_tensors['topk_code_length'][layer_idx][layer_rows] = 0

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

        def _copy_topk_rows(
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
                old: int(
                    self.cache_tensors['topk_code_length'][layer_idx][old].item()
                )
                for old in normalized.keys()
            }
            _index_copy_rows(self.cache_tensors['topk_code_length'][layer_idx], 0)
            _copy_topk_rows(self.topk_codes[layer_idx], row_lengths)

    def trim_prefill_padding(self, extend_seq_lens: list[int], padded_q_len: int) -> None:
        super().trim_prefill_padding(extend_seq_lens, padded_q_len)
        if not extend_seq_lens or all(int(x) == int(padded_q_len) for x in extend_seq_lens):
            return
        trims = torch.tensor(
            [int(padded_q_len) - int(x) for x in extend_seq_lens],
            dtype=torch.int32,
        )
        for layer_idx in range(self.num_layers):
            active = self.cache_tensors['topk_code_length'][layer_idx][
                :self.curr_batch_size
            ]
            active.sub_(trims.to(device=active.device, non_blocking=True))

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
        seqlen_tensor = self.cache_tensors['topk_code_length'][layer_idx][
            :self.curr_batch_size
        ]
        seqlen_tensor.add_(key_states.shape[1])
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
            self.cache_tensors['topk_code_length'][key_layer_idx][
                :self.curr_batch_size
            ],
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
            self.cache_tensors['topk_code_length'][key_layer_idx][
                :self.curr_batch_size
            ],
        )

        return query_out1, query_out2

    def _decode_append_hash_k(self, key_states: torch.Tensor, layer_idx: int):
        hash_encode_append_decode_k(
            key_states,
            self.topk_codes[layer_idx],
            self.metadata_tensors['hash_weights'][layer_idx],
            self.metadata_tensors[f'packbit_aux_tensor_{key_states.device.index}'],
            self.cache_tensors['topk_code_length'][layer_idx][
                :self.curr_batch_size
            ],
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
        need_overlap_current_query = (
            self.record_overlap_stats
            and self._pending_overlap_head_mask[layer_idx] is not None
        )
        encode_current_query = (
            self.layers_gpu_head_ids[layer_idx].numel() > 0
            or need_overlap_current_query
        )
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

        self.cache_tensors['topk_code_length'][layer_idx][
            :self.curr_batch_size
        ].add_(1)

        torch.cuda.nvtx.range_pop()

        return prefetch_query_code, current_query_code

    def _record_prefetch_overlap_candidate(
        self,
        layer_idx: int,
        prefetch_topk_indices: torch.Tensor,
        gather_mask: torch.Tensor,
    ) -> None:
        k = int(self.topk_prefetch_k_host)
        total_heads = self.curr_batch_size * self.num_key_value_heads
        mask_cpu = (
            gather_mask[:total_heads]
            .detach()
            .to(dtype=torch.bool)
            .cpu()
            .clone()
        )
        if k <= 0 or not bool(mask_cpu.any().item()):
            self._pending_overlap_prefetch[layer_idx] = None
            self._pending_overlap_head_mask[layer_idx] = None
            return

        indices_cpu = (
            prefetch_topk_indices[:, :, :k]
            .reshape(total_heads, k)
            .detach()
            .to(dtype=torch.int32)
            .cpu()
            .clone()
        )
        self._pending_overlap_prefetch[layer_idx] = indices_cpu
        self._pending_overlap_head_mask[layer_idx] = mask_cpu

    def _compute_real_sparse_topk_for_overlap(
        self,
        query: torch.Tensor,
        layer_idx: int,
        head_mask: torch.Tensor,
    ) -> torch.Tensor:
        device_idx = query.device.index
        seq_len_tensor = (
            self.cache_tensors["topk_code_length"][layer_idx][
                :self.curr_batch_size
            ]
            - 1
        ).clamp_min(0)
        seq_len = int(seq_len_tensor.max().item())
        k = int(self.topk_prefetch_k_host)
        if seq_len <= 0 or k <= 0:
            return torch.empty(
                (self.curr_batch_size * self.num_key_value_heads, 0),
                dtype=torch.int32,
                device="cpu",
            )

        seq_len_tensor = seq_len_tensor.to(device=query.device, dtype=torch.int32)
        k_tensor = self.metadata_tensors[f"topk_prefetch_k_{device_idx}"][
            :self.curr_batch_size
        ].to(device=query.device, dtype=torch.int32)
        score_buf = self.metadata_tensors[f"gpu_topk_scores_{device_idx}"][
            :self.curr_batch_size
        ]
        index_buf = self.metadata_tensors[f"gpu_topk_indices_{device_idx}"][
            :self.curr_batch_size
        ]
        value_buf = self.metadata_tensors[f"gpu_topk_values_{device_idx}"][
            :self.curr_batch_size
        ]
        KVLib.static_hamming_score_mask(
            self.topk_codes[layer_idx][: self.curr_batch_size],
            query,
            head_mask,
            score_buf,
            seq_len_tensor,
            self.rbits,
            torch.finfo(torch.float16).max,
            0.0,
            0,
            0,
            self.config.sparse_attention_config.sink_budget,
            self.config.sparse_attention_config.recent_budget,
        )
        KVLib.batch_topk_masked(
            score_buf,
            head_mask,
            index_buf,
            value_buf,
            seq_len_tensor.repeat_interleave(self.num_key_value_heads),
            k_tensor.repeat_interleave(self.num_key_value_heads),
            False,
        )
        max_k = int(k_tensor.max().item())
        return (
            index_buf[:, :, :max_k]
            .reshape(self.curr_batch_size * self.num_key_value_heads, -1)
            .detach()
            .to(dtype=torch.int32)
            .cpu()
            .clone()
        )

    def _record_real_overlap_for_current_layer(
        self,
        layer_idx: int,
        current_query,
    ) -> None:
        prefetch_topk = self._pending_overlap_prefetch[layer_idx]
        head_mask = self._pending_overlap_head_mask[layer_idx]
        self._pending_overlap_prefetch[layer_idx] = None
        self._pending_overlap_head_mask[layer_idx] = None
        if prefetch_topk is None or head_mask is None or current_query is None:
            return

        active_rows = head_mask.numpy().astype(bool)
        if not active_rows.any():
            return

        real_topk = self._compute_real_sparse_topk_for_overlap(
            current_query,
            layer_idx,
            head_mask.to(current_query.device, non_blocking=True),
        )
        if real_topk.numel() == 0:
            return

        real_topk_np = real_topk.numpy()[active_rows]
        prefetch_topk_np = prefetch_topk.numpy()[active_rows]
        active_bh_indices = np.flatnonzero(active_rows).astype(np.int32)
        common_k = min(real_topk_np.shape[1], prefetch_topk_np.shape[1])
        if common_k <= 0:
            return
        real_topk_np = real_topk_np[:, :common_k]
        prefetch_topk_np = prefetch_topk_np[:, :common_k]
        if real_topk_np.shape[0] == 0:
            return

        metrics = _compute_overlap_metrics(
            real_topk_np,
            prefetch_topk_np,
        )
        metrics["active_bh_indices"] = [int(v) for v in active_bh_indices.tolist()]
        self._decode_overlap_metrics[layer_idx] = metrics

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
        max_k = int(self.topk_prefetch_k_host if is_prefetch else self.topk_current_k_host)
        if max_k <= 0:
            return torch.empty(
                (self.curr_batch_size, self.num_key_value_heads, 0),
                dtype=torch.int32,
                device=query.device,
            )
        active_k = k[:self.curr_batch_size]
        max_k = min(max_k, self.max_prefetch_topk_len if is_prefetch else self.max_current_topk_len)
        score_buf = self.metadata_tensors[f'gpu_topk_scores_{device_idx}'][
            :self.curr_batch_size
        ]
        index_buf = self.metadata_tensors[f'gpu_topk_indices_{device_idx}'][
            :self.curr_batch_size
        ]
        value_buf = self.metadata_tensors[f'gpu_topk_values_{device_idx}'][
            :self.curr_batch_size
        ]
        # checked
        KVLib.static_hamming_score_mask(
            self.topk_codes[layer_idx][: self.curr_batch_size],
            query,
            mask,
            score_buf,
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
            score_buf,
            mask,
            index_buf,
            value_buf,
            self.cache_tensors['topk_code_length'][layer_idx][
                :self.curr_batch_size
            ].repeat_interleave(self.num_key_value_heads),
            active_k.repeat_interleave(self.num_key_value_heads),
            False,
        )
        topk_indices = index_buf
        active = topk_indices.reshape(
            self.curr_batch_size * self.num_key_value_heads,
            self.max_buffer_len,
        )
        per_head_seq_lens = self.cache_tensors['topk_code_length'][layer_idx][
            :self.curr_batch_size
        ].repeat_interleave(self.num_key_value_heads).to(
            device=topk_indices.device,
            dtype=topk_indices.dtype,
        )
        # Keep decode CUDA graph capture safe: do not use tensor .item() here.
        # The masked_fill handles zero-length rows without a CPU-side branch.
        row_max_indices = (per_head_seq_lens - 1).clamp_min(0).view(-1, 1)
        active.clamp_min_(0)
        active.copy_(torch.minimum(active, row_max_indices))
        active.masked_fill_((per_head_seq_lens <= 0).view(-1, 1), 0)
        return topk_indices[:, :, :max_k].contiguous()

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
