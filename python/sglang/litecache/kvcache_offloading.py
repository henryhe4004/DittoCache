from typing import Dict, Optional, Union, Any
import os
import csv
import math
import torch
import logging
import pandas as pd

from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig

from .kvcache_full_attn import CustomStaticCache
import sgl_kernel.kvlib as KVLib
from sglang.jit_kernel.triton_kernels.cache.check_reuse import (
    check_reuse_with_importance,
    check_reuse_head_threshold_with_gpu_head,
)


GPU_PAGE_SIZE = 65536
USE_FIXED_THRESHOLDS = False if os.environ.get(
    "USE_FIXED_THRESHOLDS") is None else int(
        os.environ["USE_FIXED_THRESHOLDS"]) > 0
USE_INTRA_GQA_AGGREGATION = True if os.environ.get(
    "USE_INTRA_GQA_AGGREGATION") is None else int(
        os.environ["USE_INTRA_GQA_AGGREGATION"]) > 0
DISABLE_PERSISTENT_CACHING = False if os.environ.get(
    "DISABLE_PERSISTENT_CACHING") is None else int(
        os.environ["DISABLE_PERSISTENT_CACHING"]) > 0


def create_aligned_cuda_tensor(data_numel, dtype, device, pagesize):
    elem_size = dtype.itemsize
    data_size = data_numel * elem_size

    min_total_bytes = data_size + (pagesize - 1)
    total_bytes = ((min_total_bytes + pagesize - 1) // pagesize) * pagesize

    aligned_numel = (total_bytes + elem_size - 1) // elem_size

    raw_data = torch.zeros((aligned_numel,), dtype=dtype, device=device)

    data_ptr = raw_data.data_ptr()
    aligned_data_ptr = ((data_ptr + pagesize - 1) // pagesize) * pagesize
    offset_bytes = aligned_data_ptr - data_ptr
    offset_bytes = ((offset_bytes + elem_size - 1) // elem_size) * elem_size
    aligned_data_ptr = data_ptr + offset_bytes

    align_skip_numel = offset_bytes // elem_size
    aligned_data = raw_data[align_skip_numel:align_skip_numel + data_numel]

    assert aligned_data.data_ptr() % pagesize == 0
    assert aligned_data.numel() == data_numel

    return raw_data, aligned_data


class OffloadingCache(CustomStaticCache):

    def __init__(
        self,
        config: PretrainedConfig,
        custom_config: Any,
        device: torch.device = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device,
                                                   int]]] = None,
    ) -> None:
        super().__init__(
            config,
            custom_config,
            device,
            layer_device_map,
        )
        self.topk_ratio = self.config.sparse_attention_config.token_budget
        if self.config.sparse_attention_config.token_budget < 1:
            self.max_sparse_tokens = int(self.config.kvcache_manager_config.max_tokens * self.topk_ratio)
        else:
            self.max_sparse_tokens = int(self.topk_ratio)
        self.max_sparse_tokens = max(self.max_sparse_tokens, (self.config.sparse_attention_config.sink_budget + self.config.sparse_attention_config.recent_budget + 1) * self.config.kvcache_manager_config.max_batch_size)

    def _plan_topk_used_gpu_memory(self):
        raise NotImplementedError

    def _init_kv_placement(self):
        # Read head importance and cosine similarity data
        cosine_file = os.path.join(self.config.offload_config.attn_pattern_path, "heads_cosine_similarity.csv")
        k_head_importance_file = os.path.join(self.config.offload_config.attn_pattern_path, "k_heads_importance.tsv")
        q_head_importance_file = os.path.join(self.config.offload_config.attn_pattern_path, "q_heads_importance.tsv")

        head_importance = pd.read_csv(k_head_importance_file, sep='\t', header=None).to_numpy()
        q_head_importance = pd.read_csv(q_head_importance_file, sep='\t', header=None).to_numpy()
        head_cos = pd.read_csv(cosine_file, sep=',', header=None).to_numpy()

        # Process head importance and cosine similarity
        head_cos = torch.from_numpy(head_cos)
        head_importance = torch.from_numpy(head_importance)
        self.layers_q_importance = [
            torch.from_numpy(q_head_importance[l]).to(self.layer_devices[l])
            for l in range(self.num_layers)
        ]
        # Normalize q head importance within GQA groups
        for l in range(self.num_layers):
            layer_q_importance = self.layers_q_importance[l].view(
                self.num_key_value_heads, -1)
            qsum = layer_q_importance.sum(dim=-1)
            qsum[qsum == 0] = qsum[qsum == 0] + 1e-8
            layer_q_importance = layer_q_importance / qsum.unsqueeze(-1)
            self.layers_q_importance[l] = layer_q_importance.view(-1)

        # Set reuse thresholds for kv heads
        stacked_reuse_thresholds = []
        self.layers_reuse_thresholds = []
        for l in range(self.num_layers):
            reuse_thresholds = head_importance[l]
            high = math.acos(self.config.offload_config.reuse_threshold_upper)
            low = math.acos(self.config.offload_config.reuse_threshold_lower)
            reuse_thresholds = torch.cos(
                low + (high - low) *
                reuse_thresholds**self.config.offload_config.decay_p)
            if USE_FIXED_THRESHOLDS:
                reuse_thresholds[:] = self.config.offload_config.reuse_threshold_upper
            self.layers_reuse_thresholds.append(
                reuse_thresholds.to(self.layer_devices[l]))
            stacked_reuse_thresholds.append(reuse_thresholds)
        stacked_reuse_thresholds = torch.stack(stacked_reuse_thresholds, dim=0)
        print(f"Reuse thresholds:\n{stacked_reuse_thresholds}")

        # Calculate reuse difficulty and hard-to-reuse heads
        hard2reuse_head_mask = stacked_reuse_thresholds > head_cos - self.config.offload_config.cosine_padding
        if DISABLE_PERSISTENT_CACHING:
            hard2reuse_head_mask[:] = False
        num_hard2reuse_head = torch.sum(hard2reuse_head_mask, dim=1)
        reuse_difficulty = stacked_reuse_thresholds - head_cos
        print(f"#Hard-to-reuse-heads:{num_hard2reuse_head.sum().item()}")

        # ----> 显存充足的理想情况下，哪些 head 应该被放到 GPU
        layers_gpu_head_mask = []
        num_gpu_heads = 0
        sorted_hids = torch.argsort(reuse_difficulty,
                                    dim=-1,
                                    stable=True,
                                    descending=True)
        for l in range(self.num_layers):
            if l < self.config.offload_config.num_skip_layers:
                head_gpu_mask = torch.ones((self.num_key_value_heads, ),
                                           dtype=torch.bool,
                                           device="cpu")
            else:
                head_gpu_mask = torch.zeros((self.num_key_value_heads, ),
                                            dtype=torch.bool,
                                            device="cpu")
                num_unoverlapped = max(
                    num_hard2reuse_head[l] -
                    self.config.offload_config.num_overlapped_heads, 0)
                if num_unoverlapped > 0:
                    head_gpu_mask[sorted_hids[l, :num_unoverlapped]] = True
            layers_gpu_head_mask.append(head_gpu_mask)
            num_gpu_heads += head_gpu_mask.sum().item()

        # Calculate GPU memory requirements for skip layers
        numel_one_layer = (2 * self.config.kvcache_manager_config.max_tokens *
                           self.num_key_value_heads * self.head_dim)
        skip_layers_mem = self.config.offload_config.num_skip_layers * numel_one_layer * self.dtype.itemsize
        assert skip_layers_mem <= self.mem_budget, \
            f"max_tokens={self.config.kvcache_manager_config.max_tokens}\n" \
            f"non-prefetchable layers ({self.config.offload_config.num_skip_layers * self.num_key_value_heads} heads) require "\
            f"{skip_layers_mem / 1024**3:.2f} GB GPU memory, but only {self.mem_budget / 1024**3:.2f} GB left!"
        self.mem_budget -= skip_layers_mem
        print(
            f"Non-prefetchable layer ({self.config.offload_config.num_skip_layers * self.num_key_value_heads} heads) "\
            f"consumed GPU memory: {skip_layers_mem / 1024**3:.2f} GB. " \
            f"{self.mem_budget / 1024**3:.2f} GB budget left.")

        # ----> 如果显存不足，调整放置策略

        # Calculate memory for top-k buffer and kvcache
        topk_buffer_mem_per_head = (
            2 * self.max_sparse_tokens * self.head_dim * self.dtype.itemsize)
        kvcache_mem_per_head = (
            2 * self.config.kvcache_manager_config.max_tokens * self.head_dim * self.dtype.itemsize)
        num_gpu_heads_remained = num_gpu_heads - self.config.offload_config.num_skip_layers * self.num_key_value_heads
        num_cpu_heads_total = self.num_layers * self.num_key_value_heads - num_gpu_heads
        remained_layers_mem = num_gpu_heads_remained * kvcache_mem_per_head + num_cpu_heads_total * topk_buffer_mem_per_head

        # Adjust GPU head placement if memory is insufficient
        if remained_layers_mem > self.mem_budget:
            num_heads_remained = self.num_key_value_heads * (
                self.num_layers - self.config.offload_config.num_skip_layers)
            # Ensure minimum memory for top-k buffers
            assert self.mem_budget >= topk_buffer_mem_per_head * num_heads_remained, \
                f"max_sparsse_token={self.max_sparse_tokens} " \
                f"require at least {topk_buffer_mem_per_head * num_heads_remained / 1024**3:.2f} GB GPU memory, " \
                f"but only {self.mem_budget / 1024**3:.2f} GB left!"

            # Calculate available heads based on memory budget
            available_heads = int(
                (self.mem_budget -
                 num_heads_remained * topk_buffer_mem_per_head) /
                (kvcache_mem_per_head - topk_buffer_mem_per_head))
            print(
                f"{num_gpu_heads_remained} on-GPU heads need GPU memory: " \
                f"{remained_layers_mem / 1024**3:.2f} GB. " \
                f"However, only {self.max_gpu_memory_size / 1024**3:.2f} GB budget is available, "
                f"which only supports {available_heads} on-GPU heads. Re-alloc...."
            )

            # Re-allocate GPU heads based on reuse difficulty
            total_on_gpu_mask = torch.cat(
                layers_gpu_head_mask[self.config.offload_config.
                                     num_skip_layers:])
            total_on_gpu_ids = torch.nonzero(total_on_gpu_mask).view(-1)
            total_reuse_diff = reuse_difficulty[self.config.offload_config.
                                                num_skip_layers:].view(-1)
            on_gpu_reuse_diff = total_reuse_diff[total_on_gpu_ids]

            # Sort heads by reuse difficulty and select top candidates
            sort_idx = torch.argsort(on_gpu_reuse_diff,
                                     descending=True,
                                     stable=True)
            selected_gpu_head_ids = total_on_gpu_ids[
                sort_idx][:available_heads]

            # Create new GPU head mask
            new_gpu_mask = torch.zeros(
                (self.num_layers - self.config.offload_config.num_skip_layers,
                 self.num_key_value_heads),
                dtype=torch.bool,
                device="cpu").view(-1)
            new_gpu_mask[selected_gpu_head_ids] = True
            new_gpu_mask = new_gpu_mask.view(-1, self.num_key_value_heads)

            # Update head mask for non-skip layers
            for l in range(self.config.offload_config.num_skip_layers,
                           self.num_layers):
                layers_gpu_head_mask[l] = new_gpu_mask[
                    l - self.config.offload_config.num_skip_layers]

            # Update memory calculation
            remained_layers_mem = available_heads * kvcache_mem_per_head + \
                             (num_heads_remained - available_heads) * topk_buffer_mem_per_head

        # Allocate remaining memory
        self.mem_budget -= remained_layers_mem

        # Initialize final data structures for head placement
        self.num_gpu_heads = 0
        self.num_full_gpu_layers = 0
        self.layers_gpu_head_mask = []
        self.layers_gpu_bh_mask = []
        self.layers_gpu_head_ids = []
        self.layers_cpu_head_ids = []
        self.layers_mixed_head_index = []
        self.layers_mixed_head_index_cpu = []
        self.layers_full_gpu_mask = []
        self.layers_num_gpu_buffer_heads = []

        for l in range(self.num_layers):
            head_mask = layers_gpu_head_mask[l].to(self.layer_devices[l])
            self.layers_gpu_head_mask.append(head_mask)

            # Get GPU and CPU head IDs
            gpu_head_ids = torch.nonzero(head_mask).view(-1)
            cpu_head_ids = torch.nonzero(~head_mask).view(-1)

            self.layers_gpu_head_ids.append(gpu_head_ids)
            self.layers_cpu_head_ids.append(cpu_head_ids)
            self.layers_gpu_bh_mask.append(
                head_mask.repeat(
                    self.config.kvcache_manager_config.max_batch_size))

            # Update counts and indices
            num_layer_gpu_heads = gpu_head_ids.numel()
            self.num_gpu_heads += num_layer_gpu_heads
            self.layers_num_gpu_buffer_heads.append(self.num_key_value_heads -
                                                    num_layer_gpu_heads)
            print(f"Layer {l:02d} on-GPU heads: {gpu_head_ids.cpu().tolist()}, offloaded heads: {cpu_head_ids.cpu().tolist()}")

            # Determine if layer is full GPU
            if num_layer_gpu_heads >= self.num_key_value_heads:
                self.layers_full_gpu_mask.append(True)
                self.num_full_gpu_layers += 1
            else:
                self.layers_full_gpu_mask.append(False)
            # Create mixed head index
            mixed_head_index = torch.zeros((self.num_key_value_heads, ),
                                            device=self.layer_devices[l],
                                            dtype=torch.int64)
            mixed_head_index[head_mask] = torch.arange(
                0, num_layer_gpu_heads, device=mixed_head_index.device)
            mixed_head_index[~head_mask] = torch.arange(
                0,
                self.num_key_value_heads - num_layer_gpu_heads,
                device=mixed_head_index.device)
            self.layers_mixed_head_index.append(mixed_head_index.int())
            self.layers_mixed_head_index_cpu.append(mixed_head_index.int().cpu())

        # Set final counts
        print("Total on-GPU heads number:", self.num_gpu_heads)
        self.num_cpu_layers = self.num_layers - self.num_full_gpu_layers
        self.num_cpu_heads = self.num_layers * self.num_key_value_heads - self.num_gpu_heads
        print(
            f"Cache and other layer's {self.num_gpu_heads} on-GPU heads consumed GPU memory: " \
            f"{remained_layers_mem / 1024**3:.2f} GB. " \
            f"{self.mem_budget / 1024**3:.2f} GB budget left.")

    def _create_cache_tensors(self):
        # ==================== gpu heads ====================
        numel_one_head = (2 * self.config.kvcache_manager_config.max_tokens *
                          self.head_dim)
        self.cache_tensors['cache_data'] = [None for l in range(self.num_layers)]
        self.cache_tensors['cache_length'] = [None for l in range(self.num_layers)]
        for l in range(self.num_layers):
            num_gpu_heads = self.layers_gpu_head_ids[l].numel()
            if num_gpu_heads == 0:
                continue
            layer_device = self.layer_devices[l]
            cache_data = torch.zeros((numel_one_head * num_gpu_heads, ),
                                     dtype=self.dtype,
                                     device=layer_device)
            gpu_cache_length = torch.zeros((1, ),
                                           dtype=torch.int32,
                                           device=layer_device)
            self.cache_tensors['cache_data'][l] = cache_data
            self.cache_tensors['cache_length'][l] = gpu_cache_length
        self.kv_caches = [None for _ in range(self.num_layers)]

    def _create_offload_tensors(self):
        # ==================== offloaded heads ====================
        # CPU cache + GPU top-k data buffer
        numel_one_head = (2 * self.config.kvcache_manager_config.max_tokens *
                          self.head_dim)
        numel_buffer_one_head = (2 * self.max_sparse_tokens * self.head_dim)

        self.cache_tensors['cpu_cache_data'] = [None for _ in range(self.num_layers)]
        self.cache_tensors['cpu_cache_length'] = [None for _ in range(self.num_layers)]
        self.cache_tensors['gpu_buffer_data_raw'] = [None for _ in range(self.num_layers)]
        self.cache_tensors['gpu_buffer_data'] = [None for _ in range(self.num_layers)]
        self.cache_tensors['gpu_buffer_length'] = [None for _ in range(self.num_layers)]
        self.cache_tensors['gpu_buffer_ptr'] = [None for _ in range(self.num_layers)]

        for l in range(self.num_layers):
            layer_device = self.layer_devices[l]

            cache_data = torch.zeros((numel_one_head * self.num_key_value_heads, ),
                                     dtype=self.dtype,
                                     device="cpu",
                                     pin_memory=True)
            cache_length = torch.zeros((1, ),
                                       dtype=torch.int32,
                                       device=layer_device)
            self.cache_tensors['cpu_cache_data'][l] = cache_data
            self.cache_tensors['cpu_cache_length'][l] = cache_length

            num_cpu_heads = self.layers_cpu_head_ids[l].numel()
            if num_cpu_heads == 0:
                continue

            raw_data, aligned_data = create_aligned_cuda_tensor(
                num_cpu_heads * numel_buffer_one_head,
                dtype=self.dtype,
                device=self.layer_devices[l],
                pagesize=GPU_PAGE_SIZE)
            buffer_length = torch.zeros((1, ),
                                        dtype=torch.int32,
                                        device=layer_device)
            buffer_ptr = torch.zeros((1, ),
                                     dtype=torch.int32,
                                     device=layer_device)
            self.cache_tensors['gpu_buffer_data_raw'][l] = raw_data
            self.cache_tensors['gpu_buffer_data'][l] = aligned_data
            self.cache_tensors['gpu_buffer_length'][l] = buffer_length
            self.cache_tensors['gpu_buffer_ptr'][l] = buffer_ptr

        self.cpu_kv_caches = [None for _ in range(self.num_layers)]
        self.gpu_kv_buffers = [None for _ in range(self.num_layers)]
        self.cpu_indices_buffer = None

    def _create_metadata_tensors(self):
        # rope metadata
        super()._create_metadata_tensors()

        # flag for launching gather
        # flag[0] = flag for start doing gather, value = layer
        # flag[1] = gather length
        # flag[2] = current batch size (<= max batch size)
        # flag[3] = current max_seq_len
        # flag[4] = current max_buffer_len
        self.metadata_tensors['gather_engine_metadata'] = torch.full((6, ),
                                 -1,
                                 device="cpu",
                                 dtype=torch.int32,
                                 pin_memory=True)
        self.metadata_tensors['cpu_indices_data'] = torch.full((self.max_sparse_tokens * self.num_key_value_heads, ),
                                                    fill_value=-1,
                                                    dtype=torch.int64,
                                                    device="cpu",
                                                    pin_memory=True)

        # flag for gather being finished
        self.metadata_tensors['ready_flag'] = [None for _ in range(self.num_layers)]
        self.metadata_tensors['cached_query'] = [None for _ in range(self.num_layers)]
        self.metadata_tensors['gather_mask'] = [None for _ in range(self.num_layers)]
        for l in range(self.num_layers):
            self.metadata_tensors['ready_flag'][l] = torch.zeros(
                (self.config.kvcache_manager_config.max_batch_size *
                 self.num_key_value_heads, ),
                dtype=torch.bool,
                device='cpu',
                pin_memory=True,
            )
            self.metadata_tensors['cached_query'][l] = torch.zeros(
                (self.config.kvcache_manager_config.max_batch_size,
                 1,
                 self.num_heads,
                 self.head_dim, ),
                dtype=self.dtype,
                device=self.layer_devices[l]
            )
            self.metadata_tensors['gather_mask'][l] = torch.ones(
                (self.config.kvcache_manager_config.max_batch_size *
                 self.num_key_value_heads, ),
                dtype=torch.bool,
                device=self.layer_devices[l])

        for device_idx in self.unique_devices:
            self.metadata_tensors[f'query_cache_valid_{device_idx}'] = torch.zeros(
                (1, ),
                dtype=torch.bool,
                device=device_idx,
            )
            self.metadata_tensors[f'gpu_topk_scores_data_{device_idx}'] = torch.zeros(
                (self.config.kvcache_manager_config.max_tokens * self.num_key_value_heads, ),
                dtype=torch.float16,
                device=device_idx,
            )
            self.metadata_tensors[f'gpu_topk_values_data_{device_idx}'] = torch.zeros(
                (self.max_sparse_tokens * self.num_key_value_heads, ),
                dtype=torch.float16, # the raft top-k kernel only support float16 and float32
                device=device_idx,
            )
            self.metadata_tensors[f'gpu_topk_indices_data_{device_idx}'] = torch.full(
                (self.max_sparse_tokens * self.num_key_value_heads, ),
                fill_value=-1,
                dtype=torch.int32,
                device=device_idx,
            )
            self.metadata_tensors[f'topk_prefetch_k_{device_idx}'] = torch.zeros(
                (1, ),
                dtype=torch.int32,
                device=device_idx,
            )
            self.metadata_tensors[f'topk_current_k_{device_idx}'] = torch.zeros(
                (1, ),
                dtype=torch.int32,
                device=device_idx,
            )

        self.is_first_prefill_chunk = [True for _ in range(self.num_layers)]
        self.first_decode_layer_step = True

    def _create_topk_tensors(self):
        # ================ top-k retrieval metadata cache ================
        raise NotImplementedError

    def build_cache(self):
        self._plan_topk_used_gpu_memory()
        self._init_kv_placement()
        self._create_metadata_tensors()
        self._create_topk_tensors()
        self._create_cache_tensors()
        self._create_offload_tensors()
        self._init_offloading()

    def _reset_cache_tensors(self, batch_size):
         # reset gpu kv cache
        for l in range(self.num_layers):
            num_gpu_heads = self.layers_gpu_head_ids[l].numel()
            if num_gpu_heads == 0:
                continue
            cache_data = self.cache_tensors['cache_data'][l]
            self.kv_caches[l] = cache_data[:2 * batch_size * self.max_seq_len *
                                           num_gpu_heads * self.head_dim].view(
                                               2, batch_size, self.max_seq_len,
                                               num_gpu_heads, self.head_dim)
            cache_length = self.cache_tensors['cache_length'][l]
            cache_length.zero_()

    def _reset_offload_tensors(self, batch_size):
        # reset cpu kv cache
        for l in range(self.num_layers):
            cpu_cache_data = self.cache_tensors['cpu_cache_data'][l]
            self.cpu_kv_caches[l] = cpu_cache_data[:2 * batch_size *
                                                   self.max_seq_len *
                                                   self.num_key_value_heads *
                                                   self.head_dim].view(
                                                       2, batch_size,
                                                       self.max_seq_len,
                                                       self.num_key_value_heads,
                                                       self.head_dim)
            cpu_cache_length = self.cache_tensors['cpu_cache_length'][l]
            cpu_cache_length.zero_()

            num_cpu_heads = self.layers_cpu_head_ids[l].numel()
            if num_cpu_heads == 0:
                continue

            gpu_buffer_data = self.cache_tensors['gpu_buffer_data'][l]
            gpu_buffer_data.zero_()
            self.gpu_kv_buffers[
                l] = gpu_buffer_data[:2 * batch_size *
                                     self.max_buffer_len *
                                     num_cpu_heads * self.head_dim].view(
                                         2, batch_size,
                                         self.max_buffer_len,
                                         num_cpu_heads, self.head_dim)
            gpu_buffer_length = self.cache_tensors['gpu_buffer_length'][l]
            gpu_buffer_length.zero_()

            num_cpu_heads = self.layers_cpu_head_ids[l].numel()
            if num_cpu_heads == 0:
                continue

            gpu_buffer_ptr = self.cache_tensors['gpu_buffer_ptr'][l]
            gpu_buffer_ptr[0] = self.config.sparse_attention_config.sink_budget + self.config.sparse_attention_config.recent_budget

    def _reset_topk_tensors(self, batch_size):
        raise NotImplementedError

    # ====================== HF APIs ======================
    def reset(self, batch_size):
        assert batch_size <= self.config.kvcache_manager_config.max_batch_size, \
            f"batch_size ({batch_size}) should be less than max_batch_size " \
            f"({self.config.kvcache_manager_config.max_batch_size})"

        self.curr_batch_size = batch_size
        self.max_seq_len = (self.config.kvcache_manager_config.max_tokens //
                            batch_size)
        self.max_buffer_len = self.max_sparse_tokens // batch_size
        self.max_prefetch_topk_len = (self.max_buffer_len - self.config.sparse_attention_config.sink_budget - self.config.sparse_attention_config.recent_budget - 1)
        self.max_current_topk_len = self.max_buffer_len

        self._reset_cache_tensors(batch_size)
        self._reset_offload_tensors(batch_size)
        self._reset_topk_tensors(batch_size)

        self.metadata_tensors['gather_engine_metadata'][0:2] = -1
        self.metadata_tensors['gather_engine_metadata'][2] = batch_size
        self.metadata_tensors['gather_engine_metadata'][3] = self.max_seq_len
        self.metadata_tensors['gather_engine_metadata'][4] = self.max_buffer_len
        self.metadata_tensors['gather_engine_metadata'][5] = self.max_prefetch_topk_len

        self.metadata_tensors['cpu_indices_data'].fill_(-1)
        self.cpu_indices_buffer = self.metadata_tensors['cpu_indices_data'][:batch_size * 
            self.max_prefetch_topk_len * self.num_key_value_heads
        ].view(
            batch_size * self.num_key_value_heads,
            self.max_prefetch_topk_len,
        )

        for l in range(self.num_layers):
            self.metadata_tensors['ready_flag'][l][:] = False
            self.metadata_tensors['gather_mask'][l][:] = True
            self.metadata_tensors['cached_query'][l].zero_()
            self.is_first_prefill_chunk[l] = True

        for device_idx in self.unique_devices:
            self.metadata_tensors[f'gpu_topk_scores_{device_idx}'] = self.metadata_tensors[
                f'gpu_topk_scores_data_{device_idx}'][:batch_size * self.max_seq_len * self.num_key_value_heads].view(
                    batch_size, self.num_key_value_heads, self.max_seq_len,
                )
            self.metadata_tensors[f'gpu_topk_values_{device_idx}'] = self.metadata_tensors[
                f'gpu_topk_values_data_{device_idx}'][:batch_size * self.max_buffer_len * self.num_key_value_heads].view(
                    batch_size, self.num_key_value_heads, self.max_buffer_len,
                )
            self.metadata_tensors[f'gpu_topk_indices_{device_idx}'] = self.metadata_tensors[
                f'gpu_topk_indices_data_{device_idx}'][:batch_size * self.max_buffer_len * self.num_key_value_heads].view(
                    batch_size, self.num_key_value_heads, self.max_buffer_len,
                )

        self.total_seqlen = 0
        self.first_decode_layer_step = True

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if self.layers_gpu_head_ids[layer_idx].numel() > 0:
            return self.cache_tensors['cache_length'][layer_idx].item()
        else:
            return self.cache_tensors['cpu_cache_length'][layer_idx].item()

    # ==================== Copy Engine ====================
    def _init_offloading(self):
        # prefill offloading still uses cuda event
        self.default_stream_event = torch.cuda.Event()
        self.transfer_event = torch.cuda.Event()
        self.transfer_stream = torch.cuda.Stream()
        self.prefill_copy_buffer = None

        # gather and copy engine
        self.cpu_gather_engine = None
        if self.num_cpu_layers > 0:
            print(
                f"Set {self.config.offload_config.num_omp_threads} omp threads for CPUGatherEngine")
            self.cpu_gather_engine = KVLib.CPUGatherEngineV3(
                self.config.offload_config.num_omp_threads,
                self.cache_tensors['cpu_cache_data'],
                self.cache_tensors['gpu_buffer_data'],
                self.layers_mixed_head_index_cpu,
                self.layers_num_gpu_buffer_heads,
                self.metadata_tensors['cpu_indices_data'],
                self.metadata_tensors['gather_engine_metadata'],
                self.metadata_tensors['ready_flag'],
                self.config.kvcache_manager_config.max_batch_size,
                self.config.sparse_attention_config.sink_budget + self.config.sparse_attention_config.recent_budget + 1,
                self.num_key_value_heads,
                self.head_dim,
                debug=True)

    def update_metadata(self, q_len: int, is_prefill=False, layer_idx: int = 0):
        super().update_metadata(q_len, is_prefill, layer_idx)
        if is_prefill:
            self.first_decode_layer_step = True
        else:
            curr_seqlen = self.get_seq_length(0)
            if self.topk_ratio < 1:
                topk_prefetch_k = int(curr_seqlen * self.topk_ratio) - self.config.sparse_attention_config.sink_budget - self.config.sparse_attention_config.recent_budget - 1
                topk_current_k = int(curr_seqlen * self.topk_ratio)
            else:
                topk_prefetch_k = int(self.topk_ratio) - self.config.sparse_attention_config.sink_budget - self.config.sparse_attention_config.recent_budget - 1
                topk_current_k = int(self.topk_ratio)
            topk_prefetch_k = max(min(topk_prefetch_k, self.max_prefetch_topk_len), 0)
            topk_current_k = max(min(topk_current_k, self.max_current_topk_len), self.config.sparse_attention_config.sink_budget + self.config.sparse_attention_config.recent_budget + 1)

            for device_idx in self.unique_devices:
                if self.first_decode_layer_step:
                    self.metadata_tensors[f'query_cache_valid_{device_idx}'][0] = False
                    self.first_decode_layer_step = False
                else:
                    self.metadata_tensors[f'query_cache_valid_{device_idx}'][0] = True
                self.metadata_tensors[f'topk_prefetch_k_{device_idx}'][0] = topk_prefetch_k
                self.metadata_tensors[f'topk_current_k_{device_idx}'][0] = topk_current_k

            for l in range(self.num_layers):
                num_cpu_heads = self.layers_cpu_head_ids[l].numel()
                if num_cpu_heads == 0:
                    continue
                buffer_append_ptr = self.cache_tensors['gpu_buffer_ptr'][l]
                if buffer_append_ptr[0].item() >= (self.config.sparse_attention_config.sink_budget + self.config.sparse_attention_config.recent_budget + 1):
                    buffer_append_ptr[0] = self.config.sparse_attention_config.sink_budget

        # print("\nupdate metadata", q_len)
        # print("cache_length", self.cache_tensors['cache_length'])
        # print("cpu_cache_length", self.cache_tensors['cpu_cache_length'])
        # print("topk_code_length", self.cache_tensors['topk_code_length'])
        # print("gpu_buffer_ptr", self.cache_tensors['gpu_buffer_ptr'])
        # print("query_cache_valid", self.metadata_tensors[f'query_cache_valid_{self.layer_devices[0]}'])
        # print("topk_prefetch_k", self.metadata_tensors[f'topk_prefetch_k_{self.layer_devices[0]}'])
        # print("topk_current_k", self.metadata_tensors[f'topk_current_k_{self.layer_devices[0]}'])

    # =====================================================
    def append_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        key_states = key_states.view(self.curr_batch_size, -1,
                                     self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(self.curr_batch_size, -1,
                                         self.num_key_value_heads,
                                         self.head_dim)
        prefill_len = key_states.shape[1]

        if self.layers_gpu_head_ids[layer_idx].numel() > 0:
            torch.cuda.nvtx.range_push("append gpu cache")
            assert prefill_len <= self.max_seq_len, \
                f"input kv states length {prefill_len}, " \
                f"should be less than max_seq_len = {self.max_seq_len}"
            self.kv_caches[layer_idx][0, :,
                                    :prefill_len, :, :].copy_(key_states[:, :,
                                    self.layers_gpu_head_ids[layer_idx], :])
            self.kv_caches[layer_idx][1, :,
                                    :prefill_len, :, :].copy_(value_states[:, :,
                                    self.layers_gpu_head_ids[layer_idx], :])
            seq_len_tensor = self.cache_tensors['cache_length'][layer_idx]
            seq_len_tensor[0] = prefill_len
            torch.cuda.nvtx.range_pop()

        if not self.layers_full_gpu_mask[layer_idx]:
            sink = self.config.sparse_attention_config.sink_budget
            recent = min(self.config.sparse_attention_config.recent_budget, prefill_len - sink)

            assert prefill_len <= self.max_seq_len, \
                f"input kv states length {prefill_len}, " \
                f"should be less than max_seq_len = {self.max_seq_len}"

            with torch.cuda.stream(self.transfer_stream):
                for bsz in range(key_states.shape[0]):
                    # pytorch doesn't support cudaMemcpy2DAsync, so offload batch by batch
                    self.cpu_kv_caches[layer_idx][
                        0, bsz:bsz + 1, :prefill_len,
                        ...].copy_(key_states[bsz:bsz + 1, ...], non_blocking=True)
                    self.cpu_kv_caches[layer_idx][
                        1, bsz:bsz + 1, :prefill_len,
                        ...].copy_(value_states[bsz:bsz + 1, ...], non_blocking=True)
                self.transfer_event.record(self.transfer_stream)

            self.prefill_copy_buffer = (key_states, value_states)
            cpu_seq_len_tensor = self.cache_tensors['cpu_cache_length'][layer_idx]
            cpu_seq_len_tensor[0] = prefill_len

            # 2. save sink recent tokens (checked)
            torch.cuda.nvtx.range_push("append sink recent")
            if sink > 0:
                self.gpu_kv_buffers[layer_idx][
                    0, :, :sink, :, :].copy_(key_states[:, :sink, self.layers_cpu_head_ids[layer_idx], :])
                self.gpu_kv_buffers[layer_idx][
                    1, :, :sink, :, :].copy_(value_states[:, :sink, self.layers_cpu_head_ids[layer_idx], :])
            if recent > 0:
                self.gpu_kv_buffers[layer_idx][
                    0, :, sink:sink + recent, :, :].copy_(key_states[:, -recent:, self.layers_cpu_head_ids[layer_idx], :])
                self.gpu_kv_buffers[layer_idx][
                    1, :, sink:sink + recent, :, :].copy_(value_states[:, -recent:, self.layers_cpu_head_ids[layer_idx], :])
            torch.cuda.nvtx.range_pop()

        return key_states, value_states

    def append_topk_cache_prefill(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        values_states: torch.Tensor,
        layer_idx: int,
    ):
        raise NotImplementedError

    def sync_offload_prefill(self):
        torch.cuda.nvtx.range_push("prefill_sync")
        if self.prefill_copy_buffer is not None:
            cur_stream = torch.cuda.current_stream()
            torch.cuda.nvtx.range_push("wait event")
            cur_stream.wait_event(self.transfer_event)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("get buffer")
            k, v = self.prefill_copy_buffer
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("delete")
            self.prefill_copy_buffer = None
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

    def append_gpu_cache_decode(self, key_states: torch.Tensor,
                                   value_states: torch.Tensor, layer_idx: int):
        torch.cuda.nvtx.range_push("append gpu cache")
        if self.layers_full_gpu_mask[layer_idx]:
            KVLib.kvcache_append_tensor_pos(
                self.kv_caches[layer_idx], key_states, value_states,
                self.cache_tensors['cache_length'][layer_idx])
        else:
            KVLib.kvcache_append_tensor_pos_head_sparse(
                self.kv_caches[layer_idx], key_states, value_states,
                self.layers_gpu_head_ids[layer_idx],
                self.cache_tensors['cache_length'][layer_idx])
        self.cache_tensors['cache_length'][layer_idx] += 1
        torch.cuda.nvtx.range_pop()

    def append_topk_cache_decode(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        prefetch_query_states: Optional[torch.Tensor] = None,
        current_query_states: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError

    def cache_query_and_update(
        self,
        query_states: torch.Tensor,
        layer_idx: int,
    ):
        gather_mask = self.metadata_tensors['gather_mask'][layer_idx][
            :self.curr_batch_size * self.num_key_value_heads]

        torch.cuda.nvtx.range_push("check reuse")
        if USE_INTRA_GQA_AGGREGATION and self.num_key_value_heads < self.num_heads:
            check_reuse_with_importance(
                query_states,
                self.metadata_tensors['cached_query'][layer_idx][:self.curr_batch_size],
                self.layers_gpu_head_mask[layer_idx],
                self.layers_q_importance[layer_idx],
                gather_mask.view(self.curr_batch_size,
                                self.num_key_value_heads),
                self.layers_reuse_thresholds[layer_idx],
                self.metadata_tensors[f'query_cache_valid_{query_states.device.index}'])
        else:
            check_reuse_head_threshold_with_gpu_head(
                query_states,
                self.metadata_tensors['cached_query'][layer_idx][:self.curr_batch_size],
                self.layers_gpu_head_mask[layer_idx],
                gather_mask.view(self.curr_batch_size,
                                self.num_key_value_heads),
                self.layers_reuse_thresholds[layer_idx],
                self.metadata_tensors[f'query_cache_valid_{query_states.device.index}'])
        torch.cuda.nvtx.range_pop()

        return gather_mask

    def append_cpu_cache_decode_and_wait(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int
    ):
        torch.cuda.nvtx.range_push("append and wait data")
        print(f'[jhe] before decode append wait layer {layer_idx}')
        KVLib.decode_append_offload_tensor_pos_wait(
            key_states,
            value_states,
            self.gpu_kv_buffers[layer_idx],
            self.cpu_kv_caches[layer_idx],
            self.cache_tensors['gpu_buffer_ptr'][layer_idx],
            self.cache_tensors['cpu_cache_length'][layer_idx],
            self.metadata_tensors['ready_flag'][layer_idx],
            self.layers_cpu_head_ids[layer_idx],
        )
        print(f'[jhe] after decode append wait layer {layer_idx}')
        self.cache_tensors['cpu_cache_length'][layer_idx] += 1
        self.cache_tensors['gpu_buffer_ptr'][layer_idx] += 1
        torch.cuda.nvtx.range_pop()

    def launch_prefetch(self, indices: torch.Tensor,
                        head_mask: torch.Tensor,
                        prefetch_layer_idx: int):
        torch.cuda.nvtx.range_push("real indices")
        device_idx = indices.device.index
        KVLib.static_launch_prefetch(
            indices,
            head_mask,
            self.metadata_tensors[f'topk_prefetch_k_{device_idx}'],
            self.cpu_indices_buffer,
            self.metadata_tensors['gather_engine_metadata'],
            self.metadata_tensors['ready_flag'][prefetch_layer_idx],
            self.curr_batch_size,
            self.max_seq_len,
            self.num_key_value_heads,
            prefetch_layer_idx,
        )
        torch.cuda.nvtx.range_pop()

    def compute_topk(self, query, layer_idx, mask, is_prefetch=False):
        raise NotImplementedError

    def append_decode(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        prefetch_query_states: Optional[torch.Tensor] = None,
        current_query_states: Optional[torch.Tensor] = None,
    ):
        key_states = key_states.view(self.curr_batch_size, 1,
                                     self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(self.curr_batch_size, 1,
                                     self.num_key_value_heads, self.head_dim)

        # 1. gpu kvcache append (checked)
        if self.layers_gpu_head_ids[layer_idx].numel() > 0:
            self.append_gpu_cache_decode(key_states, value_states,
                                            layer_idx)

        # 2. gpu topk cache append and process query (checked)
        prefetch_query, current_query = self.append_topk_cache_decode(
            key_states, value_states, layer_idx, prefetch_query_states,
            current_query_states)

        # 3. compute prefetch top-k indices (checked)
        next_layer_idx = (layer_idx + 1) % self.num_layers
        if not self.layers_full_gpu_mask[next_layer_idx]:
            gather_mask = self.cache_query_and_update(
                prefetch_query_states, next_layer_idx)
            prefetch_topk_indices = self.compute_topk(
                prefetch_query,
                next_layer_idx,
                gather_mask,
                is_prefetch=True,
            )

        # 4. append cpu kvcache, gpu recent buffer, and sync prefetch (checked)
        if not self.layers_full_gpu_mask[layer_idx]:
            torch.cuda.nvtx.range_push(f"append cpu and wait layer{layer_idx}")
            self.append_cpu_cache_decode_and_wait(key_states, value_states, layer_idx)
            torch.cuda.nvtx.range_pop()

        # 5. launch prefetching (checked)
        if not self.layers_full_gpu_mask[next_layer_idx]:
            torch.cuda.nvtx.range_push("real indices")
            self.launch_prefetch(prefetch_topk_indices, gather_mask,
                                 next_layer_idx)
            torch.cuda.nvtx.range_pop()

        # 6. compute top-k for current layer (checked)
        self.current_topk_indices = None
        if self.layers_gpu_head_ids[layer_idx].numel() > 0:
            self.current_topk_indices = self.compute_topk(
                current_query,
                layer_idx,
                self.layers_gpu_bh_mask[layer_idx][:self.curr_batch_size *
                                                   self.num_key_value_heads],
                is_prefetch=False
            )

    def get_attention_data(self, layer_idx: int, device_idx: int):
        if self.layers_gpu_head_ids[layer_idx].numel() > 0:
            kcache = self.kv_caches[layer_idx][0]
            vcache = self.kv_caches[layer_idx][1]
            topk_index = self.current_topk_indices
            topk_count = self.metadata_tensors[f'topk_current_k_{device_idx}']
        else:
            kcache = None
            vcache = None
            topk_index = None
            topk_count = None

        if self.layers_cpu_head_ids[layer_idx].numel() > 0:
            kbuffer = self.gpu_kv_buffers[layer_idx][0]
            vbuffer = self.gpu_kv_buffers[layer_idx][1]
            buffer_count = self.metadata_tensors[f'topk_current_k_{device_idx}']
        else:
            kbuffer = None
            vbuffer = None
            buffer_count = None

        mask = self.layers_gpu_head_mask[layer_idx]
        mixed_head_ids = self.layers_mixed_head_index[layer_idx]

        return (
            kcache,
            vcache,
            topk_index,
            topk_count,
            kbuffer,
            vbuffer,
            buffer_count,
            mask,
            mixed_head_ids,
        )

