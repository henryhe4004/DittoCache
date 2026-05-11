from typing import Dict, Optional, Union, Any
import os
import csv
import math
import time
import torch
import logging
import pandas as pd

from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig

from .kvcache_full_attn import CustomStaticCache
from .transfer_stats import (
    record_decode_transfer_step,
    reset_transfer_stats,
    set_transfer_stats_enabled,
    transfer_stats_enabled,
)
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

logger = logging.getLogger(__name__)


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
        self.selective_start_len = max(
            int(getattr(self.config.sparse_attention_config, "selective_start_len", 0)),
            0,
        )
        # Max consecutive reuse count before forcing one gather refresh for each KV head.
        # <=0 disables this forcing path entirely.
        raw_max_reuse_count = int(
            getattr(self.config.offload_config, "max_reuse_count", 0)
        )
        self.max_reuse_count = 0 if raw_max_reuse_count <= 0 else raw_max_reuse_count

        # Debug switch for diagnosing potential CPUGather stalls.
        self.debug_cpugather = os.environ.get("LITECACHE_DEBUG_CPUGATHER", "0") == "1"
        # Coarse-grained stage tracing for timeout diagnosis.
        self.debug_stall = os.environ.get("LITECACHE_DEBUG_STALL", "0") == "1"
        self.debug_stall_min_ms = float(
            os.environ.get("LITECACHE_DEBUG_STALL_MIN_MS", "0")
        )
        self.debug_batch = os.environ.get("LITECACHE_DEBUG_BATCH", "0") == "1"
        self._warned_ragged_decode_batch = False
        # Safety switch: bypass prefetch kernel launch to keep system runnable.
        self.disable_prefetch = os.environ.get("LITECACHE_DISABLE_PREFETCH", "0") == "1"
        # Keep a host mirror of prefetch-k to avoid CUDA tensor .item() during graph capture.
        self.topk_prefetch_k_host = 0
        self.topk_current_k_host = 0
        self.topk_prefetch_k_host_per_row = []
        self.topk_current_k_host_per_row = []
        max_batch = int(self.config.kvcache_manager_config.max_batch_size)
        self.cache_length_host = [[0] * max_batch for _ in range(self.num_layers)]
        self.cpu_cache_length_host = [[0] * max_batch for _ in range(self.num_layers)]
        self._decode_transfer_step_idx = 0
        self.record_overlap_stats = (
            os.environ.get("LITECACHE_RECORD_OVERLAP_STATS", "0") == "1"
        )
        self.record_transfer_stats = (
            os.environ.get("LITECACHE_RECORD_TRANSFER_STATS", "0") == "1"
        ) or self.record_overlap_stats
        set_transfer_stats_enabled(self.record_transfer_stats)
        self._pending_overlap_prefetch = [None for _ in range(self.num_layers)]
        self._pending_overlap_head_mask = [None for _ in range(self.num_layers)]
        self._decode_overlap_metrics = [None for _ in range(self.num_layers)]

    def _plan_topk_used_gpu_memory(self):
        raise NotImplementedError

    def _resolve_head_threshold_csv_path(self) -> str | None:
        explicit = os.environ.get("LITECACHE_HEAD_THRESHOLD_CSV_FILE")
        if explicit:
            out_path = explicit
        else:
            csv_path_env = os.environ.get("LITECACHE_TRANSFER_STATS_CSV_FILE")
            if csv_path_env:
                root, ext = os.path.splitext(csv_path_env)
                if ext.lower() == ".csv":
                    out_path = f"{root}_head_thresholds.csv"
                else:
                    out_path = f"{csv_path_env}_head_thresholds.csv"
            else:
                json_path_env = os.environ.get("LITECACHE_TRANSFER_STATS_FILE")
                if json_path_env:
                    root, ext = os.path.splitext(json_path_env)
                    if ext.lower() in {".json", ".jsonl"}:
                        out_path = f"{root}_head_thresholds.csv"
                    else:
                        out_path = f"{json_path_env}_head_thresholds.csv"
                else:
                    return None

        if self.attn_tp_size > 1:
            root, ext = os.path.splitext(out_path)
            return f"{root}.tp{self.attn_tp_rank:02d}{ext or '.csv'}"
        return out_path

    def _write_head_thresholds_csv(
        self,
        stacked_reuse_thresholds: torch.Tensor,
        head_cos: torch.Tensor,
        hard2reuse_head_mask: torch.Tensor,
        reuse_difficulty: torch.Tensor,
    ) -> None:
        out_path = self._resolve_head_threshold_csv_path()
        if not out_path:
            return

        rows = []
        thresholds_cpu = stacked_reuse_thresholds.detach().cpu()
        cos_cpu = head_cos.detach().cpu()
        hard_cpu = hard2reuse_head_mask.detach().cpu()
        diff_cpu = reuse_difficulty.detach().cpu()

        for layer_idx in range(self.num_layers):
            gpu_mask = self.layers_gpu_head_mask[layer_idx].detach().cpu().to(torch.bool)
            for head_idx in range(self.num_key_value_heads):
                global_head_idx = int(self.kv_head_start + head_idx)
                rows.append(
                    {
                        "layer_idx": int(layer_idx),
                        "head_idx": global_head_idx,
                        "reuse_threshold": float(thresholds_cpu[layer_idx, head_idx].item()),
                        "head_cosine": float(cos_cpu[layer_idx, head_idx].item()),
                        "reuse_difficulty": float(diff_cpu[layer_idx, head_idx].item()),
                        "hard_to_reuse": int(bool(hard_cpu[layer_idx, head_idx].item())),
                        "placed_on_gpu": int(bool(gpu_mask[head_idx].item())),
                        "offloaded_to_cpu": int(not bool(gpu_mask[head_idx].item())),
                    }
                )

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "layer_idx",
                    "head_idx",
                    "reuse_threshold",
                    "head_cosine",
                    "reuse_difficulty",
                    "hard_to_reuse",
                    "placed_on_gpu",
                    "offloaded_to_cpu",
                ],
            )
            writer.writeheader()
            if rows:
                writer.writerows(rows)

        print(f"Saved head thresholds CSV to: {out_path}")

    @property
    def _per_token_head_kv_bytes(self) -> int:
        return 2 * self.head_dim * self.dtype.itemsize

    def _debug_cpugather_log(self, stage: str, layer_idx: int, **kwargs):
        if not self.debug_cpugather:
            return
        ts = time.time()
        details = " ".join(f"{k}={v}" for k, v in kwargs.items())
        msg = f"[LiteCacheCPUGather] ts={ts:.6f} stage={stage} layer={layer_idx}"
        if details:
            msg += f" {details}"
        print(msg, flush=True)

    def _safe_stream_query(self, stream):
        # stream.query() is not permitted during CUDA graph capture.
        if not self.debug_cpugather:
            return None
        try:
            if torch.cuda.is_current_stream_capturing():
                return "capturing"
        except Exception:
            pass
        try:
            return stream.query()
        except Exception as exc:
            return f"query_err={type(exc).__name__}"

    def _debug_stall_log(self, stage: str, layer_idx: int, **kwargs):
        if not self.debug_stall:
            return
        ts = time.time()
        details = " ".join(f"{k}={v}" for k, v in kwargs.items())
        msg = f"[LiteCacheStall] ts={ts:.6f} stage={stage} layer={layer_idx}"
        if details:
            msg += f" {details}"
        print(msg, flush=True)

    def _debug_stall_duration(self, stage: str, layer_idx: int, t0: float, **kwargs):
        if not self.debug_stall:
            return
        ms = (time.perf_counter() - t0) * 1000.0
        if ms < self.debug_stall_min_ms:
            return
        self._debug_stall_log(stage, layer_idx, ms=f"{ms:.2f}", **kwargs)

    def _debug_ready_flag_snapshot(
        self,
        stage: str,
        layer_idx: int,
        total_heads: Optional[int] = None,
    ):
        if not self.debug_cpugather:
            return
        if total_heads is None:
            total_heads = self.curr_batch_size * self.num_key_value_heads
        ready = self.metadata_tensors["ready_flag"][layer_idx][:total_heads]
        ready_true = int(ready.sum().item())
        self._debug_cpugather_log(
            stage,
            layer_idx,
            ready_true=ready_true,
            ready_total=total_heads,
        )

    def _init_kv_placement(self):
        # Read head importance and cosine similarity data
        cosine_file = os.path.join(self.config.offload_config.attn_pattern_path, "heads_cosine_similarity.csv")
        k_head_importance_file = os.path.join(self.config.offload_config.attn_pattern_path, "k_heads_importance.tsv")
        q_head_importance_file = os.path.join(self.config.offload_config.attn_pattern_path, "q_heads_importance.tsv")

        head_importance = pd.read_csv(k_head_importance_file, sep='\t', header=None).to_numpy()
        q_head_importance = pd.read_csv(q_head_importance_file, sep='\t', header=None).to_numpy()
        head_cos = pd.read_csv(cosine_file, sep=',', header=None).to_numpy()

        # TP-aware slicing: auxiliary stats are often stored in global-head layout.
        # Convert them to this TP rank's local head range when needed.
        kv_cols = head_importance.shape[1]
        if kv_cols != self.num_key_value_heads and kv_cols == self.total_num_key_value_heads:
            kv_start = int(self.kv_head_start)
            kv_end = kv_start + int(self.num_key_value_heads)
            head_importance = head_importance[:, kv_start:kv_end]
            head_cos = head_cos[:, kv_start:kv_end]

        q_cols = q_head_importance.shape[1]
        if q_cols != self.num_heads:
            if q_cols == self.total_num_heads and self.total_num_key_value_heads > 0:
                # Preferred path: map Q heads by GQA groups aligned to KV head shards.
                if q_cols % self.total_num_key_value_heads != 0:
                    raise ValueError(
                        f"q_heads_importance columns={q_cols} not divisible by total_num_key_value_heads="
                        f"{self.total_num_key_value_heads}"
                    )
                gqa_group = q_cols // self.total_num_key_value_heads
                q_start = int(self.kv_head_start) * gqa_group
                q_end = q_start + int(self.num_key_value_heads) * gqa_group
                q_head_importance = q_head_importance[:, q_start:q_end]
            elif self.attn_tp_size > 1 and q_cols % self.attn_tp_size == 0:
                # Fallback: contiguous split by TP rank.
                local_q = q_cols // self.attn_tp_size
                q_start = self.attn_tp_rank * local_q
                q_end = q_start + local_q
                q_head_importance = q_head_importance[:, q_start:q_end]

        if head_importance.shape[1] != self.num_key_value_heads:
            raise ValueError(
                f"head_importance columns={head_importance.shape[1]} mismatch local num_key_value_heads="
                f"{self.num_key_value_heads} (total={self.total_num_key_value_heads}, tp={self.attn_tp_rank}/{self.attn_tp_size})"
            )
        if head_cos.shape[1] != self.num_key_value_heads:
            raise ValueError(
                f"head_cos columns={head_cos.shape[1]} mismatch local num_key_value_heads="
                f"{self.num_key_value_heads} (total={self.total_num_key_value_heads}, tp={self.attn_tp_rank}/{self.attn_tp_size})"
            )
        if q_head_importance.shape[1] != self.num_heads:
            raise ValueError(
                f"q_head_importance columns={q_head_importance.shape[1]} mismatch local num_heads={self.num_heads} "
                f"(total={self.total_num_heads}, tp={self.attn_tp_rank}/{self.attn_tp_size})"
            )

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
            reuse_thresholds = head_importance[l].clone()
            if USE_FIXED_THRESHOLDS:
                reuse_thresholds[:] = self.config.offload_config.reuse_threshold_upper
            else:
                # Keep threshold mapping identical to myTransformer offloading:
                # angle-space interpolation between lower/upper with decay_p.
                high = math.acos(self.config.offload_config.reuse_threshold_upper)
                low = math.acos(self.config.offload_config.reuse_threshold_lower)
                reuse_thresholds = torch.cos(
                    low + (high - low) *
                    reuse_thresholds**self.config.offload_config.decay_p)
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
            gpu_global_head_ids = self._local_kv_head_ids_to_global(gpu_head_ids).cpu().tolist()
            cpu_global_head_ids = self._local_kv_head_ids_to_global(cpu_head_ids).cpu().tolist()
            print(
                f"[TP{self.attn_tp_rank}] Layer {l:02d} on-GPU heads: {gpu_global_head_ids}, "
                f"offloaded heads: {cpu_global_head_ids}"
            )

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
        print(f"[TP{self.attn_tp_rank}] Total on-GPU heads number:", self.num_gpu_heads)
        self.num_cpu_layers = self.num_layers - self.num_full_gpu_layers
        self.num_cpu_heads = self.num_layers * self.num_key_value_heads - self.num_gpu_heads
        print(
            f"Cache and other layer's {self.num_gpu_heads} on-GPU heads consumed GPU memory: " \
            f"{remained_layers_mem / 1024**3:.2f} GB. " \
            f"{self.mem_budget / 1024**3:.2f} GB budget left.")

        self._write_head_thresholds_csv(
            stacked_reuse_thresholds=stacked_reuse_thresholds,
            head_cos=head_cos,
            hard2reuse_head_mask=hard2reuse_head_mask,
            reuse_difficulty=reuse_difficulty,
        )

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
            gpu_cache_length = torch.zeros((self.config.kvcache_manager_config.max_batch_size, ),
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
            cache_length = torch.zeros((self.config.kvcache_manager_config.max_batch_size, ),
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
            buffer_length = torch.zeros((self.config.kvcache_manager_config.max_batch_size, ),
                                        dtype=torch.int32,
                                        device=layer_device)
            buffer_ptr = torch.zeros((self.config.kvcache_manager_config.max_batch_size, ),
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
        # flag[6:] = per flattened KV head gather lengths for the current batch
        gather_meta_len = (
            6
            + self.config.kvcache_manager_config.max_batch_size
            * self.num_key_value_heads
        )
        self.metadata_tensors['gather_engine_metadata'] = torch.full((gather_meta_len, ),
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
        self.metadata_tensors['reuse_count'] = [None for _ in range(self.num_layers)]
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
            self.metadata_tensors['reuse_count'][l] = torch.zeros(
                (self.config.kvcache_manager_config.max_batch_size *
                 self.num_key_value_heads, ),
                dtype=torch.int32,
                device=self.layer_devices[l],
            )

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
                (self.config.kvcache_manager_config.max_batch_size, ),
                dtype=torch.int32,
                device=device_idx,
            )
            self.metadata_tensors[f'topk_current_k_{device_idx}'] = torch.zeros(
                (self.config.kvcache_manager_config.max_batch_size, ),
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
            self.cache_length_host[l] = [0] * self.config.kvcache_manager_config.max_batch_size

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
            self.cpu_cache_length_host[l] = [0] * self.config.kvcache_manager_config.max_batch_size

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
            gpu_buffer_ptr[:batch_size] = (
                self.config.sparse_attention_config.sink_budget
                + self.config.sparse_attention_config.recent_budget
            )

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

        if self.debug_batch:
            logger.info(
                "LiteCache reset: requested_batch=%d configured_max_batch=%d "
                "max_tokens=%d max_seq_len=%d max_sparse_tokens=%d max_buffer_len=%d",
                batch_size,
                self.config.kvcache_manager_config.max_batch_size,
                self.config.kvcache_manager_config.max_tokens,
                self.max_seq_len,
                self.max_sparse_tokens,
                self.max_buffer_len,
            )

        self._reset_cache_tensors(batch_size)
        self._reset_offload_tensors(batch_size)
        self._reset_topk_tensors(batch_size)

        self.metadata_tensors['gather_engine_metadata'][0:2] = -1
        self.metadata_tensors['gather_engine_metadata'][2] = batch_size
        self.metadata_tensors['gather_engine_metadata'][3] = self.max_seq_len
        self.metadata_tensors['gather_engine_metadata'][4] = self.max_buffer_len
        self.metadata_tensors['gather_engine_metadata'][5] = self.max_prefetch_topk_len
        self.metadata_tensors['gather_engine_metadata'][6:].zero_()

        self.metadata_tensors['cpu_indices_data'].fill_(-1)
        self.cpu_indices_buffer = self.metadata_tensors['cpu_indices_data'][:batch_size * 
            self.max_prefetch_topk_len * self.num_key_value_heads
        ].view(
            batch_size * self.num_key_value_heads,
            self.max_prefetch_topk_len,
        )

        for l in range(self.num_layers):
            # Reset must mark prefetch buffers as not-ready.
            # myTransformer decode kernels rely on this invariant.
            self.metadata_tensors['ready_flag'][l][:] = False
            self.metadata_tensors['gather_mask'][l][:] = True
            self.metadata_tensors['reuse_count'][l][:] = 0
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
        self._pending_overlap_prefetch = [None for _ in range(self.num_layers)]
        self._pending_overlap_head_mask = [None for _ in range(self.num_layers)]
        self._decode_overlap_metrics = [None for _ in range(self.num_layers)]

    def reset_batch_rows(self, row_indices: list[int]) -> None:
        if not row_indices:
            return
        rows = torch.tensor(row_indices, dtype=torch.long)
        sink_recent = (
            self.config.sparse_attention_config.sink_budget
            + self.config.sparse_attention_config.recent_budget
        )
        for l in range(self.num_layers):
            row_device = self.cache_tensors['cpu_cache_length'][l].device
            layer_rows = rows.to(device=row_device, non_blocking=True)
            if self.cache_tensors['cache_length'][l] is not None:
                self.cache_tensors['cache_length'][l][layer_rows] = 0
            self.cache_tensors['cpu_cache_length'][l][layer_rows] = 0
            if self.cache_tensors['gpu_buffer_ptr'][l] is not None:
                self.cache_tensors['gpu_buffer_ptr'][l][layer_rows] = sink_recent
            if self.cache_tensors['gpu_buffer_length'][l] is not None:
                self.cache_tensors['gpu_buffer_length'][l][layer_rows] = 0
            for row in row_indices:
                self.cache_length_host[l][row] = 0
                self.cpu_cache_length_host[l][row] = 0
                start = row * self.num_key_value_heads
                end = start + self.num_key_value_heads
                self.metadata_tensors['ready_flag'][l][start:end] = False
                self.metadata_tensors['gather_mask'][l][start:end] = True
                self.metadata_tensors['reuse_count'][l][start:end] = 0
                self.metadata_tensors['cached_query'][l][row].zero_()

    def move_batch_rows(self, old_to_new_rows: dict[int, int]) -> None:
        if not old_to_new_rows:
            return

        max_batch = int(self.config.kvcache_manager_config.max_batch_size)
        normalized = {
            int(old): int(new)
            for old, new in old_to_new_rows.items()
            if int(old) != int(new)
        }
        if not normalized:
            return
        bad_rows = [
            row
            for pair in normalized.items()
            for row in pair
            if row < 0 or row >= max_batch
        ]
        if bad_rows:
            raise RuntimeError(
                f"LiteCache batch-row remap out of range: {normalized}, "
                f"max_batch_size={max_batch}"
            )

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

        def _index_copy_head_rows(tensor: torch.Tensor | None) -> None:
            if tensor is None:
                return
            old_indices: list[int] = []
            new_indices: list[int] = []
            for old, new in normalized.items():
                old_indices.extend(
                    range(
                        old * self.num_key_value_heads,
                        (old + 1) * self.num_key_value_heads,
                    )
                )
                new_indices.extend(
                    range(
                        new * self.num_key_value_heads,
                        (new + 1) * self.num_key_value_heads,
                    )
                )
            old_tensor = torch.tensor(
                old_indices, dtype=torch.long, device=tensor.device
            )
            new_tensor = torch.tensor(
                new_indices, dtype=torch.long, device=tensor.device
            )
            src = tensor.index_select(0, old_tensor).clone()
            tensor.index_copy_(0, new_tensor, src)

        def _copy_kv_cache_rows(
            tensor: torch.Tensor | None,
            row_lengths: list[int],
        ) -> None:
            if tensor is None:
                return
            snapshots = []
            seq_cap = int(tensor.size(2))
            for old, new in normalized.items():
                valid_len = max(0, min(int(row_lengths[old]), seq_cap))
                if valid_len == 0:
                    continue
                snapshots.append(
                    (
                        new,
                        valid_len,
                        tensor[:, old : old + 1, :valid_len].clone(),
                    )
                )
            for new, valid_len, src in snapshots:
                tensor[:, new : new + 1, :valid_len].copy_(src)

        for layer_idx in range(self.num_layers):
            _index_copy_rows(self.cache_tensors['cache_length'][layer_idx], 0)
            _index_copy_rows(self.cache_tensors['cpu_cache_length'][layer_idx], 0)
            _index_copy_rows(self.cache_tensors['gpu_buffer_ptr'][layer_idx], 0)
            _index_copy_rows(self.cache_tensors['gpu_buffer_length'][layer_idx], 0)
            _copy_kv_cache_rows(
                self.kv_caches[layer_idx],
                self.cache_length_host[layer_idx],
            )
            _copy_kv_cache_rows(
                self.cpu_kv_caches[layer_idx],
                self.cpu_cache_length_host[layer_idx],
            )
            _index_copy_rows(self.gpu_kv_buffers[layer_idx], 1)
            _index_copy_rows(self.metadata_tensors['cached_query'][layer_idx], 0)
            _index_copy_head_rows(self.metadata_tensors['ready_flag'][layer_idx])
            _index_copy_head_rows(self.metadata_tensors['gather_mask'][layer_idx])
            _index_copy_head_rows(self.metadata_tensors['reuse_count'][layer_idx])

            cache_lengths = [
                self.cache_length_host[layer_idx][old]
                for old in normalized.keys()
            ]
            cpu_cache_lengths = [
                self.cpu_cache_length_host[layer_idx][old]
                for old in normalized.keys()
            ]
            for new, value in zip(normalized.values(), cache_lengths):
                self.cache_length_host[layer_idx][new] = int(value)
            for new, value in zip(normalized.values(), cpu_cache_lengths):
                self.cpu_cache_length_host[layer_idx][new] = int(value)

    def trim_prefill_padding(self, extend_seq_lens: list[int], padded_q_len: int) -> None:
        if not extend_seq_lens or all(int(x) == int(padded_q_len) for x in extend_seq_lens):
            return
        if len(extend_seq_lens) != self.curr_batch_size:
            raise RuntimeError(
                "LiteCache trim_prefill_padding length mismatch: "
                f"extend_seq_lens={extend_seq_lens}, "
                f"curr_batch_size={self.curr_batch_size}."
            )
        trims = torch.tensor(
            [int(padded_q_len) - int(x) for x in extend_seq_lens],
            dtype=torch.int32,
        )
        sink_recent = (
            self.config.sparse_attention_config.sink_budget
            + self.config.sparse_attention_config.recent_budget
        )

        def _align_for_gpu_buffer(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
            if src.shape == dst.shape:
                return src
            if (
                src.ndim == 4
                and src.shape[0] == dst.shape[0]
                and src.shape[1] == dst.shape[2]
                and src.shape[2] == dst.shape[1]
                and src.shape[3] == dst.shape[3]
            ):
                return src.transpose(1, 2).contiguous()
            raise RuntimeError(
                "LiteCache trim_prefill_padding buffer shape mismatch: "
                f"src={tuple(src.shape)} dst={tuple(dst.shape)}"
            )

        for layer_idx in range(self.num_layers):
            for name, host_lengths in (
                ("cache_length", self.cache_length_host),
                ("cpu_cache_length", self.cpu_cache_length_host),
            ):
                length_tensor = self.cache_tensors.get(name, [None])[layer_idx]
                if length_tensor is None:
                    continue
                active = length_tensor[:self.curr_batch_size]
                layer_trims = trims.to(device=active.device, non_blocking=True)
                active.sub_(layer_trims)
                for row, trim in enumerate(trims.tolist()):
                    host_lengths[layer_idx][row] -= int(trim)

            if self.layers_full_gpu_mask[layer_idx]:
                continue

            cpu_head_ids = self.layers_cpu_head_ids[layer_idx]
            if cpu_head_ids.numel() == 0:
                continue
            cpu_head_ids_cpu = cpu_head_ids.cpu()
            for row in range(self.curr_batch_size):
                new_len = int(self.cpu_cache_length_host[layer_idx][row])
                sink = min(self.config.sparse_attention_config.sink_budget, new_len)
                recent = min(
                    self.config.sparse_attention_config.recent_budget,
                    max(new_len - sink, 0),
                )
                self.cache_tensors['gpu_buffer_ptr'][layer_idx][row] = sink_recent
                if self.cache_tensors['gpu_buffer_length'][layer_idx] is not None:
                    self.cache_tensors['gpu_buffer_length'][layer_idx][row] = 0
                if sink > 0:
                    sink_key_states = self.cpu_kv_caches[layer_idx][
                        0, row:row + 1, :sink, cpu_head_ids_cpu, :
                    ]
                    sink_value_states = self.cpu_kv_caches[layer_idx][
                        1, row:row + 1, :sink, cpu_head_ids_cpu, :
                    ]
                    sink_key_target = self.gpu_kv_buffers[layer_idx][
                        0, row:row + 1, :sink, :, :
                    ]
                    sink_value_target = self.gpu_kv_buffers[layer_idx][
                        1, row:row + 1, :sink, :, :
                    ]
                    sink_key_target.copy_(
                        _align_for_gpu_buffer(sink_key_states, sink_key_target)
                    )
                    sink_value_target.copy_(
                        _align_for_gpu_buffer(sink_value_states, sink_value_target)
                    )
                if recent > 0:
                    recent_start = new_len - recent
                    recent_key_states = self.cpu_kv_caches[layer_idx][
                        0, row:row + 1, recent_start:new_len, cpu_head_ids_cpu, :
                    ]
                    recent_value_states = self.cpu_kv_caches[layer_idx][
                        1, row:row + 1, recent_start:new_len, cpu_head_ids_cpu, :
                    ]
                    recent_key_target = self.gpu_kv_buffers[layer_idx][
                        0, row:row + 1, sink:sink + recent, :, :
                    ]
                    recent_value_target = self.gpu_kv_buffers[layer_idx][
                        1, row:row + 1, sink:sink + recent, :, :
                    ]
                    recent_key_target.copy_(
                        _align_for_gpu_buffer(recent_key_states, recent_key_target)
                    )
                    recent_value_target.copy_(
                        _align_for_gpu_buffer(recent_value_states, recent_value_target)
                    )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        active = slice(0, self.curr_batch_size)
        if self.layers_gpu_head_ids[layer_idx].numel() > 0:
            return int(min(self.cache_length_host[layer_idx][active]))
        else:
            return int(min(self.cpu_cache_length_host[layer_idx][active]))

    def _active_cache_length_tensor(self, name: str, layer_idx: int) -> torch.Tensor:
        return self.cache_tensors[name][layer_idx][:self.curr_batch_size]

    def _set_active_host_lengths(self, host_lengths, layer_idx: int, value: int) -> None:
        for idx in range(self.curr_batch_size):
            host_lengths[layer_idx][idx] = int(value)

    def _set_active_host_lengths_from_tensor(self, host_lengths, layer_idx: int, values: torch.Tensor) -> None:
        for idx, value in enumerate(values.detach().cpu().tolist()):
            host_lengths[layer_idx][idx] = int(value)

    def _increment_active_host_lengths(self, host_lengths, layer_idx: int, delta: int = 1) -> None:
        for idx in range(self.curr_batch_size):
            host_lengths[layer_idx][idx] += int(delta)

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
            # CPUGatherEngineV3 pybind requires list[Tensor].
            # Keep tensor refs on self so C++ side pointers remain valid.
            self.cpu_gather_gpu_buffers = []
            for layer_idx, gpu_buffer_data in enumerate(
                    self.cache_tensors['gpu_buffer_data']):
                if gpu_buffer_data is None:
                    # Use 0-sized tensor so C++ wrapper converts it to nullopt
                    # and skips GDR pin/map for layers without offload buffers.
                    gpu_buffer_data = torch.empty(
                        (0, ),
                        dtype=self.dtype,
                        device=self.layer_devices[layer_idx],
                    )
                self.cpu_gather_gpu_buffers.append(gpu_buffer_data)

            print(
                f"Set {self.config.offload_config.num_omp_threads} omp threads for CPUGatherEngine")
            self.cpu_gather_engine = KVLib.CPUGatherEngineV3(
                self.config.offload_config.num_omp_threads,
                self.cache_tensors['cpu_cache_data'],
                self.cpu_gather_gpu_buffers,
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
        if self.layers_gpu_head_ids[layer_idx].numel() > 0:
            active_seq_lens = self.cache_length_host[layer_idx][
                : self.curr_batch_size
            ]
        else:
            active_seq_lens = self.cpu_cache_length_host[layer_idx][
                : self.curr_batch_size
            ]
        for device_idx in self.unique_devices:
            rope_device = self.metadata_tensors[f"rope_offsets_{device_idx}"].device
            if self.layers_gpu_head_ids[layer_idx].numel() > 0:
                rope_offsets = self._active_cache_length_tensor(
                    "cache_length", layer_idx
                ).to(device=rope_device, non_blocking=True)
            else:
                rope_offsets = self._active_cache_length_tensor(
                    "cpu_cache_length", layer_idx
                ).to(device=rope_device, non_blocking=True)
            self.metadata_tensors[f"rope_offsets_{device_idx}"][
                : self.curr_batch_size
            ] = rope_offsets
        if is_prefill:
            self.first_decode_layer_step = True
            self._decode_transfer_step_idx = 0
            if self.record_transfer_stats:
                reset_transfer_stats()
            if self.debug_batch and layer_idx == 0:
                logger.info(
                    "LiteCache metadata prefill: batch=%d q_len=%d seq_lens=%s",
                    self.curr_batch_size,
                    q_len,
                    active_seq_lens,
                )
        else:
            active_seq_lens = [int(x) for x in active_seq_lens]
            topk_prefetch_ks = []
            topk_current_ks = []
            for curr_seqlen in active_seq_lens:
                # Clamp sink/recent by row sequence length to avoid invalid
                # budgets on short prompts (e.g. seq_len < sink+recent).
                sink = min(self.config.sparse_attention_config.sink_budget, curr_seqlen)
                recent = min(
                    self.config.sparse_attention_config.recent_budget,
                    max(curr_seqlen - sink, 0),
                )
                keep_budget = min(curr_seqlen, sink + recent + 1)
                selective_start_len = min(self.selective_start_len, curr_seqlen)
                if curr_seqlen <= selective_start_len:
                    raw_k = curr_seqlen
                    topk_prefetch_k = raw_k - keep_budget
                else:
                    extra_len = curr_seqlen - selective_start_len
                    if self.topk_ratio < 1:
                        raw_k = selective_start_len + int(extra_len * self.topk_ratio)
                    else:
                        raw_k = selective_start_len + int(self.topk_ratio)
                    raw_k = max(raw_k, selective_start_len)
                    topk_prefetch_k = raw_k - keep_budget

                topk_current_k = raw_k
                topk_prefetch_k = max(
                    min(topk_prefetch_k, self.max_prefetch_topk_len), 0
                )
                topk_current_k = max(
                    min(topk_current_k, self.max_current_topk_len), keep_budget
                )
                topk_current_k = min(topk_current_k, curr_seqlen)
                topk_prefetch_ks.append(int(topk_prefetch_k))
                topk_current_ks.append(int(topk_current_k))

            self.topk_prefetch_k_host_per_row = topk_prefetch_ks
            self.topk_current_k_host_per_row = topk_current_ks
            self.topk_prefetch_k_host = max(topk_prefetch_ks, default=0)
            self.topk_current_k_host = max(topk_current_ks, default=0)
            if self.debug_batch and layer_idx == 0:
                logger.info(
                    "LiteCache metadata decode: batch=%d q_len=%d seq_lens=%s "
                    "prefetch_k=%s current_k=%s max_prefetch_k=%d max_current_k=%d",
                    self.curr_batch_size,
                    q_len,
                    active_seq_lens,
                    topk_prefetch_ks,
                    topk_current_ks,
                    self.topk_prefetch_k_host,
                    self.topk_current_k_host,
                )

            is_first_decode_layer_step = self.first_decode_layer_step
            for device_idx in self.unique_devices:
                self.metadata_tensors[f'query_cache_valid_{device_idx}'].fill_(
                    not is_first_decode_layer_step
                )
                prefetch_tensor = torch.tensor(
                    topk_prefetch_ks,
                    dtype=torch.int32,
                    device=device_idx,
                )
                current_tensor = torch.tensor(
                    topk_current_ks,
                    dtype=torch.int32,
                    device=device_idx,
                )
                self.metadata_tensors[f'topk_prefetch_k_{device_idx}'][
                    :self.curr_batch_size
                ].copy_(prefetch_tensor)
                self.metadata_tensors[f'topk_current_k_{device_idx}'][
                    :self.curr_batch_size
                ].copy_(current_tensor)
            self.first_decode_layer_step = False

            for l in range(self.num_layers):
                num_cpu_heads = self.layers_cpu_head_ids[l].numel()
                if num_cpu_heads == 0:
                    continue
                buffer_append_ptr = self.cache_tensors['gpu_buffer_ptr'][l]
                limit = self.config.sparse_attention_config.sink_budget + self.config.sparse_attention_config.recent_budget + 1
                active_ptr = buffer_append_ptr[:self.curr_batch_size]
                active_ptr.copy_(
                    torch.where(
                        active_ptr >= limit,
                        torch.full_like(
                            active_ptr,
                            self.config.sparse_attention_config.sink_budget,
                        ),
                        active_ptr,
                    )
                )

        # print("\nupdate metadata", q_len)
        # print("cache_length", self.cache_tensors['cache_length'])
        # print("cpu_cache_length", self.cache_tensors['cpu_cache_length'])
        # print("topk_code_length", self.cache_tensors['topk_code_length'])
        # print("gpu_buffer_ptr", self.cache_tensors['gpu_buffer_ptr'])
        # print("query_cache_valid", self.metadata_tensors[f'query_cache_valid_{self.layer_devices[0]}'])
        # print("topk_prefetch_k", self.metadata_tensors[f'topk_prefetch_k_{self.layer_devices[0]}'])
        # print("topk_current_k", self.metadata_tensors[f'topk_current_k_{self.layer_devices[0]}'])

    def record_decode_transfer_step(self) -> None:
        if not transfer_stats_enabled():
            return

        total_heads = self.curr_batch_size * self.num_key_value_heads
        per_token_head_bytes = self._per_token_head_kv_bytes
        prefetch_k = max(int(self.topk_prefetch_k_host), 0)
        layer_h2d_bytes = [0] * self.num_layers
        layer_d2h_bytes = [0] * self.num_layers
        layer_prefetch_heads = [0] * self.num_layers
        layer_offloaded_heads = [0] * self.num_layers

        for layer_idx in range(self.num_layers):
            cpu_heads = int(self.layers_cpu_head_ids[layer_idx].numel())
            if cpu_heads <= 0:
                continue

            offloaded_heads = self.curr_batch_size * cpu_heads
            layer_offloaded_heads[layer_idx] = offloaded_heads
            layer_d2h_bytes[layer_idx] = offloaded_heads * per_token_head_bytes

            if prefetch_k <= 0 or self.disable_prefetch:
                continue

            gather_mask = self.metadata_tensors["gather_mask"][layer_idx][:total_heads]
            gather_heads = int(gather_mask.to(torch.int32).sum().item())
            layer_prefetch_heads[layer_idx] = gather_heads
            layer_h2d_bytes[layer_idx] = gather_heads * prefetch_k * per_token_head_bytes

        overlap_summary = self._summarize_decode_overlap_step()
        layer_selected_tokens = [0] * self.num_layers
        layer_recalled_tokens = [0] * self.num_layers
        layer_hit_tokens = [0] * self.num_layers
        layer_hit_rates = [0.0] * self.num_layers
        layer_overlap_sum_real_tokens = [0] * self.num_layers
        layer_overlap_sum_prefetch_tokens = [0] * self.num_layers
        layer_overlap_sum_hit_tokens = [0] * self.num_layers
        layer_overlap_recall_tokens = [0] * self.num_layers
        layer_overlap_prefetch_h2d_bytes_est = [0] * self.num_layers
        layer_overlap_recall_h2d_bytes_est = [0] * self.num_layers
        hit_source = "none"
        if isinstance(overlap_summary, dict):
            for layer_metrics in overlap_summary.get("layers", []):
                layer_idx = int(layer_metrics.get("layer_idx", -1))
                if layer_idx < 0 or layer_idx >= self.num_layers:
                    continue
                selected_tokens = int(layer_metrics.get("union_real_count", 0))
                hit_tokens = int(layer_metrics.get("union_intersection_count", 0))
                selected_tokens = max(selected_tokens, 0)
                hit_tokens = max(min(hit_tokens, selected_tokens), 0)
                recalled_tokens = max(selected_tokens - hit_tokens, 0)
                layer_selected_tokens[layer_idx] = selected_tokens
                layer_recalled_tokens[layer_idx] = recalled_tokens
                layer_hit_tokens[layer_idx] = hit_tokens
                layer_hit_rates[layer_idx] = (
                    float(hit_tokens / selected_tokens) if selected_tokens > 0 else 0.0
                )
                sum_real_tokens = int(layer_metrics.get("sum_real_count", selected_tokens))
                sum_prefetch_tokens = int(layer_metrics.get("sum_prefetch_count", 0))
                sum_hit_tokens = int(layer_metrics.get("sum_intersection_count", hit_tokens))
                sum_real_tokens = max(sum_real_tokens, 0)
                sum_prefetch_tokens = max(sum_prefetch_tokens, 0)
                sum_hit_tokens = max(min(sum_hit_tokens, sum_real_tokens), 0)
                recall_tokens = max(sum_real_tokens - sum_hit_tokens, 0)
                layer_overlap_sum_real_tokens[layer_idx] = sum_real_tokens
                layer_overlap_sum_prefetch_tokens[layer_idx] = sum_prefetch_tokens
                layer_overlap_sum_hit_tokens[layer_idx] = sum_hit_tokens
                layer_overlap_recall_tokens[layer_idx] = recall_tokens
                layer_overlap_prefetch_h2d_bytes_est[layer_idx] = (
                    sum_prefetch_tokens * per_token_head_bytes
                )
                layer_overlap_recall_h2d_bytes_est[layer_idx] = (
                    recall_tokens * per_token_head_bytes
                )
            hit_source = "overlap_union"

        selected_tokens = int(sum(layer_selected_tokens))
        recalled_tokens = int(sum(layer_recalled_tokens))
        hit_tokens = int(sum(layer_hit_tokens))
        hit_rate = float(hit_tokens / selected_tokens) if selected_tokens > 0 else 0.0

        self._decode_transfer_step_idx += 1
        step = {
            "step": int(self._decode_transfer_step_idx),
            "seq_len": int(self.get_seq_length(0)),
            "prefetch_k": int(prefetch_k),
            "h2d_bytes": int(sum(layer_h2d_bytes)),
            "d2h_bytes": int(sum(layer_d2h_bytes)),
            "total_bytes": int(sum(layer_h2d_bytes) + sum(layer_d2h_bytes)),
            "layer_h2d_bytes": [int(v) for v in layer_h2d_bytes],
            "layer_d2h_bytes": [int(v) for v in layer_d2h_bytes],
            "layer_prefetch_heads": [int(v) for v in layer_prefetch_heads],
            "layer_offloaded_heads": [int(v) for v in layer_offloaded_heads],
            "selected_tokens": int(selected_tokens),
            "recalled_tokens": int(recalled_tokens),
            "hit_tokens": int(hit_tokens),
            "hit_rate": float(hit_rate),
            "layer_selected_tokens": [int(v) for v in layer_selected_tokens],
            "layer_recalled_tokens": [int(v) for v in layer_recalled_tokens],
            "layer_hit_tokens": [int(v) for v in layer_hit_tokens],
            "layer_hit_rates": [float(v) for v in layer_hit_rates],
            "layer_overlap_sum_real_tokens": [int(v) for v in layer_overlap_sum_real_tokens],
            "layer_overlap_sum_prefetch_tokens": [
                int(v) for v in layer_overlap_sum_prefetch_tokens
            ],
            "layer_overlap_sum_hit_tokens": [int(v) for v in layer_overlap_sum_hit_tokens],
            "layer_overlap_recall_tokens": [int(v) for v in layer_overlap_recall_tokens],
            "layer_overlap_prefetch_h2d_bytes_est": [
                int(v) for v in layer_overlap_prefetch_h2d_bytes_est
            ],
            "layer_overlap_recall_h2d_bytes_est": [
                int(v) for v in layer_overlap_recall_h2d_bytes_est
            ],
            "overlap_prefetch_h2d_bytes_est": int(
                sum(layer_overlap_prefetch_h2d_bytes_est)
            ),
            "overlap_recall_h2d_bytes_est": int(
                sum(layer_overlap_recall_h2d_bytes_est)
            ),
            "hit_source": hit_source,
        }
        if overlap_summary is not None:
            step["overlap"] = overlap_summary
        record_decode_transfer_step(step)
        self._decode_overlap_metrics = [None for _ in range(self.num_layers)]

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

        prefill_old_lengths = None
        gpu_seq_offsets = None
        if self.layers_gpu_head_ids[layer_idx].numel() > 0:
            torch.cuda.nvtx.range_push("append gpu cache")
            seq_len_tensor = self._active_cache_length_tensor('cache_length', layer_idx)
            gpu_seq_offsets = seq_len_tensor.clone()
            prefill_old_lengths = gpu_seq_offsets.detach().cpu().tolist()
            new_gpu_seq_lens = gpu_seq_offsets + prefill_len
            max_new_gpu_seq_len = int(new_gpu_seq_lens.max().item())
            assert max_new_gpu_seq_len <= self.max_seq_len, \
                f"input kv states length {prefill_len} + max current seq length {int(gpu_seq_offsets.max().item())}, " \
                f"should be less than max_seq_len = {self.max_seq_len}"
            gpu_head_ids = self.layers_gpu_head_ids[layer_idx]
            if gpu_head_ids.numel() > 0 and int(gpu_head_ids.max().item()) >= key_states.shape[2]:
                raise RuntimeError(
                    f"LiteCache GPU head id out of range at layer={layer_idx}: "
                    f"head_ids={gpu_head_ids.detach().cpu().tolist()}, "
                    f"num_key_value_heads={key_states.shape[2]}"
                )
            for row in range(self.curr_batch_size):
                start = int(gpu_seq_offsets[row].item())
                end = start + prefill_len
                self.kv_caches[layer_idx][
                    0, row:row + 1, start:end, :, :
                ].copy_(key_states[row:row + 1, :, gpu_head_ids, :])
                self.kv_caches[layer_idx][
                    1, row:row + 1, start:end, :, :
                ].copy_(value_states[row:row + 1, :, gpu_head_ids, :])
            seq_len_tensor.copy_(new_gpu_seq_lens)
            self._set_active_host_lengths_from_tensor(
                self.cache_length_host, layer_idx, new_gpu_seq_lens
            )
            torch.cuda.nvtx.range_pop()

        if not self.layers_full_gpu_mask[layer_idx]:
            cpu_seq_len_tensor = self._active_cache_length_tensor(
                'cpu_cache_length', layer_idx
            )
            cpu_seq_offsets = cpu_seq_len_tensor.clone()
            if prefill_old_lengths is None:
                prefill_old_lengths = cpu_seq_offsets.detach().cpu().tolist()
            if gpu_seq_offsets is not None and not torch.equal(gpu_seq_offsets, cpu_seq_offsets):
                raise RuntimeError(
                    f"LiteCache GPU/CPU seq length mismatch at layer={layer_idx}: "
                    f"gpu={gpu_seq_offsets.detach().cpu().tolist()} "
                    f"cpu={cpu_seq_offsets.detach().cpu().tolist()}"
                )
            new_cpu_seq_lens = cpu_seq_offsets + prefill_len
            max_new_cpu_seq_len = int(new_cpu_seq_lens.max().item())
            assert max_new_cpu_seq_len <= self.max_seq_len, \
                f"input kv states length {prefill_len} + max current seq length {int(cpu_seq_offsets.max().item())}, " \
                f"should be less than max_seq_len = {self.max_seq_len}"

            with torch.cuda.stream(self.transfer_stream):
                for bsz in range(key_states.shape[0]):
                    start = int(cpu_seq_offsets[bsz].item())
                    end = start + prefill_len
                    # pytorch doesn't support cudaMemcpy2DAsync, so offload batch by batch
                    self.cpu_kv_caches[layer_idx][
                        0, bsz:bsz + 1, start:end,
                        ...].copy_(key_states[bsz:bsz + 1, ...], non_blocking=True)
                    self.cpu_kv_caches[layer_idx][
                        1, bsz:bsz + 1, start:end,
                        ...].copy_(value_states[bsz:bsz + 1, ...], non_blocking=True)
                self.transfer_event.record(self.transfer_stream)

            self.prefill_copy_buffer = (key_states, value_states)
            cpu_seq_len_tensor.copy_(new_cpu_seq_lens)
            self._set_active_host_lengths_from_tensor(
                self.cpu_cache_length_host, layer_idx, new_cpu_seq_lens
            )
            # Ensure D2H prefill copies are visible before rebuilding sink/recent windows.
            self.transfer_event.synchronize()

            # 2. save sink recent tokens (checked)
            torch.cuda.nvtx.range_push("append sink recent")
            cpu_head_ids = self.layers_cpu_head_ids[layer_idx]
            cpu_head_ids_cpu = cpu_head_ids.cpu()
            if cpu_head_ids.numel() > 0 and int(cpu_head_ids.max().item()) >= key_states.shape[2]:
                raise RuntimeError(
                    f"LiteCache CPU head id out of range at layer={layer_idx}: "
                    f"head_ids={cpu_head_ids.detach().cpu().tolist()}, "
                    f"num_key_value_heads={key_states.shape[2]}"
                )
            def _align_for_gpu_buffer(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
                if src.shape == dst.shape:
                    return src
                # Some runtime stacks return [B, H, T, D] for sparse head slices.
                # Normalize to the buffer layout [B, T, H, D].
                if (
                    src.ndim == 4
                    and src.shape[0] == dst.shape[0]
                    and src.shape[1] == dst.shape[2]
                    and src.shape[2] == dst.shape[1]
                    and src.shape[3] == dst.shape[3]
                ):
                    return src.transpose(1, 2).contiguous()
                raise RuntimeError(
                    f"LiteCache prefill shape mismatch at layer={layer_idx}: "
                    f"src={tuple(src.shape)} dst={tuple(dst.shape)}"
                )

            sink_recent = (
                self.config.sparse_attention_config.sink_budget
                + self.config.sparse_attention_config.recent_budget
            )
            for row in range(self.curr_batch_size):
                new_cpu_seq_len = int(new_cpu_seq_lens[row].item())
                sink = min(
                    self.config.sparse_attention_config.sink_budget,
                    new_cpu_seq_len,
                )
                recent = min(
                    self.config.sparse_attention_config.recent_budget,
                    max(new_cpu_seq_len - sink, 0),
                )
                self.cache_tensors['gpu_buffer_ptr'][layer_idx][row] = sink_recent
                if self.cache_tensors['gpu_buffer_length'][layer_idx] is not None:
                    self.cache_tensors['gpu_buffer_length'][layer_idx][row] = 0
                if sink > 0:
                    sink_key_states = self.cpu_kv_caches[layer_idx][
                        0, row:row + 1, :sink, cpu_head_ids_cpu, :
                    ]
                    sink_value_states = self.cpu_kv_caches[layer_idx][
                        1, row:row + 1, :sink, cpu_head_ids_cpu, :
                    ]
                    sink_key_target = self.gpu_kv_buffers[layer_idx][
                        0, row:row + 1, :sink, :, :
                    ]
                    sink_value_target = self.gpu_kv_buffers[layer_idx][
                        1, row:row + 1, :sink, :, :
                    ]
                    sink_key_target.copy_(
                        _align_for_gpu_buffer(sink_key_states, sink_key_target)
                    )
                    sink_value_target.copy_(
                        _align_for_gpu_buffer(sink_value_states, sink_value_target)
                    )
                if recent > 0:
                    recent_start = new_cpu_seq_len - recent
                    recent_key_states = self.cpu_kv_caches[layer_idx][
                        0, row:row + 1, recent_start:new_cpu_seq_len, cpu_head_ids_cpu, :
                    ]
                    recent_value_states = self.cpu_kv_caches[layer_idx][
                        1, row:row + 1, recent_start:new_cpu_seq_len, cpu_head_ids_cpu, :
                    ]
                    recent_key_target = self.gpu_kv_buffers[layer_idx][
                        0, row:row + 1, sink:sink + recent, :, :
                    ]
                    recent_value_target = self.gpu_kv_buffers[layer_idx][
                        1, row:row + 1, sink:sink + recent, :, :
                    ]
                    recent_key_target.copy_(
                        _align_for_gpu_buffer(recent_key_states, recent_key_target)
                    )
                    recent_value_target.copy_(
                        _align_for_gpu_buffer(recent_value_states, recent_value_target)
                    )
            torch.cuda.nvtx.range_pop()

        if not hasattr(self, "_last_prefill_old_lengths"):
            self._last_prefill_old_lengths = {}
        self._last_prefill_old_lengths[layer_idx] = [
            int(x) for x in (prefill_old_lengths or [0] * self.curr_batch_size)
        ]

        # For external chunked prefill, attention in this step must see all
        # prefix tokens accumulated so far.
        if self.layers_full_gpu_mask[layer_idx]:
            full_seq_len = int(
                self._active_cache_length_tensor("cache_length", layer_idx).max().item()
            )
            return (
                self.kv_caches[layer_idx][0, :self.curr_batch_size, :full_seq_len, :, :],
                self.kv_caches[layer_idx][1, :self.curr_batch_size, :full_seq_len, :, :],
            )

        full_seq_len = int(
            self._active_cache_length_tensor("cpu_cache_length", layer_idx).max().item()
        )
        if full_seq_len <= prefill_len:
            return key_states, value_states

        layer_device = key_states.device
        full_key_states = self.cpu_kv_caches[layer_idx][
            0, :self.curr_batch_size, :full_seq_len, :, :
        ].to(
            layer_device,
            non_blocking=True,
        )
        full_value_states = self.cpu_kv_caches[layer_idx][
            1, :self.curr_batch_size, :full_seq_len, :, :
        ].to(
            layer_device,
            non_blocking=True,
        )
        return full_key_states, full_value_states

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
                self._active_cache_length_tensor('cache_length', layer_idx))
        else:
            KVLib.kvcache_append_tensor_pos_head_sparse(
                self.kv_caches[layer_idx], key_states, value_states,
                self.layers_gpu_head_ids[layer_idx],
                self._active_cache_length_tensor('cache_length', layer_idx))
        self._active_cache_length_tensor('cache_length', layer_idx).add_(1)
        self._increment_active_host_lengths(self.cache_length_host, layer_idx)
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
        reuse_count = self.metadata_tensors['reuse_count'][layer_idx][
            :self.curr_batch_size * self.num_key_value_heads]

        torch.cuda.nvtx.range_push("check reuse")
        if USE_INTRA_GQA_AGGREGATION and self.num_key_value_heads < self.num_heads:
            check_reuse_with_importance(
                query_states,
                self.metadata_tensors['cached_query'][layer_idx][:self.curr_batch_size],
                self.layers_gpu_head_mask[layer_idx],
                self.layers_q_importance[layer_idx],
                reuse_count,
                gather_mask.view(self.curr_batch_size,
                                self.num_key_value_heads),
                self.layers_reuse_thresholds[layer_idx],
                self.metadata_tensors[f'query_cache_valid_{query_states.device.index}'],
                self.max_reuse_count)
        else:
            check_reuse_head_threshold_with_gpu_head(
                query_states,
                self.metadata_tensors['cached_query'][layer_idx][:self.curr_batch_size],
                self.layers_gpu_head_mask[layer_idx],
                reuse_count,
                gather_mask.view(self.curr_batch_size,
                                self.num_key_value_heads),
                self.layers_reuse_thresholds[layer_idx],
                self.metadata_tensors[f'query_cache_valid_{query_states.device.index}'],
                self.max_reuse_count)
        torch.cuda.nvtx.range_pop()

        return gather_mask

    def append_cpu_cache_decode_and_wait(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int
    ):
        torch.cuda.nvtx.range_push(f"decode_append_and_wait_layer{layer_idx}")
        total_heads = self.curr_batch_size * self.num_key_value_heads
        if self.debug_cpugather:
            ptr_before = int(
                self._active_cache_length_tensor('gpu_buffer_ptr', layer_idx)
                .min()
                .item()
            )
            cpu_len_before = int(
                self._active_cache_length_tensor('cpu_cache_length', layer_idx)
                .min()
                .item()
            )
            self._debug_cpugather_log(
                "wait_enter",
                layer_idx,
                ptr=ptr_before,
                cpu_len=cpu_len_before,
            )
        self._debug_ready_flag_snapshot("wait_ready_before", layer_idx, total_heads)
        stream = torch.cuda.current_stream(device=key_states.device)
        self._debug_cpugather_log(
            "wait_stream_before",
            layer_idx,
            stream_ready=self._safe_stream_query(stream),
        )
        t0 = time.perf_counter()
        torch.cuda.nvtx.range_push(
            f"wait_op:decode_append_offload_tensor_pos_wait[layer={layer_idx}]"
        )
        KVLib.decode_append_offload_tensor_pos_wait(
            key_states,
            value_states,
            self.gpu_kv_buffers[layer_idx],
            self.cpu_kv_caches[layer_idx],
            self._active_cache_length_tensor('gpu_buffer_ptr', layer_idx),
            self._active_cache_length_tensor('cpu_cache_length', layer_idx),
            self.metadata_tensors['ready_flag'][layer_idx],
            self.layers_cpu_head_ids[layer_idx],
        )
        torch.cuda.nvtx.range_pop()
        wait_ms = (time.perf_counter() - t0) * 1000.0
        self._debug_cpugather_log(
            "wait_exit",
            layer_idx,
            wait_ms=f"{wait_ms:.2f}",
        )
        if wait_ms > 5000:
            self._debug_cpugather_log(
                "SUSPECT_CPUGATHER_BLOCK",
                layer_idx,
                wait_ms=f"{wait_ms:.2f}",
            )
        self._debug_cpugather_log(
            "wait_kernel_enqueued",
            layer_idx,
            stream_ready=self._safe_stream_query(stream),
        )
        self._debug_ready_flag_snapshot("wait_ready_after_enqueue", layer_idx, total_heads)
        self._active_cache_length_tensor('cpu_cache_length', layer_idx).add_(1)
        self._increment_active_host_lengths(self.cpu_cache_length_host, layer_idx)
        self._active_cache_length_tensor('gpu_buffer_ptr', layer_idx).add_(1)
        torch.cuda.nvtx.range_pop()

    def launch_prefetch(self, indices: torch.Tensor,
                        head_mask: torch.Tensor,
                        prefetch_layer_idx: int):
        torch.cuda.nvtx.range_push("real indices")
        device_idx = indices.device.index
        total_heads = self.curr_batch_size * self.num_key_value_heads
        self._debug_cpugather_log(
            "prefetch_enter",
            prefetch_layer_idx,
            device=device_idx,
            indices_shape=tuple(indices.shape),
            indices_dtype=str(indices.dtype),
            head_mask_shape=tuple(head_mask.shape),
            head_mask_dtype=str(head_mask.dtype),
        )
        stream = torch.cuda.current_stream(device=indices.device)
        self._debug_cpugather_log(
            "prefetch_stream_before",
            prefetch_layer_idx,
            stream_ready=self._safe_stream_query(stream),
        )
        self._debug_ready_flag_snapshot("prefetch_ready_before", prefetch_layer_idx, total_heads)

        prefetch_k_shape = int(indices.shape[-1]) if indices.ndim >= 2 else 0
        k_tensor = self.metadata_tensors[f"topk_prefetch_k_{device_idx}"]
        requested_prefetch_k = int(self.topk_prefetch_k_host)
        try:
            if not torch.cuda.is_current_stream_capturing():
                requested_prefetch_k = int(
                    k_tensor[: self.curr_batch_size].max().item()
                )
        except Exception:
            # Fall back to host mirror when stream capture state/query is unavailable.
            pass
        prefetch_k = max(
            0,
            min(requested_prefetch_k, prefetch_k_shape, self.max_prefetch_topk_len),
        )
        if requested_prefetch_k != prefetch_k:
            k_tensor[: self.curr_batch_size].clamp_(0, prefetch_k)

        if indices.dtype != torch.int32:
            t_cast = time.perf_counter()
            self._debug_cpugather_log("prefetch_cast_indices_enter", prefetch_layer_idx)
            indices = indices.to(torch.int32)
            self._debug_cpugather_log(
                "prefetch_cast_indices_exit",
                prefetch_layer_idx,
                ms=f"{(time.perf_counter() - t_cast) * 1000.0:.2f}",
            )
        if not indices.is_contiguous():
            t_contig = time.perf_counter()
            self._debug_cpugather_log("prefetch_contig_indices_enter", prefetch_layer_idx)
            indices = indices.contiguous()
            self._debug_cpugather_log(
                "prefetch_contig_indices_exit",
                prefetch_layer_idx,
                ms=f"{(time.perf_counter() - t_contig) * 1000.0:.2f}",
            )
        if head_mask.dtype != torch.bool:
            t_mask_cast = time.perf_counter()
            self._debug_cpugather_log("prefetch_cast_mask_enter", prefetch_layer_idx)
            head_mask = head_mask.to(torch.bool)
            self._debug_cpugather_log(
                "prefetch_cast_mask_exit",
                prefetch_layer_idx,
                ms=f"{(time.perf_counter() - t_mask_cast) * 1000.0:.2f}",
            )
        if not head_mask.is_contiguous():
            t_mask_contig = time.perf_counter()
            self._debug_cpugather_log("prefetch_contig_mask_enter", prefetch_layer_idx)
            head_mask = head_mask.contiguous()
            self._debug_cpugather_log(
                "prefetch_contig_mask_exit",
                prefetch_layer_idx,
                ms=f"{(time.perf_counter() - t_mask_contig) * 1000.0:.2f}",
            )

        # Switch to static_launch_prefetch for myTransformer parity.
        # Keep the previous real_indices path commented below for fallback/debug.
        if prefetch_k != prefetch_k_shape:
            indices = indices[..., :prefetch_k].contiguous()
            self._debug_cpugather_log(
                "prefetch_k_clamped",
                prefetch_layer_idx,
                requested_k=requested_prefetch_k,
                shape_k=prefetch_k_shape,
                use_k=prefetch_k,
            )

        if self.disable_prefetch:
            self.metadata_tensors["ready_flag"][prefetch_layer_idx][:total_heads] = True
            self._debug_cpugather_log(
                "prefetch_skip_disabled",
                prefetch_layer_idx,
                shape_k=prefetch_k_shape,
            )
            torch.cuda.nvtx.range_pop()
            return

        if prefetch_k == 0:
            self.metadata_tensors["ready_flag"][prefetch_layer_idx][:total_heads] = True
            self._debug_cpugather_log(
                "prefetch_skip_zero",
                prefetch_layer_idx,
                requested_k=requested_prefetch_k,
                shape_k=prefetch_k_shape,
            )
            torch.cuda.nvtx.range_pop()
            return

        self._debug_cpugather_log(
            "prefetch_launch",
            prefetch_layer_idx,
            prefetch_k=prefetch_k,
            requested_k=requested_prefetch_k,
            indices_shape=tuple(indices.shape),
        )
        if self.debug_cpugather:
            try:
                indices_2d = indices.reshape(-1, indices.shape[-1])
                launch_k = prefetch_k
                if launch_k > 0:
                    used_indices = indices_2d[:, :launch_k]
                    idx_min = int(used_indices.min().item())
                    idx_max = int(used_indices.max().item())
                    neg_count = int((used_indices < 0).sum().item())
                else:
                    idx_min = 0
                    idx_max = -1
                    neg_count = 0
                seq_cap = (
                    int(
                        self.cache_tensors["topk_code_length"][
                            prefetch_layer_idx
                        ][: self.curr_batch_size]
                        .max()
                        .item()
                    )
                    if "topk_code_length" in self.cache_tensors
                    else -1
                )
                self._debug_cpugather_log(
                    "prefetch_index_stats",
                    prefetch_layer_idx,
                    launch_k=launch_k,
                    idx_min=idx_min,
                    idx_max=idx_max,
                    neg=neg_count,
                    seq_cap=seq_cap,
                )
            except Exception as exc:
                self._debug_cpugather_log(
                    "prefetch_index_stats_err",
                    prefetch_layer_idx,
                    err=repr(exc),
                )
        if self.debug_cpugather:
            self._debug_cpugather_log(
                "prefetch_native_call_enter",
                prefetch_layer_idx,
                stream_ready=self._safe_stream_query(stream),
                gather_meta=tuple(
                    int(x)
                    for x in self.metadata_tensors["gather_engine_metadata"][:6]
                ),
            )
        # KVLib.real_indices_and_launch_prefetch(
        #     indices,
        #     head_mask,
        #     self.cpu_indices_buffer,
        #     self.metadata_tensors['gather_engine_metadata'],
        #     self.metadata_tensors['ready_flag'][prefetch_layer_idx],
        #     self.max_seq_len,
        #     self.curr_batch_size,
        #     self.num_key_value_heads,
        #     prefetch_layer_idx,
        # )
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
        self._debug_cpugather_log(
            "prefetch_native_call_exit",
            prefetch_layer_idx,
            stream_ready=self._safe_stream_query(stream),
        )
        torch.cuda.nvtx.range_pop()

    def _batch_topk_masked_compat(
        self,
        data: torch.Tensor,
        bh_mask: torch.Tensor,
        k: int,
        largest: bool = True,
    ) -> torch.Tensor:
        max_k = int(data.size(2))
        if k > max_k:
            k = max_k
        if k <= 0:
            return torch.empty(
                (data.size(0), data.size(1), 0),
                dtype=torch.int32,
                device=data.device,
            )

        if not hasattr(self, "_batch_topk_masked_param_count"):
            import inspect
            self._batch_topk_masked_param_count = len(
                inspect.signature(KVLib.batch_topk_masked).parameters
            )

        # RAFT masked top-k kernel supports fp16/fp32 (not bf16).
        if data.dtype == torch.bfloat16:
            topk_data = data.to(torch.float16)
        else:
            topk_data = data
        if not topk_data.is_contiguous():
            topk_data = topk_data.contiguous()
        if bh_mask.device != topk_data.device:
            bh_mask = bh_mask.to(topk_data.device, non_blocking=True)
        if bh_mask.dtype != torch.bool:
            bh_mask = bh_mask.to(torch.bool)
        if not bh_mask.is_contiguous():
            bh_mask = bh_mask.contiguous()

        if self._batch_topk_masked_param_count <= 4:
            # myTransformer-style API: returns indices tensor directly.
            return KVLib.batch_topk_masked(topk_data, bh_mask, k, largest)

        # sgl-kernel API: requires output buffers + real_len/real_k.
        out_index = torch.empty(
            (topk_data.size(0), topk_data.size(1), k),
            dtype=torch.int32,
            device=topk_data.device,
        )
        out_values = torch.empty(
            (topk_data.size(0), topk_data.size(1), k),
            dtype=topk_data.dtype,
            device=topk_data.device,
        )
        real_len = torch.tensor([topk_data.size(2)], dtype=torch.int32, device=topk_data.device)
        real_k = torch.tensor([k], dtype=torch.int32, device=topk_data.device)
        KVLib.batch_topk_masked(
            topk_data,
            bh_mask,
            out_index,
            out_values,
            real_len,
            real_k,
            largest,
        )
        return out_index

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
        self._debug_stall_log("decode_enter", layer_idx)
        key_states = key_states.view(self.curr_batch_size, 1,
                                     self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(self.curr_batch_size, 1,
                                     self.num_key_value_heads, self.head_dim)

        # 1. gpu kvcache append (checked)
        if self.layers_gpu_head_ids[layer_idx].numel() > 0:
            t_step = time.perf_counter()
            self._debug_stall_log("step1_gpu_append_enter", layer_idx)
            self.append_gpu_cache_decode(key_states, value_states,
                                            layer_idx)
            self._debug_stall_duration("step1_gpu_append_exit", layer_idx, t_step)

        # 2. gpu topk cache append and process query (checked)
        t_step = time.perf_counter()
        self._debug_stall_log("step2_topk_append_enter", layer_idx)
        prefetch_query, current_query = self.append_topk_cache_decode(
            key_states, value_states, layer_idx, prefetch_query_states,
            current_query_states)
        self._debug_stall_duration("step2_topk_append_exit", layer_idx, t_step)

        # 3. compute prefetch top-k indices (checked)
        next_layer_idx = (layer_idx + 1) % self.num_layers
        should_prefetch = (
            (not self.layers_full_gpu_mask[next_layer_idx])
            and (not self.disable_prefetch)
            and (int(self.topk_prefetch_k_host) > 0)
        )
        if should_prefetch:
            t_step = time.perf_counter()
            self._debug_stall_log("step3_prefetch_topk_enter", layer_idx, next_layer=next_layer_idx)
            gather_mask = self.cache_query_and_update(
                prefetch_query_states, next_layer_idx)
            prefetch_topk_indices = self.compute_topk(
                prefetch_query,
                next_layer_idx,
                gather_mask,
                is_prefetch=True,
            )
            if self.record_overlap_stats:
                self._record_prefetch_overlap_candidate(
                    next_layer_idx,
                    prefetch_topk_indices,
                    gather_mask,
                )
            self._debug_stall_duration(
                "step3_prefetch_topk_exit",
                layer_idx,
                t_step,
                next_layer=next_layer_idx,
                prefetch_shape=tuple(prefetch_topk_indices.shape),
            )

        # 4. append cpu kvcache, gpu recent buffer, and sync prefetch (checked)
        if not self.layers_full_gpu_mask[layer_idx]:
            t_step = time.perf_counter()
            self._debug_stall_log("step4_cpu_append_wait_enter", layer_idx)
            torch.cuda.nvtx.range_push(f"append cpu and wait layer{layer_idx}")
            self.append_cpu_cache_decode_and_wait(key_states, value_states, layer_idx)
            torch.cuda.nvtx.range_pop()
            self._debug_stall_duration("step4_cpu_append_wait_exit", layer_idx, t_step)

        # 5. launch prefetching (checked)
        if not self.layers_full_gpu_mask[next_layer_idx]:
            if should_prefetch:
                t_step = time.perf_counter()
                self._debug_stall_log("step5_launch_prefetch_enter", layer_idx, next_layer=next_layer_idx)
                torch.cuda.nvtx.range_push("real indices")
                self.launch_prefetch(prefetch_topk_indices, gather_mask,
                                     next_layer_idx)
                torch.cuda.nvtx.range_pop()
                self._debug_stall_duration("step5_launch_prefetch_exit", layer_idx, t_step, next_layer=next_layer_idx)
            else:
                total_heads = self.curr_batch_size * self.num_key_value_heads
                self.metadata_tensors["ready_flag"][next_layer_idx][:total_heads] = True

        # 6. compute top-k for current layer (checked)
        self.current_topk_indices = None
        if self.record_overlap_stats:
            self._record_real_overlap_for_current_layer(
                layer_idx,
                current_query,
            )
        if self.layers_gpu_head_ids[layer_idx].numel() > 0:
            t_step = time.perf_counter()
            self._debug_stall_log("step6_current_topk_enter", layer_idx)
            self.current_topk_indices = self.compute_topk(
                current_query,
                layer_idx,
                self.layers_gpu_bh_mask[layer_idx][:self.curr_batch_size *
                                                   self.num_key_value_heads],
                is_prefetch=False
            )
            if self.current_topk_indices is not None and self.current_topk_indices.dim() == 2:
                # decode kernels expect gather_idx as [batch, kv_heads, k].
                expected_rows = self.curr_batch_size * self.num_key_value_heads
                if self.current_topk_indices.size(0) != expected_rows:
                    raise RuntimeError(
                        "Unexpected top-k shape for decode attention: "
                        f"got {tuple(self.current_topk_indices.shape)}, "
                        f"expected first dim {expected_rows} "
                        f"(batch_size={self.curr_batch_size}, "
                        f"num_key_value_heads={self.num_key_value_heads})."
                    )
                self.current_topk_indices = self.current_topk_indices.view(
                    self.curr_batch_size,
                    self.num_key_value_heads,
                    self.current_topk_indices.size(1),
                ).contiguous()
            self._debug_stall_duration(
                "step6_current_topk_exit",
                layer_idx,
                t_step,
                current_shape=tuple(self.current_topk_indices.shape),
            )

        self._debug_stall_log("decode_exit", layer_idx)

    def _record_prefetch_overlap_candidate(
        self,
        layer_idx: int,
        prefetch_topk_indices: torch.Tensor,
        gather_mask: torch.Tensor,
    ) -> None:
        del layer_idx, prefetch_topk_indices, gather_mask

    def _record_real_overlap_for_current_layer(
        self,
        layer_idx: int,
        current_query,
    ) -> None:
        del layer_idx, current_query

    def _summarize_decode_overlap_step(self):
        if not self.record_overlap_stats:
            return None

        layer_metrics = []
        mean_recall = []
        mean_precision = []
        mean_jaccard = []
        union_recall = []
        union_precision = []
        union_jaccard = []

        for layer_idx, metrics in enumerate(self._decode_overlap_metrics):
            if metrics is None:
                continue
            layer_metrics.append(
                {
                    "layer_idx": int(layer_idx),
                    **metrics,
                }
            )
            mean_recall.append(float(metrics["mean_recall"]))
            mean_precision.append(float(metrics["mean_precision"]))
            mean_jaccard.append(float(metrics["mean_jaccard"]))
            union_recall.append(float(metrics["union_recall"]))
            union_precision.append(float(metrics["union_precision"]))
            union_jaccard.append(float(metrics["union_jaccard"]))

        if not layer_metrics:
            return None

        def _avg(values):
            return float(sum(values) / len(values)) if values else 0.0

        return {
            "matched_layers": int(len(layer_metrics)),
            "mean_recall": _avg(mean_recall),
            "mean_precision": _avg(mean_precision),
            "mean_jaccard": _avg(mean_jaccard),
            "union_recall": _avg(union_recall),
            "union_precision": _avg(union_precision),
            "union_jaccard": _avg(union_jaccard),
            "layers": layer_metrics,
        }

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
