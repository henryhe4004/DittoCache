from typing import Dict, Optional, Union, Any

import json
import os
import time
import torch
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig
import sgl_kernel.kvlib as KVLib

from sglang.ditto.tp_head_mapping import (
    local_kv_head_ids,
    query_head_order,
    resolve_tp_kv_head_orders,
)

DEBUG_LOG_PATH = "/tmp/ditto_debug.log"
DEBUG_SESSION_ID = "9e2373"


def _detect_attention_tp_info() -> tuple[int, int]:
    """Best-effort detection of attention TP rank/size."""
    try:
        from sglang.srt.layers.dp_attention import (  # pylint: disable=import-outside-toplevel
            get_attention_tp_rank,
            get_attention_tp_size,
        )

        tp_size = int(get_attention_tp_size())
        tp_rank = int(get_attention_tp_rank())
        if tp_size > 0 and tp_rank >= 0:
            return tp_rank, tp_size
    except Exception:
        pass
    return 0, 1


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


class CustomStaticCache(Cache):
    def __init__(
        self,
        config: PretrainedConfig,
        custom_config: Any,
        device: torch.device = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__(layers=[])

        self.model_config = config
        self.config = custom_config
        self.dtype = self.model_config.torch_dtype

        # ==================== load model config ====================
        self.num_layers = self.model_config.num_hidden_layers
        if hasattr(config, "qk_nope_head_dim"):
            self.head_dim = self.model_config.qk_nope_head_dim + self.model_config.qk_rope_head_dim
        elif hasattr(config, "head_dim"):
            self.head_dim = self.model_config.head_dim
        elif hasattr(config, "kv_channels"):
            self.head_dim = self.model_config.kv_channels
        else:
            self.head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads

        if hasattr(config, "num_key_value_heads"):
            total_num_kv_heads = int(self.model_config.num_key_value_heads)
        elif hasattr(config, "multi_query_group_num"):
            total_num_kv_heads = int(self.model_config.multi_query_group_num)
        else:
            total_num_kv_heads = int(self.model_config.num_attention_heads)

        total_num_heads = int(self.model_config.num_attention_heads)
        attn_tp_rank, attn_tp_size = _detect_attention_tp_info()
        if total_num_heads <= 0 or total_num_kv_heads <= 0:
            raise ValueError(
                "Invalid Ditto attention head config: "
                f"num_attention_heads={total_num_heads}, "
                f"num_key_value_heads={total_num_kv_heads}."
            )

        self.total_num_heads = total_num_heads
        self.total_num_key_value_heads = total_num_kv_heads
        self.attn_tp_rank = attn_tp_rank
        self.attn_tp_size = attn_tp_size

        if attn_tp_size <= 0:
            raise ValueError(f"Invalid attn_tp_size={attn_tp_size}")
        if total_num_heads % attn_tp_size != 0:
            raise ValueError(
                f"Ditto TP requires num_attention_heads={total_num_heads} to be "
                f"divisible by attn_tp_size={attn_tp_size}"
            )

        self.num_heads = total_num_heads // attn_tp_size
        self.query_head_start = attn_tp_rank * self.num_heads

        if attn_tp_size <= 1:
            self.num_key_value_heads = total_num_kv_heads
            self.kv_head_start = 0
            self.kv_head_replicas = 1
        elif total_num_kv_heads >= attn_tp_size:
            # Partition KV heads across TP ranks.
            if total_num_kv_heads % attn_tp_size != 0:
                raise ValueError(
                    f"num_key_value_heads={total_num_kv_heads} is not divisible by attn_tp_size={attn_tp_size}"
                )
            self.num_key_value_heads = total_num_kv_heads // attn_tp_size
            self.kv_head_start = attn_tp_rank * self.num_key_value_heads
            self.kv_head_replicas = 1
        else:
            # Replicate KV heads when tp_size > kv_heads.
            if attn_tp_size % total_num_kv_heads != 0:
                raise ValueError(
                    f"attn_tp_size={attn_tp_size} is not divisible by num_key_value_heads={total_num_kv_heads}"
                )
            self.num_key_value_heads = 1
            replicate = attn_tp_size // total_num_kv_heads
            self.kv_head_replicas = replicate
            self.kv_head_start = attn_tp_rank // replicate

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"Ditto TP requires local num_heads={self.num_heads} to be "
                f"divisible by local num_key_value_heads={self.num_key_value_heads}"
            )

        layer_kv_head_orders = resolve_tp_kv_head_orders(
            self.total_num_key_value_heads,
            self.num_layers,
        )
        self.local_kv_head_ids_by_layer = tuple(
            local_kv_head_ids(
                self.total_num_key_value_heads,
                self.attn_tp_rank,
                self.attn_tp_size,
                layer_idx=layer_idx,
            )
            for layer_idx in range(self.num_layers)
        )
        if self.total_num_key_value_heads < self.attn_tp_size:
            replicated_query_heads = tuple(
                range(self.query_head_start, self.query_head_start + self.num_heads)
            )
            self.local_query_head_ids_by_layer = tuple(
                replicated_query_heads for _ in range(self.num_layers)
            )
        else:
            self.local_query_head_ids_by_layer = tuple(
                query_head_order(
                    layer_kv_head_orders[layer_idx][
                        self.attn_tp_rank
                        * self.num_key_value_heads : (self.attn_tp_rank + 1)
                        * self.num_key_value_heads
                    ],
                    self.total_num_heads,
                    self.total_num_key_value_heads,
                )
                for layer_idx in range(self.num_layers)
            )
        # Kept for compatibility with extensions that only support one layout.
        self.local_kv_head_ids = self.local_kv_head_ids_by_layer[0]
        self.local_query_head_ids = self.local_query_head_ids_by_layer[0]

        # ==================== set layer devices ====================
        self.layer_devices = []
        for l in range(self.num_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[l]
                self.layer_devices.append(layer_device)
            else:
                layer_device = torch.device(device)
                self.layer_devices.append(layer_device.index)
        self.unique_devices = set(self.layer_devices)
        if getattr(self.config, "enable_cuda_graph", False):
            assert len(self.unique_devices) == 1, "cuda graph only support single device"

        # ==================== tensors for cuda graph ====================
        self.metadata_tensors: Dict[str, torch.Tensor] = {}
        self.cache_tensors: Dict[str, Any] = {}

        # ==================== metadata ====================
        self.curr_batch_size = 1
        self.cur_q_len = 0
        self.cache_length_host = []
        self.max_seq_len = self.config.kvcache_manager_config.max_tokens
        self.mem_budget = int(
            self.config.kvcache_manager_config.gpu_memory_budget * 1024 * 1024 * 1024
        )

    def _slice_local_kv_head_tensor(
        self,
        tensor: torch.Tensor,
        *,
        layer_idx: int,
        tensor_name: str = "tensor",
        head_dim: int = 0,
    ) -> torch.Tensor:
        """Slice a head-major tensor from global KV-head layout to this TP rank's local heads."""
        if tensor is None or self.attn_tp_size <= 1:
            return tensor
        if tensor.ndim <= head_dim:
            raise ValueError(
                f"{tensor_name} ndim={tensor.ndim} does not contain head_dim={head_dim}"
            )

        head_count = int(tensor.shape[head_dim])
        if head_count == self.num_key_value_heads:
            return tensor
        if head_count != self.total_num_key_value_heads:
            raise ValueError(
                f"{tensor_name} head dimension={head_count} mismatch local/global kv heads "
                f"({self.num_key_value_heads}/{self.total_num_key_value_heads})"
            )

        head_ids = torch.tensor(
            self.local_kv_head_ids_by_layer[layer_idx],
            dtype=torch.long,
            device=tensor.device,
        )
        return tensor.index_select(head_dim, head_ids).contiguous()

    def _local_kv_head_ids_to_global(
        self,
        head_ids: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        if head_ids is None:
            return head_ids
        mapping = torch.tensor(
            self.local_kv_head_ids_by_layer[layer_idx],
            dtype=torch.long,
            device=head_ids.device,
        )
        return mapping.index_select(0, head_ids.to(torch.long))

    def build_cache(self):
        self._create_metadata_tensors()
        self._create_cache_tensors()

    def _create_metadata_tensors(self):
        # ==================== rope ====================
        for device_idx in self.unique_devices:
            rope_indptr = torch.arange(
                (self.config.kvcache_manager_config.max_batch_size + 1),
                dtype=torch.int32,
                device=device_idx,
            )
            rope_offsets = torch.zeros(
                (self.config.kvcache_manager_config.max_batch_size,),
                dtype=torch.int32,
                device=device_idx,
            )
            self.metadata_tensors[f"rope_indptr_{device_idx}"] = rope_indptr
            self.metadata_tensors[f"rope_offsets_{device_idx}"] = rope_offsets

    def _create_cache_tensors(self):
        # ==================== kvcache ====================
        numel_one_layer = (
            2
            * self.config.kvcache_manager_config.max_tokens
            * self.num_key_value_heads
            * self.head_dim
        )
        mem_one_layer = numel_one_layer * self.dtype.itemsize
        mem_total = mem_one_layer * self.num_layers
        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H1",
            location="kvcache_full_attn.py:_create_cache_tensors",
            message="Computed KV cache memory requirement",
            data={
                "max_tokens": int(self.config.kvcache_manager_config.max_tokens),
                "gpu_memory_budget_gb": float(self.config.kvcache_manager_config.gpu_memory_budget),
                "dtype": str(self.dtype),
                "dtype_itemsize": int(self.dtype.itemsize),
                "num_layers": int(self.num_layers),
                "num_kv_heads": int(self.num_key_value_heads),
                "head_dim": int(self.head_dim),
                "mem_total_gb": float(mem_total / 1024 / 1024 / 1024),
            },
        )
        # endregion
        if mem_total > self.mem_budget:
            # region agent log
            _debug_log(
                run_id="pre-fix",
                hypothesis_id="H2",
                location="kvcache_full_attn.py:_create_cache_tensors",
                message="KV cache memory budget check failed",
                data={
                    "mem_total_gb": float(mem_total / 1024 / 1024 / 1024),
                    "mem_budget_gb": float(self.mem_budget / 1024 / 1024 / 1024),
                },
            )
            # endregion
            raise ValueError(
                f"GPU mem budget {self.config.kvcache_manager_config.gpu_memory_budget} GB "
                f"is not enough for {self.config.kvcache_manager_config.max_tokens} "
                f"tokens, {mem_total / 1024 / 1024 / 1024} GB needed"
            )

        self.cache_tensors["cache_data"] = [None for _ in range(self.num_layers)]
        self.cache_tensors["cache_length"] = [None for _ in range(self.num_layers)]
        max_batch = int(self.config.kvcache_manager_config.max_batch_size)
        self.cache_length_host = [[0] * max_batch for _ in range(self.num_layers)]
        for l in range(self.num_layers):
            layer_device = self.layer_devices[l]
            cache_data = torch.zeros(
                (numel_one_layer,),
                dtype=self.dtype,
                device=layer_device,
            )
            cache_length = torch.zeros(
                (max_batch,),
                dtype=torch.int32,
                device=layer_device,
            )
            self.cache_tensors["cache_data"][l] = cache_data
            self.cache_tensors["cache_length"][l] = cache_length
        self.kv_caches = [None for _ in range(self.num_layers)]

    def _reset_cache_tensors(self, batch_size: int):
        # reset gpu kv cache
        for l in range(self.num_layers):
            cache_data = self.cache_tensors["cache_data"][l]
            self.kv_caches[l] = cache_data[
                : 2 * batch_size * self.max_seq_len * self.num_key_value_heads * self.head_dim
            ].view(
                2,
                batch_size,
                self.max_seq_len,
                self.num_key_value_heads,
                self.head_dim,
            )
            cache_length = self.cache_tensors["cache_length"][l]
            cache_length.zero_()
            for row in range(len(self.cache_length_host[l])):
                self.cache_length_host[l][row] = 0

    # ====================== HF APIs ======================
    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        active = slice(0, self.curr_batch_size)
        return int(min(self.cache_length_host[layer_idx][active]))

    def get_seq_lengths(self, layer_idx: Optional[int] = 0) -> list[int]:
        return [
            int(x)
            for x in self.cache_length_host[layer_idx][: self.curr_batch_size]
        ]

    def get_max_length(self) -> Optional[int]:
        return self.max_seq_len

    def reset(self, batch_size: int):
        assert (
            batch_size <= self.config.kvcache_manager_config.max_batch_size
        ), f"batch_size ({batch_size}) should be less than max_batch_size ({self.config.kvcache_manager_config.max_batch_size})"
        self.curr_batch_size = batch_size
        self.max_seq_len = self.config.kvcache_manager_config.max_tokens // batch_size

        self._reset_cache_tensors(batch_size)

    def reset_batch_rows(self, row_indices: list[int]) -> None:
        if not row_indices:
            return
        max_batch = int(self.config.kvcache_manager_config.max_batch_size)
        bad_rows = [int(row) for row in row_indices if row < 0 or row >= max_batch]
        if bad_rows:
            raise RuntimeError(
                f"Ditto full-attn reset rows out of range: {bad_rows}, "
                f"max_batch_size={max_batch}"
            )
        for layer_idx in range(self.num_layers):
            rows = torch.tensor(
                row_indices,
                dtype=torch.long,
                device=self.cache_tensors["cache_length"][layer_idx].device,
            )
            self.cache_tensors["cache_length"][layer_idx][rows] = 0
            for row in row_indices:
                self.cache_length_host[layer_idx][int(row)] = 0

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
                f"Ditto full-attn row remap out of range: {normalized}, "
                f"max_batch_size={max_batch}"
            )

        def _index_copy_rows(tensor: torch.Tensor, dim: int) -> None:
            old_rows = torch.tensor(
                list(normalized.keys()), dtype=torch.long, device=tensor.device
            )
            new_rows = torch.tensor(
                list(normalized.values()), dtype=torch.long, device=tensor.device
            )
            src = tensor.index_select(dim, old_rows).clone()
            tensor.index_copy_(dim, new_rows, src)

        for layer_idx in range(self.num_layers):
            _index_copy_rows(self.cache_tensors["cache_length"][layer_idx], 0)
            kv_cache = self.kv_caches[layer_idx]
            if kv_cache is not None:
                snapshots = []
                seq_cap = int(kv_cache.size(2))
                for old, new in normalized.items():
                    valid_len = max(
                        0,
                        min(int(self.cache_length_host[layer_idx][old]), seq_cap),
                    )
                    if valid_len == 0:
                        continue
                    snapshots.append(
                        (
                            new,
                            valid_len,
                            kv_cache[:, old : old + 1, :valid_len].clone(),
                        )
                    )
                for new, valid_len, src in snapshots:
                    kv_cache[:, new : new + 1, :valid_len].copy_(src)

            old_lengths = [
                self.cache_length_host[layer_idx][old]
                for old in normalized.keys()
            ]
            for new, value in zip(normalized.values(), old_lengths):
                self.cache_length_host[layer_idx][new] = int(value)

    # =====================================================
    # call this before cuda graph
    def update_metadata(self, q_len: int, is_prefill: bool = False, layer_idx: int = 0):
        del is_prefill  # unused for now
        self.cur_q_len = q_len
        for device_idx in self.unique_devices:
            device = torch.device("cuda", int(device_idx))
            rope_indptr = torch.arange(
                0,
                (self.curr_batch_size + 1) * q_len,
                q_len,
                dtype=torch.int32,
                device=device,
            )
            self.metadata_tensors[f"rope_indptr_{device_idx}"][
                : self.curr_batch_size + 1
            ] = rope_indptr
            self.metadata_tensors[f"rope_offsets_{device_idx}"][
                : self.curr_batch_size
            ].copy_(
                self.cache_tensors["cache_length"][layer_idx][
                    : self.curr_batch_size
                ].to(device=device, dtype=torch.int32, non_blocking=True)
            )

    def get_rope_metadata(self, device: Optional[torch.device] = None):
        if device is None:
            device = self.layer_devices[0]
        indptr = self.metadata_tensors[f"rope_indptr_{device.index}"][
            : self.curr_batch_size + 1
        ]
        offsets = self.metadata_tensors[f"rope_offsets_{device.index}"][
            : self.curr_batch_size
        ]
        return indptr, offsets

    def get_cur_batch_size(self) -> int:
        return self.curr_batch_size

    def get_cur_q_len(self) -> int:
        return self.cur_q_len

    def get_seqlen_tensor(self, layer_idx: int):
        return self.cache_tensors["cache_length"][layer_idx][: self.curr_batch_size]

    def append_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        q_len = self.cur_q_len
        seq_len_tensor = self.cache_tensors["cache_length"][layer_idx]
        active_seq_lens = seq_len_tensor[: self.curr_batch_size]
        extend_seq_lens = getattr(self, "_current_extend_seq_lens", None)
        if extend_seq_lens is None:
            row_lens = [int(q_len)] * self.curr_batch_size
        else:
            if len(extend_seq_lens) != self.curr_batch_size:
                raise RuntimeError(
                    "Ditto full-attn extend length mismatch: "
                    f"extend_seq_lens={extend_seq_lens}, "
                    f"curr_batch_size={self.curr_batch_size}."
                )
            row_lens = [int(x) for x in extend_seq_lens]
        old_lengths = [int(x) for x in self.cache_length_host[layer_idx][: self.curr_batch_size]]
        max_new_seq_len = max(old + row_len for old, row_len in zip(old_lengths, row_lens))
        if max_new_seq_len > self.max_seq_len:
            raise AssertionError(
                f"input kv states max length {max_new_seq_len}, "
                f"should be less than max_seq_len = {self.max_seq_len}"
            )
        key_states = key_states.view(
            self.curr_batch_size,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        value_states = value_states.view(
            self.curr_batch_size,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        for row, (old_len, row_len) in enumerate(zip(old_lengths, row_lens)):
            if row_len <= 0:
                continue
            self.kv_caches[layer_idx][
                0, row : row + 1, old_len : old_len + row_len, :, :
            ] = key_states[row : row + 1, :row_len]
            self.kv_caches[layer_idx][
                1, row : row + 1, old_len : old_len + row_len, :, :
            ] = value_states[row : row + 1, :row_len]
            self.cache_length_host[layer_idx][row] = old_len + row_len
        new_lengths = torch.tensor(
            [old + row_len for old, row_len in zip(old_lengths, row_lens)],
            dtype=active_seq_lens.dtype,
            device=active_seq_lens.device,
        )
        active_seq_lens.copy_(new_lengths)
        if not hasattr(self, "_last_prefill_old_lengths"):
            self._last_prefill_old_lengths = {}
        if not hasattr(self, "_last_prefill_row_lengths"):
            self._last_prefill_row_lengths = {}
        self._last_prefill_old_lengths[layer_idx] = old_lengths
        self._last_prefill_row_lengths[layer_idx] = row_lens
        return (
            self.kv_caches[layer_idx][0, :, :max_new_seq_len, :, :],
            self.kv_caches[layer_idx][1, :, :max_new_seq_len, :, :],
        )

    def append_decode(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        key_states = key_states.view(
            self.curr_batch_size,
            1,
            self.num_key_value_heads,
            self.head_dim,
        )
        value_states = value_states.view(
            self.curr_batch_size,
            1,
            self.num_key_value_heads,
            self.head_dim,
        )
        KVLib.kvcache_append_tensor_pos(
            self.kv_caches[layer_idx],
            key_states,
            value_states,
            self.cache_tensors["cache_length"][layer_idx][: self.curr_batch_size],
        )
        self.cache_tensors["cache_length"][layer_idx][: self.curr_batch_size] += 1
        return self.kv_caches[layer_idx][0, ...], self.kv_caches[layer_idx][1, ...]

    def record_decode_step(self) -> None:
        for layer_idx in range(self.num_layers):
            for row in range(self.curr_batch_size):
                self.cache_length_host[layer_idx][row] += 1


def prepare_cache_for_generation(  # HF-style helper, kept for compatibility
    self,
    generation_config: GenerationConfig,
    model_kwargs: Dict,
    assistant_model,
    batch_size: int,
    max_cache_length: int,
    device: torch.device,
) -> bool:
    if not hasattr(self, "_cache"):

        def get_layer_device_map(execution_device_map: Optional[dict] = None):
            if execution_device_map is None or len(execution_device_map) <= 1:
                return None
            layer_device_map: Dict[int, Union[str, torch.device, int]] = {}
            for layer in execution_device_map:
                for idx in range(self.config.num_hidden_layers):
                    if f".{idx}." in f"{layer}.":
                        layer_device_map[idx] = execution_device_map[layer]
                        break
            for idx in range(self.config.num_hidden_layers):
                if idx not in layer_device_map:
                    raise RuntimeError(
                        f"layer {idx} has not been mapped to a device."
                    )
            return layer_device_map

        execution_device_map = None
        if hasattr(self, "hf_device_map"):
            main_device = [
                d
                for d in self.hf_device_map.values()
                if d not in ["cpu", "disk"]
            ][0]
            execution_device_map = {
                name: main_device if device in ["cpu", "disk"] else device
                for name, device in self.hf_device_map.items()
            }

        layer_device_map = get_layer_device_map(execution_device_map)
        self._cache = CustomStaticCache(
            config=self.config.get_text_config(),
            custom_config=generation_config.custom_config,
            device=device,
            layer_device_map=layer_device_map,
        )
        self._cache.build_cache()

    self._cache.reset(batch_size)
    cache_name = "past_key_values"
    model_kwargs[cache_name] = self._cache
    return True
