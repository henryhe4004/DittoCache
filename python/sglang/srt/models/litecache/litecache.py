"""
LiteCache model integration.

This entry binds to SGLang-internal ports of myTransformer LiteCache files:
- Cache implementations in `sglang.litecache.*`
- Model frameworks in both non-duohead offloading and duohead variants
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional, Tuple, Type

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.litecache.config_utils import ensure_litecache_custom_config
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.litecache.awq_linear import (
    replace_litecache_linears_with_awq,
    should_enable_litecache_awq,
)
from sglang.srt.models.transformers import replace_linear_class

logger = logging.getLogger(__name__)


def _debug_batch_enabled() -> bool:
    return os.environ.get("LITECACHE_DEBUG_BATCH", "0") == "1"


def _finalize_litecache_awq_modules(root: nn.Module) -> int:
    finalized = 0
    for module in root.modules():
        if not getattr(module, "_litecache_awq_linear", False):
            continue
        process_fn = getattr(module, "process_weights_after_loading", None)
        if callable(process_fn):
            process_fn()
            finalized += 1
    return finalized


def _replace_litecache_linears_with_tp(model: nn.Module, quant_config) -> int:
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return 0

    style_by_leaf = {
        "q_proj": "colwise",
        "k_proj": "colwise",
        "v_proj": "colwise",
        "o_proj": "rowwise",
        "gate_proj": "colwise",
        "up_proj": "colwise",
        "down_proj": "rowwise",
    }

    replaced = 0
    named_modules = list(model.named_modules())
    module_index = dict(named_modules)
    for full_name, module in named_modules:
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = full_name.split(".")[-1]
        style = style_by_leaf.get(leaf_name)
        if style is None:
            continue
        if "." not in full_name:
            continue

        parent_name, attr_name = full_name.rsplit(".", 1)
        parent = module_index.get(parent_name)
        if parent is None:
            continue

        new_module = replace_linear_class(module, style, quant_config)
        # TP linear biases are loaded later via each parameter's weight_loader.
        # Eagerly copying HF full bias into a sharded TP bias breaks colwise layers.
        setattr(parent, attr_name, new_module)
        replaced += 1

    if replaced > 0:
        logger.info(
            "LiteCache TP enabled: replaced %d linear modules with TP-aware layers (tp_size=%d).",
            replaced,
            tp_size,
        )
    return replaced


def _get_variant_name(config: PretrainedConfig) -> str:
    """
    Resolve LiteCache top-k variant with explicit override priority:
    1) `config.litecache_variant`
    2) `config.custom_config.litecache_variant` / `config.custom_config.topk_variant`
    3) env `LITECACHE_VARIANT`
    4) fallback: "offloading"
    """

    supported = {"offloading", "loki", "hash", "infinigen", "quest", "fullattn"}
    aliases = {
        "full_attn": "fullattn",
        "flash_attn": "fullattn",
        "flashattn": "fullattn",
        "full": "fullattn",
    }

    def _normalize(value):
        if value is None:
            return None
        s = str(value).strip().lower()
        s = aliases.get(s, s)
        return s or None

    def _get_custom_value(key: str):
        custom = getattr(config, "custom_config", None)
        if custom is None:
            return None
        if isinstance(custom, dict):
            return custom.get(key)
        return getattr(custom, key, None)

    candidates = [
        ("config.litecache_variant", getattr(config, "litecache_variant", None)),
        ("config.custom_config.litecache_variant", _get_custom_value("litecache_variant")),
        ("config.custom_config.topk_variant", _get_custom_value("topk_variant")),
        ("env.LITECACHE_VARIANT", os.environ.get("LITECACHE_VARIANT")),
    ]

    for source, raw in candidates:
        variant = _normalize(raw)
        if variant is None:
            continue
        if variant not in supported:
            raise ValueError(
                f"Unknown LiteCache variant from {source}: {raw!r}. "
                f"Supported: {sorted(supported)}"
            )
        return variant

    return "offloading"


def _get_offloading_method_name(config: PretrainedConfig) -> str:
    supported = {"hash", "loki", "infinigen", "quest"}

    def _normalize(value):
        if value is None:
            return None
        s = str(value).strip().lower()
        return s or None

    def _get_custom_value(key: str):
        custom = getattr(config, "custom_config", None)
        if custom is None:
            return None
        if isinstance(custom, dict):
            return custom.get(key)
        return getattr(custom, key, None)

    candidates = [
        ("config.offloading_method", getattr(config, "offloading_method", None)),
        ("config.custom_config.offloading_method", _get_custom_value("offloading_method")),
        ("env.LITECACHE_OFFLOADING_METHOD", os.environ.get("LITECACHE_OFFLOADING_METHOD")),
    ]

    for source, raw in candidates:
        method = _normalize(raw)
        if method is None:
            continue
        if method == "offloading":
            return "hash"
        if method.startswith("offloading-"):
            method = method[len("offloading-") :]
        elif method.endswith("-offloading"):
            method = method[: -len("-offloading")]

        if method not in supported:
            raise ValueError(
                f"Unknown offloading method from {source}: {raw!r}. "
                f"Supported: {sorted(supported)}"
            )
        return method

    return "hash"


def _ensure_llama_compatible_config(config: PretrainedConfig) -> None:
    """
    LiteCache duohead classes are based on HF Llama modules.
    Qwen2 config misses a few Llama-only fields; patch safe defaults for bring-up.
    """

    if not hasattr(config, "pretraining_tp"):
        setattr(config, "pretraining_tp", 1)
    if not hasattr(config, "attention_bias"):
        setattr(config, "attention_bias", False)
    if not hasattr(config, "mlp_bias"):
        setattr(config, "mlp_bias", False)
    if not hasattr(config, "head_dim"):
        num_heads = int(getattr(config, "num_attention_heads", 0) or 0)
        hidden_size = int(getattr(config, "hidden_size", 0) or 0)
        if num_heads <= 0 or hidden_size <= 0 or hidden_size % num_heads != 0:
            raise ValueError(
                "Cannot infer `head_dim` for LiteCacheLlamaForCausalLM: "
                f"hidden_size={hidden_size}, num_attention_heads={num_heads}."
            )
        setattr(config, "head_dim", hidden_size // num_heads)


def _prepare_litecache_forward_inputs(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
) -> tuple[torch.Tensor, torch.Tensor, Optional[list[int]]]:
    """Convert SGLang's flat batch tensors to LiteCache's [bsz, q_len] layout."""

    batch_size = int(forward_batch.batch_size)
    if batch_size <= 0:
        raise ValueError(f"LiteCache batch_size must be positive, got {batch_size}.")

    if batch_size == 1:
        return input_ids[None, ...], positions[None, ...], None

    if forward_batch.forward_mode.is_decode():
        if input_ids.numel() != batch_size or positions.numel() != batch_size:
            raise NotImplementedError(
                "LiteCache static batching expects one decode token per request. "
                f"Got input_tokens={input_ids.numel()}, positions={positions.numel()}, "
                f"batch_size={batch_size}."
            )
        return input_ids.view(batch_size, 1), positions.view(batch_size, 1), None

    if forward_batch.forward_mode.is_extend():
        extend_seq_lens = forward_batch.extend_seq_lens_cpu
        if extend_seq_lens is None or len(extend_seq_lens) != batch_size:
            raise NotImplementedError(
                "LiteCache static batching requires per-request extend lengths. "
                f"Got extend_seq_lens_cpu={extend_seq_lens}, batch_size={batch_size}."
            )

        extend_seq_lens = [int(x) for x in extend_seq_lens]
        q_len = max(extend_seq_lens)
        if q_len <= 0:
            raise NotImplementedError(
                "LiteCache batching requires positive extend lengths. "
                f"Got extend_seq_lens={extend_seq_lens}."
            )

        expected_tokens = sum(extend_seq_lens)
        if input_ids.numel() != expected_tokens or positions.numel() != expected_tokens:
            raise NotImplementedError(
                "LiteCache batching received inconsistent extend metadata. "
                f"Expected {expected_tokens} tokens from extend_seq_lens={extend_seq_lens}; "
                f"got input_tokens={input_ids.numel()}, "
                f"positions={positions.numel()}."
            )
        if all(x == q_len for x in extend_seq_lens):
            return input_ids.view(batch_size, q_len), positions.view(batch_size, q_len), None

        padded_input_ids = input_ids.new_zeros((batch_size, q_len))
        padded_positions = positions.new_zeros((batch_size, q_len))
        token_start = 0
        for row, row_len in enumerate(extend_seq_lens):
            token_end = token_start + row_len
            padded_input_ids[row, :row_len] = input_ids[token_start:token_end]
            padded_positions[row, :row_len] = positions[token_start:token_end]
            if row_len < q_len:
                pad_count = q_len - row_len
                pad_start = int(padded_positions[row, row_len - 1].item()) + 1
                padded_positions[row, row_len:] = torch.arange(
                    pad_start,
                    pad_start + pad_count,
                    dtype=positions.dtype,
                    device=positions.device,
                )
            token_start = token_end
        return padded_input_ids, padded_positions, extend_seq_lens

    raise NotImplementedError(
        f"LiteCache static batching does not support forward_mode={forward_batch.forward_mode}."
    )


def _flatten_litecache_hidden_states(
    model_outputs,
    extend_seq_lens: Optional[list[int]] = None,
) -> torch.Tensor:
    hidden_states = model_outputs.last_hidden_state
    if hidden_states.dim() == 3:
        if extend_seq_lens is None or hidden_states.shape[1] == 1:
            hidden_states = hidden_states[:, -1, :]
        else:
            row_ids = torch.arange(
                hidden_states.shape[0],
                dtype=torch.long,
                device=hidden_states.device,
            )
            token_ids = torch.tensor(
                [x - 1 for x in extend_seq_lens],
                dtype=torch.long,
                device=hidden_states.device,
            )
            hidden_states = hidden_states[row_ids, token_ids, :]
    return hidden_states.reshape(-1, hidden_states.shape[-1])


def _sync_litecache_cache_rows(
    owner,
    rids: list[str],
    reset_cache: bool,
    max_cache_batch_size: int,
    prune_absent_rids: bool,
) -> None:
    """Keep LiteCache cache rows aligned with SGLang's current forward rows.

    SGLang can run a prefill batch that only contains new requests while other
    requests are still decoding. That prefill batch starts at row 0, so blindly
    resetting row 0 can erase a still-live decode request. Move such occupants
    to free rows before resetting rows for new requests.
    """

    current_rids = set(rids)
    if len(rids) > max_cache_batch_size:
        raise RuntimeError(
            "LiteCache forward batch exceeds configured cache batch size: "
            f"batch_rids={rids}, max_batch_size={max_cache_batch_size}."
        )

    if reset_cache:
        owner._cache.reset(max_cache_batch_size)
        owner._clear_decode_cuda_graphs()
        owner._cache_batch_size = max_cache_batch_size
        owner._active_rids = current_rids
        owner._active_rid_to_row = {rid: idx for idx, rid in enumerate(rids)}
        return

    if not rids:
        if prune_absent_rids:
            owner._active_rids = set()
            owner._active_rid_to_row = {}
        return

    target_by_rid = {rid: idx for idx, rid in enumerate(rids)}
    target_rows = set(target_by_rid.values())
    active_map = dict(owner._active_rid_to_row)
    row_remap: dict[int, int] = {}

    for rid, target_row in target_by_rid.items():
        old_row = active_map.get(rid)
        if old_row is not None and old_row != target_row:
            row_remap[int(old_row)] = int(target_row)

    victim_rids = [
        rid
        for rid, old_row in active_map.items()
        if rid not in target_by_rid and int(old_row) in target_rows
    ]
    occupied_static_rows = {
        int(old_row)
        for rid, old_row in active_map.items()
        if rid not in target_by_rid and rid not in victim_rids
    }
    available_rows = [
        row
        for row in range(max_cache_batch_size)
        if row not in target_rows and row not in occupied_static_rows
    ]
    if len(available_rows) < len(victim_rids):
        raise RuntimeError(
            "LiteCache has no free cache rows for dynamic batching. "
            f"rids={rids}, active_rid_to_row={active_map}, "
            f"max_batch_size={max_cache_batch_size}. "
            "Set SGLANG_MAX_RUNNING_REQUESTS <= LITECACHE_MAX_BATCH_SIZE and "
            "LITECACHE_MAX_TOKENS >= max_prompt_tokens * LITECACHE_MAX_BATCH_SIZE."
        )

    for rid, new_row in zip(victim_rids, available_rows):
        row_remap[int(active_map[rid])] = int(new_row)

    if _debug_batch_enabled() and (row_remap or victim_rids):
        logger.info(
            "LiteCache row sync: rids=%s target_by_rid=%s victims=%s "
            "row_remap=%s active_before=%s prune_absent=%s",
            rids,
            target_by_rid,
            victim_rids,
            row_remap,
            active_map,
            prune_absent_rids,
        )

    if row_remap and hasattr(owner._cache, "move_batch_rows"):
        owner._cache.move_batch_rows(row_remap)

    if hasattr(owner._cache, "reset_batch_rows"):
        new_rows = [
            target_row
            for rid, target_row in target_by_rid.items()
            if rid not in owner._active_rids
        ]
        if _debug_batch_enabled() and new_rows:
            logger.info(
                "LiteCache new rows detected: rids=%s new_rows=%s "
                "active_before=%s available_rows=%s",
                rids,
                new_rows,
                active_map,
                available_rows,
            )
        owner._cache.reset_batch_rows(new_rows)

    updated_map: dict[str, int] = {}
    for rid, old_row in owner._active_rid_to_row.items():
        if prune_absent_rids and rid not in current_rids:
            continue
        updated_map[rid] = int(row_remap.get(int(old_row), int(old_row)))
    for rid, target_row in target_by_rid.items():
        updated_map[rid] = int(target_row)

    owner._active_rid_to_row = updated_map
    owner._active_rids = set(updated_map)


def _release_litecache_finished_rid(owner, rid: str) -> None:
    """Logically release a finished request's LiteCache row immediately.

    Without this, LiteCache only prunes stale rids on a later decode batch via
    `prune_absent_rids=True`. Under high-concurrency serving, a new prefill can
    arrive in that gap and observe a full `_active_rid_to_row`, even though the
    finished requests have already returned 200 to the client.

    Keep this logical-only: free the row in the active map right away, but do
    not eagerly reset cache storage or clear decode graphs here. The next
    `_sync_litecache_cache_rows()` call already treats rows whose rid is absent
    from `_active_rids` as new/free rows and will reset them before reuse.
    Eagerly zeroing cache state here can perturb the in-flight dynamic-batching
    transition and was observed to regress follow-up requests.
    """

    rid = str(rid)
    row = owner._active_rid_to_row.pop(rid, None)
    if row is None:
        return

    owner._active_rids.discard(rid)

    if _debug_batch_enabled():
        logger.info(
            "LiteCache logically released finished rid=%s row=%s active_after=%s",
            rid,
            row,
            sorted(owner._active_rids),
        )


class LiteCacheLlamaForCausalLM(nn.Module):
    """
    SGLang model entry that delegates execution+cache to internal LiteCache
    framework (`modeling_llama_offloading_duohead.py`).
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        _ = prefix

        _ensure_llama_compatible_config(config)

        variant = _get_variant_name(config)
        logger.info("Using LiteCache variant=%s", variant)

        from sglang.litecache.kvcache_full_attn import CustomStaticCache
        from sglang.litecache.kvcache_hash import HashOffloadingCache as HashOffloadingCacheOffloading
        from sglang.litecache.kvcache_offloading_hash import (
            HashOffloadingCache as HashOffloadingCacheDuohead,
        )
        from sglang.litecache.kvcache_offloading_infinigen import (
            InfiniGenOffloadingCache,
        )
        from sglang.litecache.kvcache_offloading_loki import LokiOffloadingCache
        from sglang.litecache.kvcache_offloading_quest import QuestOffloadingCache
        from sglang.srt.models.litecache import (
            modeling_llama_full as full_impl,
            modeling_llama_offloading as offloading_impl,
            modeling_llama_offloading_duohead as duohead_impl,
        )

        variant_to_cls = {
            "fullattn": getattr(full_impl, "FullAttentionLlamaForCausalLM"),
            "offloading": getattr(offloading_impl, "OffloadingLlamaForCausalLM"),
            "hash": getattr(duohead_impl, "HashLlamaForCausalLM"),
            "loki": getattr(duohead_impl, "LokiLlamaForCausalLM"),
            "infinigen": getattr(duohead_impl, "InfiniGenLlamaForCausalLM"),
            "quest": getattr(duohead_impl, "QuestLlamaForCausalLM"),
        }
        offloading_method_to_cache_cls: dict[str, Type] = {
            "hash": HashOffloadingCacheOffloading,
            "loki": LokiOffloadingCache,
            "infinigen": InfiniGenOffloadingCache,
            "quest": QuestOffloadingCache,
        }
        variant_to_cache_cls: dict[str, Type] = {
            "fullattn": CustomStaticCache,
            "hash": HashOffloadingCacheDuohead,
            "loki": LokiOffloadingCache,
            "infinigen": InfiniGenOffloadingCache,
            "quest": QuestOffloadingCache,
        }
        if variant not in variant_to_cls:
            raise ValueError(
                f"Unknown litecache_variant={variant!r}. "
                f"Supported: {sorted(variant_to_cls.keys())}"
            )

        self.model: nn.Module = variant_to_cls[variant](config)
        self._tp_enabled = _replace_litecache_linears_with_tp(self.model, quant_config) > 0
        self._awq_enabled = False
        if should_enable_litecache_awq(quant_config):
            replaced = replace_litecache_linears_with_awq(self.model, quant_config)
            self._awq_enabled = replaced > 0
            if self._awq_enabled:
                logger.info("LiteCache Llama AWQ route enabled. replaced_linears=%d", replaced)
        self.logits_processor = LogitsProcessor(config)

        self._variant = variant
        self._offloading_method = (
            _get_offloading_method_name(config) if variant == "offloading" else None
        )
        if variant == "offloading":
            self._cache_cls = offloading_method_to_cache_cls[self._offloading_method]
            logger.info(
                "Using LiteCache offloading backend=%s",
                self._offloading_method,
            )
        else:
            self._cache_cls = variant_to_cache_cls[variant]
        self._hf_config = (
            config.get_text_config() if hasattr(config, "get_text_config") else config
        )
        self._custom_config = ensure_litecache_custom_config(
            getattr(config, "custom_config", None),
            self._hf_config,
        )
        if self._awq_enabled and bool(getattr(self._custom_config, "enable_cuda_graph", False)):
            logger.warning(
                "LiteCache AWQ path currently disables internal CUDA graph for stability."
            )
            self._custom_config.enable_cuda_graph = False
        self._cache = None
        self._cache_batch_size: Optional[int] = None
        self._active_rids: set[str] = set()
        self._active_rid_to_row: dict[str, int] = {}

    def _ensure_cache(self, device: torch.device) -> None:
        if self._cache is not None:
            return

        if device.type != "cuda":
            raise RuntimeError(
                f"LiteCache currently requires CUDA device, but got {device}."
            )

        device_idx = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        self._cache = self._cache_cls(
            config=self._hf_config,
            custom_config=self._custom_config,
            device=device_idx,
            layer_device_map=None,
        )
        self._cache.build_cache()
        logger.info(
            "Initialized LiteCache cache: variant=%s offloading_method=%s device=cuda:%d",
            self._variant,
            self._offloading_method,
            device_idx,
        )

    def _maybe_reset_cache(
        self, positions: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        if self._cache is None:
            raise RuntimeError("LiteCache cache is not initialized.")

        batch_size = int(forward_batch.batch_size)
        max_cache_batch_size = int(
            self._custom_config.kvcache_manager_config.max_batch_size
        )
        is_new_sequence = bool(
            forward_batch.forward_mode.is_extend()
            and positions.numel() > 0
            and int(torch.min(positions).item()) == 0
        )
        rids = [str(rid) for rid in list(forward_batch.rids or [])]
        current_rids = set(rids)
        reset_cache = self._cache_batch_size is None or (
            is_new_sequence and not self._active_rids
        )

        if _debug_batch_enabled():
            pos_min = int(torch.min(positions).item()) if positions.numel() > 0 else None
            pos_max = int(torch.max(positions).item()) if positions.numel() > 0 else None
            logger.info(
                "LiteCache batch bridge: mode=%s batch=%d max_cache_batch=%d "
                "is_new=%s reset=%s rids=%s active_rids=%s pos=[%s,%s]",
                forward_batch.forward_mode,
                batch_size,
                max_cache_batch_size,
                is_new_sequence,
                reset_cache,
                rids,
                sorted(self._active_rids),
                pos_min,
                pos_max,
            )

        _sync_litecache_cache_rows(
            self,
            rids,
            reset_cache,
            max_cache_batch_size,
            prune_absent_rids=forward_batch.forward_mode.is_decode(),
        )
        self._cache.curr_batch_size = batch_size
        if "gather_engine_metadata" in getattr(self._cache, "metadata_tensors", {}):
            self._cache.metadata_tensors["gather_engine_metadata"][2] = batch_size
        if _debug_batch_enabled():
            logger.info(
                "LiteCache cache view: active_batch=%d cache_batch=%s "
                "max_seq_len=%s max_buffer_len=%s gather_meta=%s",
                batch_size,
                self._cache_batch_size,
                getattr(self._cache, "max_seq_len", None),
                getattr(self._cache, "max_buffer_len", None),
                tuple(
                    int(x)
                    for x in getattr(self._cache, "metadata_tensors", {})
                    .get("gather_engine_metadata", torch.empty(0, dtype=torch.int32))[:6]
                    .detach()
                    .cpu()
                    .tolist()
                ),
            )

    def _clear_decode_cuda_graphs(self) -> None:
        """Drop LiteCache decode graphs captured for a previous request."""
        for module in self.model.modules():
            graphs = getattr(module, "_graphs", None)
            if isinstance(graphs, dict) and graphs:
                graphs.clear()

    def release_finished_rid(self, rid: str) -> None:
        _release_litecache_finished_rid(self, rid)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        _ = input_embeds

        if get_embedding:
            raise NotImplementedError(
                "get_embedding is not supported yet for LiteCache models."
            )

        self._ensure_cache(input_ids.device)
        model_input_ids, model_positions, extend_seq_lens = _prepare_litecache_forward_inputs(
            input_ids,
            positions,
            forward_batch,
        )
        if _debug_batch_enabled():
            logger.info(
                "LiteCache forward input: batch=%d input_shape=%s pos_shape=%s "
                "extend_seq_lens=%s",
                int(forward_batch.batch_size),
                tuple(model_input_ids.shape),
                tuple(model_positions.shape),
                extend_seq_lens,
            )
        self._maybe_reset_cache(model_positions, forward_batch)

        self._cache._current_extend_seq_lens = extend_seq_lens
        try:
            model_outputs = self.model.model(
                input_ids=model_input_ids,
                position_ids=model_positions,
                past_key_values=self._cache,
                use_cache=True,
                return_dict=True,
            )
        finally:
            self._cache._current_extend_seq_lens = None
        if extend_seq_lens is not None and hasattr(self._cache, "trim_prefill_padding"):
            self._cache.trim_prefill_padding(
                extend_seq_lens,
                padded_q_len=model_input_ids.shape[1],
            )
        hidden_states = _flatten_litecache_hidden_states(model_outputs, extend_seq_lens)

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.model.lm_head,
            forward_batch,
        )


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        matched_param_names = set()
        loaded = 0
        for name, loaded_weight in weights:
            candidate_names = [name, f"model.{name}"]
            if name.startswith("model."):
                candidate_names.append(f"model.model.{name[6:]}")
            else:
                candidate_names.append(f"model.model.{name}")

            for candidate in candidate_names:
                if candidate in params_dict:
                    param = params_dict[candidate]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    matched_param_names.add(candidate)
                    loaded += 1
                    break

        missing = [n for n in params_dict.keys() if n not in matched_param_names]
        non_next_missing = [n for n in missing if ".next_" not in n]
        logger.info(
            "LiteCache loaded %d tensors (model params=%d, missing=%d, non_next_missing=%d, sample_missing=%s).",
            loaded,
            len(params_dict),
            len(missing),
            len(non_next_missing),
            missing[:24],
        )
        if non_next_missing:
            logger.info("LiteCache non-next missing sample: %s", non_next_missing[:24])
        if self._awq_enabled:
            finalized = _finalize_litecache_awq_modules(self)
            logger.info("LiteCache AWQ post-load finalize done. modules=%d", finalized)


class LiteCacheQwen2ForCausalLM(nn.Module):
    """
    Dedicated LiteCache entry for Qwen2/Qwen2.5 checkpoints.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        _ = prefix

        variant = _get_variant_name(config)
        logger.info("Using LiteCache variant=%s (qwen2)", variant)

        from sglang.litecache.kvcache_full_attn import CustomStaticCache
        from sglang.litecache.kvcache_hash import HashOffloadingCache as HashOffloadingCacheOffloading
        from sglang.litecache.kvcache_offloading_hash import (
            HashOffloadingCache as HashOffloadingCacheDuohead,
        )
        from sglang.litecache.kvcache_offloading_infinigen import (
            InfiniGenOffloadingCache,
        )
        from sglang.litecache.kvcache_offloading_loki import LokiOffloadingCache
        from sglang.litecache.kvcache_offloading_quest import QuestOffloadingCache
        from sglang.srt.models.litecache import (
            modeling_qwen2_full as full_impl,
            modeling_qwen2_offloading as offloading_impl,
            modeling_qwen2_offloading_duohead as duohead_impl,
        )

        variant_to_cls = {
            "fullattn": getattr(full_impl, "FullAttentionQwen2ForCausalLM"),
            "offloading": getattr(offloading_impl, "OffloadingQwen2ForCausalLM"),
            "hash": getattr(duohead_impl, "HashQwen2ForCausalLM"),
            "loki": getattr(duohead_impl, "LokiQwen2ForCausalLM"),
            "infinigen": getattr(duohead_impl, "InfiniGenQwen2ForCausalLM"),
            "quest": getattr(duohead_impl, "QuestQwen2ForCausalLM"),
        }
        offloading_method_to_cache_cls: dict[str, Type] = {
            "hash": HashOffloadingCacheOffloading,
            "loki": LokiOffloadingCache,
            "infinigen": InfiniGenOffloadingCache,
            "quest": QuestOffloadingCache,
        }
        variant_to_cache_cls: dict[str, Type] = {
            "fullattn": CustomStaticCache,
            "hash": HashOffloadingCacheDuohead,
            "loki": LokiOffloadingCache,
            "infinigen": InfiniGenOffloadingCache,
            "quest": QuestOffloadingCache,
        }
        if variant not in variant_to_cls:
            raise ValueError(
                f"Unknown litecache_variant={variant!r}. "
                f"Supported: {sorted(variant_to_cls.keys())}"
            )

        self.model: nn.Module = variant_to_cls[variant](config)
        self._tp_enabled = _replace_litecache_linears_with_tp(self.model, quant_config) > 0
        self._awq_enabled = False
        if should_enable_litecache_awq(quant_config):
            replaced = replace_litecache_linears_with_awq(self.model, quant_config)
            self._awq_enabled = replaced > 0
            if self._awq_enabled:
                logger.info("LiteCache Qwen2 AWQ route enabled. replaced_linears=%d", replaced)
        self.logits_processor = LogitsProcessor(config)

        self._variant = variant
        self._offloading_method = (
            _get_offloading_method_name(config) if variant == "offloading" else None
        )
        if variant == "offloading":
            self._cache_cls = offloading_method_to_cache_cls[self._offloading_method]
            logger.info(
                "Using LiteCache offloading backend=%s",
                self._offloading_method,
            )
        else:
            self._cache_cls = variant_to_cache_cls[variant]
        self._hf_config = (
            config.get_text_config() if hasattr(config, "get_text_config") else config
        )
        self._custom_config = ensure_litecache_custom_config(
            getattr(config, "custom_config", None),
            self._hf_config,
        )
        if self._awq_enabled and bool(getattr(self._custom_config, "enable_cuda_graph", False)):
            logger.warning(
                "LiteCache AWQ path currently disables internal CUDA graph for stability."
            )
            self._custom_config.enable_cuda_graph = False
        self._cache = None
        self._cache_batch_size: Optional[int] = None
        self._active_rids: set[str] = set()
        self._active_rid_to_row: dict[str, int] = {}

    def _ensure_cache(self, device: torch.device) -> None:
        if self._cache is not None:
            return

        if device.type != "cuda":
            raise RuntimeError(
                f"LiteCache currently requires CUDA device, but got {device}."
            )

        device_idx = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        self._cache = self._cache_cls(
            config=self._hf_config,
            custom_config=self._custom_config,
            device=device_idx,
            layer_device_map=None,
        )
        self._cache.build_cache()
        logger.info(
            "Initialized LiteCache cache: variant=%s offloading_method=%s device=cuda:%d",
            self._variant,
            self._offloading_method,
            device_idx,
        )

    def _maybe_reset_cache(
        self, positions: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        if self._cache is None:
            raise RuntimeError("LiteCache cache is not initialized.")

        batch_size = int(forward_batch.batch_size)
        max_cache_batch_size = int(
            self._custom_config.kvcache_manager_config.max_batch_size
        )
        is_new_sequence = bool(
            forward_batch.forward_mode.is_extend()
            and positions.numel() > 0
            and int(torch.min(positions).item()) == 0
        )
        rids = [str(rid) for rid in list(forward_batch.rids or [])]
        current_rids = set(rids)
        reset_cache = self._cache_batch_size is None or (
            is_new_sequence and not self._active_rids
        )

        if _debug_batch_enabled():
            pos_min = int(torch.min(positions).item()) if positions.numel() > 0 else None
            pos_max = int(torch.max(positions).item()) if positions.numel() > 0 else None
            logger.info(
                "LiteCache batch bridge: mode=%s batch=%d max_cache_batch=%d "
                "is_new=%s reset=%s rids=%s active_rids=%s pos=[%s,%s]",
                forward_batch.forward_mode,
                batch_size,
                max_cache_batch_size,
                is_new_sequence,
                reset_cache,
                rids,
                sorted(self._active_rids),
                pos_min,
                pos_max,
            )

        _sync_litecache_cache_rows(
            self,
            rids,
            reset_cache,
            max_cache_batch_size,
            prune_absent_rids=forward_batch.forward_mode.is_decode(),
        )
        self._cache.curr_batch_size = batch_size
        if "gather_engine_metadata" in getattr(self._cache, "metadata_tensors", {}):
            self._cache.metadata_tensors["gather_engine_metadata"][2] = batch_size
        if _debug_batch_enabled():
            logger.info(
                "LiteCache cache view: active_batch=%d cache_batch=%s "
                "max_seq_len=%s max_buffer_len=%s gather_meta=%s",
                batch_size,
                self._cache_batch_size,
                getattr(self._cache, "max_seq_len", None),
                getattr(self._cache, "max_buffer_len", None),
                tuple(
                    int(x)
                    for x in getattr(self._cache, "metadata_tensors", {})
                    .get("gather_engine_metadata", torch.empty(0, dtype=torch.int32))[:6]
                    .detach()
                    .cpu()
                    .tolist()
                ),
            )

    def _clear_decode_cuda_graphs(self) -> None:
        """Drop LiteCache decode graphs captured for a previous request."""
        for module in self.model.modules():
            graphs = getattr(module, "_graphs", None)
            if isinstance(graphs, dict) and graphs:
                graphs.clear()

    def release_finished_rid(self, rid: str) -> None:
        _release_litecache_finished_rid(self, rid)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        _ = input_embeds

        if get_embedding:
            raise NotImplementedError(
                "get_embedding is not supported yet for LiteCache models."
            )

        self._ensure_cache(input_ids.device)
        model_input_ids, model_positions, extend_seq_lens = _prepare_litecache_forward_inputs(
            input_ids,
            positions,
            forward_batch,
        )
        if _debug_batch_enabled():
            logger.info(
                "LiteCache forward input: batch=%d input_shape=%s pos_shape=%s "
                "extend_seq_lens=%s",
                int(forward_batch.batch_size),
                tuple(model_input_ids.shape),
                tuple(model_positions.shape),
                extend_seq_lens,
            )
        self._maybe_reset_cache(model_positions, forward_batch)

        self._cache._current_extend_seq_lens = extend_seq_lens
        try:
            model_outputs = self.model.model(
                input_ids=model_input_ids,
                position_ids=model_positions,
                past_key_values=self._cache,
                use_cache=True,
                return_dict=True,
            )
        finally:
            self._cache._current_extend_seq_lens = None
        if extend_seq_lens is not None and hasattr(self._cache, "trim_prefill_padding"):
            self._cache.trim_prefill_padding(
                extend_seq_lens,
                padded_q_len=model_input_ids.shape[1],
            )
        hidden_states = _flatten_litecache_hidden_states(model_outputs, extend_seq_lens)

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.model.lm_head,
            forward_batch,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        matched_param_names = set()
        loaded = 0
        for name, loaded_weight in weights:
            candidate_names = [name, f"model.{name}"]
            if name.startswith("model."):
                candidate_names.append(f"model.model.{name[6:]}")
            else:
                candidate_names.append(f"model.model.{name}")

            for candidate in candidate_names:
                if candidate in params_dict:
                    param = params_dict[candidate]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    matched_param_names.add(candidate)
                    loaded += 1
                    break

        missing = [n for n in params_dict.keys() if n not in matched_param_names]
        non_next_missing = [n for n in missing if ".next_" not in n]
        logger.info(
            "LiteCache loaded %d tensors (model params=%d, missing=%d, non_next_missing=%d, sample_missing=%s).",
            loaded,
            len(params_dict),
            len(missing),
            len(non_next_missing),
            missing[:24],
        )
        if non_next_missing:
            logger.info("LiteCache non-next missing sample: %s", non_next_missing[:24])
        if self._awq_enabled:
            finalized = _finalize_litecache_awq_modules(self)
            logger.info("LiteCache AWQ post-load finalize done. modules=%d", finalized)


EntryClass = [LiteCacheLlamaForCausalLM, LiteCacheQwen2ForCausalLM]
