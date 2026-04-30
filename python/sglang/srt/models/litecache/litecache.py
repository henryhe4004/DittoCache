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
        if batch_size != 1:
            raise NotImplementedError(
                "LiteCache SGLang bridge currently supports batch_size=1 only. "
                f"Got batch_size={batch_size}."
            )

        is_new_sequence = bool(
            forward_batch.forward_mode.is_extend()
            and positions.numel() > 0
            and int(torch.min(positions).item()) == 0
        )

        if self._cache_batch_size != batch_size or is_new_sequence:
            self._cache.reset(batch_size)
            self._cache_batch_size = batch_size

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
        self._maybe_reset_cache(positions, forward_batch)

        model_outputs = self.model.model(
            input_ids=input_ids[None, ...],
            position_ids=positions[None, ...],
            past_key_values=self._cache,
            use_cache=True,
            return_dict=True,
        )
        hidden_states = model_outputs.last_hidden_state[0, ...]

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
        if batch_size != 1:
            raise NotImplementedError(
                "LiteCache SGLang bridge currently supports batch_size=1 only. "
                f"Got batch_size={batch_size}."
            )

        is_new_sequence = bool(
            forward_batch.forward_mode.is_extend()
            and positions.numel() > 0
            and int(torch.min(positions).item()) == 0
        )

        if self._cache_batch_size != batch_size or is_new_sequence:
            self._cache.reset(batch_size)
            self._cache_batch_size = batch_size

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
        self._maybe_reset_cache(positions, forward_batch)

        model_outputs = self.model.model(
            input_ids=input_ids[None, ...],
            position_ids=positions[None, ...],
            past_key_values=self._cache,
            use_cache=True,
            return_dict=True,
        )
        hidden_states = model_outputs.last_hidden_state[0, ...]

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
