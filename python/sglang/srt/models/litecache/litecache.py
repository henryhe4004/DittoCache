"""
LiteCache model integration.

This entry binds to SGLang-internal ports of myTransformer duohead files:
- Cache implementations in `sglang.litecache.*`
- Model framework in `sglang.srt.models.litecache.modeling_llama_offloading_duohead`
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


def _get_variant_name(config: PretrainedConfig) -> str:
    # Allow setting via HF config.json: "litecache_variant": "loki"
    v = getattr(config, "litecache_variant", None)
    if v is None:
        return "loki"
    return str(v).lower()


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
        _ = quant_config
        _ = prefix

        variant = _get_variant_name(config)
        logger.info("Using LiteCache duohead variant=%s", variant)
        from sglang.srt.models.litecache import (
            modeling_llama_offloading_duohead as duohead_impl,
        )

        variant_to_cls = {
            "hash": getattr(duohead_impl, "HashLlamaForCausalLM"),
            "loki": getattr(duohead_impl, "LokiLlamaForCausalLM"),
            "infinigen": getattr(duohead_impl, "InfiniGenLlamaForCausalLM"),
            "quest": getattr(duohead_impl, "QuestLlamaForCausalLM"),
        }
        if variant not in variant_to_cls:
            raise ValueError(
                f"Unknown litecache_variant={variant!r}. "
                f"Supported: {sorted(variant_to_cls.keys())}"
            )

        self.model: nn.Module = variant_to_cls[variant](config)
        self.logits_processor = LogitsProcessor(config)

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
            raise NotImplementedError("get_embedding is not supported yet for LiteCacheLlamaForCausalLM.")

        hidden_states = self.model.model(
            input_ids[None, ...],
            use_cache=False,
            position_ids=positions[None, ...],
            return_dict=False,
        )[0][0, ...]

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.model.lm_head,
            forward_batch,
        )


EntryClass = LiteCacheLlamaForCausalLM

