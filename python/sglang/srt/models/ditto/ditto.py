"""
Ditto model integration.

This entry binds to SGLang-internal ports of internal prototype Ditto files:
- Cache implementations in `sglang.ditto.*`
- Model frameworks in both non-duohead offloading and duohead variants
"""

from __future__ import annotations

import gc
import logging
import os
import re
from typing import Iterable, NamedTuple, Optional, Tuple, Type

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.ditto.config_utils import ensure_ditto_custom_config
from sglang.ditto.tp_head_mapping import (
    query_head_order,
    reorder_head_axis,
    resolve_tp_kv_head_orders,
)
from sglang.srt.distributed import (
    get_pp_group,
    get_pp_indices,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.ditto.awq_linear import (
    replace_ditto_linears_with_awq,
    should_enable_ditto_awq,
)
from sglang.srt.models.transformers import replace_linear_class

logger = logging.getLogger(__name__)


def _debug_batch_enabled() -> bool:
    return os.environ.get("DITTO_DEBUG_BATCH", "0") == "1"


def _finalize_ditto_awq_modules(root: nn.Module) -> int:
    finalized = 0
    for module in root.modules():
        if not getattr(module, "_ditto_awq_linear", False):
            continue
        process_fn = getattr(module, "process_weights_after_loading", None)
        if callable(process_fn):
            process_fn()
            finalized += 1
    return finalized


def _configure_ditto_pipeline_stage(
    model: nn.Module,
    config: PretrainedConfig,
    pp_group,
) -> tuple[int, int]:
    """Keep only this PP rank's transformer layers and remap cache layer ids."""

    backbone = model.model
    total_layers = int(config.num_hidden_layers)
    start_layer, end_layer = get_pp_indices(
        total_layers,
        pp_group.rank_in_group,
        pp_group.world_size,
    )

    if not 0 <= start_layer < end_layer <= total_layers:
        raise ValueError(
            "Ditto requires at least one layer per PP stage; "
            f"got layers=[{start_layer},{end_layer})/{total_layers}."
        )

    for global_layer_idx in range(total_layers):
        if not start_layer <= global_layer_idx < end_layer:
            backbone.layers[global_layer_idx] = PPMissingLayer(return_tuple=True)

    local_layers = [backbone.layers[i] for i in range(start_layer, end_layer)]
    for local_layer_idx, layer in enumerate(local_layers):
        attention = layer.self_attn
        attention.global_layer_idx = start_layer + local_layer_idx
        attention.layer_idx = local_layer_idx

    # Cross-layer query prediction can only reference modules owned by this
    # process. The final local layer wraps to local layer 0; that layer is made
    # resident by the PP cache policy, avoiding a boundary H2D
    # dependency that cannot be overlapped without an extra PP side channel.
    for local_layer_idx, layer in enumerate(local_layers):
        attention = layer.self_attn
        if not hasattr(attention, "next_input_layernorm"):
            continue
        next_layer = local_layers[(local_layer_idx + 1) % len(local_layers)]
        attention.next_input_layernorm = next_layer.input_layernorm
        attention.next_q_proj = next_layer.self_attn.q_proj
        attention.next_rotary_emb = next_layer.self_attn.rotary_emb

    backbone.start_layer = start_layer
    backbone.end_layer = end_layer
    backbone.local_num_layers = end_layer - start_layer
    backbone.is_first_pp_rank = pp_group.is_first_rank
    backbone.is_last_pp_rank = pp_group.is_last_rank

    if not pp_group.is_first_rank:
        backbone.embed_tokens = PPMissingLayer()
    if not pp_group.is_last_rank:
        backbone.norm = PPMissingLayer()
        model.lm_head = PPMissingLayer()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return start_layer, end_layer


class _DittoTPHeadLayout(NamedTuple):
    tp_rank: int
    tp_size: int
    attn_tp_rank: int
    attn_tp_size: int
    total_num_heads: int
    total_num_key_value_heads: int
    local_num_heads: int
    local_num_key_value_heads: int
    query_head_start: int
    kv_head_start: int
    kv_head_replicas: int

    @property
    def kv_linear_replicated(self) -> bool:
        return self.tp_size > 1 and self.total_num_key_value_heads < self.tp_size

    @property
    def kv_linear_tp_rank(self) -> int:
        if not self.kv_linear_replicated:
            return self.tp_rank
        return self.kv_head_start

    @property
    def kv_linear_tp_size(self) -> int:
        if not self.kv_linear_replicated:
            return self.tp_size
        return self.total_num_key_value_heads


def _get_ditto_text_config(config: PretrainedConfig) -> PretrainedConfig:
    return config.get_text_config() if hasattr(config, "get_text_config") else config


def _get_ditto_total_kv_heads(config: PretrainedConfig) -> int:
    if hasattr(config, "num_key_value_heads"):
        return int(config.num_key_value_heads)
    if hasattr(config, "multi_query_group_num"):
        return int(config.multi_query_group_num)
    return int(config.num_attention_heads)


def _get_ditto_tensor_parallel_info() -> tuple[int, int]:
    try:
        return (
            int(get_tensor_model_parallel_rank()),
            int(get_tensor_model_parallel_world_size()),
        )
    except Exception:
        return 0, 1


def _get_ditto_attention_parallel_info() -> tuple[int, int, bool, int]:
    try:
        from sglang.srt.layers.dp_attention import (  # pylint: disable=import-outside-toplevel
            get_attention_cp_size,
            get_attention_tp_rank,
            get_attention_tp_size,
            is_dp_attention_enabled,
        )

        return (
            int(get_attention_tp_rank()),
            int(get_attention_tp_size()),
            bool(is_dp_attention_enabled()),
            int(get_attention_cp_size()),
        )
    except Exception:
        tp_rank, tp_size = _get_ditto_tensor_parallel_info()
        return tp_rank, tp_size, False, 1


def _maybe_get_global_server_args():
    try:
        from sglang.srt.server_args import get_global_server_args  # pylint: disable=import-outside-toplevel

        return get_global_server_args()
    except Exception:
        return None


def _build_ditto_tp_head_layout(config: PretrainedConfig) -> _DittoTPHeadLayout:
    text_config = _get_ditto_text_config(config)
    tp_rank, tp_size = _get_ditto_tensor_parallel_info()
    attn_tp_rank, attn_tp_size, dp_attention_enabled, attn_cp_size = (
        _get_ditto_attention_parallel_info()
    )

    if tp_size <= 0:
        raise ValueError(f"Invalid tensor parallel size: {tp_size}")
    if attn_tp_size <= 0:
        raise ValueError(f"Invalid attention tensor parallel size: {attn_tp_size}")

    if tp_size > 1:
        if dp_attention_enabled or attn_tp_size != tp_size or attn_tp_rank != tp_rank:
            raise NotImplementedError(
                "Ditto TP currently supports pure single-node tensor parallel only. "
                f"Got tp_rank/size={tp_rank}/{tp_size}, "
                f"attn_tp_rank/size={attn_tp_rank}/{attn_tp_size}, "
                f"dp_attention_enabled={dp_attention_enabled}."
            )
        if attn_cp_size != 1:
            raise NotImplementedError(
                "Ditto TP does not support attention context parallelism yet. "
                f"Got attn_cp_size={attn_cp_size}."
            )

        server_args = _maybe_get_global_server_args()
        if server_args is not None:
            if int(getattr(server_args, "nnodes", 1)) != 1:
                raise NotImplementedError(
                    "Ditto TP currently supports single-node tensor parallel only. "
                    f"Got nnodes={getattr(server_args, 'nnodes', None)}."
                )
            if bool(getattr(server_args, "enable_attn_tp_input_scattered", False)):
                raise NotImplementedError(
                    "Ditto TP bypasses SGLang's standard attention backend and "
                    "does not support enable_attn_tp_input_scattered yet."
                )

    total_num_heads = int(text_config.num_attention_heads)
    total_num_kv_heads = _get_ditto_total_kv_heads(text_config)
    if total_num_heads <= 0 or total_num_kv_heads <= 0:
        raise ValueError(
            "Invalid Ditto attention head config: "
            f"num_attention_heads={total_num_heads}, "
            f"num_key_value_heads={total_num_kv_heads}."
        )
    if total_num_heads % attn_tp_size != 0:
        raise ValueError(
            f"Ditto TP requires num_attention_heads={total_num_heads} to be "
            f"divisible by attn_tp_size={attn_tp_size}."
        )

    local_num_heads = total_num_heads // attn_tp_size
    query_head_start = attn_tp_rank * local_num_heads
    if attn_tp_size <= 1:
        local_num_kv_heads = total_num_kv_heads
        kv_head_start = 0
        kv_head_replicas = 1
    elif total_num_kv_heads >= attn_tp_size:
        if total_num_kv_heads % attn_tp_size != 0:
            raise ValueError(
                f"Ditto TP requires num_key_value_heads={total_num_kv_heads} "
                f"to be divisible by attn_tp_size={attn_tp_size}."
            )
        local_num_kv_heads = total_num_kv_heads // attn_tp_size
        kv_head_start = attn_tp_rank * local_num_kv_heads
        kv_head_replicas = 1
    else:
        if attn_tp_size % total_num_kv_heads != 0:
            raise ValueError(
                f"Ditto TP requires attn_tp_size={attn_tp_size} to be "
                f"divisible by num_key_value_heads={total_num_kv_heads} "
                "when KV heads are replicated."
            )
        local_num_kv_heads = 1
        kv_head_replicas = attn_tp_size // total_num_kv_heads
        kv_head_start = attn_tp_rank // kv_head_replicas

    if local_num_heads % local_num_kv_heads != 0:
        raise ValueError(
            f"Ditto TP requires local num_heads={local_num_heads} to be "
            f"divisible by local num_key_value_heads={local_num_kv_heads}."
        )

    return _DittoTPHeadLayout(
        tp_rank=tp_rank,
        tp_size=tp_size,
        attn_tp_rank=attn_tp_rank,
        attn_tp_size=attn_tp_size,
        total_num_heads=total_num_heads,
        total_num_key_value_heads=total_num_kv_heads,
        local_num_heads=local_num_heads,
        local_num_key_value_heads=local_num_kv_heads,
        query_head_start=query_head_start,
        kv_head_start=kv_head_start,
        kv_head_replicas=kv_head_replicas,
    )


def _validate_ditto_tp_runtime(config: PretrainedConfig) -> None:
    layout = _build_ditto_tp_head_layout(config)
    if layout.tp_size > 1:
        logger.info(
            "Ditto TP layout: tp=%d/%d q_heads=%d:%d kv_heads=%d:%d "
            "kv_replicas=%d.",
            layout.tp_rank,
            layout.tp_size,
            layout.query_head_start,
            layout.query_head_start + layout.local_num_heads,
            layout.kv_head_start,
            layout.kv_head_start + layout.local_num_key_value_heads,
            layout.kv_head_replicas,
        )


def _maybe_disable_ditto_cuda_graph_for_tp(custom_config, tp_enabled: bool) -> None:
    if not tp_enabled or not bool(getattr(custom_config, "enable_cuda_graph", False)):
        return
    if os.environ.get("DITTO_TP_ENABLE_CUDA_GRAPH", "0") == "1":
        logger.warning(
            "Ditto TP internal CUDA graph is enabled by DITTO_TP_ENABLE_CUDA_GRAPH=1."
        )
        return
    logger.warning(
        "Ditto TP disables internal Ditto CUDA graph by default. "
        "Set DITTO_TP_ENABLE_CUDA_GRAPH=1 to opt in after validating NCCL graph replay."
    )
    custom_config.enable_cuda_graph = False


def _refresh_ditto_prefetch_aliases(model: nn.Module) -> int:
    backbone = getattr(model, "model", model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        return 0

    try:
        num_layers = len(layers)
    except TypeError:
        return 0
    if num_layers <= 0:
        return 0

    start_layer = getattr(backbone, "start_layer", 0)
    end_layer = getattr(backbone, "end_layer", num_layers)
    refreshed = 0
    for idx in range(start_layer, end_layer):
        layer = layers[idx]
        next_layer = layers[idx + 1 if idx + 1 < end_layer else start_layer]
        attn = getattr(layer, "self_attn", None)
        next_attn = getattr(next_layer, "self_attn", None)
        if attn is None or next_attn is None:
            continue
        if not hasattr(attn, "next_q_proj"):
            continue
        attn.next_input_layernorm = getattr(next_layer, "input_layernorm", None)
        attn.next_q_proj = getattr(next_attn, "q_proj", None)
        attn.next_rotary_emb = getattr(next_attn, "rotary_emb", None)
        refreshed += 1
    return refreshed


def _ditto_load_params_dict(model: nn.Module) -> dict[str, torch.nn.Parameter]:
    try:
        iterator = model.named_parameters(remove_duplicate=False)
    except TypeError:
        iterator = model.named_parameters()
    return {
        name: param
        for name, param in iterator
        if ".next_" not in name
    }


def _maybe_reorder_ditto_attention_weight(
    name: str,
    loaded_weight: torch.Tensor,
    config: PretrainedConfig,
    kv_orders: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    """Apply the layer's semantic head permutation to attention projections."""
    layer_match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    if layer_match is None:
        return loaded_weight
    layer_idx = int(layer_match.group(1))
    if layer_idx >= len(kv_orders):
        raise ValueError(
            f"Weight {name!r} refers to layer {layer_idx}, but only "
            f"{len(kv_orders)} Ditto head mappings were configured"
        )
    kv_order = kv_orders[layer_idx]
    if kv_order == tuple(range(len(kv_order))):
        return loaded_weight

    text_config = _get_ditto_text_config(config)
    total_q_heads = int(text_config.num_attention_heads)
    total_kv_heads = _get_ditto_total_kv_heads(text_config)
    q_order = query_head_order(kv_order, total_q_heads, total_kv_heads)

    if name.endswith((".q_proj.weight", ".q_proj.bias")):
        return reorder_head_axis(loaded_weight, q_order, axis=0)
    if name.endswith(
        (
            ".k_proj.weight",
            ".k_proj.bias",
            ".v_proj.weight",
            ".v_proj.bias",
        )
    ):
        return reorder_head_axis(loaded_weight, kv_order, axis=0)
    if name.endswith(".o_proj.weight"):
        return reorder_head_axis(loaded_weight, q_order, axis=1)
    return loaded_weight


def _replace_ditto_linears_with_tp(
    model: nn.Module,
    quant_config,
    config: PretrainedConfig,
) -> int:
    _, tp_size = _get_ditto_tensor_parallel_info()
    if tp_size <= 1:
        return 0
    layout = _build_ditto_tp_head_layout(config)

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
    named_modules = list(model.named_modules(remove_duplicate=False))
    module_index = dict(named_modules)
    for full_name, module in named_modules:
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = full_name.split(".")[-1]
        style = style_by_leaf.get(leaf_name)
        if style is None:
            continue
        if ".next_" in full_name:
            continue
        if "." not in full_name:
            continue

        parent_name, attr_name = full_name.rsplit(".", 1)
        parent = module_index.get(parent_name)
        if parent is None:
            continue

        tp_kwargs = {}
        if leaf_name in {"k_proj", "v_proj"} and layout.kv_linear_replicated:
            tp_kwargs = {
                "tp_rank": layout.kv_linear_tp_rank,
                "tp_size": layout.kv_linear_tp_size,
            }

        new_module = replace_linear_class(module, style, quant_config, **tp_kwargs)
        # TP linear biases are loaded later via each parameter's weight_loader.
        # Eagerly copying HF full bias into a sharded TP bias breaks colwise layers.
        setattr(parent, attr_name, new_module)
        replaced += 1

    if replaced > 0:
        logger.info(
            "Ditto TP enabled: replaced %d linear modules with TP-aware layers "
            "(tp_size=%d, kv_linear_replicated=%s, kv_linear_tp=%d/%d).",
            replaced,
            tp_size,
            layout.kv_linear_replicated,
            layout.kv_linear_tp_rank,
            layout.kv_linear_tp_size,
        )
    return replaced


def _get_variant_name(config: PretrainedConfig) -> str:
    """
    Resolve Ditto top-k variant with explicit override priority:
    1) `config.ditto_variant`
    2) `config.custom_config.ditto_variant` / `config.custom_config.topk_variant`
    3) env `DITTO_VARIANT`
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
        ("config.ditto_variant", getattr(config, "ditto_variant", None)),
        ("config.custom_config.ditto_variant", _get_custom_value("ditto_variant")),
        ("config.custom_config.topk_variant", _get_custom_value("topk_variant")),
        ("env.DITTO_VARIANT", os.environ.get("DITTO_VARIANT")),
    ]

    for source, raw in candidates:
        variant = _normalize(raw)
        if variant is None:
            continue
        if variant not in supported:
            raise ValueError(
                f"Unknown Ditto variant from {source}: {raw!r}. "
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
        ("env.DITTO_OFFLOADING_METHOD", os.environ.get("DITTO_OFFLOADING_METHOD")),
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
    Ditto duohead classes are based on HF Llama modules.
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
                "Cannot infer `head_dim` for DittoLlamaForCausalLM: "
                f"hidden_size={hidden_size}, num_attention_heads={num_heads}."
            )
        setattr(config, "head_dim", hidden_size // num_heads)


def _prepare_ditto_forward_inputs(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
) -> tuple[torch.Tensor, torch.Tensor, Optional[list[int]]]:
    """Convert SGLang's flat batch tensors to Ditto's [bsz, q_len] layout."""

    batch_size = int(forward_batch.batch_size)
    if batch_size <= 0:
        raise ValueError(f"Ditto batch_size must be positive, got {batch_size}.")

    if batch_size == 1:
        return input_ids[None, ...], positions[None, ...], None

    if forward_batch.forward_mode.is_decode():
        if input_ids.numel() != batch_size or positions.numel() != batch_size:
            raise NotImplementedError(
                "Ditto static batching expects one decode token per request. "
                f"Got input_tokens={input_ids.numel()}, positions={positions.numel()}, "
                f"batch_size={batch_size}."
            )
        return input_ids.view(batch_size, 1), positions.view(batch_size, 1), None

    if forward_batch.forward_mode.is_extend():
        extend_seq_lens = forward_batch.extend_seq_lens_cpu
        if extend_seq_lens is None or len(extend_seq_lens) != batch_size:
            raise NotImplementedError(
                "Ditto static batching requires per-request extend lengths. "
                f"Got extend_seq_lens_cpu={extend_seq_lens}, batch_size={batch_size}."
            )

        extend_seq_lens = [int(x) for x in extend_seq_lens]
        q_len = max(extend_seq_lens)
        if q_len <= 0:
            raise NotImplementedError(
                "Ditto batching requires positive extend lengths. "
                f"Got extend_seq_lens={extend_seq_lens}."
            )

        expected_tokens = sum(extend_seq_lens)
        if input_ids.numel() != expected_tokens or positions.numel() != expected_tokens:
            raise NotImplementedError(
                "Ditto batching received inconsistent extend metadata. "
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
        f"Ditto static batching does not support forward_mode={forward_batch.forward_mode}."
    )


def _flatten_ditto_hidden_states(
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


def _get_ditto_pp_stage_inputs(
    owner,
    model_input_ids: torch.Tensor,
    pp_proxy_tensors: Optional[PPProxyTensors],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if owner.pp_group.is_first_rank:
        return model_input_ids, None
    if pp_proxy_tensors is None:
        raise RuntimeError(
            f"Ditto PP rank {owner.pp_group.rank_in_group} did not receive hidden_states."
        )

    hidden_states = pp_proxy_tensors["hidden_states"]
    batch_size, seq_len = model_input_ids.shape
    if hidden_states.dim() == 2:
        expected_tokens = batch_size * seq_len
        if hidden_states.shape[0] != expected_tokens:
            raise RuntimeError(
                "Ditto PP hidden-state shape mismatch: "
                f"shape={tuple(hidden_states.shape)}, expected_tokens={expected_tokens}."
            )
        hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
    elif hidden_states.dim() != 3:
        raise RuntimeError(
            "Ditto PP expects hidden_states with rank 2 or 3, "
            f"got shape={tuple(hidden_states.shape)}."
        )
    if hidden_states.shape[:2] != (batch_size, seq_len):
        raise RuntimeError(
            "Ditto PP hidden-state shape mismatch: "
            f"shape={tuple(hidden_states.shape)}, "
            f"expected_batch_seq={(batch_size, seq_len)}."
        )
    return None, hidden_states


def _sync_ditto_cache_rows(
    owner,
    rids: list[str],
    reset_cache: bool,
    max_cache_batch_size: int,
    prune_absent_rids: bool,
) -> None:
    """Keep Ditto cache rows aligned with SGLang's current forward rows.

    SGLang can run a prefill batch that only contains new requests while other
    requests are still decoding. That prefill batch starts at row 0, so blindly
    resetting row 0 can erase a still-live decode request. Move such occupants
    to free rows before resetting rows for new requests.
    """

    current_rids = set(rids)
    if len(rids) > max_cache_batch_size:
        raise RuntimeError(
            "Ditto forward batch exceeds configured cache batch size: "
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
            "Ditto has no free cache rows for dynamic batching. "
            f"rids={rids}, active_rid_to_row={active_map}, "
            f"max_batch_size={max_cache_batch_size}. "
            "Set SGLANG_MAX_RUNNING_REQUESTS <= DITTO_MAX_BATCH_SIZE and "
            "DITTO_MAX_TOKENS >= max_prompt_tokens * DITTO_MAX_BATCH_SIZE."
        )

    for rid, new_row in zip(victim_rids, available_rows):
        row_remap[int(active_map[rid])] = int(new_row)

    if _debug_batch_enabled() and (row_remap or victim_rids):
        logger.info(
            "Ditto row sync: rids=%s target_by_rid=%s victims=%s "
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
                "Ditto new rows detected: rids=%s new_rows=%s "
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


def _release_ditto_finished_rid(owner, rid: str) -> None:
    """Logically release a finished request's Ditto row immediately.

    Without this, Ditto only prunes stale rids on a later decode batch via
    `prune_absent_rids=True`. Under high-concurrency serving, a new prefill can
    arrive in that gap and observe a full `_active_rid_to_row`, even though the
    finished requests have already returned 200 to the client.

    Keep this logical-only: free the row in the active map right away, but do
    not eagerly reset cache storage or clear decode graphs here. The next
    `_sync_ditto_cache_rows()` call already treats rows whose rid is absent
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
            "Ditto logically released finished rid=%s row=%s active_after=%s",
            rid,
            row,
            sorted(owner._active_rids),
        )


class DittoLlamaForCausalLM(nn.Module):
    """
    SGLang model entry that delegates execution+cache to internal Ditto
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
        _validate_ditto_tp_runtime(config)

        variant = _get_variant_name(config)
        logger.info("Using Ditto variant=%s", variant)

        from sglang.ditto.kvcache_full_attn import CustomStaticCache
        from sglang.ditto.kvcache_hash import HashOffloadingCache as HashOffloadingCacheOffloading
        from sglang.ditto.kvcache_offloading_hash import (
            HashOffloadingCache as HashOffloadingCacheDuohead,
        )
        from sglang.ditto.kvcache_offloading_infinigen import (
            InfiniGenOffloadingCache,
        )
        from sglang.ditto.kvcache_offloading_loki import LokiOffloadingCache
        from sglang.ditto.kvcache_offloading_quest import QuestOffloadingCache
        from sglang.srt.models.ditto import (
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
                f"Unknown ditto_variant={variant!r}. "
                f"Supported: {sorted(variant_to_cls.keys())}"
            )

        self.pp_group = get_pp_group()
        self.model: nn.Module = variant_to_cls[variant](config)
        self.start_layer, self.end_layer = _configure_ditto_pipeline_stage(
            self.model,
            config,
            self.pp_group,
        )
        self._tp_enabled = (
            _replace_ditto_linears_with_tp(self.model, quant_config, config) > 0
        )
        self._awq_enabled = False
        if should_enable_ditto_awq(quant_config):
            replaced = replace_ditto_linears_with_awq(self.model, quant_config)
            self._awq_enabled = replaced > 0
            if self._awq_enabled:
                logger.info("Ditto Llama AWQ route enabled. replaced_linears=%d", replaced)
        refreshed = _refresh_ditto_prefetch_aliases(self.model)
        if refreshed:
            logger.info("Ditto refreshed %d prefetch projection aliases.", refreshed)
        self.logits_processor = LogitsProcessor(config)

        self._variant = variant
        self._offloading_method = (
            _get_offloading_method_name(config) if variant == "offloading" else None
        )
        if variant == "offloading":
            self._cache_cls = offloading_method_to_cache_cls[self._offloading_method]
            logger.info(
                "Using Ditto offloading backend=%s",
                self._offloading_method,
            )
        else:
            self._cache_cls = variant_to_cache_cls[variant]
        self._hf_config = (
            config.get_text_config() if hasattr(config, "get_text_config") else config
        )
        self._custom_config = ensure_ditto_custom_config(
            getattr(config, "custom_config", None),
            self._hf_config,
        )
        self._hf_config._ditto_pp_start_layer = self.start_layer
        self._hf_config._ditto_pp_end_layer = self.end_layer
        global_skip_layers = int(self._custom_config.offload_config.num_skip_layers)
        local_skip_layers = max(
            min(global_skip_layers - self.start_layer, self.end_layer - self.start_layer),
            0,
        )
        if (
            self.pp_group.world_size > 1
            and variant != "fullattn"
            and self._custom_config.offload_config.prefetch_mode == "cross_layer"
            and self._custom_config.offload_config.resident_policy != "none"
        ):
            local_skip_layers = max(local_skip_layers, 1)
        self._custom_config.offload_config.num_skip_layers = local_skip_layers
        logger.info(
            "Ditto PP stage configured: rank=%d/%d layers=[%d,%d) "
            "cache_layers=%d local_skip_layers=%d",
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
            self.start_layer,
            self.end_layer,
            self.end_layer - self.start_layer,
            local_skip_layers,
        )
        if self._awq_enabled and bool(getattr(self._custom_config, "enable_cuda_graph", False)):
            logger.warning(
                "Ditto AWQ path currently disables internal CUDA graph for stability."
            )
            self._custom_config.enable_cuda_graph = False
        _maybe_disable_ditto_cuda_graph_for_tp(self._custom_config, self._tp_enabled)
        self._cache = None
        self._cache_batch_size: Optional[int] = None
        self._active_rids: set[str] = set()
        self._active_rid_to_row: dict[str, int] = {}

    def _ensure_cache(self, device: torch.device) -> None:
        if self._cache is not None:
            return

        if device.type != "cuda":
            raise RuntimeError(
                f"Ditto currently requires CUDA device, but got {device}."
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
            "Initialized Ditto cache: variant=%s offloading_method=%s device=cuda:%d",
            self._variant,
            self._offloading_method,
            device_idx,
        )

    def _maybe_reset_cache(
        self, positions: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        if self._cache is None:
            raise RuntimeError("Ditto cache is not initialized.")

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
                "Ditto batch bridge: mode=%s batch=%d max_cache_batch=%d "
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

        _sync_ditto_cache_rows(
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
                "Ditto cache view: active_batch=%d cache_batch=%s "
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
        """Drop Ditto decode graphs captured for a previous request."""
        for module in self.model.modules():
            graphs = getattr(module, "_graphs", None)
            if isinstance(graphs, dict) and graphs:
                graphs.clear()

    def release_finished_rid(self, rid: str) -> None:
        _release_ditto_finished_rid(self, rid)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput | PPProxyTensors:
        _ = input_embeds
        if get_embedding:
            raise NotImplementedError(
                "get_embedding is not supported yet for Ditto models."
            )

        self._ensure_cache(input_ids.device)
        model_input_ids, model_positions, extend_seq_lens = _prepare_ditto_forward_inputs(
            input_ids,
            positions,
            forward_batch,
        )
        if _debug_batch_enabled():
            logger.info(
                "Ditto forward input: batch=%d input_shape=%s pos_shape=%s "
                "extend_seq_lens=%s",
                int(forward_batch.batch_size),
                tuple(model_input_ids.shape),
                tuple(model_positions.shape),
                extend_seq_lens,
            )
        self._maybe_reset_cache(model_positions, forward_batch)
        stage_input_ids, stage_input_embeds = _get_ditto_pp_stage_inputs(
            self, model_input_ids, pp_proxy_tensors
        )

        self._cache._current_extend_seq_lens = extend_seq_lens
        try:
            model_outputs = self.model.model(
                input_ids=stage_input_ids,
                inputs_embeds=stage_input_embeds,
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
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": model_outputs.last_hidden_state.contiguous()}
            )
        hidden_states = _flatten_ditto_hidden_states(model_outputs, extend_seq_lens)

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.model.lm_head,
            forward_batch,
        )


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = _ditto_load_params_dict(self)
        kv_orders = resolve_tp_kv_head_orders(
            _get_ditto_total_kv_heads(self._hf_config),
            int(_get_ditto_text_config(self._hf_config).num_hidden_layers),
        )
        matched_param_names = set()
        loaded = 0
        for name, loaded_weight in weights:
            loaded_weight = _maybe_reorder_ditto_attention_weight(
                name,
                loaded_weight,
                self._hf_config,
                kv_orders,
            )
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
            "Ditto loaded %d tensors (model params=%d, missing=%d, non_next_missing=%d, sample_missing=%s).",
            loaded,
            len(params_dict),
            len(missing),
            len(non_next_missing),
            missing[:24],
        )
        if non_next_missing:
            logger.info("Ditto non-next missing sample: %s", non_next_missing[:24])
        if self._awq_enabled:
            finalized = _finalize_ditto_awq_modules(self)
            logger.info("Ditto AWQ post-load finalize done. modules=%d", finalized)


class DittoQwen2ForCausalLM(nn.Module):
    """
    Dedicated Ditto entry for Qwen2/Qwen2.5 checkpoints.
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
        _validate_ditto_tp_runtime(config)
        logger.info("Using Ditto variant=%s (qwen2)", variant)

        from sglang.ditto.kvcache_full_attn import CustomStaticCache
        from sglang.ditto.kvcache_hash import HashOffloadingCache as HashOffloadingCacheOffloading
        from sglang.ditto.kvcache_offloading_hash import (
            HashOffloadingCache as HashOffloadingCacheDuohead,
        )
        from sglang.ditto.kvcache_offloading_infinigen import (
            InfiniGenOffloadingCache,
        )
        from sglang.ditto.kvcache_offloading_loki import LokiOffloadingCache
        from sglang.ditto.kvcache_offloading_quest import QuestOffloadingCache
        from sglang.srt.models.ditto import (
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
                f"Unknown ditto_variant={variant!r}. "
                f"Supported: {sorted(variant_to_cls.keys())}"
            )

        self.pp_group = get_pp_group()
        self.model: nn.Module = variant_to_cls[variant](config)
        self.start_layer, self.end_layer = _configure_ditto_pipeline_stage(
            self.model,
            config,
            self.pp_group,
        )
        self._tp_enabled = (
            _replace_ditto_linears_with_tp(self.model, quant_config, config) > 0
        )
        self._awq_enabled = False
        if should_enable_ditto_awq(quant_config):
            replaced = replace_ditto_linears_with_awq(self.model, quant_config)
            self._awq_enabled = replaced > 0
            if self._awq_enabled:
                logger.info("Ditto Qwen2 AWQ route enabled. replaced_linears=%d", replaced)
        refreshed = _refresh_ditto_prefetch_aliases(self.model)
        if refreshed:
            logger.info("Ditto refreshed %d prefetch projection aliases.", refreshed)
        self.logits_processor = LogitsProcessor(config)

        self._variant = variant
        self._offloading_method = (
            _get_offloading_method_name(config) if variant == "offloading" else None
        )
        if variant == "offloading":
            self._cache_cls = offloading_method_to_cache_cls[self._offloading_method]
            logger.info(
                "Using Ditto offloading backend=%s",
                self._offloading_method,
            )
        else:
            self._cache_cls = variant_to_cache_cls[variant]
        self._hf_config = (
            config.get_text_config() if hasattr(config, "get_text_config") else config
        )
        self._custom_config = ensure_ditto_custom_config(
            getattr(config, "custom_config", None),
            self._hf_config,
        )
        self._hf_config._ditto_pp_start_layer = self.start_layer
        self._hf_config._ditto_pp_end_layer = self.end_layer
        global_skip_layers = int(self._custom_config.offload_config.num_skip_layers)
        local_skip_layers = max(
            min(global_skip_layers - self.start_layer, self.end_layer - self.start_layer),
            0,
        )
        if (
            self.pp_group.world_size > 1
            and variant != "fullattn"
            and self._custom_config.offload_config.prefetch_mode == "cross_layer"
            and self._custom_config.offload_config.resident_policy != "none"
        ):
            local_skip_layers = max(local_skip_layers, 1)
        self._custom_config.offload_config.num_skip_layers = local_skip_layers
        logger.info(
            "Ditto PP stage configured: rank=%d/%d layers=[%d,%d) "
            "cache_layers=%d local_skip_layers=%d",
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
            self.start_layer,
            self.end_layer,
            self.end_layer - self.start_layer,
            local_skip_layers,
        )
        if self._awq_enabled and bool(getattr(self._custom_config, "enable_cuda_graph", False)):
            logger.warning(
                "Ditto AWQ path currently disables internal CUDA graph for stability."
            )
            self._custom_config.enable_cuda_graph = False
        _maybe_disable_ditto_cuda_graph_for_tp(self._custom_config, self._tp_enabled)
        self._cache = None
        self._cache_batch_size: Optional[int] = None
        self._active_rids: set[str] = set()
        self._active_rid_to_row: dict[str, int] = {}

    def _ensure_cache(self, device: torch.device) -> None:
        if self._cache is not None:
            return

        if device.type != "cuda":
            raise RuntimeError(
                f"Ditto currently requires CUDA device, but got {device}."
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
            "Initialized Ditto cache: variant=%s offloading_method=%s device=cuda:%d",
            self._variant,
            self._offloading_method,
            device_idx,
        )

    def _maybe_reset_cache(
        self, positions: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        if self._cache is None:
            raise RuntimeError("Ditto cache is not initialized.")

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
                "Ditto batch bridge: mode=%s batch=%d max_cache_batch=%d "
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

        _sync_ditto_cache_rows(
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
                "Ditto cache view: active_batch=%d cache_batch=%s "
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
        """Drop Ditto decode graphs captured for a previous request."""
        for module in self.model.modules():
            graphs = getattr(module, "_graphs", None)
            if isinstance(graphs, dict) and graphs:
                graphs.clear()

    def release_finished_rid(self, rid: str) -> None:
        _release_ditto_finished_rid(self, rid)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput | PPProxyTensors:
        _ = input_embeds
        if get_embedding:
            raise NotImplementedError(
                "get_embedding is not supported yet for Ditto models."
            )

        self._ensure_cache(input_ids.device)
        model_input_ids, model_positions, extend_seq_lens = _prepare_ditto_forward_inputs(
            input_ids,
            positions,
            forward_batch,
        )
        if _debug_batch_enabled():
            logger.info(
                "Ditto forward input: batch=%d input_shape=%s pos_shape=%s "
                "extend_seq_lens=%s",
                int(forward_batch.batch_size),
                tuple(model_input_ids.shape),
                tuple(model_positions.shape),
                extend_seq_lens,
            )
        self._maybe_reset_cache(model_positions, forward_batch)
        stage_input_ids, stage_input_embeds = _get_ditto_pp_stage_inputs(
            self, model_input_ids, pp_proxy_tensors
        )

        self._cache._current_extend_seq_lens = extend_seq_lens
        try:
            model_outputs = self.model.model(
                input_ids=stage_input_ids,
                inputs_embeds=stage_input_embeds,
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
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": model_outputs.last_hidden_state.contiguous()}
            )
        hidden_states = _flatten_ditto_hidden_states(model_outputs, extend_seq_lens)

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.model.lm_head,
            forward_batch,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = _ditto_load_params_dict(self)
        kv_orders = resolve_tp_kv_head_orders(
            _get_ditto_total_kv_heads(self._hf_config),
            int(_get_ditto_text_config(self._hf_config).num_hidden_layers),
        )
        matched_param_names = set()
        loaded = 0
        for name, loaded_weight in weights:
            loaded_weight = _maybe_reorder_ditto_attention_weight(
                name,
                loaded_weight,
                self._hf_config,
                kv_orders,
            )
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
            "Ditto loaded %d tensors (model params=%d, missing=%d, non_next_missing=%d, sample_missing=%s).",
            loaded,
            len(params_dict),
            len(missing),
            len(non_next_missing),
            missing[:24],
        )
        if non_next_missing:
            logger.info("Ditto non-next missing sample: %s", non_next_missing[:24])
        if self._awq_enabled:
            finalized = _finalize_ditto_awq_modules(self)
            logger.info("Ditto AWQ post-load finalize done. modules=%d", finalized)


EntryClass = [DittoLlamaForCausalLM, DittoQwen2ForCausalLM]
