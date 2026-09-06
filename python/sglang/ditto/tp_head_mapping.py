from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Sequence

import torch


TP_KV_HEAD_ORDER_ENV = "DITTO_TP_KV_HEAD_ORDER"
TP_KV_HEAD_ORDER_FILE_ENV = "DITTO_TP_KV_HEAD_ORDER_FILE"
RESIDENT_KV_HEADS_FILE_ENV = "DITTO_RESIDENT_HEADS_FILE"


def _validate_order(
    order: Sequence[int],
    total_kv_heads: int,
    *,
    source: str,
) -> tuple[int, ...]:
    try:
        normalized = tuple(int(value) for value in order)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must contain integer head IDs") from exc

    expected = tuple(range(total_kv_heads))
    if len(normalized) != total_kv_heads or tuple(sorted(normalized)) != expected:
        raise ValueError(
            f"{source} must permute {list(expected)}, got {list(normalized)}"
        )
    return normalized


@lru_cache(maxsize=8)
def _load_layer_orders(path: str, total_kv_heads: int) -> tuple[tuple[int, ...], ...]:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if isinstance(payload, dict):
        payload = payload.get("orders")
    if not isinstance(payload, list) or not payload:
        raise ValueError(
            f"{TP_KV_HEAD_ORDER_FILE_ENV} must contain a non-empty JSON list "
            "of per-layer permutations, or an object with an 'orders' list"
        )
    return tuple(
        _validate_order(
            order,
            total_kv_heads,
            source=f"{TP_KV_HEAD_ORDER_FILE_ENV} layer {layer_idx}",
        )
        for layer_idx, order in enumerate(payload)
    )


def resolve_tp_kv_head_order(
    total_kv_heads: int,
    layer_idx: int | None = None,
) -> tuple[int, ...]:
    """Return the logical-to-original KV-head permutation for Ditto TP."""
    total_kv_heads = int(total_kv_heads)
    if total_kv_heads <= 0:
        raise ValueError(f"total_kv_heads must be positive, got {total_kv_heads}")

    order_file = os.environ.get(TP_KV_HEAD_ORDER_FILE_ENV, "").strip()
    if order_file:
        if layer_idx is None:
            raise ValueError(
                f"layer_idx is required when {TP_KV_HEAD_ORDER_FILE_ENV} is set"
            )
        orders = _load_layer_orders(order_file, total_kv_heads)
        if layer_idx < 0 or layer_idx >= len(orders):
            raise ValueError(
                f"{TP_KV_HEAD_ORDER_FILE_ENV} has {len(orders)} layers, "
                f"but layer {layer_idx} was requested"
            )
        return orders[layer_idx]

    raw = os.environ.get(TP_KV_HEAD_ORDER_ENV, "").strip()
    if not raw or raw.lower() in {"linear", "contiguous", "identity"}:
        return tuple(range(total_kv_heads))
    if raw.lower().startswith("balanced:"):
        raw = raw.split(":", 1)[1]

    try:
        order = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as exc:
        raise ValueError(
            f"{TP_KV_HEAD_ORDER_ENV} must be a comma-separated integer permutation, got {raw!r}"
        ) from exc

    return _validate_order(order, total_kv_heads, source=TP_KV_HEAD_ORDER_ENV)


def resolve_tp_kv_head_orders(
    total_kv_heads: int,
    num_layers: int,
) -> tuple[tuple[int, ...], ...]:
    """Return one logical-to-original KV-head permutation per model layer."""
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    order_file = os.environ.get(TP_KV_HEAD_ORDER_FILE_ENV, "").strip()
    if order_file:
        orders = _load_layer_orders(order_file, int(total_kv_heads))
        if len(orders) != num_layers:
            raise ValueError(
                f"{TP_KV_HEAD_ORDER_FILE_ENV} must define exactly {num_layers} layers, "
                f"got {len(orders)}"
            )
        return orders
    order = resolve_tp_kv_head_order(total_kv_heads)
    return tuple(order for _ in range(num_layers))


@lru_cache(maxsize=8)
def _load_resident_kv_heads(
    path: str,
    total_kv_heads: int,
) -> tuple[tuple[int, ...], ...]:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if isinstance(payload, dict):
        payload = payload.get("resident_heads")
    if not isinstance(payload, list) or not payload:
        raise ValueError(
            f"{RESIDENT_KV_HEADS_FILE_ENV} must contain a non-empty JSON list "
            "of per-layer original head IDs, or an object with a 'resident_heads' list"
        )

    layers = []
    for layer_idx, heads in enumerate(payload):
        if not isinstance(heads, list):
            raise ValueError(
                f"{RESIDENT_KV_HEADS_FILE_ENV} layer {layer_idx} must be a list"
            )
        try:
            normalized = tuple(int(head) for head in heads)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{RESIDENT_KV_HEADS_FILE_ENV} layer {layer_idx} must contain integer head IDs"
            ) from exc
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                f"{RESIDENT_KV_HEADS_FILE_ENV} layer {layer_idx} contains duplicate heads: "
                f"{list(normalized)}"
            )
        invalid = [head for head in normalized if not 0 <= head < total_kv_heads]
        if invalid:
            raise ValueError(
                f"{RESIDENT_KV_HEADS_FILE_ENV} layer {layer_idx} contains invalid heads "
                f"{invalid}; expected IDs in [0, {total_kv_heads})"
            )
        layers.append(tuple(sorted(normalized)))
    return tuple(layers)


def resolve_resident_kv_heads(
    total_kv_heads: int,
    num_layers: int,
) -> tuple[tuple[int, ...], ...] | None:
    """Return explicit per-layer resident original head IDs when configured."""
    path = os.environ.get(RESIDENT_KV_HEADS_FILE_ENV, "").strip()
    if not path:
        return None
    layers = _load_resident_kv_heads(path, int(total_kv_heads))
    if len(layers) != num_layers:
        raise ValueError(
            f"{RESIDENT_KV_HEADS_FILE_ENV} must define exactly {num_layers} layers, "
            f"got {len(layers)}"
        )
    return layers


def local_kv_head_ids(
    total_kv_heads: int,
    tp_rank: int,
    tp_size: int,
    layer_idx: int | None = None,
) -> tuple[int, ...]:
    """Return original KV-head IDs owned by one rank after permutation."""
    order = resolve_tp_kv_head_order(total_kv_heads, layer_idx=layer_idx)
    if tp_size <= 1:
        return order
    if total_kv_heads < tp_size:
        if order != tuple(range(total_kv_heads)):
            raise ValueError(
                f"{TP_KV_HEAD_ORDER_ENV} does not support replicated KV heads "
                f"(kv_heads={total_kv_heads}, tp_size={tp_size})"
            )
        replicas = tp_size // total_kv_heads
        return (tp_rank // replicas,)
    if total_kv_heads % tp_size != 0:
        raise ValueError(
            f"KV heads ({total_kv_heads}) must be divisible by TP size ({tp_size})"
        )
    local_count = total_kv_heads // tp_size
    start = tp_rank * local_count
    return order[start : start + local_count]


def query_head_order(
    kv_head_order: Sequence[int],
    total_query_heads: int,
    total_kv_heads: int,
) -> tuple[int, ...]:
    """Expand a KV-head permutation to its complete GQA query-head groups."""
    if total_query_heads % total_kv_heads != 0:
        raise ValueError(
            f"Query heads ({total_query_heads}) must be divisible by KV heads ({total_kv_heads})"
        )
    group_size = total_query_heads // total_kv_heads
    return tuple(
        query_head
        for kv_head in kv_head_order
        for query_head in range(kv_head * group_size, (kv_head + 1) * group_size)
    )


def reorder_head_axis(
    tensor: torch.Tensor,
    head_order: Sequence[int],
    *,
    axis: int,
) -> torch.Tensor:
    """Reorder a packed head axis while preserving each head's feature block."""
    head_count = len(head_order)
    axis = axis % tensor.ndim
    axis_size = int(tensor.shape[axis])
    if axis_size % head_count != 0:
        raise ValueError(
            f"Tensor shape {tuple(tensor.shape)} axis {axis} cannot be split into {head_count} heads"
        )
    width = axis_size // head_count
    moved = tensor.movedim(axis, 0)
    unpacked = moved.reshape(head_count, width, *moved.shape[1:])
    index = torch.tensor(head_order, dtype=torch.long, device=tensor.device)
    reordered = unpacked.index_select(0, index).reshape(moved.shape)
    return reordered.movedim(0, axis).contiguous()
