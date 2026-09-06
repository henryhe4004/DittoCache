from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from sglang.srt.model_loader.weight_utils import default_weight_loader

logger = logging.getLogger(__name__)

_AWQ_TARGET_LINEAR_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def _get_quant_name(quant_config) -> Optional[str]:
    if quant_config is None:
        return None
    get_name = getattr(quant_config, "get_name", None)
    if callable(get_name):
        try:
            return str(get_name()).lower()
        except TypeError:
            return str(get_name).lower()
    return None


def should_enable_ditto_awq(quant_config) -> bool:
    return _get_quant_name(quant_config) in {"awq", "awq_marlin"}


class DittoAWQLinear(nn.Module):
    """
    Minimal AWQ linear adapter for Ditto model paths that are built on
    HuggingFace nn.Linear modules.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        quant_config,
        params_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.input_size = int(in_features)
        self.output_size = int(out_features)
        self.input_size_per_partition = self.input_size
        self.output_size_per_partition = self.output_size
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self._ditto_awq_linear = True
        self._weights_post_processed = False

        quant_name = _get_quant_name(quant_config)
        if quant_name is None:
            raise ValueError("DittoAWQLinear requires a valid quantization config.")
        self._quant_name = quant_name

        from sglang.srt.layers.quantization.awq import (
            AWQLinearMethod,
            AWQMarlinLinearMethod,
        )

        if quant_name == "awq_marlin":
            self.quant_method = AWQMarlinLinearMethod(quant_config)
        elif quant_name == "awq":
            self.quant_method = AWQLinearMethod(quant_config)
        else:
            raise ValueError(f"Unsupported quantization method for Ditto AWQ: {quant_name}")

        self.quant_method.create_weights(
            self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=[self.output_size],
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.output_size, dtype=self.params_dtype))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, quant_config):
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            quant_config=quant_config,
            params_dtype=linear.weight.dtype,
        )
        if linear.bias is not None:
            module.bias.data.copy_(linear.bias.data)
        return module

    def weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor):
        default_weight_loader(param, loaded_weight)

    def process_weights_after_loading(self) -> None:
        if self._weights_post_processed:
            return
        self.quant_method.process_weights_after_loading(self)
        self._weights_post_processed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, x, bias=self.bias)

    def forward_with_out(self, x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.forward(x)
        if out is not None:
            out.copy_(y)
            return out
        return y


def replace_ditto_linears_with_awq(model: nn.Module, quant_config) -> int:
    if not should_enable_ditto_awq(quant_config):
        return 0

    replaced = 0
    named_modules = list(model.named_modules(remove_duplicate=False))
    module_index = dict(named_modules)
    for full_name, module in named_modules:
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = full_name.split(".")[-1]
        if leaf_name not in _AWQ_TARGET_LINEAR_NAMES:
            continue
        if full_name.endswith("lm_head"):
            continue
        if ".next_" in full_name:
            continue
        if "." not in full_name:
            continue

        parent_name, attr_name = full_name.rsplit(".", 1)
        parent = module_index.get(parent_name)
        if parent is None:
            continue

        awq_linear = DittoAWQLinear.from_linear(module, quant_config)
        setattr(parent, attr_name, awq_linear)
        replaced += 1

    if replaced > 0:
        logger.info(
            "Ditto AWQ enabled: replaced %d nn.Linear modules with DittoAWQLinear.",
            replaced,
        )
    return replaced


def ditto_linear_forward(
    x: torch.Tensor,
    linear: nn.Module,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if hasattr(linear, "forward_with_out"):
        return linear.forward_with_out(x, out=out)

    if isinstance(linear, nn.Linear):
        if out is None:
            return linear(x)

        weight = linear.weight.T
        bias = linear.bias
        torch.matmul(x, weight, out=out)
        if bias is not None:
            out.add_(bias.unsqueeze(0))
        return out

    y = linear(x)
    if isinstance(y, tuple):
        y = y[0]
    if out is not None:
        out.copy_(y)
        return out
    return y
