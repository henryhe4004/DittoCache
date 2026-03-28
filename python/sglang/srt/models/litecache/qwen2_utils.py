from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm


class CustomerQwen2MLP(Qwen2MLP):
    def __init__(self, config):
        super().__init__(config)
        self.torch_dtype = config.torch_dtype
        self.hidden_act = config.hidden_act
        assert self.hidden_act in ["silu"]

        self.converted = False
        self._graph_buffers = {}

    def forward(self, x, is_prefill=False):
        if not self.converted:
            _fuse_gate_up_proj(self)
        if is_prefill:
            return _dense_ffn_prefill_forward(self, x)

        bsz = x.shape[0]
        if bsz not in self._graph_buffers:
            _ffn_prepare_cuda_graph_metadata(
                self,
                bsz,
                x.dtype,
                x.device,
            )
        return _dense_ffn_decode_forward(self, x)


class CustomQwen2RMSNorm(Qwen2RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self._graph_buffers = {}

    def forward(self, hidden_states, is_prefill=False):
        if is_prefill:
            return _layernorm_prefill_forward(self, hidden_states)

        bsz = hidden_states.shape[0]
        if bsz not in self._graph_buffers:
            _layernorm_prepare_cuda_graph_metadata(
                self,
                bsz,
                hidden_states.shape[-1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        return _layernorm_decode_forward(self, hidden_states)


class CustomQwen2RotaryEmbedding(nn.Module):
    """RoPE adapter matching myTransformer's cache metadata API."""

    def __init__(self, config):
        super().__init__()
        import flashinfer

        if getattr(config, "rope_scaling", None) is not None:
            rope_scaling = config.rope_scaling
            self.rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        assert self.rope_type in ["default", "llama3", "linear"]

        self.fn = None
        self.fn_kwargs = {}
        if self.rope_type == "linear":
            self.fn_kwargs["interleave"] = False
            self.fn_kwargs["rope_scale"] = config.rope_scaling["factor"]
            self.fn_kwargs["rope_theta"] = config.rope_theta
            self.fn = flashinfer.apply_rope_inplace
        elif self.rope_type == "llama3":
            self.fn_kwargs["interleave"] = False
            self.fn_kwargs["high_freq_factor"] = config.rope_scaling["high_freq_factor"]
            self.fn_kwargs["low_freq_factor"] = config.rope_scaling["low_freq_factor"]
            self.fn_kwargs["rope_theta"] = config.rope_theta
            self.fn_kwargs["rope_scale"] = config.rope_scaling["factor"]
            self.fn_kwargs["old_context_len"] = config.rope_scaling["original_max_position_embeddings"]
            self.fn = flashinfer.apply_llama31_rope_inplace
        else:
            self.fn_kwargs["interleave"] = False
            self.fn_kwargs["rope_scale"] = 1
            self.fn_kwargs["rope_theta"] = config.rope_theta
            self.fn = flashinfer.apply_rope_inplace

    def forward(self, query_states, key_states, past_key_values):
        indptr, offsets = past_key_values.get_rope_metadata(query_states.device)
        self.fn(query_states, key_states, indptr, offsets, **self.fn_kwargs)
        return query_states, key_states


def _custom_linear_forward_wrapper(x, linear, out):
    weight = linear.weight.T
    bias = linear.bias
    torch.matmul(x, weight, out=out)
    if bias is not None:
        out.add_(bias.unsqueeze(0))


def _ffn_prepare_cuda_graph_metadata(
    self,
    bsz,
    dtype=torch.bfloat16,
    device="cuda",
):
    self._graph_buffers[bsz] = {
        "gate_up_states": torch.zeros(
            (bsz, self.intermediate_size * 2),
            device=device,
            dtype=dtype,
        ),
        "activated_states": torch.zeros(
            (bsz, self.intermediate_size),
            device=device,
            dtype=dtype,
        ),
    }


def _fuse_gate_up_proj(self):
    import flashinfer

    device = self.down_proj.weight.device
    self.gate_up_proj = nn.Linear(
        self.hidden_size,
        self.intermediate_size * 2,
        bias=self.gate_proj.bias is not None,
        dtype=self.torch_dtype,
        device=device,
    )
    self.gate_up_proj.weight.data[: self.intermediate_size, :] = self.gate_proj.weight.data
    self.gate_up_proj.weight.data[self.intermediate_size :, :] = self.up_proj.weight.data
    self.fn = flashinfer.activation.silu_and_mul
    del self.gate_proj
    del self.up_proj
    torch.cuda.empty_cache()
    self.converted = True


def _dense_ffn_prefill_forward(self, x):
    chunk_size = 8192
    bsz = x.shape[0]
    if bsz > chunk_size:
        num_chunks = (bsz + chunk_size - 1) // chunk_size
        chunk_intermediate1 = torch.zeros(
            (chunk_size, self.intermediate_size * 2),
            dtype=x.dtype,
            device=x.device,
        )
        chunk_intermediate2 = torch.zeros(
            (chunk_size, self.intermediate_size),
            dtype=x.dtype,
            device=x.device,
        )
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, bsz)
            torch.matmul(
                x[start:end, ...],
                self.gate_up_proj.weight.T.data,
                out=chunk_intermediate1[: end - start],
            )
            self.fn(
                chunk_intermediate1[: end - start],
                out=chunk_intermediate2[: end - start],
            )
            torch.matmul(
                chunk_intermediate2[: end - start],
                self.down_proj.weight.T.data,
                out=x[start:end, ...],
            )
    else:
        x = self.gate_up_proj(x)
        x = self.fn(x)
        x = self.down_proj(x)
    return x


def _dense_ffn_decode_forward(self, hidden_states):
    bsz = hidden_states.shape[0]
    _custom_linear_forward_wrapper(
        hidden_states,
        self.gate_up_proj,
        out=self._graph_buffers[bsz]["gate_up_states"],
    )
    self.fn(
        self._graph_buffers[bsz]["gate_up_states"],
        out=self._graph_buffers[bsz]["activated_states"],
    )
    _custom_linear_forward_wrapper(
        self._graph_buffers[bsz]["activated_states"],
        self.down_proj,
        out=hidden_states,
    )
    return hidden_states


def _layernorm_prepare_cuda_graph_metadata(
    self,
    bsz,
    hidden_size,
    dtype=torch.bfloat16,
    device="cuda",
):
    self._graph_buffers[bsz] = {
        "output": torch.zeros(
            (bsz, hidden_size),
            device=device,
            dtype=dtype,
        ),
    }


def _layernorm_prefill_forward(self, hidden_states):
    import flashinfer

    chunk_size = 8192
    bsz = hidden_states.shape[0]
    if bsz > chunk_size:
        num_chunks = (bsz + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, bsz)
            hidden_states[start:end] = flashinfer.norm.rmsnorm(
                hidden_states[start:end],
                self.weight,
                self.variance_epsilon,
            )
    else:
        hidden_states = flashinfer.norm.rmsnorm(
            hidden_states,
            self.weight,
            self.variance_epsilon,
        )
    return hidden_states


def _layernorm_decode_forward(self, hidden_states):
    import flashinfer

    bsz = hidden_states.shape[0]
    flashinfer.norm.rmsnorm(
        hidden_states,
        self.weight,
        self.variance_epsilon,
        out=self._graph_buffers[bsz]["output"],
    )
    return self._graph_buffers[bsz]["output"]
