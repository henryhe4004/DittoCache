from __future__ import annotations

import torch
import torch.nn as nn


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
