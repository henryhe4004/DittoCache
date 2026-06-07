from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.models.ditto.awq_linear import ditto_linear_forward


def _custom_linear_forward_wrapper(x, linear, out):
    return ditto_linear_forward(x, linear, out=out)


def _get_local_intermediate_size(self) -> int:
    if hasattr(self, "gate_proj") and hasattr(self.gate_proj, "output_size_per_partition"):
        return int(self.gate_proj.output_size_per_partition)
    if hasattr(self, "gate_up_proj") and hasattr(self.gate_up_proj, "output_size_per_partition"):
        return int(self.gate_up_proj.output_size_per_partition) // 2
    return int(self.intermediate_size)


def _ffn_prepare_cuda_graph_metadata(
    self,
    bsz,
    dtype=torch.bfloat16,
    device="cuda",
):
    use_unfused_awq = bool(getattr(self, "use_unfused_awq", False))
    local_intermediate_size = _get_local_intermediate_size(self)
    if use_unfused_awq:
        self._graph_buffers[bsz] = {
            "gate_states": torch.zeros(
                (bsz, local_intermediate_size),
                device=device,
                dtype=dtype,
            ),
            "up_states": torch.zeros(
                (bsz, local_intermediate_size),
                device=device,
                dtype=dtype,
            ),
            "activated_states": torch.zeros(
                (bsz, local_intermediate_size),
                device=device,
                dtype=dtype,
            ),
        }
        return

    self._graph_buffers[bsz] = {
        "gate_up_states": torch.zeros(
            (bsz, local_intermediate_size * 2),
            device=device,
            dtype=dtype,
        ),
        "activated_states": torch.zeros(
            (bsz, local_intermediate_size),
            device=device,
            dtype=dtype,
        ),
    }


def _fuse_gate_up_proj(self):
    import flashinfer

    if getattr(self, "use_unfused_awq", False):
        self.converted = True
        return
    if not isinstance(self.gate_proj, nn.Linear) or not isinstance(self.up_proj, nn.Linear):
        # AWQ path keeps gate/up split and does not fuse into a dense gate_up_proj.
        self.use_unfused_awq = True
        self.converted = True
        return

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
    local_intermediate_size = _get_local_intermediate_size(self)
    if getattr(self, "use_unfused_awq", False):
        if bsz > chunk_size:
            num_chunks = (bsz + chunk_size - 1) // chunk_size
            gate_chunk = torch.zeros(
                (chunk_size, local_intermediate_size),
                dtype=x.dtype,
                device=x.device,
            )
            up_chunk = torch.zeros(
                (chunk_size, local_intermediate_size),
                dtype=x.dtype,
                device=x.device,
            )
            act_chunk = torch.zeros(
                (chunk_size, local_intermediate_size),
                dtype=x.dtype,
                device=x.device,
            )
            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, bsz)
                size = end - start
                _custom_linear_forward_wrapper(
                    x[start:end, ...],
                    self.gate_proj,
                    out=gate_chunk[:size],
                )
                _custom_linear_forward_wrapper(
                    x[start:end, ...],
                    self.up_proj,
                    out=up_chunk[:size],
                )
                act_chunk[:size].copy_(gate_chunk[:size])
                act_chunk[:size].sigmoid_()
                act_chunk[:size].mul_(gate_chunk[:size])
                act_chunk[:size].mul_(up_chunk[:size])
                _custom_linear_forward_wrapper(
                    act_chunk[:size],
                    self.down_proj,
                    out=x[start:end, ...],
                )
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            x = F.silu(gate) * up
            x = self.down_proj(x)
        return x

    if bsz > chunk_size:
        num_chunks = (bsz + chunk_size - 1) // chunk_size
        chunk_intermediate1 = torch.zeros(
            (chunk_size, local_intermediate_size * 2),
            dtype=x.dtype,
            device=x.device,
        )
        chunk_intermediate2 = torch.zeros(
            (chunk_size, local_intermediate_size),
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
    if getattr(self, "use_unfused_awq", False):
        _custom_linear_forward_wrapper(
            hidden_states,
            self.gate_proj,
            out=self._graph_buffers[bsz]["gate_states"],
        )
        _custom_linear_forward_wrapper(
            hidden_states,
            self.up_proj,
            out=self._graph_buffers[bsz]["up_states"],
        )
        self._graph_buffers[bsz]["activated_states"].copy_(
            self._graph_buffers[bsz]["gate_states"]
        )
        self._graph_buffers[bsz]["activated_states"].sigmoid_()
        self._graph_buffers[bsz]["activated_states"].mul_(
            self._graph_buffers[bsz]["gate_states"]
        )
        self._graph_buffers[bsz]["activated_states"].mul_(
            self._graph_buffers[bsz]["up_states"]
        )
        _custom_linear_forward_wrapper(
            self._graph_buffers[bsz]["activated_states"],
            self.down_proj,
            out=hidden_states,
        )
        return hidden_states

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
