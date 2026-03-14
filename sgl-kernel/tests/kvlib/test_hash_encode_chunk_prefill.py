from sglang.jit_kernel.triton_kernels.hash.prefill_encode import (
    hash_encode_append_prefill,
)
import torch


def torch_hash_encode(data: torch.Tensor, hash_weight: torch.Tensor,
                      packbit_aux_tensor: torch.Tensor):
    output_dtype = torch.int32
    chunk_size = 32

    RBIT = hash_weight.shape[-1]
    chunk_num = RBIT // chunk_size

    BSZ, SEQ, HEAD, HEAD_DIM = data.shape
    output_shape = (BSZ, SEQ, HEAD, chunk_num, chunk_size)

    key_code = torch.einsum("bshd,hdr->bshr", data, hash_weight) > 0
    packbit_key_code = key_code.reshape(*output_shape)
    packbit_key_code = packbit_key_code * packbit_aux_tensor
    packbit_key_code = packbit_key_code.sum(dim=-1, dtype=output_dtype)

    return packbit_key_code


torch.cuda.set_device(0)
torch.manual_seed(42)
device = "cuda"
dtype = torch.bfloat16

BSZ = 16
HEAD = 8
HEAD_DIM = 128
RBIT = 256
hash_weight = torch.normal(
    0,
    2,
    size=(HEAD, HEAD_DIM, RBIT),
    device=device,
    dtype=dtype,
)
packbit_aux_tensor = torch.pow(
    2, torch.arange(0, 32, 1, dtype=torch.int32, device=device))

SEQ = 15000
CHUNK_SIZE = 4096
key_states = torch.randn((BSZ, SEQ, HEAD, HEAD_DIM),
                         dtype=dtype,
                         device=device)
torch_output = torch.empty((BSZ, SEQ * 2, HEAD, int(RBIT / 32)),
                            dtype=torch.int32,
                            device=device)
triton_output = torch.empty((BSZ, SEQ * 2, HEAD, int(RBIT / 32)),
                            dtype=torch.int32,
                            device=device)
seqlen = torch.zeros((1,), dtype=torch.int32, device=device)

num_chunks = (SEQ + CHUNK_SIZE - 1) // CHUNK_SIZE
for i in range(0, num_chunks):
    begin = i * CHUNK_SIZE
    end = min((i + 1) * CHUNK_SIZE, SEQ)
    print(i, begin, end)

    torch_code = torch_hash_encode(key_states[:, begin:end, ...], hash_weight, packbit_aux_tensor)
    hash_encode_append_prefill(key_states[:, begin:end, ...], triton_output, hash_weight, seqlen, packbit_aux_tensor)
    assert (torch_code == triton_output[:, begin:end, :, :]).all()
    seqlen[0] = seqlen[0].item() + end - begin
    torch_output[:, begin:end, ...] = torch_code

assert (torch_output[:, :SEQ, ...] == triton_output[:, :SEQ, ...]).all()
