import torch
import sgl_kernel.kvlib as capi
from functools import partial


def bench(func):
    import time
    import numpy as np

    for i in range(5):
        func()

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        func()
    torch.cuda.synchronize()
    t1 = time.time()
    print((t1 - t0) * 1000 / 100)


def torch_block_id_to_token_id(block_id, block_size, num_sink, num_recent,
                               seq_length):
    b, h, num_blocks = block_id.shape
    middle_length = num_blocks * block_size
    block_id = block_id.view(b, h, num_blocks,
                             1).expand(b, h, num_blocks, block_size)
    block_offset = torch.arange(0,
                                block_size,
                                device=block_id.device,
                                dtype=block_id.dtype).view(
                                    1, 1, 1, block_size)
    token_id = block_id * block_size + block_offset + num_sink
    token_id = token_id.view(b, h, middle_length)
    sink_id = torch.arange(0,
                           num_sink,
                           device=block_id.device,
                           dtype=block_id.dtype).view(1, 1, num_sink).expand(
                               b, h, num_sink)
    recent_id = torch.arange(seq_length - num_recent,
                             seq_length,
                             device=block_id.device,
                             dtype=block_id.dtype).view(1, 1,
                                                        num_recent).expand(
                                                            b, h, num_recent)

    token_id = torch.cat([sink_id, token_id, recent_id], dim=-1)
    return token_id


b = 32
h = 8
s = 4000
k = 400
bs = 8
nb = k // bs
head_id = torch.randperm(h * b)[:h * b // 2].cuda()
head_mask = torch.zeros(b * h, device="cuda", dtype=torch.bool)
head_mask[head_id] = True
block_id = torch.randint(0,
                         s // bs, (b, h, nb),
                         device="cuda",
                         dtype=torch.int32)
print(block_id.shape)

sink = 4
recent = 64
torch_out = torch_block_id_to_token_id(block_id, bs, sink, recent, s)
triton_out = capi.block_id_to_token_id_head_mask(
    block_id, bs, sink, recent, s, head_mask)
assert torch.equal(
    torch_out.view(b * h, -1)[head_mask, :],
    triton_out.view(b * h, -1)[head_mask, :])

sink = 0
recent = 64
torch_out = torch_block_id_to_token_id(block_id, bs, sink, recent, s)
triton_out = capi.block_id_to_token_id_head_mask(
    block_id, bs, sink, recent, s, head_mask)
assert torch.equal(
    torch_out.view(b * h, -1)[head_mask, :],
    triton_out.view(b * h, -1)[head_mask, :])

sink = 4
recent = 0
torch_out = torch_block_id_to_token_id(block_id, bs, sink, recent, s)
triton_out = capi.block_id_to_token_id_head_mask(
    block_id, bs, sink, recent, s, head_mask)
assert torch.equal(
    torch_out.view(b * h, -1)[head_mask, :],
    triton_out.view(b * h, -1)[head_mask, :])

sink = 0
recent = 0
torch_out = torch_block_id_to_token_id(block_id, bs, sink, recent, s)
triton_out = capi.block_id_to_token_id_head_mask(
    block_id, bs, sink, recent, s, head_mask)
assert torch.equal(
    torch_out.view(b * h, -1)[head_mask, :],
    triton_out.view(b * h, -1)[head_mask, :])

sink = 4
recent = 64
bench(partial(torch_block_id_to_token_id, block_id, bs, sink, recent, s))
bench(
    partial(capi.block_id_to_token_id_head_mask, block_id, bs,
            sink, recent, s, head_mask))

sink = 0
recent = 0
bench(partial(torch_block_id_to_token_id, block_id, bs, sink, recent, s))
bench(
    partial(capi.block_id_to_token_id_head_mask, block_id, bs,
            sink, recent, s, head_mask))
