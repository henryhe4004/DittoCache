import torch
from functools import partial
from test_hamming_static import hash_encode
import sgl_kernel.kvlib as capi

torch.cuda.set_device(6)
torch.manual_seed(42)


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
    latency = (t1 - t0) / 100
    print(latency * 1000)
    return latency


def torch_hamming_distance(key, query, hash_weight):
    h = query.shape[2]
    b, s, hk, _ = key.shape
    gqa = h // hk
    rbit = hash_weight.shape[-1]
    key = torch.matmul(key, hash_weight) > 0
    query = torch.matmul(query, hash_weight) > 0
    key = key.view(b, s, hk, 1, rbit).expand(-1, -1, -1, gqa,
                                             -1).reshape(b, s, h, rbit)
    hamming_distance = query.to(torch.float16) - key.to(torch.float16)
    hamming_distance = hamming_distance.abs().sum(dim=-1)
    return hamming_distance


def encode(key, query, hash_weight):
    packbit_aux_tensor = torch.pow(
        2, torch.arange(0, 32, 1, dtype=torch.int32, device="cuda"))
    key_code = hash_encode(key, hash_weight, packbit_aux_tensor)
    query_code = hash_encode(query, hash_weight, packbit_aux_tensor)
    return key_code, query_code


if __name__ == "__main__":
    b = 16
    rbit = 128
    h = 28
    hk = 4
    gqa = h // hk
    s = 4000
    key = torch.randn(b, s, hk, 128).to(torch.float16).cuda()
    query = torch.randn(b, 1, h, 128).to(torch.float16).cuda()

    mask = torch.zeros((b * hk, ), dtype=torch.bool, device='cuda')
    index = torch.arange(0, b * hk, 2, device='cuda')
    mask[index] = True

    hash_weight = torch.normal(
        0,
        2,
        size=(128, rbit),
        device=key.device,
        dtype=key.dtype,
    )
    torch_output = torch_hamming_distance(key, query, hash_weight)
    torch_output = torch_output.transpose(1, 2).view(b, hk, gqa, -1).sum(2)
    torch_output = torch_output.reshape(-1, s)

    key_code, query_code = encode(key, query, hash_weight)
    torch.cuda.synchronize()

    my_output2 = capi.hamming_score_head_mask(
        key_code, query_code, mask, rbit, s, 0, 0
    ).view(-1, s)
    print(torch_output)
    print(my_output2)
    print((my_output2[mask, :] - torch_output[mask, :]).abs().max())

    bench(
        partial(
            capi.hamming_score,
            key_code,
            query_code,
            rbit,
            s,
            0,
            0,
        )
    )

    mask[:] = True
    print("#gather heads", mask.sum().item())
    bench(
        partial(
            capi.hamming_score_head_mask,
            key_code,
            query_code,
            mask,
            rbit,
            s,
            0,
            0,
        )
    )
