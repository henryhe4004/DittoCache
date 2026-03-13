import torch
from functools import partial
from collections import Counter
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
    print((t1 - t0) * 1000 / 100)


BSZ = 16
NUM_HEAD = 8

for MAX_SEQ in [8000, 16000, 32000, 64000, 128000, 256000]:
    MAX_SEL = int(MAX_SEQ * 0.1)
    SEQ = MAX_SEQ // 2
    SEL = MAX_SEL // 2

    print(f"--------> Test seqlen = {SEQ}")

    data = torch.randn(BSZ, NUM_HEAD, MAX_SEQ, dtype=torch.float16,
                    device='cuda')  #  * 1000
    mask = torch.zeros((BSZ * NUM_HEAD, ), dtype=torch.bool, device='cuda')
    index = torch.arange(0, BSZ * NUM_HEAD, 2, device='cuda')
    my_output = torch.zeros((BSZ * NUM_HEAD, MAX_SEL), dtype=torch.int32, device='cuda')
    my_output_values = torch.zeros((BSZ * NUM_HEAD, MAX_SEL), dtype=torch.float16, device='cuda')
    real_len = torch.tensor([SEQ], dtype=torch.int32, device='cuda')
    real_sel = torch.tensor([SEL], dtype=torch.int32, device='cuda')

    mask[index] = True

    full_output = capi.batch_topk(
        data[..., :SEQ].contiguous(), SEL, False
    )
    full_sorted_indices = full_output.sort(dim=-1).values.view(BSZ * NUM_HEAD, -1)
    # print(full_sorted_indices)

    capi.batch_topk_masked(
        data, mask, my_output, my_output_values, real_len, real_sel, False
    )
    my_sorted_indices = my_output[..., :SEL].sort(dim=-1).values.view(BSZ * NUM_HEAD, -1)
    assert my_sorted_indices.min().item() >= 0
    assert my_sorted_indices.max().item() < SEQ
    # print(my_sorted_indices)
    # print(my_output[..., SEL:])

    diff = 0
    ref = full_sorted_indices[mask, :]
    real = my_sorted_indices[mask, :]
    print(real.max(), real.min())
    for i in range(ref.shape[0]):
        t = ref[i, :].tolist()
        m = real[i, :].tolist()
        c = Counter(t + m)
        diff = diff + 2 * len(c) - len(t) - len(m)
    print(f"diff: {diff} / total: {MAX_SEL * ref.shape[0]}")

    mask[:] = True
    print("#gather heads", mask.sum().item())
    bench(
        partial(
            capi.batch_topk_masked,
            data,
            mask,
            my_output,
            my_output_values,
            real_len,
            real_sel,
            False,
        )
    )

    mask[:] = False
    mask[index] = True
    print("#gather heads", mask.sum().item())
    bench(
        partial(
            capi.batch_topk_masked,
            data,
            mask,
            my_output,
            my_output_values,
            real_len,
            real_sel,
            False,
        )
    )
