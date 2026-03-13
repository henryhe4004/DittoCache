import torch
from functools import partial
import sgl_kernel.kvlib as capi

from collections import Counter

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
SEQ = 4000
SEL = 400
NUM_HEAD = 4
data = torch.randn(BSZ, NUM_HEAD, SEQ, dtype=torch.float16,
                   device='cuda')  #  * 1000
mask = torch.zeros((BSZ * NUM_HEAD, ), dtype=torch.bool, device='cuda')
index = torch.arange(0, BSZ * NUM_HEAD, 2, device='cuda')
mask[index] = True
my_output = torch.zeros((BSZ * NUM_HEAD, SEL), dtype=torch.int32, device='cuda')
my_output_values = torch.zeros((BSZ * NUM_HEAD, SEL), dtype=torch.float16, device='cuda')
real_len = torch.tensor([SEQ], dtype=torch.int32, device='cuda')
real_sel = torch.tensor([SEL], dtype=torch.int32, device='cuda')

full_output = capi.batch_topk(data, SEL, False)
full_sorted_indices = full_output.sort(dim=-1).values.view(BSZ * NUM_HEAD, -1)
print(full_sorted_indices)

capi.batch_topk_masked(data, mask, my_output, my_output_values, real_len, real_sel, False)
my_sorted_indices = my_output.sort(dim=-1).values.view(BSZ * NUM_HEAD, -1)
print(my_sorted_indices)

diff = 0
ref = full_sorted_indices[mask, :]
real = my_sorted_indices[mask, :]
for i in range(ref.shape[0]):
    t = ref[i, :].tolist()
    m = real[i, :].tolist()
    c = Counter(t + m)
    diff = diff + 2 * len(c) - len(t) - len(m)
print(f"diff: {diff} / total: {SEL * ref.shape[0]}")

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
