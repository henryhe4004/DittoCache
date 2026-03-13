import torch
import time
import numpy as np

b = 1
s = 128000
h = 8
d = 128

for r in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
    res = []
    for _ in range(50):
        cpu_data = torch.empty((2, b, int(s * r), h, d), dtype=torch.float16)
        gpu_buffer = torch.empty((2, b, int(s * r), h, d),
                                 dtype=torch.float16,
                                 device="cuda")
        torch.cuda.synchronize()
        tic = time.time()
        gpu_buffer.copy_(cpu_data)
        torch.cuda.synchronize()
        toc = time.time()
        res.append(toc - tic)
    duration = np.mean(res[5:])
    bandwidth = cpu_data.element_size() * cpu_data.numel(
    ) / 1024 / 1024 / 1024 / duration
    print(
        f"b={b}, s={s}, sparse={r:.2f}, time={duration*1000000} us, bandwidth={bandwidth} GB/s"
    )
