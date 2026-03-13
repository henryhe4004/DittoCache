import torch


def create_aligned_cuda_tensor(data_numel, dtype, device, pagesize):
    # Calculate raw data size in bytes
    data_size = data_numel * dtype.itemsize
    total_bytes = (data_size + pagesize - 1)  # Minimum required bytes
    aligned_numel = (total_bytes + dtype.itemsize - 1) // dtype.itemsize

    # Allocate the buffer (page-aligned due to byte-size rounding)
    raw_data = torch.zeros((aligned_numel, ), dtype=dtype, device=device)

    # Calculate required alignment offset
    data_ptr = raw_data.data_ptr()
    aligned_data_ptr = (data_ptr + pagesize -
                        1) // pagesize * pagesize  # Next page-aligned address

    offset_bytes = aligned_data_ptr - data_ptr
    align_skip_numel = offset_bytes // dtype.itemsize

    # Slice to get aligned tensor (ensures aligned_data starts at aligned address)
    aligned_data = raw_data[align_skip_numel:align_skip_numel + data_numel]

    return raw_data, aligned_data


b, s, h, d = 1, 3371, 4, 128
dtype = torch.float16
device = "cuda"
pagesize = 65536

raw_datas = []
aligned_datas = []

layer = 32

for _ in range(layer):
    data_numel = 2 * b * s * h * d
    raw_data, aligned_data = create_aligned_cuda_tensor(
        data_numel, dtype, device, pagesize)

    raw_datas.append(raw_data)
    aligned_datas.append(aligned_data)

    assert aligned_data.data_ptr() % pagesize == 0, aligned_data.data_ptr()
