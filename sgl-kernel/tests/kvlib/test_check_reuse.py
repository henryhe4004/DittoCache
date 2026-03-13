import torch
from myTransformer.kernels.cache.check_reuse import check_reuse_head_threshold_with_gpu_head


if __name__ == "__main__":
    curr_query = torch.randn((1, 1, 9, 128),
                             device="cuda",
                             dtype=torch.float16)
    prev_query = torch.randn((1, 1, 9, 128),
                             device="cuda",
                             dtype=torch.float16)

    prev_query_cloned = prev_query.clone()
    mask = torch.zeros((1, 3), dtype=bool, device="cuda")
    cosine = torch.cosine_similarity(curr_query, prev_query,
                                     dim=-1).view(curr_query.shape[2])

    threshold = torch.tensor(
        [1, -1, 1, -1, 1, 1, -1, -1, 1],
        device=curr_query.device,
        dtype=torch.float64,
    )
    query_cache_valid = torch.zeros((1, ), dtype=bool, device="cuda")
    gpu_head_mask = torch.Tensor([0, 0, 0])
    gpu_head_mask = gpu_head_mask.bool().to(curr_query.device)

    mask[:] = False
    query_cache_valid[:] = False

    print(curr_query)
    print(prev_query)
    print(cosine)
    print(threshold)
    print(gpu_head_mask)
    print(query_cache_valid)

    torch.cuda.synchronize()

    check_reuse_head_threshold_with_gpu_head(
        curr_query, prev_query, gpu_head_mask, mask, threshold, query_cache_valid)
    print("------------------------")
    print(mask)
    for h in range(9):
        print(h)
        print(curr_query[0, 0, h, :4])
        print(prev_query[0, 0, h, :4])
