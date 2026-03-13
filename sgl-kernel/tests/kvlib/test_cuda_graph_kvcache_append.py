import torch
import time
import sgl_kernel.kvlib as capi

def cuda_graph_kvcache_append_demo():
    # 测试参数
    s_values = [1000, 5000, 10000, 50000]
    B, H, D = 16, 8, 128
    for s in s_values:
        print("-" * 40)
        print(f"Test s = {s}......", end="")

        data_src = torch.randn(2, B, s, H, D, device='cuda', dtype=torch.bfloat16)
        k = torch.zeros(B, 1, H, D, device='cuda', dtype=torch.bfloat16)
        v = torch.zeros(B, 1, H, D, device='cuda', dtype=torch.bfloat16)

        torch.cuda.synchronize()
        cache0 = torch.zeros(2, B, s, H, D, device='cuda', dtype=torch.bfloat16)
        for i in range(s):
            k[:] = data_src[0, :, i:i+1, ...]
            v[:] = data_src[1, :, i:i+1, ...]
            cache0[0, :, i:i+1, ...] = k
            cache0[1, :, i:i+1, ...] = v
        torch.cuda.synchronize()
        assert torch.allclose(data_src, cache0)

        # 使用CUDA Graph
        cache1 = torch.zeros(2, B, s, H, D, device='cuda', dtype=torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        idx = torch.tensor([0], dtype=torch.int32, device='cuda')
        k[:] = data_src[0, :, 0:1, ...]
        v[:] = data_src[1, :, 0:1, ...]
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            capi.kvcache_append_tensor_pos(cache1, k, v, idx)
            idx += 1
        torch.cuda.synchronize()

        # 重放阶段
        for i in range(s):
            k[:] = data_src[0, :, i:i+1, ...]
            v[:] = data_src[1, :, i:i+1, ...]
            graph.replay()
        torch.cuda.synchronize()
        assert torch.allclose(data_src, cache1)

        print("Passed")


def cuda_graph_kvcache_append_masked_demo():
    # 测试参数
    s_values = [1000, 5000, 10000, 50000]
    B, H, D = 16, 8, 128
    for s in s_values:
        print("-" * 40)
        print(f"Test s = {s}......", end="")

        data_src = torch.randn(2, B, s, H, D, device='cuda', dtype=torch.bfloat16)
        k = torch.zeros(B, 1, H, D, device='cuda', dtype=torch.bfloat16)
        v = torch.zeros(B, 1, H, D, device='cuda', dtype=torch.bfloat16)
        head_ids = torch.randperm(H)[:H // 2].to('cuda')

        torch.cuda.synchronize()
        cache0 = torch.zeros(2, B, s, H // 2, D, device='cuda', dtype=torch.bfloat16)
        for i in range(s):
            k[:] = data_src[0, :, i:i+1, ...]
            v[:] = data_src[1, :, i:i+1, ...]
            cache0[0, :, i:i+1, ...] = k[:, :, head_ids, :]
            cache0[1, :, i:i+1, ...] = v[:, :, head_ids, :]
        torch.cuda.synchronize()
        assert torch.allclose(data_src[:, :, :, head_ids], cache0)

        # 使用CUDA Graph
        cache1 = torch.zeros(2, B, s, H // 2, D, device='cuda', dtype=torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        idx = torch.tensor([0], dtype=torch.int32, device='cuda')
        k[:] = data_src[0, :, 0:1, ...]
        v[:] = data_src[1, :, 0:1, ...]
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            capi.kvcache_append_tensor_pos_head_sparse(cache1, k, v, head_ids, idx)
            idx += 1
        torch.cuda.synchronize()

        # 重放阶段
        for i in range(s):
            k[:] = data_src[0, :, i:i+1, ...]
            v[:] = data_src[1, :, i:i+1, ...]
            graph.replay()
        torch.cuda.synchronize()
        assert torch.allclose(data_src[:, :, :, head_ids], cache1)

        print("Passed")


if __name__ == "__main__":
    print("Test kvcache append......")
    cuda_graph_kvcache_append_demo()

    print("Test kvcache masked append......")
    cuda_graph_kvcache_append_masked_demo()
