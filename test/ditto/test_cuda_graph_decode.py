"""Numerical regression: the token that captures a graph must execute it too."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("variant", ["fullattn", "offloading"])
@pytest.mark.parametrize("last_stage", [False, True])
def test_capture_step_returns_current_token(variant, last_stage):
    from sglang.srt.models.ditto.full_attn_ops import llm_decode_forward
    from sglang.srt.models.ditto.offloading_ops import llm_sparse_offloading_decode_forward

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.Identity()
            self.self_attn = SimpleNamespace(q_proj=nn.Identity(), rotary_emb=nn.Identity())

        def forward(self, hidden_states, past_key_value):
            return hidden_states + 3

    class Norm(nn.Module):
        def forward(self, hidden_states, is_prefill):
            return hidden_states * 2

    model = SimpleNamespace(layers=[Layer()], norm=Norm(), _graphs={},
                            _graph_buffers={}, is_last_pp_rank=last_stage)
    cache = SimpleNamespace(config=SimpleNamespace(enable_cuda_graph=True),
                            attn_tp_size=1, update_metadata=lambda q_len: None,
                            record_decode_transfer_step=lambda: None,
                            record_decode_step=lambda: None)
    forward = llm_decode_forward if variant == "fullattn" else llm_sparse_offloading_decode_forward
    # Warmup, capture, replay: distinct values reveal stale warmup output.
    for value in (1.0, 4.0, 9.0):
        hidden = torch.full((1, 1, 4), value, device="cuda")
        output = forward(model, inputs_embeds=hidden, past_key_values=cache)
        expected = (hidden + 3) * (2 if last_stage else 1)
        torch.testing.assert_close(output.last_hidden_state, expected)
    assert len(model._graphs) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("initial_k", [0, 2])
@pytest.mark.parametrize("backend", ["kvcache_hash", "kvcache_offloading_hash"])
@pytest.mark.parametrize("is_prefetch", [False, True])
def test_hash_graph_topk_can_grow_after_capture(monkeypatch, initial_k, backend, is_prefetch):
    import importlib

    implementation = importlib.import_module(f"sglang.ditto.{backend}")

    capacity = 32
    indices = torch.zeros((1, 1, capacity), dtype=torch.int32, device="cuda")
    k = torch.tensor([initial_k], dtype=torch.int32, device="cuda")
    columns = torch.arange(capacity, dtype=torch.int32, device="cuda").view(1, 1, -1)

    # Isolate graph buffer lifetime/shape from hash scoring and native top-k.
    # This deterministic top-k still reads the requested k from a CUDA tensor.
    def select_indices(score, mask, output, values, lengths, requested_k, largest):
        output.copy_(torch.where(columns < requested_k[0], columns, 0))

    monkeypatch.setattr(implementation.KVLib, "static_hamming_score_mask", lambda *a: None)
    monkeypatch.setattr(implementation.KVLib, "batch_topk_masked", select_indices)
    cache = object.__new__(implementation.HashOffloadingCache)
    device = torch.cuda.current_device()
    cache.config = SimpleNamespace(enable_cuda_graph=True, sparse_attention_config=
                                   SimpleNamespace(sink_budget=0, recent_budget=0))
    cache.curr_batch_size = cache.num_key_value_heads = 1
    cache.max_buffer_len = capacity
    cache.max_prefetch_topk_len = cache.max_current_topk_len = capacity
    cache.topk_current_k_host = cache.topk_prefetch_k_host = initial_k
    cache.rbits = 32
    cache.layers_hash_cache = [torch.zeros((1, 64, 1, 1), dtype=torch.int32, device="cuda")]
    cache.layers_hash_cache_length_tensor = [torch.full((1,), 64, dtype=torch.int32, device="cuda")]
    cache.topk_codes = cache.layers_hash_cache
    cache.cache_tensors = {"topk_code_length": cache.layers_hash_cache_length_tensor}
    cache.metadata_tensors = {
        f"topk_current_k_{device}": k,
        f"topk_prefetch_k_{device}": k,
        f"gpu_topk_scores_{device}": torch.zeros((1, 1, 64), device="cuda"),
        f"gpu_topk_indices_{device}": indices,
        f"gpu_topk_values_{device}": torch.zeros_like(indices),
    }
    query = torch.zeros((1, 1, 1, 1), dtype=torch.int32, device="cuda")
    mask = torch.ones(1, dtype=torch.bool, device="cuda")
    cache.compute_topk(query, 0, mask, is_prefetch=is_prefetch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = cache.compute_topk(query, 0, mask, is_prefetch=is_prefetch)
        total = selected.sum()
    # The device-side length changes without recapturing Python shape logic.
    for next_k in (initial_k, 5, 9):
        k.fill_(next_k)
        graph.replay()
        assert total.item() == sum(range(next_k))


@pytest.mark.parametrize("metrics_enabled", [False, True])
def test_offloading_host_lengths_advance_once_per_decode(monkeypatch, metrics_enabled):
    import sglang.ditto.kvcache_offloading as implementation

    cache = object.__new__(implementation.OffloadingCache)
    cache.num_layers = 3
    cache.curr_batch_size = 2
    cache.layers_gpu_head_ids = [torch.tensor([0]), torch.tensor([0]), torch.tensor([])]
    cache.layers_full_gpu_mask = [True, False, False]
    cache.cache_length_host = [[10, 20, 99], [10, 20, 99], [0, 0, 99]]
    cache.cpu_cache_length_host = [[0, 0, 99], [10, 20, 99], [10, 20, 99]]
    cache._decode_transfer_step_idx = 0
    # Observe host state at the metrics gate, without running the unrelated
    # transfer accounting. Mirrors must already be current with metrics off.
    observations = []
    def metrics_gate():
        observations.append(cache.cache_length_host[0][0])
        if metrics_enabled:
            raise StopIteration
        return False
    monkeypatch.setattr(implementation, "transfer_stats_enabled", metrics_gate)
    for _ in range(3):
        if metrics_enabled:
            with pytest.raises(StopIteration):
                cache.record_decode_transfer_step()
        else:
            cache.record_decode_transfer_step()
    assert observations == [11, 12, 13]
    assert cache._decode_transfer_step_idx == 3
    assert cache.cache_length_host == [[13, 23, 99], [13, 23, 99], [0, 0, 99]]
    assert cache.cpu_cache_length_host == [[0, 0, 99], [13, 23, 99], [13, 23, 99]]
