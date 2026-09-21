from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.ditto.kvcache_full_attn import CustomStaticCache
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models.ditto.ditto import (
    _configure_ditto_pipeline_stage,
    _get_ditto_pp_stage_inputs,
)


class _Attention(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(2, 2, bias=False)
        self.rotary_emb = nn.Identity()
        self.next_input_layernorm = None
        self.next_q_proj = None
        self.next_rotary_emb = None


class _Layer(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(2)
        self.self_attn = _Attention(layer_idx)


class _Backbone(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 2)
        self.layers = nn.ModuleList([_Layer(i) for i in range(num_layers)])
        self.norm = nn.LayerNorm(2)


class _CausalLM(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.model = _Backbone(num_layers)
        self.lm_head = nn.Linear(2, 8, bias=False)


def _pp_group(rank: int, size: int):
    return SimpleNamespace(
        rank_in_group=rank,
        world_size=size,
        is_first_rank=rank == 0,
        is_last_rank=rank == size - 1,
    )


def test_configure_middle_pipeline_stage_keeps_global_module_slots():
    model = _CausalLM(6)
    start_layer, end_layer = _configure_ditto_pipeline_stage(
        model,
        SimpleNamespace(num_hidden_layers=6),
        _pp_group(rank=1, size=3),
    )

    assert (start_layer, end_layer) == (2, 4)
    assert isinstance(model.model.embed_tokens, PPMissingLayer)
    assert isinstance(model.lm_head, PPMissingLayer)
    assert isinstance(model.model.norm, PPMissingLayer)
    assert all(
        isinstance(model.model.layers[i], PPMissingLayer) for i in (0, 1, 4, 5)
    )

    first_local = model.model.layers[2]
    second_local = model.model.layers[3]
    assert first_local.self_attn.layer_idx == 0
    assert first_local.self_attn.global_layer_idx == 2
    assert second_local.self_attn.layer_idx == 1
    assert second_local.self_attn.global_layer_idx == 3
    assert first_local.self_attn.next_q_proj is second_local.self_attn.q_proj
    assert second_local.self_attn.next_q_proj is first_local.self_attn.q_proj


def test_configure_last_pipeline_stage_keeps_norm_and_lm_head():
    model = _CausalLM(4)
    start_layer, end_layer = _configure_ditto_pipeline_stage(
        model,
        SimpleNamespace(num_hidden_layers=4),
        _pp_group(rank=1, size=2),
    )

    assert (start_layer, end_layer) == (2, 4)
    assert not isinstance(model.model.norm, PPMissingLayer)
    assert not isinstance(model.lm_head, PPMissingLayer)


def test_pp_stage_inputs_reshape_flat_hidden_states():
    owner = SimpleNamespace(pp_group=_pp_group(rank=1, size=2))
    input_ids = torch.zeros((2, 3), dtype=torch.long)
    hidden_states = torch.arange(24, dtype=torch.float32).view(6, 4)

    stage_input_ids, stage_input_embeds = _get_ditto_pp_stage_inputs(
        owner,
        input_ids,
        PPProxyTensors({"hidden_states": hidden_states}),
    )

    assert stage_input_ids is None
    assert stage_input_embeds.shape == (2, 3, 4)
    torch.testing.assert_close(stage_input_embeds.view(6, 4), hidden_states)


def test_pp_stage_inputs_reject_bad_shape():
    owner = SimpleNamespace(pp_group=_pp_group(rank=1, size=2))
    input_ids = torch.zeros((2, 3), dtype=torch.long)

    with pytest.raises(RuntimeError, match="expected_tokens=6"):
        _get_ditto_pp_stage_inputs(
            owner,
            input_ids,
            PPProxyTensors({"hidden_states": torch.zeros((5, 4))}),
        )


def test_cache_uses_pipeline_local_layer_count():
    model_config = SimpleNamespace(
        torch_dtype=torch.float16,
        num_hidden_layers=8,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        _ditto_pp_start_layer=2,
        _ditto_pp_end_layer=5,
    )
    custom_config = SimpleNamespace(
        enable_cuda_graph=False,
        kvcache_manager_config=SimpleNamespace(
            max_tokens=16,
            gpu_memory_budget=1,
        ),
    )

    cache = CustomStaticCache(
        config=model_config,
        custom_config=custom_config,
        device="cpu",
    )

    assert cache.total_num_layers == 8
    assert cache.num_layers == 3
    assert cache.global_layer_ids == [2, 3, 4]
    assert cache.global_layer_idx(1) == 3


def test_tp_accepts_pipeline_parallelism(monkeypatch):
    from sglang.srt.models.ditto import ditto

    monkeypatch.setattr(ditto, '_get_ditto_tensor_parallel_info', lambda: (1, 2))
    monkeypatch.setattr(ditto, '_get_ditto_attention_parallel_info', lambda: (1, 2, False, 1))
    monkeypatch.setattr(ditto, '_maybe_get_global_server_args', lambda: SimpleNamespace(nnodes=1, pp_size=2))
    layout = ditto._build_ditto_tp_head_layout(
        SimpleNamespace(num_attention_heads=8, num_key_value_heads=4)
    )
    assert (layout.local_num_heads, layout.local_num_key_value_heads) == (4, 2)


@pytest.mark.parametrize('rank,size', [(0, 1), (0, 2), (1, 2), (5, 8)])
def test_pp_cache_uses_global_layer_tp_orders(monkeypatch, tmp_path, rank, size):
    import json
    from sglang.ditto import kvcache_full_attn
    from sglang.ditto.tp_head_mapping import local_kv_head_ids

    orders = [[0, 1, 2, 3], [1, 0, 3, 2], [3, 2, 1, 0], [2, 0, 3, 1]]
    if size == 1 or size > 4:
        orders = [list(range(4)) for _ in range(4)]
    path = tmp_path / 'orders.json'
    path.write_text(json.dumps(orders))
    monkeypatch.setenv('DITTO_TP_KV_HEAD_ORDER_FILE', str(path))
    monkeypatch.setattr(kvcache_full_attn, '_detect_attention_tp_info', lambda: (rank, size))
    cache = CustomStaticCache(
        config=SimpleNamespace(torch_dtype=torch.float32, num_hidden_layers=4,
            hidden_size=16, num_attention_heads=8, num_key_value_heads=4,
            _ditto_pp_start_layer=2, _ditto_pp_end_layer=4),
        custom_config=SimpleNamespace(enable_cuda_graph=False,
            kvcache_manager_config=SimpleNamespace(max_tokens=16, gpu_memory_budget=1)),
        device='cpu',
    )
    expected = tuple(local_kv_head_ids(4, rank, size, layer_idx=i) for i in (2, 3))
    assert cache.local_kv_head_ids_by_layer == expected
    if size <= 4:
        assert cache.local_query_head_ids_by_layer == tuple(
            tuple(q for h in heads for q in (2 * h, 2 * h + 1)) for heads in expected
        )
    for layer_idx in range(2):
        actual = cache._slice_local_kv_head_tensor(torch.arange(4), layer_idx=layer_idx)
        torch.testing.assert_close(actual, torch.tensor(expected[layer_idx]))


def test_prefetch_aliases_refresh_after_tp_replacement():
    from sglang.srt.models.ditto.ditto import _refresh_ditto_prefetch_aliases

    model = _CausalLM(6)
    _configure_ditto_pipeline_stage(model, SimpleNamespace(num_hidden_layers=6), _pp_group(1, 3))
    for i in (2, 3):
        model.model.layers[i].self_attn.q_proj = nn.Linear(2, 1, bias=False)
    assert _refresh_ditto_prefetch_aliases(model) == 2
    assert model.model.layers[2].self_attn.next_q_proj is model.model.layers[3].self_attn.q_proj
    assert model.model.layers[3].self_attn.next_q_proj is model.model.layers[2].self_attn.q_proj


def test_transfer_stats_paths_distinguish_tp_and_pp(monkeypatch):
    from pathlib import Path
    from sglang.ditto import transfer_stats
    from sglang.srt import distributed

    monkeypatch.setattr(transfer_stats, '_detect_tp_rank_suffix', lambda: (1, 2))
    monkeypatch.setattr(distributed, 'get_pp_group', lambda: _pp_group(1, 2))
    path = transfer_stats._with_tp_rank_suffix(Path('stats.json'))
    assert path.name == 'stats.pp01.tp01.json'
    assert transfer_stats._with_tp_rank_suffix(path) == path


@pytest.mark.parametrize("shape", [(1, 6, 4), (2, 4, 4), (2, 3, 4, 1)])
def test_pp_stage_inputs_reject_invalid_batched_shape(shape):
    owner = SimpleNamespace(pp_group=_pp_group(1, 2))
    with pytest.raises(RuntimeError, match="Ditto PP"):
        _get_ditto_pp_stage_inputs(
            owner, torch.zeros((2, 3), dtype=torch.long),
            PPProxyTensors({"hidden_states": torch.zeros(shape)}),
        )


def test_pp_stage_inputs_require_proxy_only_after_first_stage():
    tokens = torch.zeros((2, 3), dtype=torch.long)
    assert _get_ditto_pp_stage_inputs(
        SimpleNamespace(pp_group=_pp_group(0, 2)), tokens, None
    ) == (tokens, None)
    with pytest.raises(RuntimeError, match="did not receive hidden_states"):
        _get_ditto_pp_stage_inputs(
            SimpleNamespace(pp_group=_pp_group(1, 2)), tokens, None
        )


def test_pp_rejects_empty_stage_before_mutating_model(monkeypatch):
    monkeypatch.delenv("SGLANG_PP_LAYER_PARTITION", raising=False)
    model = _CausalLM(2)
    with pytest.raises(ValueError, match="at least one layer"):
        _configure_ditto_pipeline_stage(
            model, SimpleNamespace(num_hidden_layers=2), _pp_group(0, 3)
        )
    assert all(isinstance(layer, _Layer) for layer in model.model.layers)


def test_single_local_layer_prefetch_wraps_to_itself(monkeypatch):
    monkeypatch.setenv("SGLANG_PP_LAYER_PARTITION", "2,1,3")
    model = _CausalLM(6)
    assert _configure_ditto_pipeline_stage(
        model, SimpleNamespace(num_hidden_layers=6), _pp_group(1, 3)
    ) == (2, 3)
    attention = model.model.layers[2].self_attn
    assert attention.layer_idx == 0
    assert attention.global_layer_idx == 2
    assert attention.next_q_proj is attention.q_proj


@pytest.mark.parametrize("last_stage", [False, True])
def test_pp_chunked_prefill_preserves_batched_hidden_states(last_stage):
    from sglang.srt.models.ditto.full_attn_ops import llm_prefill_forward

    class AddLayer(nn.Module):
        def forward(self, hidden_states, past_key_value):
            return hidden_states + 1

    class Norm(nn.Module):
        def forward(self, hidden_states, is_prefill):
            return hidden_states * 2

    backbone = SimpleNamespace(
        layers=[PPMissingLayer(), AddLayer(), AddLayer(), PPMissingLayer()],
        start_layer=1, end_layer=3, is_last_pp_rank=last_stage, norm=Norm(),
    )
    lengths_seen = []
    cache = SimpleNamespace(
        config=SimpleNamespace(chunk_prefill_size=2),
        _current_extend_seq_lens=[5, 3],
    )
    cache.update_metadata = lambda q_len: lengths_seen.append(
        (q_len, list(cache._current_extend_seq_lens))
    )
    hidden = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
    output = llm_prefill_forward(backbone, inputs_embeds=hidden, past_key_values=cache)
    expected = hidden + 2
    if last_stage:
        expected = torch.stack([expected[0, 4], expected[1, 2]]).unsqueeze(1) * 2
    torch.testing.assert_close(output.last_hidden_state, expected)
    assert lengths_seen == [(2, [2, 2]), (2, [2, 1]), (1, [1, 0])]
    assert cache._current_extend_seq_lens == [5, 3]
