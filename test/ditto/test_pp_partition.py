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
