from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("build_tp_head_mapping_cost_model.py")
SPEC = importlib.util.spec_from_file_location("tp_head_cost_model", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class HeadMappingCostModelTest(unittest.TestCase):
    def _write_rank_profile(
        self, directory: Path, rank: int, heads: list[int], misses: list[int]
    ) -> Path:
        steps = []
        for step_idx in range(1, 5):
            active_heads = sum(misses)
            steps.append(
                {
                    "step": step_idx,
                    "attn_tp_rank": rank,
                    "attn_tp_size": 2,
                    "seq_len": 128000 + step_idx,
                    "prefetch_k": 12800,
                    "layer_h2d_bytes": [active_heads * 12800 * 512],
                    "layer_d2h_bytes": [len(heads) * 512],
                    "layer_prefetch_heads": [active_heads],
                    "layer_offloaded_heads": [len(heads)],
                    "local_kv_head_ids_by_layer": [heads],
                    "layer_prefetch_head_masks": [misses],
                }
            )
        path = directory / f"profile.tp{rank:02d}.json"
        path.write_text(json.dumps({"steps": steps}), encoding="utf-8")
        return path

    def test_balances_correlated_heavy_heads_across_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            rank0 = self._write_rank_profile(
                directory, 0, [0, 1, 2, 3], [1, 1, 1, 1]
            )
            rank1 = self._write_rank_profile(
                directory, 1, [4, 5, 6, 7], [0, 0, 0, 0]
            )
            samples, num_layers, num_heads, token_bytes = MODULE.load_step_samples(
                [str(rank0), str(rank1)], skip_steps=0
            )
            self.assertEqual((num_layers, num_heads, token_bytes), (1, 8, 512))

            config = MODULE.CostConfig(
                h2d_gbps=(22.0, 22.0),
                d2h_gbps=(22.0, 22.0),
                h2d_launch_us=2.0,
                d2h_launch_us=1.0,
                active_head_us=0.5,
                shrinkage=0.5,
                tail_quantile=0.95,
                tail_weight=0.1,
                imbalance_weight=0.05,
                worst_profile_weight=0.25,
            )
            group0, group1, _ = MODULE.choose_partition(
                layer_idx=0,
                samples=samples,
                resident=set(),
                per_token_head_bytes=token_bytes,
                config=config,
                reference_order=list(range(8)),
            )
            heavy_heads = {0, 1, 2, 3}
            self.assertEqual(len(set(group0) & heavy_heads), 2)
            self.assertEqual(len(set(group1) & heavy_heads), 2)

    def test_rejects_incomplete_tp_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            rank0 = self._write_rank_profile(
                directory, 0, [0, 1, 2, 3], [1, 1, 1, 1]
            )
            with self.assertRaisesRegex(ValueError, "Incomplete TP profile"):
                MODULE.load_step_samples([str(rank0)], skip_steps=0)


if __name__ == "__main__":
    unittest.main()
