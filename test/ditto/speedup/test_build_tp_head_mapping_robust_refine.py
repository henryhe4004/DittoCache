from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
import build_tp_head_mapping_robust_refine as robust  # noqa: E402


class RobustHeadMappingTest(unittest.TestCase):
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

    def test_linear_reference_and_no_resident_file_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            rank0 = self._write_rank_profile(
                directory, 0, [0, 1, 2, 3], [1, 1, 1, 1]
            )
            rank1 = self._write_rank_profile(
                directory, 1, [4, 5, 6, 7], [0, 0, 0, 0]
            )
            output = directory / "mapping.json"
            argv = [
                "build_tp_head_mapping_robust_refine.py",
                "--transfer-json",
                str(rank0),
                "--transfer-json",
                str(rank1),
                "--output",
                str(output),
                "--skip-steps",
                "0",
                "--skip-layers",
                "0",
                "--max-changes",
                "1",
                "--min-split-improvement-pct",
                "0",
            ]
            with mock.patch.object(sys, "argv", argv):
                robust.main()

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIsNone(payload["metadata"]["reference_mapping"])
            self.assertIsNone(payload["metadata"]["resident_heads_file"])
            self.assertEqual(payload["metadata"]["selected_layers"], [0])
            self.assertEqual(len(payload["orders"]), 1)
            self.assertEqual(sorted(payload["orders"][0]), list(range(8)))


if __name__ == "__main__":
    unittest.main()
