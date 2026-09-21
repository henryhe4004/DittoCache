from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SPEEDUP_DIR = Path(__file__).parent
if str(SPEEDUP_DIR) not in sys.path:
    sys.path.insert(0, str(SPEEDUP_DIR))

SCRIPT_PATH = SPEEDUP_DIR / "build_tp_head_mapping_timeline.py"
SPEC = importlib.util.spec_from_file_location("tp_head_timeline", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _config(**overrides):
    values = {
        "h2d_gbps": (22.0, 22.0),
        "d2h_gbps": (22.0, 22.0),
        "h2d_launch_us": 2.0,
        "d2h_launch_us": 1.0,
        "active_head_us": 0.5,
        "tail_quantile": 0.95,
        "tail_weight": 0.0,
        "worst_profile_weight": 0.0,
        "beam_width": 4,
        "local_passes": 0,
    }
    values.update(overrides)
    return MODULE.TimelineConfig(**values)


def _sample(profile: str) -> object:
    return MODULE.StepSample(
        profile=profile,
        step=1,
        seq_len=1,
        prefetch_k=1,
        masks=((0, 0), (0, 0), (0, 0)),
    )


class HeadMappingTimelineTest(unittest.TestCase):
    def test_joint_search_alternates_rank_to_hide_next_layer_transfer(self) -> None:
        partitions = (
            ((0,), (1,)),
            ((1,), (0,)),
        )
        h2d = np.zeros((3, 2, 1, 2), dtype=np.float64)
        h2d[1, 0, 0] = (10.0, 0.0)
        h2d[1, 1, 0] = (0.0, 10.0)
        h2d[2, 0, 0] = (10.0, 0.0)
        h2d[2, 1, 0] = (0.0, 10.0)
        costs = MODULE.TransferCosts(
            partitions=partitions,
            h2d_us=h2d,
            d2h_us=np.zeros_like(h2d),
        )
        timing = MODULE.TimelineTiming(
            pre_wait_us=np.zeros((1, 3), dtype=np.float64),
            post_launch_us=np.zeros((1, 3), dtype=np.float64),
            tp_sync_us=np.zeros((1, 3), dtype=np.float64),
            resolved_profiles={},
            source=None,
        )
        references = [[0, 1], [0, 1], [0, 1]]

        path, metrics = MODULE.optimize_timeline(
            costs=costs,
            timing=timing,
            samples=[_sample("seq4K")],
            references=references,
            skip_layers=1,
            config=_config(),
        )

        self.assertEqual(path, (0, 0, 1))
        self.assertAlmostEqual(metrics["reference_mean_timeline_us"], 20.0)
        self.assertAlmostEqual(metrics["optimized_mean_timeline_us"], 10.0)
        self.assertAlmostEqual(metrics["objective"], -0.5)

    def test_regret_normalization_equalizes_sequence_buckets(self) -> None:
        profile_indices = {
            "seq4K": np.asarray([0]),
            "seq128K": np.asarray([1]),
        }
        reference = np.asarray([10.0, 100.0])
        candidates = np.asarray(
            [
                [8.0, 100.0],
                [10.0, 90.0],
            ]
        )

        objective, _ = MODULE._score_end_batch(
            candidates,
            reference,
            reference,
            profile_indices,
            _config(),
        )

        self.assertAlmostEqual(float(objective[0]), -0.10)
        self.assertAlmostEqual(float(objective[1]), -0.05)
        self.assertLess(float(objective[0]), float(objective[1]))

    def test_timing_json_resolves_sequence_specific_values(self) -> None:
        samples = [_sample("/tmp/model-seq4K-transfer.json")]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "timing.json"
            path.write_text(
                json.dumps(
                    {
                        "default": {"pre_wait_us": 1.0},
                        "profiles": {
                            "4K": {
                                "post_launch_us": [2.0, 3.0, 4.0],
                                "tp_sync_us": 5.0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            timing = MODULE.load_timeline_timing(
                str(path),
                samples,
                num_layers=3,
                pre_wait_us=0.0,
                post_launch_us=0.0,
                tp_sync_us=0.0,
            )

        np.testing.assert_allclose(timing.pre_wait_us, [[1.0, 1.0, 1.0]])
        np.testing.assert_allclose(timing.post_launch_us, [[2.0, 3.0, 4.0]])
        np.testing.assert_allclose(timing.tp_sync_us, [[5.0, 5.0, 5.0]])


if __name__ == "__main__":
    unittest.main()
