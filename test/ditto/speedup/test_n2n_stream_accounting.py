"""Regression for interleaved batch output counts and wall-clock completion."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "n2n_offloading", Path(__file__).with_name("n2n_offloading.py")
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class FakeEngine:
    def __init__(self, chunks):
        self.chunks = chunks

    def generate(self, **kwargs):
        return iter(self.chunks)


def test_interleaved_requests_counted_once_and_timed_to_last(monkeypatch):
    chunks = [
        {"index": index, "meta_info": {"id": str(index), "completion_tokens": count}}
        for index, count in [(0, 1), (1, 1), (0, 3), (1, 2), (1, 4), (1, 4)]
    ]
    clock = iter(range(7))
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(clock))
    result = bench.generate_with_internal_forward_timing(FakeEngine(chunks), ["a", "b"], {}, 2)
    assert result[2] == 7  # Latest cumulative counts: 3 + 4, not last chunk's 4.
    assert result[1] == 5  # Last token is request 1 at t=5; t=6 duplicates metadata.
    assert result[4] == pytest.approx(5000 / 3)  # Within-request intervals only.


def test_single_request_without_id(monkeypatch):
    clock = iter([0, 1, 2])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(clock))
    result = bench.generate_with_internal_forward_timing(
        FakeEngine([{"meta_info": {"completion_tokens": n}} for n in [1, 2]]), "a", {}, 1
    )
    assert result[2] == 2
    assert result[1] == 2
    assert result[4] == 1000
