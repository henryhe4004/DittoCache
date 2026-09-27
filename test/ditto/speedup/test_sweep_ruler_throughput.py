"""CPU regressions for endpoint routing and trustworthy sweep accounting."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "sweep_ruler_throughput", Path(__file__).with_name("sweep_ruler_throughput.py")
)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_custom_server_is_forwarded(monkeypatch):
    monkeypatch.setattr("sys.argv", ["sweep", "--server-base-url", "http://127.0.0.1:30184/"])
    cmd = sweep.build_client_cmd(sweep.parse_args(), 2, 4)
    assert cmd[cmd.index("--server") + 1] == "http://127.0.0.1:30184"


@pytest.mark.parametrize("requested, expected", [(0, 2), (4, 4)])
def test_duration_and_incomplete_request_accounting(tmp_path, requested, expected):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(row) for row in [
        {"ok": True, "status": 200, "latency_ms": 10},
        {"ok": False, "status": 503, "latency_ms": 20},
    ]))
    result = sweep.summarize_run(1, requested, 0, 1, raw, tmp_path / "client.log")
    assert result["total_requests"] == expected
    assert result["ok_requests"] == 1
    assert result["failed_requests"] == expected - 1


def test_failed_client_makes_sweep_fail_and_retains_csv(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["sweep", "--concurrencies", "1", "--total-requests", "1", "--output-dir", str(tmp_path), "--no-plot"])
    def fail_client(cmd, **kwargs):
        raw = Path(cmd[cmd.index("--log-file") + 1])
        raw.write_text(json.dumps({"ok": False, "status": 503}) + "\n")
        return 0  # The historical client reports HTTP failures through JSON only.
    monkeypatch.setattr(sweep, "run_client_with_tee", fail_client)
    monkeypatch.setattr(sweep, "wait_for_server_idle", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="Throughput sweep failed"):
        sweep.main()
    assert (tmp_path / "ruler_throughput.csv").is_file()
