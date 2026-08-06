from __future__ import annotations

import socket
from pathlib import Path

import pytest

from quant_bench.artifacts import verify_run
from quant_bench.config import load_config
from quant_bench.experiments import Experiment


def test_offline_quickstart_generates_complete_verified_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def blocked_socket(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("offline quickstart attempted network access")

    monkeypatch.setattr(socket, "socket", blocked_socket)
    config = load_config("crypto_smoke_v1")
    config.experiment.workspace = tmp_path
    result = Experiment(config).run()
    assert result.status == "completed"
    assert result.metrics["sample_count"] == 10
    assert result.manifest_path.is_file()
    assert (result.run_dir / "resolved_config.yaml").is_file()
    assert (result.run_dir / "dataset_manifest.json").is_file()
    assert (result.run_dir / "model" / "model.npz").is_file()
    assert (result.run_dir / "predictions.csv").is_file()
    assert (result.run_dir / "backtest" / "returns.csv").is_file()
    assert verify_run(result.run_dir) == []
