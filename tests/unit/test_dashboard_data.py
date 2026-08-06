from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_bench.dashboard.data import discover_workspace, load_run


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_discovers_canonical_and_nested_runtime_runs(tmp_path: Path) -> None:
    canonical = tmp_path / "runs" / "offline-1"
    canonical.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z"],
            "net_return": [0.01],
            "equity": [101_000.0],
        }
    ).to_csv(canonical / "returns.csv", index=False)
    (canonical / "resolved_config.yaml").write_text(
        """
experiment:
  name: Offline smoke
frequency:
  id: 1h
model:
  model_id: numpy_linear_v1
runtime:
  mode: research
""".strip(),
        encoding="utf-8",
    )
    _write_json(
        canonical / "run_manifest.json",
        {
            "run_id": "offline-1",
            "status": "completed",
            "protocol_id": "p1",
            "dataset_sha256": "a" * 64,
            "artifacts": [{"path": "backtest/returns.csv"}],
        },
    )
    (canonical / "backtest").mkdir()
    (canonical / "returns.csv").replace(canonical / "backtest" / "returns.csv")

    runtime = tmp_path / "runs" / "runtime" / "1h" / "ModelA"
    runtime.mkdir(parents=True)
    pd.DataFrame(
        {"timestamp": [1_700_000_000], "strategy_equity": [100.0]}
    ).to_csv(runtime / "metrics.csv", index=False)
    _write_json(
        runtime / "run_manifest.json",
        {
            "strategy_id": "ModelA",
            "method_family": "traditional_ml",
            "frequency": "1h",
            "mode": "demo",
            "status": "ACTIVE",
            "metrics_path": "metrics.csv",
        },
    )

    inventory = discover_workspace(tmp_path)

    assert [(run.kind, run.run_id) for run in inventory.runs] == [
        ("canonical", "offline-1"),
        ("runtime", "ModelA"),
    ]
    canonical_source = inventory.runs[0]
    assert canonical_source.display_name == "Offline smoke"
    assert canonical_source.files["metrics"].name == "returns.csv"
    runtime_data = load_run(inventory.runs[1])
    assert runtime_data.metrics["strategy_equity"].tolist() == [100.0]


def test_manifest_path_escape_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside.csv"
    outside.write_text("timestamp,strategy_equity\n1,100\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "runtime" / "ModelA"
    _write_json(
        run_dir / "run_manifest.json",
        {
            "strategy_id": "ModelA",
            "method_family": "traditional_ml",
            "frequency": "1h",
            "metrics_path": "../../../outside.csv",
        },
    )

    inventory = discover_workspace(tmp_path)

    assert len(inventory.runs) == 1
    assert "metrics" not in inventory.runs[0].files
    assert any(issue.code == "PATH_ESCAPE" for issue in inventory.issues)


def test_empty_csv_is_reported_without_crashing(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "runtime" / "ModelA"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.csv").write_text("", encoding="utf-8")
    _write_json(
        run_dir / "run_manifest.json",
        {
            "strategy_id": "ModelA",
            "method_family": "traditional_ml",
            "frequency": "1h",
            "metrics_path": "metrics.csv",
        },
    )

    source = discover_workspace(tmp_path).runs[0]
    data = load_run(source)

    assert data.metrics.empty
    assert [issue.code for issue in data.issues] == ["EMPTY_CSV"]
