from pathlib import Path

import pandas as pd
import pytest

from quant_bench.contracts.csv import PREDICTIONS_V1, RUNTIME_METRICS_V1
from quant_bench.runtime.core.run_manifest import write_run_manifest


def test_csv_contract_reports_missing_columns() -> None:
    frame = pd.DataFrame(columns=["timestamp", "symbol", "score"])

    assert PREDICTIONS_V1.missing_columns(frame.columns) == ("label",)
    with pytest.raises(ValueError, match="label"):
        PREDICTIONS_V1.validate(frame)


def test_runtime_metrics_contract_keeps_identity_and_units() -> None:
    assert "total_equity_usdt" in RUNTIME_METRICS_V1.required_columns
    assert RUNTIME_METRICS_V1.required_columns[-3:] == (
        "method_family",
        "strategy_id",
        "frequency",
    )


def test_runtime_manifest_normalizes_paths_and_versions_schema(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    absolute_metric = run_dir / "metrics" / "metrics.csv"

    write_run_manifest(
        run_dir,
        strategy_id="demo",
        method_family="traditional_ml",
        frequency="1h",
        mode="paper",
        paths={"metrics_path": str(absolute_metric)},
    )

    manifest = pd.read_json(run_dir / "run_manifest.json", typ="series")
    assert manifest["metrics_path"] == "metrics/metrics.csv"
    assert manifest["csv_schema_version"] == "quant-bench.csv.v1"


def test_runtime_manifest_rejects_path_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes run directory"):
        write_run_manifest(
            tmp_path / "run",
            strategy_id="demo",
            method_family="traditional_ml",
            frequency="1h",
            mode="paper",
            paths={"metrics_path": "../outside.csv"},
        )
