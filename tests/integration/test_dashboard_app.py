from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parents[2] / "src" / "quant_bench" / "dashboard" / "app.py"


def _run_app(workspace: Path) -> AppTest:
    with patch.dict(os.environ, {"QUANT_BENCH_DASHBOARD_ROOT": str(workspace)}):
        return AppTest.from_file(
            str(APP_PATH),
            default_timeout=15,
        ).run(timeout=15)


def test_dashboard_empty_workspace_smoke(tmp_path: Path) -> None:
    app = _run_app(tmp_path)

    assert not app.exception
    assert any("Awesome TradingAI" in title.value for title in app.title)


def test_dashboard_canonical_workspace_smoke(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo"
    (run_dir / "backtest").mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00"],
            "gross_return": [0.01, 0.02],
            "net_return": [0.009, 0.019],
            "turnover": [0.1, 0.1],
            "cost": [0.001, 0.001],
            "positions": [1, 1],
            "equity": [1.009, 1.028171],
        }
    ).to_csv(run_dir / "backtest" / "returns.csv", index=False)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "demo",
                "status": "completed",
                "protocol_id": "protocol-v1",
                "dataset_sha256": "a" * 64,
                "artifacts": [{"path": "backtest/returns.csv"}],
            }
        ),
        encoding="utf-8",
    )

    app = _run_app(tmp_path)

    assert not app.exception
    assert any("Workspace Overview" in title.value for title in app.title)
    assert app.dataframe
