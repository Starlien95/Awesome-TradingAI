from __future__ import annotations

import json
import socket
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from quant_bench.cli.app import main


def test_fingpt_research_fixture_runs_full_selection_and_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def blocked_socket(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("FinGPT research attempted network access")

    monkeypatch.setattr(socket, "socket", blocked_socket)
    workspace = tmp_path / "workspace"
    assert main(["fingpt", "init", "--workspace", str(workspace)]) == 0
    root = workspace / "methods" / "fingpt_news" / "research"
    config = root / "config.yaml"
    output = tmp_path / "run"

    assert main(["fingpt", "validate", "--config", str(config)]) == 0
    assert main(
        ["fingpt", "backtest", "--config", str(config), "--output", str(output)]
    ) == 0

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    chosen = json.loads((output / "chosen_config.json").read_text(encoding="utf-8"))
    quality = json.loads((output / "data_quality.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED"
    assert manifest["method_family"] == "fingpt_news"
    assert manifest["csv_schema_version"] == "quant-bench.csv.v1"
    assert chosen["test_metrics_used_for_selection"] is False
    assert quality["validation_test_overlap"] is False
    assert (output / "validation" / "parameter_trials.csv").is_file()
    assert (output / "per_symbol_tuned" / "portfolio_summary.csv").is_file()
    assert (output / "checksums.sha256").is_file()

    metrics = pd.read_csv(output / manifest["metrics_path"])
    signals = pd.read_csv(output / manifest["signals_path"])
    trades = pd.read_csv(output / manifest["trades_path"])
    assert {
        "initial_capital_usdt", "cash_total_usdt", "holdings_value_usdt",
        "total_equity_usdt", "method_family", "strategy_id", "frequency",
    }.issubset(metrics.columns)
    assert {"coin", "score", "signal_label", "price_at_pred", "true_return"}.issubset(
        signals.columns
    )
    assert {"trade_id", "coin", "side", "notional_usdt", "success"}.issubset(trades.columns)
    assert metrics["strategy_equity"].tolist() == pytest.approx(
        metrics["total_equity_usdt"].tolist()
    )
    first_local = datetime.fromtimestamp(
        float(metrics["bar_timestamp"].iloc[0]),
        ZoneInfo("Asia/Shanghai"),
    )
    assert (first_local.hour, first_local.minute, first_local.second) == (0, 0, 0)
    assert signals["cycle_id"].nunique() == signals["bar_timestamp"].nunique()

    assert main(
        ["fingpt", "backtest", "--config", str(config), "--output", str(output)]
    ) == 1
