from __future__ import annotations

from types import MethodType, SimpleNamespace

import pandas as pd

from quant_bench.contracts.csv import RUNTIME_METRICS_V1, RUNTIME_SIGNALS_V1
from quant_bench.runtime.core.metrics import MetricsTracker
from quant_bench.runtime.core.signal_tracker import SignalTracker


def test_metrics_tracker_writes_runtime_identity_and_absolute_aliases(tmp_path) -> None:
    tracker = MetricsTracker.__new__(MetricsTracker)
    tracker.identifier = "demo"
    tracker.timeframe = "1h"
    tracker.method_family = "traditional_ml"
    tracker.initial_usdt = 100.0
    tracker.external_cash_buffer_usdt = 0.0
    tracker.baseline_btc_qty = 1.0
    tracker.btc_instId = "BTC-USDT"
    tracker.capital_mode = "configured_budget"
    tracker.api = SimpleNamespace(get_current_price=lambda _symbol: 50.0)
    tracker.history_records = []
    tracker._csv_path = str(tmp_path / "metrics.csv")
    tracker._portfolio_snapshot = MethodType(
        lambda _self, _prices: {
            "cash_total_usdt": 40.0,
            "total_equity_usdt": 500.0,
            "cash_available_for_strategy": 40.0,
            "strategy_cash_ledger": 40.0,
            "holdings_value_usdt": 60.0,
            "strategy_equity": 100.0,
            "strategy_positions": {"ETH-USDT": 1.0},
            "strategy_accounting_mode": "trade_ledger",
            "account_holdings_value_usdt": 460.0,
        },
        tracker,
    )
    tracker._refresh_initial_usdt_if_needed = MethodType(
        lambda _self, _cash: None,
        tracker,
    )

    tracker.record_cycle(1_700_000_000.0, {"BTC-USDT": 50.0}, 1_699_999_200.0)

    record = tracker.history_records[-1]
    assert not RUNTIME_METRICS_V1.missing_columns(record)
    assert record["total_equity_usdt"] == record["strategy_equity"] == 100.0
    assert record["method_family"] == "traditional_ml"
    tracker.export_snapshot()
    assert tuple(pd.read_csv(tracker._csv_path).columns[: len(RUNTIME_METRICS_V1.required_columns)]) == (
        RUNTIME_METRICS_V1.required_columns
    )


def test_signal_tracker_writes_runtime_identity_and_aliases(tmp_path) -> None:
    tracker = SignalTracker(
        identifier="demo",
        timeframe="1h",
        log_dir=str(tmp_path),
        method_family="traditional_ml",
    )

    tracker.record_signals(
        [{"coin": "ETH-USDT", "score": 0.25}],
        {"ETH-USDT": 2_000.0},
        1_700_000_000.0,
        1_699_999_200.0,
    )

    record = tracker._pending_buffer[-1]
    assert not RUNTIME_SIGNALS_V1.missing_columns(record)
    assert record["signal_score"] == 0.25
    assert record["strategy_id"] == "demo"
    tracker.export_csv()
    assert tuple(pd.read_csv(tracker._csv_path).columns) == RUNTIME_SIGNALS_V1.required_columns
