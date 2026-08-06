"""Versioned CSV artifact contracts shared by producers and readers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

CSV_SCHEMA_VERSION = "quant-bench.csv.v1"


@dataclass(frozen=True)
class CsvContract:
    """A stable per-artifact column contract.

    Required columns define interoperability. Optional and extension columns may
    coexist without changing the schema id.
    """

    schema_id: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()

    def missing_columns(self, columns: Iterable[str]) -> tuple[str, ...]:
        available = set(columns)
        return tuple(column for column in self.required_columns if column not in available)

    def validate(self, frame: pd.DataFrame) -> None:
        missing = self.missing_columns(frame.columns)
        if missing:
            raise ValueError(f"{self.schema_id} missing columns: {', '.join(missing)}")

    def ordered_columns(self, columns: Iterable[str] = ()) -> tuple[str, ...]:
        available = list(dict.fromkeys(columns))
        known = [*self.required_columns, *self.optional_columns]
        return tuple([*known, *(column for column in available if column not in known)])


PREDICTIONS_V1 = CsvContract(
    "quant-bench.predictions.v1",
    ("timestamp", "symbol", "score", "label"),
)

RETURNS_V1 = CsvContract(
    "quant-bench.returns.v1",
    ("timestamp", "gross_return", "net_return", "turnover", "cost", "positions", "equity"),
)

RUNTIME_METRICS_V1 = CsvContract(
    "quant-bench.runtime.metrics.v1",
    (
        "timestamp",
        "datetime",
        "bar_timestamp",
        "bar_datetime",
        "initial_capital_usdt",
        "cash_total_usdt",
        "holdings_value_usdt",
        "total_equity_usdt",
        "strategy_equity",
        "strategy_pnl",
        "strategy_returns_pct",
        "baseline_equity",
        "baseline_pnl",
        "baseline_returns_pct",
        "btc_price",
        "active_positions",
        "gross_exposure",
        "cash_ratio",
        "method_family",
        "strategy_id",
        "frequency",
    ),
)

RUNTIME_SIGNALS_V1 = CsvContract(
    "quant-bench.runtime.signals.v1",
    (
        "cycle_id",
        "timestamp",
        "datetime",
        "bar_timestamp",
        "bar_datetime",
        "coin",
        "score",
        "signal_label",
        "signal_score",
        "price_at_pred",
        "price_future",
        "true_return",
        "labeled",
        "method_family",
        "strategy_id",
        "frequency",
    ),
)

RUNTIME_TRADES_V1 = CsvContract(
    "quant-bench.runtime.trades.v1",
    (
        "trade_id",
        "timestamp",
        "datetime",
        "trade_date",
        "coin",
        "side",
        "price",
        "qty",
        "notional_usdt",
        "fee_usdt",
        "slippage_bps",
        "cash_before",
        "cash_after",
        "position_qty_before",
        "position_qty_after",
        "reason",
        "success",
    ),
)

RUNTIME_VOLUME_V1 = CsvContract(
    "quant-bench.runtime.volume.v1",
    (
        "timestamp",
        "datetime",
        "cycle_volume_usdt",
        "cumulative_total_volume_usdt",
        "daily_total_volume_usdt",
        "per_coin_cumulative_json",
        "per_coin_daily_json",
    ),
)

CSV_CONTRACTS = {
    contract.schema_id: contract
    for contract in (
        PREDICTIONS_V1,
        RETURNS_V1,
        RUNTIME_METRICS_V1,
        RUNTIME_SIGNALS_V1,
        RUNTIME_TRADES_V1,
        RUNTIME_VOLUME_V1,
    )
}
