"""Versioned column contract used by every local dashboard adapter."""

from __future__ import annotations

DASHBOARD_VIEW_SCHEMA_VERSION = "1.0.0"

PERFORMANCE_COLUMNS = (
    "datetime_utc",
    "equity",
    "equity_normalized",
    "return_pct",
    "period_return",
    "baseline_equity",
    "baseline_normalized",
    "baseline_return_pct",
    "alpha_pct",
    "drawdown_pct",
)

PERFORMANCE_ALIASES = {
    "datetime_utc": (
        "bar_timestamp",
        "timestamp",
        "bar_datetime",
        "datetime_utc",
        "datetime",
    ),
    "equity": (
        "strategy_equity",
        "total_equity_usdt",
        "strategy_equity_usdt",
        "equity",
    ),
    "return_pct": (
        "strategy_returns_pct",
        "cumulative_return_pct",
        "return_pct",
    ),
    "period_return": (
        "net_return",
        "strategy_period_return",
        "period_return",
    ),
    "baseline_equity": (
        "baseline_equity",
        "benchmark_equity",
        "btc_equity",
    ),
    "baseline_return_pct": (
        "baseline_returns_pct",
        "benchmark_return_pct",
    ),
}

SIGNAL_COLUMNS = (
    "datetime_utc",
    "symbol",
    "score",
    "label",
    "signal_label",
)

SIGNAL_ALIASES = {
    "symbol": ("coin", "symbol", "asset"),
    "score": ("score", "signal_score", "final_score"),
    "label": ("label", "true_return", "target"),
    "signal_label": ("signal_label", "model_label"),
}
