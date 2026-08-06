"""Validation-only parameter selection for FinGPT sentiment strategies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_bench.methods.fingpt_news.research.config import FinGPTResearchConfig
from quant_bench.methods.fingpt_news.research.engine import (
    apply_signal_config,
    build_daily_scores,
    run_capital_backtest,
)


@dataclass(frozen=True)
class TuningResult:
    aggregation: dict[str, float | int]
    signal: dict[str, Any]
    threshold: float
    selection_status: str
    trials: pd.DataFrame
    validation_scores: pd.DataFrame
    validation_signals: pd.DataFrame


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _clean_signal(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (None if item is None or (isinstance(item, float) and np.isnan(item)) else item)
        for key, item in value.items()
    }


def _eligible(row: dict[str, Any], cfg: FinGPTResearchConfig) -> bool:
    guard = cfg.selection
    return bool(
        int(row.get("trades", 0) or 0) >= guard.min_trades
        and int(row.get("model_signal_days", 0) or 0) >= guard.min_signal_days
        and int(row.get("classified_news", 0) or 0) >= guard.min_classified_news
    )


def _sort_trials(frame: pd.DataFrame, objective: str) -> pd.DataFrame:
    objective_column = {
        "sharpe": "strategy_sharpe",
        "total_return": "strategy_total_return",
        "calmar": "strategy_calmar",
        "excess_total_return": "excess_total_return",
    }[objective]
    work = frame.copy()
    work["__objective"] = pd.to_numeric(work[objective_column], errors="coerce").fillna(-np.inf)
    work["__drawdown"] = pd.to_numeric(work["strategy_max_drawdown"], errors="coerce").fillna(-np.inf)
    work["__excess"] = pd.to_numeric(work["excess_total_return"], errors="coerce").fillna(-np.inf)
    work = work.sort_values(
        ["eligible", "__objective", "__drawdown", "__excess", "threshold", "trial"],
        ascending=[False, False, False, False, True, True],
        kind="mergesort",
    )
    return work.drop(columns=["__objective", "__drawdown", "__excess"]).reset_index(drop=True)


def tune_global(
    sentiment: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: FinGPTResearchConfig,
) -> TuningResult:
    rows: list[dict[str, Any]] = []
    score_cache: dict[str, pd.DataFrame] = {}
    trial = 0
    for aggregation in cfg.aggregation.combinations():
        aggregation_key = _json(aggregation)
        daily_scores = build_daily_scores(sentiment, prices, cfg.symbols, aggregation)
        score_cache[aggregation_key] = daily_scores
        for raw_signal in cfg.signals.variant_combinations():
            signal = _clean_signal(raw_signal)
            for threshold in cfg.signals.thresholds:
                trial += 1
                signals = apply_signal_config(daily_scores, cfg, signal, float(threshold))
                result = run_capital_backtest(prices, signals, cfg)
                summary = result.portfolio_summary.iloc[0].to_dict()
                row: dict[str, Any] = {
                    "trial": trial,
                    "aggregation_json": aggregation_key,
                    "signal_json": _json(signal),
                    "threshold": float(threshold),
                    "daily_positive": int((signals["model_label"] == "positive").sum()),
                    "daily_neutral": int((signals["model_label"] == "neutral").sum()),
                    "daily_negative": int((signals["model_label"] == "negative").sum()),
                    **summary,
                }
                row["eligible"] = _eligible(row, cfg)
                rows.append(row)
    trials = _sort_trials(pd.DataFrame(rows), cfg.selection.objective)
    if trials.empty:
        raise RuntimeError("parameter grid produced no trials")
    best = trials.iloc[0]
    status = "selected" if bool(best["eligible"]) else "coverage_fallback"
    aggregation = json.loads(str(best["aggregation_json"]))
    signal = json.loads(str(best["signal_json"]))
    validation_scores = score_cache[str(best["aggregation_json"])]
    validation_signals = apply_signal_config(validation_scores, cfg, signal, float(best["threshold"]))
    return TuningResult(
        aggregation=aggregation,
        signal=signal,
        threshold=float(best["threshold"]),
        selection_status=status,
        trials=trials,
        validation_scores=validation_scores,
        validation_signals=validation_signals,
    )


def tune_per_symbol_thresholds(
    daily_scores: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: FinGPTResearchConfig,
    signal: dict[str, Any],
    fallback_threshold: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    selected: dict[str, float] = {}
    for symbol in cfg.symbols:
        for trial_index, threshold in enumerate(cfg.signals.thresholds, start=1):
            signals = apply_signal_config(daily_scores, cfg, signal, float(threshold))
            result = run_capital_backtest(prices, signals, cfg)
            summary = result.per_symbol_summary[result.per_symbol_summary["symbol"] == symbol].iloc[0].to_dict()
            row: dict[str, Any] = {
                "symbol": symbol,
                "trial": trial_index,
                "threshold": float(threshold),
                **summary,
            }
            row["eligible"] = _eligible(row, cfg)
            rows.append(row)
        symbol_trials = _sort_trials(pd.DataFrame([row for row in rows if row["symbol"] == symbol]), cfg.selection.objective)
        eligible = symbol_trials[symbol_trials["eligible"]]
        selected[symbol] = (
            float(eligible.iloc[0]["threshold"]) if not eligible.empty else float(fallback_threshold)
        )
    trials = pd.DataFrame(rows)
    if not trials.empty:
        trials = trials.sort_values(["symbol", "eligible", "trial"], ascending=[True, False, True])
    return selected, trials.reset_index(drop=True)
