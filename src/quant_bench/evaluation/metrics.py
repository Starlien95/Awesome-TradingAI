"""Dependency-light benchmark metrics with explicit sample counts."""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from quant_bench.config.models import BenchmarkProtocolSpec, FrequencySpec


def _safe_correlation(left: pd.Series, right: pd.Series, *, rank: bool = False) -> float | None:
    aligned = pd.concat([left, right], axis=1).dropna()
    if len(aligned) < 2 or aligned.iloc[:, 0].nunique() < 2 or aligned.iloc[:, 1].nunique() < 2:
        return None
    if rank:
        aligned = aligned.rank(method="average")
    value = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return None if pd.isna(value) else float(value)


def _signal_metrics(joined: pd.DataFrame) -> dict[str, float | int | None]:
    finite = joined[["score", "label"]].dropna()
    mse = float(((finite["score"] - finite["label"]) ** 2).mean()) if len(finite) else None
    mae = float((finite["score"] - finite["label"]).abs().mean()) if len(finite) else None
    daily_ic: list[float] = []
    daily_rank_ic: list[float] = []
    for _, group in finite.groupby(level="timestamp"):
        ic = _safe_correlation(group["score"], group["label"])
        rank_ic = _safe_correlation(group["score"], group["label"], rank=True)
        if ic is not None:
            daily_ic.append(ic)
        if rank_ic is not None:
            daily_rank_ic.append(rank_ic)
    return {
        "sample_count": len(finite),
        "coverage": float(len(finite) / len(joined)) if len(joined) else 0.0,
        "mse": mse,
        "mae": mae,
        "ic_mean": float(np.mean(daily_ic)) if daily_ic else None,
        "ic_std": float(np.std(daily_ic, ddof=1)) if len(daily_ic) > 1 else None,
        "rank_ic_mean": float(np.mean(daily_rank_ic)) if daily_rank_ic else None,
        "rank_ic_std": float(np.std(daily_rank_ic, ddof=1)) if len(daily_rank_ic) > 1 else None,
        "ic_period_count": len(daily_ic),
    }


def _portfolio_returns(joined: pd.DataFrame, protocol: BenchmarkProtocolSpec) -> pd.DataFrame:
    previous_weights: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    strategy = protocol.strategy
    one_way_cost = (protocol.costs.open_cost + protocol.costs.close_cost) / 2.0
    one_way_cost += protocol.costs.slippage_bps / 10_000.0

    for timestamp, group in joined.dropna(subset=["score", "label"]).groupby(level="timestamp", sort=True):
        table = group.reset_index(level="timestamp", drop=True)
        candidates = table[table["score"] > strategy.threshold].nlargest(strategy.top_k, "score")
        if len(candidates):
            weight = strategy.risk_degree / len(candidates)
            weights = {str(symbol): weight for symbol in candidates.index}
            gross_return = float((candidates["label"].astype(float) * weight).sum())
        else:
            weights = {}
            gross_return = 0.0
        symbols = set(previous_weights) | set(weights)
        turnover = 0.5 * sum(
            abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0)) for symbol in symbols
        )
        fees = turnover * one_way_cost
        net_return = gross_return - fees
        rows.append(
            {
                "timestamp": timestamp,
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "cost": fees,
                "positions": len(weights),
            }
        )
        previous_weights = weights
    result = pd.DataFrame(rows)
    if not result.empty:
        result["equity"] = protocol.initial_cash * (1.0 + result["net_return"]).cumprod()
    return result


def _portfolio_metrics(returns: pd.DataFrame, frequency: FrequencySpec) -> dict[str, float | int | None]:
    if returns.empty:
        return {
            "period_count": 0,
            "total_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "max_drawdown": None,
            "turnover_mean": None,
        }
    values = returns["net_return"].to_numpy(dtype=float)
    total_return = float(np.prod(1.0 + values) - 1.0)
    annualized_return = float((1.0 + total_return) ** (frequency.annualization / len(values)) - 1.0)
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    volatility = standard_deviation * sqrt(frequency.annualization)
    sharpe = (
        float(np.mean(values) / standard_deviation * sqrt(frequency.annualization))
        if len(values) > 1 and standard_deviation > 0
        else None
    )
    equity = np.cumprod(1.0 + values)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "period_count": len(values),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(np.min(drawdown)),
        "turnover_mean": float(np.mean(returns["turnover"].to_numpy(dtype=float))),
        "cost_total": float(np.sum(returns["cost"].to_numpy(dtype=float))),
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    labels: pd.Series,
    protocol: BenchmarkProtocolSpec,
    frequency: FrequencySpec,
) -> tuple[dict[str, float | int | None], pd.DataFrame, pd.DataFrame]:
    prediction_table = predictions.set_index(["timestamp", "symbol"])[["score"]].sort_index()
    joined = prediction_table.join(labels.rename("label"), how="left")
    returns = _portfolio_returns(joined, protocol)
    metrics = {**_signal_metrics(joined), **_portfolio_metrics(returns, frequency)}
    return metrics, joined.reset_index(), returns
