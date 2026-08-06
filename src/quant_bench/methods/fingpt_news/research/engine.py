"""Dependency-light FinGPT sentiment aggregation and paper-backtest engine."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_bench.methods.fingpt_news.research.config import (
    BacktestSpec,
    DateWindow,
    FinGPTResearchConfig,
)

LABEL_SCORE = {"negative": -1, "neutral": 0, "positive": 1}
LABELS = tuple(LABEL_SCORE)


@dataclass(frozen=True)
class InputBundle:
    validation_prices: pd.DataFrame
    test_prices: pd.DataFrame
    validation_sentiment: pd.DataFrame
    test_sentiment: pd.DataFrame
    quality: dict[str, Any]


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    per_symbol_summary: pd.DataFrame
    portfolio: pd.DataFrame
    portfolio_summary: pd.DataFrame
    trades: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _window_mask(values: pd.Series, window: DateWindow) -> pd.Series:
    return (values >= pd.Timestamp(window.start)) & (values < pd.Timestamp(window.end))


def _load_prices(path: Path, cfg: FinGPTResearchConfig, window: DateWindow) -> pd.DataFrame:
    columns = cfg.inputs
    header = pd.read_csv(path, nrows=0)
    required = {columns.price_date_column, columns.price_symbol_column, columns.price_close_column}
    missing = sorted(required - set(header.columns))
    if missing:
        raise ValueError(f"price CSV is missing required columns: {missing}")
    frame = pd.read_csv(path, usecols=list(required), low_memory=False).rename(
        columns={
            columns.price_date_column: "date",
            columns.price_symbol_column: "symbol",
            columns.price_close_column: "close",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.floor("D")
    frame["symbol"] = frame["symbol"].fillna("").astype(str).str.strip().str.upper()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame[_window_mask(frame["date"], window) & frame["symbol"].isin(cfg.symbols)].copy()
    if frame[["date", "symbol", "close"]].isna().any().any():
        raise ValueError("price CSV contains invalid date, symbol, or close values in the selected window")
    if (~np.isfinite(frame["close"].to_numpy(dtype=float))).any() or (frame["close"] <= 0).any():
        raise ValueError("price close values must be finite and positive")
    duplicates = frame.duplicated(["date", "symbol"], keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, ["date", "symbol"]].head(5).astype(str).to_dict("records")
        raise ValueError(f"price CSV contains duplicate symbol/date rows: {examples}")
    missing_symbols = sorted(set(cfg.symbols) - set(frame["symbol"]))
    if missing_symbols:
        raise ValueError(f"price CSV has no rows for symbols in {window.start}:{window.end}: {missing_symbols}")

    calendars = [
        set(frame.loc[frame["symbol"] == symbol, "date"].tolist())
        for symbol in cfg.symbols
    ]
    common_dates = set.intersection(*calendars)
    if len(common_dates) < 2:
        raise ValueError(f"fewer than two common price dates in {window.start}:{window.end}")
    frame = frame[frame["date"].isin(common_dates)].sort_values(["date", "symbol"]).reset_index(drop=True)
    expected = len(common_dates) * len(cfg.symbols)
    if len(frame) != expected:
        raise ValueError("price intersection calendar is incomplete")
    return frame


def _sentiment_score(frame: pd.DataFrame) -> pd.Series:
    if "article_score" in frame:
        return pd.to_numeric(frame["article_score"], errors="coerce")
    if {"prob_positive", "prob_negative"}.issubset(frame.columns):
        positive = pd.to_numeric(frame["prob_positive"], errors="coerce")
        negative = pd.to_numeric(frame["prob_negative"], errors="coerce")
        return positive - negative
    if "model_score" in frame:
        return pd.to_numeric(frame["model_score"], errors="coerce")
    raise ValueError(
        "sentiment CSV requires article_score, prob_positive/prob_negative, or model_score"
    )


def _load_sentiment(
    path: Path,
    cfg: FinGPTResearchConfig,
    window: DateWindow,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, low_memory=False)
    required = {"date", "symbol"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"sentiment CSV is missing required columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.floor("D")
    frame["datetime"] = pd.to_datetime(
        frame["datetime"] if "datetime" in frame else frame["date"], errors="coerce"
    )
    frame["symbol"] = frame["symbol"].fillna("").astype(str).str.strip().str.upper()
    frame["bucket"] = (
        frame["bucket"].fillna("asset").astype(str).str.strip().str.lower()
        if "bucket" in frame
        else "asset"
    )
    invalid_buckets = sorted(set(frame["bucket"]) - {"asset", "market"})
    if invalid_buckets:
        raise ValueError(f"unsupported sentiment buckets: {invalid_buckets}")
    frame["article_score"] = _sentiment_score(frame)
    frame["relevance_weight"] = pd.to_numeric(
        frame.get("relevance_weight", 1.0),
        errors="coerce",
    ).fillna(1.0)
    frame["source"] = (
        frame["source"].fillna("unknown").astype(str)
        if "source" in frame
        else "unknown"
    )
    frame = frame[_window_mask(frame["date"], window)].copy()
    if frame.empty:
        raise ValueError(f"sentiment CSV has no rows in {window.start}:{window.end}: {path}")
    if frame[["date", "datetime", "article_score"]].isna().any().any():
        raise ValueError("sentiment CSV contains invalid date, datetime, or score values")
    scores = frame["article_score"].to_numpy(dtype=float)
    if (~np.isfinite(scores)).any() or (np.abs(scores) > 1.000001).any():
        raise ValueError("sentiment scores must be finite and within [-1, 1]")
    if (frame["relevance_weight"] < 0).any() or (~np.isfinite(frame["relevance_weight"])).any():
        raise ValueError("relevance weights must be finite and non-negative")
    invalid_assets = frame[(frame["bucket"] == "asset") & ~frame["symbol"].isin(cfg.symbols)]
    if not invalid_assets.empty:
        raise ValueError(
            "asset sentiment contains symbols outside config: "
            f"{sorted(invalid_assets['symbol'].unique().tolist())}"
        )

    original_rows = len(frame)
    if "news_id" in frame:
        frame["news_id"] = frame["news_id"].fillna("").astype(str)
        nonempty = frame["news_id"].str.len() > 0
        with_id = frame[nonempty].drop_duplicates(["bucket", "symbol", "news_id"], keep="last")
        without_id = frame[~nonempty]
        frame = pd.concat([with_id, without_id], ignore_index=True)
    else:
        dedupe_columns = ["bucket", "symbol", "date", "datetime", "source", "article_score"]
        frame = frame.drop_duplicates(dedupe_columns, keep="last")
    frame = frame.sort_values(["date", "bucket", "symbol", "datetime"]).reset_index(drop=True)
    report = {
        "path": str(path),
        "sha256": _sha256(path),
        "selected_rows": len(frame),
        "duplicates_removed": original_rows - len(frame),
        "start": frame["date"].min().strftime("%Y-%m-%d"),
        "end": frame["date"].max().strftime("%Y-%m-%d"),
        "asset_rows": int((frame["bucket"] == "asset").sum()),
        "market_rows": int((frame["bucket"] == "market").sum()),
    }
    return frame, report


def load_inputs(cfg: FinGPTResearchConfig) -> InputBundle:
    for path in (cfg.inputs.prices, cfg.inputs.validation_sentiment, cfg.inputs.test_sentiment):
        if not path.is_file():
            raise FileNotFoundError(f"input file does not exist: {path}")
    validation_prices = _load_prices(cfg.inputs.prices, cfg, cfg.validation)
    test_prices = _load_prices(cfg.inputs.prices, cfg, cfg.test)
    validation_sentiment, validation_report = _load_sentiment(
        cfg.inputs.validation_sentiment, cfg, cfg.validation
    )
    test_sentiment, test_report = _load_sentiment(cfg.inputs.test_sentiment, cfg, cfg.test)
    quality = {
        "valid": True,
        "prices": {
            "path": str(cfg.inputs.prices),
            "sha256": _sha256(cfg.inputs.prices),
            "validation_rows": len(validation_prices),
            "test_rows": len(test_prices),
            "validation_dates": int(validation_prices["date"].nunique()),
            "test_dates": int(test_prices["date"].nunique()),
        },
        "validation_sentiment": validation_report,
        "test_sentiment": test_report,
        "symbols": cfg.symbols,
        "windows_half_open": True,
        "validation_test_overlap": False,
    }
    return InputBundle(
        validation_prices=validation_prices,
        test_prices=test_prices,
        validation_sentiment=validation_sentiment,
        test_sentiment=test_sentiment,
        quality=quality,
    )


def _weighted_score(group: pd.DataFrame) -> float:
    if group.empty:
        return 0.0
    weights = group["effective_weight"].to_numpy(dtype=float)
    scores = group["article_score"].to_numpy(dtype=float)
    return float(np.average(scores, weights=weights)) if weights.sum() > 0 else float(np.mean(scores))


def _prepare_bucket(
    group: pd.DataFrame,
    *,
    top_n: int,
    min_abs_score: float,
    score_power: float,
) -> pd.DataFrame:
    work = group.copy()
    work["abs_article_score"] = work["article_score"].abs()
    if min_abs_score > 0:
        work = work[work["abs_article_score"] >= min_abs_score].copy()
    if work.empty:
        return work
    work["effective_weight"] = work["relevance_weight"] * np.power(
        work["abs_article_score"].clip(lower=1e-12), score_power
    )
    work = work.sort_values(
        ["effective_weight", "abs_article_score", "datetime"],
        ascending=[False, False, False],
    )
    return work.head(top_n) if top_n > 0 else work


def build_daily_scores(
    sentiment: pd.DataFrame,
    prices: pd.DataFrame,
    symbols: list[str],
    aggregation: dict[str, float | int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    asset_top_n = int(aggregation["asset_top_n"])
    market_top_n = int(aggregation["market_top_n"])
    min_abs_score = float(aggregation["min_abs_score"])
    score_power = float(aggregation["score_power"])
    market_rows = sentiment[sentiment["bucket"] == "market"]
    market_by_date = {date: group for date, group in market_rows.groupby("date", sort=True)}
    asset_rows = sentiment[sentiment["bucket"] == "asset"]
    asset_by_key = {
        (symbol, date): group
        for (symbol, date), group in asset_rows.groupby(["symbol", "date"], sort=True)
    }
    empty = sentiment.iloc[0:0]
    for price_row in prices[["date", "symbol"]].itertuples(index=False):
        asset = _prepare_bucket(
            asset_by_key.get((price_row.symbol, price_row.date), empty),
            top_n=asset_top_n,
            min_abs_score=min_abs_score,
            score_power=score_power,
        )
        market = _prepare_bucket(
            market_by_date.get(price_row.date, empty),
            top_n=market_top_n,
            min_abs_score=min_abs_score,
            score_power=score_power,
        )
        rows.append(
            {
                "date": price_row.date,
                "symbol": price_row.symbol,
                "asset_score": _weighted_score(asset),
                "market_score": _weighted_score(market),
                "asset_news_count": len(asset),
                "market_news_count": len(market),
                "asset_weight_sum": float(asset.get("effective_weight", pd.Series(dtype=float)).sum()),
                "market_weight_sum": float(market.get("effective_weight", pd.Series(dtype=float)).sum()),
                "classified_news": len(asset) + len(market),
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


def label_from_score(score: float, threshold: float) -> str:
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"


def apply_signal_config(
    daily_scores: pd.DataFrame,
    cfg: FinGPTResearchConfig,
    signal: dict[str, Any],
    thresholds: float | dict[str, float],
) -> pd.DataFrame:
    daily = daily_scores.copy()
    daily["market_beta"] = daily["symbol"].map(cfg.signals.market_beta).fillna(1.0).astype(float)
    variant = str(signal["variant"])
    if variant == "asset_only":
        daily["alpha"] = 1.0
        daily["final_score"] = daily["asset_score"]
    elif variant == "market_only":
        daily["alpha"] = 0.0
        daily["final_score"] = daily["market_beta"] * daily["market_score"]
    elif variant == "hybrid_fixed":
        alpha = float(signal["fixed_alpha"])
        daily["alpha"] = alpha
        daily["final_score"] = (
            alpha * daily["asset_score"]
            + (1.0 - alpha) * daily["market_beta"] * daily["market_score"]
        )
    elif variant == "hybrid_adaptive":
        k = float(signal["adaptive_k"])
        denominator = daily["asset_weight_sum"] + k
        raw_alpha = np.divide(
            daily["asset_weight_sum"],
            denominator,
            out=np.zeros(len(daily), dtype=float),
            where=denominator.to_numpy(dtype=float) > 0,
        )
        daily["alpha"] = np.clip(raw_alpha, float(signal["min_alpha"]), float(signal["max_alpha"]))
        daily["final_score"] = (
            daily["alpha"] * daily["asset_score"]
            + (1.0 - daily["alpha"]) * daily["market_beta"] * daily["market_score"]
        )
    else:
        raise ValueError(f"unsupported strategy variant: {variant}")
    if isinstance(thresholds, dict):
        missing = sorted(set(cfg.symbols) - set(thresholds))
        if missing:
            raise ValueError(f"threshold map is missing symbols: {missing}")
        daily["threshold"] = daily["symbol"].map(thresholds).astype(float)
    else:
        daily["threshold"] = float(thresholds)
    daily["model_label"] = [
        label_from_score(score, threshold)
        for score, threshold in zip(daily["final_score"], daily["threshold"], strict=True)
    ]
    if cfg.signals.conflict_policy == "neutral":
        conflict = (
            (np.sign(daily["asset_score"]) != np.sign(daily["market_score"]))
            & (daily["asset_score"].abs() >= cfg.signals.conflict_threshold)
            & (daily["market_score"].abs() >= cfg.signals.conflict_threshold)
        )
        daily.loc[conflict, "model_label"] = "neutral"
    daily["model_score"] = daily["model_label"].map(LABEL_SCORE).astype(int)
    daily["strategy_variant"] = variant
    return daily


def performance_metrics(returns: pd.Series, annualization: int) -> dict[str, float | int | None]:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float)
    if clean.empty:
        return {"period_count": 0, "total_return": None, "sharpe": None, "max_drawdown": None}
    equity = (1.0 + clean).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = len(clean) / float(annualization)
    annual_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else -1.0
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    annual_vol = std * math.sqrt(annualization)
    sharpe = float(clean.mean() / std * math.sqrt(annualization)) if std > 0 else None
    downside = clean[clean < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(clean.mean() / downside_std * math.sqrt(annualization)) if downside_std > 0 else None
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else None
    gross_profit = float(clean[clean > 0].sum())
    gross_loss = float(-clean[clean < 0].sum())
    return {
        "period_count": len(clean),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "win_rate": float((clean > 0).mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
    }


def run_capital_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: FinGPTResearchConfig,
) -> BacktestResult:
    frames: list[pd.DataFrame] = []
    spec: BacktestSpec = cfg.backtest
    one_way_cost = (spec.fee_bps + spec.slippage_bps) / 10_000.0
    per_symbol_capital = float(spec.per_symbol_capital_usdt)
    for symbol in cfg.symbols:
        px = prices[prices["symbol"] == symbol].sort_values("date").copy()
        sig = signals[signals["symbol"] == symbol].sort_values("date").copy()
        df = px.merge(sig, on=["symbol", "date"], how="left", validate="one_to_one")
        if df["model_score"].isna().any():
            raise ValueError(f"signal calendar is incomplete for {symbol}")
        raw_target = np.where(
            df["model_score"] > 0,
            1.0,
            np.where(df["model_score"] < 0, 0.0, np.nan),
        )
        df["decision_target"] = pd.Series(raw_target, index=df.index).ffill().fillna(0.0)
        df["position"] = df["decision_target"].shift(spec.signal_lag_days).fillna(0.0)
        df["position_before"] = df["position"].shift(1).fillna(0.0)
        df["asset_return"] = df["close"].pct_change().fillna(0.0)
        df["turnover"] = (df["position"] - df["position_before"]).abs()
        df["cost"] = df["turnover"] * one_way_cost
        df["strategy_return"] = df["position"] * df["asset_return"] - df["cost"]
        df["buy_hold_return"] = df["asset_return"]
        df["account_equity"] = per_symbol_capital * (1.0 + df["strategy_return"]).cumprod()
        df["account_equity_before"] = df["account_equity"].shift(1).fillna(per_symbol_capital)
        df["transaction_cost_usdt"] = df["account_equity_before"] * df["cost"]
        df["buy_hold_account_equity"] = per_symbol_capital * (1.0 + df["buy_hold_return"]).cumprod()
        df["cash_equity"] = df["account_equity"] * (1.0 - df["position"])
        df["position_value"] = df["account_equity"] * df["position"]
        df["signal_date"] = df["date"].shift(spec.signal_lag_days)
        df["executed_signal_label"] = df["model_label"].shift(spec.signal_lag_days)
        df["executed_final_score"] = df["final_score"].shift(spec.signal_lag_days)
        frames.append(df)
    daily = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)

    summary_rows: list[dict[str, Any]] = []
    for symbol, group in daily.groupby("symbol", sort=False):
        strategy = performance_metrics(group["strategy_return"], spec.annualization)
        baseline = performance_metrics(group["buy_hold_return"], spec.annualization)
        summary_rows.append(
            {
                "symbol": symbol,
                "start": group["date"].min().strftime("%Y-%m-%d"),
                "end": group["date"].max().strftime("%Y-%m-%d"),
                "days": len(group),
                "model_signal_days": int((group["model_score"] != 0).sum()),
                "classified_news": int(group["classified_news"].sum()),
                "exposure": float(group["position"].mean()),
                "trades": int((group["turnover"] > 0).sum()),
                "turnover": float(group["turnover"].sum()),
                "cost_total_usdt": float(group["transaction_cost_usdt"].sum()),
                **{f"strategy_{key}": value for key, value in strategy.items()},
                **{f"buy_hold_{key}": value for key, value in baseline.items()},
                "excess_total_return": float(strategy["total_return"] or 0.0)
                - float(baseline["total_return"] or 0.0),
            }
        )
    per_symbol_summary = pd.DataFrame(summary_rows).sort_values("symbol").reset_index(drop=True)

    portfolio = daily.groupby("date", as_index=False).agg(
        total_equity_usdt=("account_equity", "sum"),
        baseline_equity=("buy_hold_account_equity", "sum"),
        cash_total_usdt=("cash_equity", "sum"),
        holdings_value_usdt=("position_value", "sum"),
        active_positions=("position", "sum"),
        gross_exposure=("position", "mean"),
        cycle_turnover=("turnover", "sum"),
        cycle_cost=("cost", "sum"),
        cycle_cost_usdt=("transaction_cost_usdt", "sum"),
    )
    portfolio["strategy_return"] = portfolio["total_equity_usdt"].pct_change().fillna(0.0)
    portfolio["buy_hold_return"] = portfolio["baseline_equity"].pct_change().fillna(0.0)
    portfolio["cash_ratio"] = portfolio["cash_total_usdt"] / portfolio["total_equity_usdt"]
    portfolio["strategy_equity"] = portfolio["total_equity_usdt"] / spec.initial_capital_usdt
    portfolio["buy_hold_strategy_equity"] = portfolio["baseline_equity"] / spec.initial_capital_usdt
    portfolio["strategy_pnl"] = portfolio["total_equity_usdt"] - spec.initial_capital_usdt
    portfolio["baseline_pnl"] = portfolio["baseline_equity"] - spec.initial_capital_usdt
    strategy_metrics = performance_metrics(portfolio["strategy_return"], spec.annualization)
    baseline_metrics = performance_metrics(portfolio["buy_hold_return"], spec.annualization)
    portfolio_summary = pd.DataFrame(
        [
            {
                "initial_capital_usdt": spec.initial_capital_usdt,
                "final_capital_usdt": float(portfolio["total_equity_usdt"].iloc[-1]),
                "buy_hold_final_capital_usdt": float(portfolio["baseline_equity"].iloc[-1]),
                "avg_active_positions": float(portfolio["active_positions"].mean()),
                "avg_gross_exposure": float(portfolio["gross_exposure"].mean()),
                "avg_cash_ratio": float(portfolio["cash_ratio"].mean()),
                "trades": int((daily["turnover"] > 0).sum()),
                "model_signal_days": int((daily["model_score"] != 0).sum()),
                "classified_news": int(daily["classified_news"].sum()),
                "turnover": float(daily["turnover"].sum()),
                "cost_total_usdt": float(daily["transaction_cost_usdt"].sum()),
                **{f"strategy_{key}": value for key, value in strategy_metrics.items()},
                **{f"buy_hold_{key}": value for key, value in baseline_metrics.items()},
                "excess_total_return": float(strategy_metrics["total_return"] or 0.0)
                - float(baseline_metrics["total_return"] or 0.0),
            }
        ]
    )

    trade_rows: list[dict[str, Any]] = []
    for row in daily[daily["turnover"] > 0].itertuples(index=False):
        before_equity = float(row.account_equity_before)
        notional = before_equity * float(row.turnover)
        qty = notional / float(row.close)
        side = "buy" if row.position > row.position_before else "sell"
        trade_key = f"{cfg.strategy_id}|{row.symbol}|{row.date}|{side}"
        trade_rows.append(
            {
                "trade_id": hashlib.sha256(trade_key.encode()).hexdigest()[:20],
                "timestamp": int(
                    pd.Timestamp(row.date)
                    .tz_localize("Asia/Shanghai")
                    .timestamp()
                ),
                "datetime": pd.Timestamp(row.date)
                .tz_localize("Asia/Shanghai")
                .isoformat(),
                "trade_date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                "signal_date": pd.Timestamp(row.signal_date).strftime("%Y-%m-%d") if pd.notna(row.signal_date) else "",
                "coin": row.symbol,
                "side": side,
                "price": float(row.close),
                "qty": qty,
                "notional_usdt": notional,
                "fee_usdt": notional * spec.fee_bps / 10_000.0,
                "slippage_bps": spec.slippage_bps,
                "cash_before": before_equity * (1.0 - float(row.position_before)),
                "cash_after": float(row.account_equity) * (1.0 - float(row.position)),
                "position_qty_before": before_equity * float(row.position_before) / float(row.close),
                "position_qty_after": float(row.account_equity) * float(row.position) / float(row.close),
                "reason": (
                    f"delayed_{row.executed_signal_label}"
                    if pd.notna(row.executed_signal_label)
                    else "delayed_signal"
                ),
                "success": True,
                "signal_score": row.executed_final_score,
            }
        )
    trades = pd.DataFrame(trade_rows)
    return BacktestResult(daily, per_symbol_summary, portfolio, portfolio_summary, trades)


def classification_report(sentiment: pd.DataFrame) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    if not {"true_label", "model_label"}.issubset(sentiment.columns):
        return None, pd.DataFrame()
    frame = sentiment[["true_label", "model_label"]].dropna().copy()
    frame["true_label"] = frame["true_label"].astype(str).str.lower()
    frame["model_label"] = frame["model_label"].astype(str).str.lower()
    frame = frame[frame["true_label"].isin(LABELS) & frame["model_label"].isin(LABELS)]
    if frame.empty:
        return None, pd.DataFrame()
    confusion = pd.crosstab(frame["true_label"], frame["model_label"]).reindex(
        index=LABELS, columns=LABELS, fill_value=0
    )
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in LABELS:
        tp = int(confusion.at[label, label])
        fp = int(confusion[label].sum() - tp)
        fn = int(confusion.loc[label].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {"support": int(confusion.loc[label].sum()), "precision": precision,
                            "recall": recall, "f1": f1}
    report = {
        "sample_count": len(frame),
        "accuracy": float((frame["true_label"] == frame["model_label"]).mean()),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
    }
    confusion_out = confusion.rename_axis("true_label").reset_index()
    return report, confusion_out
