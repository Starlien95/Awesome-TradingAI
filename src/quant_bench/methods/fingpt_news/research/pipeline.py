"""High-level FinGPT research workflow with reproducible public artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, time
from importlib import resources
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quant_bench.methods.fingpt_news.research.config import FinGPTResearchConfig
from quant_bench.methods.fingpt_news.research.engine import (
    BacktestResult,
    apply_signal_config,
    build_daily_scores,
    classification_report,
    load_inputs,
    run_capital_backtest,
)
from quant_bench.methods.fingpt_news.research.tuning import (
    TuningResult,
    tune_global,
    tune_per_symbol_thresholds,
)
from quant_bench.runtime.core.atomic_io import atomic_write_csv, atomic_write_json, atomic_write_text
from quant_bench.runtime.core.run_manifest import write_run_manifest

RESEARCH_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _local_midnight(value: Any) -> datetime:
    return datetime.combine(pd.Timestamp(value).date(), time.min, tzinfo=RESEARCH_TIMEZONE)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) or np.isinf(value) else float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, _jsonable(payload))


def _resource(name: str) -> resources.abc.Traversable:
    return (
        resources.files("quant_bench")
        .joinpath("methods")
        .joinpath("fingpt_news")
        .joinpath("resources")
        .joinpath("research_fixture")
        .joinpath(name)
    )


def initialize_research_workspace(workspace: str | Path, *, force: bool = False) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve() / "methods" / "fingpt_news" / "research"
    data_dir = root / "data"
    files = {
        root / "config.yaml": "config.yaml",
        data_dir / "prices.csv": "prices.csv",
        data_dir / "validation_sentiment.csv": "validation_sentiment.csv",
        data_dir / "test_sentiment.csv": "test_sentiment.csv",
    }
    for destination in files:
        if destination.exists() and not force:
            raise FileExistsError(f"refusing to replace existing research workspace file: {destination}")
    for destination, source_name in files.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_resource(source_name).read_text(encoding="utf-8"), encoding="utf-8")
    return {"workspace": str(root), "written": [str(path) for path in files]}


def validate_research_inputs(config_path: str | Path) -> dict[str, Any]:
    config = FinGPTResearchConfig.from_yaml(config_path)
    bundle = load_inputs(config)
    return bundle.quality


def _write_classification(root: Path, sentiment: pd.DataFrame) -> None:
    report, confusion = classification_report(sentiment)
    if report is None:
        return
    _write_json(root / "classification_metrics.json", report)
    atomic_write_csv(confusion, root / "confusion_matrix.csv")


def _write_result(root: Path, result: BacktestResult, signals: pd.DataFrame) -> None:
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(signals, root / "daily_signals.csv")
    atomic_write_csv(result.daily, root / "daily_backtest.csv")
    atomic_write_csv(result.per_symbol_summary, root / "per_symbol_summary.csv")
    atomic_write_csv(result.portfolio, root / "portfolio_daily.csv")
    atomic_write_csv(result.portfolio_summary, root / "portfolio_summary.csv")
    atomic_write_csv(result.trades, root / "trade_list.csv")


def _canonical_metrics(
    result: BacktestResult,
    prices: pd.DataFrame,
    cfg: FinGPTResearchConfig,
) -> pd.DataFrame:
    portfolio = result.portfolio.copy()
    benchmark_symbol = "BTC-USDT" if "BTC-USDT" in cfg.symbols else cfg.symbols[0]
    benchmark = prices[prices["symbol"] == benchmark_symbol][["date", "close"]].rename(
        columns={"close": "btc_price"}
    )
    portfolio = portfolio.merge(benchmark, on="date", how="left", validate="one_to_one")
    portfolio["timestamp"] = portfolio["date"].map(
        lambda value: int(_local_midnight(value).timestamp())
    )
    portfolio["datetime"] = portfolio["date"].map(
        lambda value: _local_midnight(value).isoformat()
    )
    portfolio["bar_timestamp"] = portfolio["timestamp"]
    portfolio["bar_datetime"] = portfolio["datetime"]
    portfolio["initial_capital_usdt"] = cfg.backtest.initial_capital_usdt
    portfolio["strategy_returns_pct"] = (portfolio["strategy_equity"] - 1.0) * 100.0
    portfolio["strategy_equity"] = portfolio["total_equity_usdt"]
    portfolio["baseline_returns_pct"] = (
        portfolio["baseline_equity"] / cfg.backtest.initial_capital_usdt - 1.0
    ) * 100.0
    portfolio["method_family"] = "fingpt_news"
    portfolio["strategy_id"] = cfg.strategy_id
    portfolio["frequency"] = "1d"
    columns = [
        "timestamp", "datetime", "bar_timestamp", "bar_datetime",
        "initial_capital_usdt", "cash_total_usdt", "holdings_value_usdt", "total_equity_usdt",
        "strategy_equity", "strategy_pnl", "strategy_returns_pct",
        "baseline_equity", "baseline_pnl", "baseline_returns_pct", "btc_price",
        "active_positions", "gross_exposure", "cash_ratio",
        "method_family", "strategy_id", "frequency",
    ]
    return portfolio[columns]


def _canonical_signals(
    result: BacktestResult,
    signals: pd.DataFrame,
    cfg: FinGPTResearchConfig,
) -> pd.DataFrame:
    prices = result.daily[["date", "symbol", "close", "asset_return"]].copy()
    prices["price_future"] = prices.groupby("symbol")["close"].shift(-1)
    prices["true_return"] = prices.groupby("symbol")["asset_return"].shift(-1)
    table = signals.merge(prices, on=["date", "symbol"], how="left", validate="one_to_one")
    table = table.sort_values(["date", "symbol"]).reset_index(drop=True)
    table["cycle_id"] = pd.factorize(table["date"], sort=True)[0] + 1
    table["timestamp"] = table["date"].map(
        lambda value: int(_local_midnight(value).timestamp())
    )
    table["datetime"] = table["date"].map(
        lambda value: _local_midnight(value).isoformat()
    )
    table["bar_timestamp"] = table["timestamp"]
    table["bar_datetime"] = table["datetime"]
    table["coin"] = table["symbol"]
    table["score"] = table["final_score"]
    table["signal_label"] = table["model_label"]
    table["signal_score"] = table["model_score"]
    table["price_at_pred"] = table["close"]
    table["labeled"] = table["price_future"].notna()
    table["method_family"] = "fingpt_news"
    table["strategy_id"] = cfg.strategy_id
    table["frequency"] = "1d"
    columns = [
        "cycle_id", "timestamp", "datetime", "bar_timestamp", "bar_datetime",
        "coin", "score", "signal_label", "signal_score", "price_at_pred", "price_future",
        "true_return", "labeled", "method_family", "strategy_id", "frequency",
    ]
    return table[columns]


def _canonical_volume(result: BacktestResult) -> pd.DataFrame:
    dates = result.portfolio[["date"]].copy()
    symbols = sorted(result.daily["symbol"].unique().tolist())
    if result.trades.empty:
        volume = pd.DataFrame({"date": dates["date"], "cycle_volume_usdt": 0.0})
        per_symbol = pd.DataFrame(0.0, index=range(len(dates)), columns=symbols)
    else:
        grouped = result.trades.groupby("trade_date", as_index=False)["notional_usdt"].sum().rename(
            columns={"trade_date": "date", "notional_usdt": "cycle_volume_usdt"}
        )
        grouped["date"] = pd.to_datetime(grouped["date"])
        volume = dates.merge(grouped, on="date", how="left").fillna({"cycle_volume_usdt": 0.0})
        by_symbol = result.trades.pivot_table(
            index="trade_date",
            columns="coin",
            values="notional_usdt",
            aggfunc="sum",
            fill_value=0.0,
        )
        by_symbol.index = pd.to_datetime(by_symbol.index)
        per_symbol = (
            dates.set_index("date")
            .join(by_symbol, how="left")
            .reindex(columns=symbols, fill_value=0.0)
            .fillna(0.0)
            .reset_index(drop=True)
        )
    volume["timestamp"] = volume["date"].map(
        lambda value: int(_local_midnight(value).timestamp())
    )
    volume["datetime"] = volume["date"].map(
        lambda value: _local_midnight(value).isoformat()
    )
    volume["cumulative_total_volume_usdt"] = volume["cycle_volume_usdt"].cumsum()
    volume["daily_total_volume_usdt"] = volume["cycle_volume_usdt"]
    cumulative = per_symbol.cumsum()
    volume["per_coin_daily_json"] = [
        json.dumps(
            {symbol: float(row[symbol]) for symbol in symbols if float(row[symbol]) > 0.0},
            sort_keys=True,
            separators=(",", ":"),
        )
        for _, row in per_symbol.iterrows()
    ]
    volume["per_coin_cumulative_json"] = [
        json.dumps(
            {symbol: float(row[symbol]) for symbol in symbols if float(row[symbol]) > 0.0},
            sort_keys=True,
            separators=(",", ":"),
        )
        for _, row in cumulative.iterrows()
    ]
    return volume[
        ["timestamp", "datetime", "cycle_volume_usdt", "cumulative_total_volume_usdt",
         "daily_total_volume_usdt", "per_coin_cumulative_json", "per_coin_daily_json"]
    ]


def _write_canonical_outputs(
    output: Path,
    result: BacktestResult,
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: FinGPTResearchConfig,
) -> dict[str, str]:
    metrics_path = output / "metrics" / f"{cfg.strategy_id}_metrics.csv"
    signals_path = output / "signals" / f"{cfg.strategy_id}_signals.csv"
    trades_path = output / "trades" / f"{cfg.strategy_id}_trades.csv"
    volume_path = output / "volume" / f"{cfg.strategy_id}_volume.csv"
    atomic_write_csv(_canonical_metrics(result, prices, cfg), metrics_path)
    atomic_write_csv(_canonical_signals(result, signals, cfg), signals_path)
    atomic_write_csv(result.trades, trades_path)
    atomic_write_csv(_canonical_volume(result), volume_path)
    return {
        "metrics_path": str(metrics_path.relative_to(output)),
        "signals_path": str(signals_path.relative_to(output)),
        "trades_path": str(trades_path.relative_to(output)),
        "volume_path": str(volume_path.relative_to(output)),
    }


def _checksums(root: Path) -> None:
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"checksums.sha256", "run_manifest.json"}:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(root).as_posix()}")
    atomic_write_text(root / "checksums.sha256", "\n".join(rows) + "\n")


def _chosen_payload(tuning: TuningResult, cfg: FinGPTResearchConfig) -> dict[str, Any]:
    return {
        "aggregation": tuning.aggregation,
        "signal": tuning.signal,
        "threshold": tuning.threshold,
        "selection_status": tuning.selection_status,
        "objective": cfg.selection.objective,
        "selection_window": cfg.validation.model_dump(),
        "test_window": cfg.test.model_dump(),
        "test_metrics_used_for_selection": False,
    }


def run_research_pipeline(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    cfg = FinGPTResearchConfig.from_yaml(config_path)
    output = Path(output_dir).expanduser().resolve()
    if (output / "run_manifest.json").exists() and not force:
        raise FileExistsError(f"completed run already exists; pass --force to overwrite known outputs: {output}")
    output.mkdir(parents=True, exist_ok=True)
    bundle = load_inputs(cfg)
    atomic_write_text(output / "resolved_config.yaml", cfg.to_yaml())
    _write_json(output / "data_quality.json", bundle.quality)
    _write_classification(output / "validation", bundle.validation_sentiment)
    _write_classification(output / "test", bundle.test_sentiment)

    tuning = tune_global(bundle.validation_sentiment, bundle.validation_prices, cfg)
    atomic_write_csv(tuning.trials, output / "validation" / "parameter_trials.csv")
    atomic_write_csv(tuning.validation_scores, output / "validation" / "daily_scores.csv")
    atomic_write_csv(tuning.validation_signals, output / "validation" / "daily_signals.csv")
    chosen = _chosen_payload(tuning, cfg)
    _write_json(output / "chosen_config.json", chosen)

    test_scores = build_daily_scores(
        bundle.test_sentiment, bundle.test_prices, cfg.symbols, tuning.aggregation
    )
    test_signals = apply_signal_config(test_scores, cfg, tuning.signal, tuning.threshold)
    global_result = run_capital_backtest(bundle.test_prices, test_signals, cfg)
    atomic_write_csv(test_scores, output / "test" / "daily_scores.csv")
    _write_result(output / "global_tuned", global_result, test_signals)
    manifest_paths = _write_canonical_outputs(output, global_result, test_signals, bundle.test_prices, cfg)

    per_symbol_payload: dict[str, Any] | None = None
    if cfg.selection.enable_per_symbol_thresholds:
        thresholds, trials = tune_per_symbol_thresholds(
            tuning.validation_scores,
            bundle.validation_prices,
            cfg,
            tuning.signal,
            tuning.threshold,
        )
        atomic_write_csv(trials, output / "validation" / "per_symbol_threshold_trials.csv")
        per_symbol_signals = apply_signal_config(test_scores, cfg, tuning.signal, thresholds)
        per_symbol_result = run_capital_backtest(bundle.test_prices, per_symbol_signals, cfg)
        _write_result(output / "per_symbol_tuned", per_symbol_result, per_symbol_signals)
        per_symbol_payload = {
            "thresholds": thresholds,
            "fallback_threshold": tuning.threshold,
            "selection_window": cfg.validation.model_dump(),
            "test_metrics_used_for_selection": False,
        }
        _write_json(output / "per_symbol_tuned" / "chosen_config.json", per_symbol_payload)

    _checksums(output)
    write_run_manifest(
        output,
        strategy_id=cfg.strategy_id,
        method_family="fingpt_news",
        frequency="1d",
        mode="research_backtest",
        paths=manifest_paths,
        extra={
            "schema_version": cfg.schema_version,
            "status": "COMPLETED",
            "symbols": cfg.symbols,
            "selection_window": cfg.validation.model_dump(),
            "test_window": cfg.test.model_dump(),
            "model_id": cfg.model_id,
            "adapter_id": cfg.adapter_id,
            "dataset_id": cfg.dataset_id,
            "chosen_config_path": "chosen_config.json",
            "data_quality_path": "data_quality.json",
            "checksums_path": "checksums.sha256",
        },
    )
    return {
        "output_dir": str(output),
        "strategy_id": cfg.strategy_id,
        "selection": chosen,
        "per_symbol_selection": per_symbol_payload,
        "test_portfolio": _jsonable(global_result.portfolio_summary.iloc[0].to_dict()),
    }
