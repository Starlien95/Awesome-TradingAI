from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from quant_bench.methods.fingpt_news import run_fingpt_news
from quant_bench.methods.fingpt_news.execution.okx_spot_broker import PerCoinOKXSpotBroker
from quant_bench.methods.fingpt_news.news.cleaner import clean_html_text
from quant_bench.methods.fingpt_news.research.config import FinGPTResearchConfig
from quant_bench.methods.fingpt_news.research.engine import (
    apply_signal_config,
    run_capital_backtest,
)


def _config() -> FinGPTResearchConfig:
    return FinGPTResearchConfig.model_validate(
        {
            "symbols": ["BTC-USDT"],
            "inputs": {
                "prices": "prices.csv",
                "validation_sentiment": "valid.csv",
                "test_sentiment": "test.csv",
            },
            "validation": {"start": "2024-01-01", "end": "2024-02-01"},
            "test": {"start": "2025-01-01", "end": "2025-02-01"},
            "aggregation": {
                "asset_top_n": [0],
                "market_top_n": [0],
                "min_abs_score": [0.0],
                "score_power": [0.0],
            },
            "signals": {"variants": ["asset_only"], "thresholds": [0.05]},
            "backtest": {
                "initial_capital_usdt": 1000.0,
                "per_symbol_capital_usdt": 1000.0,
                "fee_bps": 10.0,
                "slippage_bps": 5.0,
                "signal_lag_days": 1,
            },
        }
    )


def test_signal_is_delayed_and_neutral_holds_position() -> None:
    cfg = _config()
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
            "symbol": ["BTC-USDT"] * 4,
            "close": [100.0, 110.0, 121.0, 100.0],
        }
    )
    scores = pd.DataFrame(
        {
            "date": prices["date"],
            "symbol": prices["symbol"],
            "asset_score": [0.5, 0.0, -0.5, 0.0],
            "market_score": [0.0] * 4,
            "asset_news_count": [1] * 4,
            "market_news_count": [0] * 4,
            "asset_weight_sum": [1.0] * 4,
            "market_weight_sum": [0.0] * 4,
            "classified_news": [1] * 4,
        }
    )
    signal = {"variant": "asset_only", "fixed_alpha": 1.0, "adaptive_k": None,
              "min_alpha": None, "max_alpha": None}
    signals = apply_signal_config(scores, cfg, signal, 0.05)
    result = run_capital_backtest(prices, signals, cfg)
    daily = result.daily

    assert daily["position"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert daily.loc[0, "strategy_return"] == 0.0
    assert daily.loc[1, "strategy_return"] == pytest.approx(0.10 - 0.0015)
    assert daily.loc[2, "strategy_return"] == pytest.approx(0.10)
    assert daily.loc[3, "strategy_return"] == pytest.approx(-0.0015)
    assert result.trades["side"].tolist() == ["buy", "sell"]
    assert result.trades.iloc[0]["notional_usdt"] == pytest.approx(1000.0)
    assert result.trades.iloc[0]["fee_usdt"] == pytest.approx(1.0)
    assert result.per_symbol_summary.iloc[0]["cost_total_usdt"] > 0.0


def test_config_rejects_lookahead_and_overlapping_windows() -> None:
    payload = _config().model_dump(mode="json")
    payload["backtest"]["signal_lag_days"] = 0
    with pytest.raises(ValidationError, match="signal_lag_days"):
        FinGPTResearchConfig.model_validate(payload)

    payload = _config().model_dump(mode="json")
    payload["validation"]["end"] = "2025-01-02"
    with pytest.raises(ValidationError, match="validation window"):
        FinGPTResearchConfig.model_validate(payload)


def test_config_resolves_input_paths_relative_to_config(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    source.write_text(_config().to_yaml(), encoding="utf-8")
    loaded = FinGPTResearchConfig.from_yaml(source)
    assert loaded.inputs.prices == tmp_path / "prices.csv"


def test_okx_broker_reconciliation_is_opt_in(tmp_path: Path) -> None:
    class NoAccountExecutor:
        identifier = "test"

        def get_holdings(self, *args: object, **kwargs: object) -> dict[str, float]:
            raise AssertionError("account access must remain disabled")

    broker = PerCoinOKXSpotBroker(
        tmp_path / "state.json",
        NoAccountExecutor(),  # type: ignore[arg-type]
        ["BTC-USDT"],
        1000.0,
        1000.0,
        10.0,
        reconcile_account_positions_enabled=False,
    )
    broker.reconcile_account_positions({"BTC-USDT": 100.0}, trade_date="2025-01-01")


def test_packaged_live_defaults_are_public_safe_and_general() -> None:
    root = run_fingpt_news.CURRENT_DIR
    payload = yaml.safe_load(
        (root / "configs" / "sentiment_sft_live.yaml").read_text(encoding="utf-8")
    )
    assert payload["runtime"]["dry_run"] is True
    assert payload["trade"]["mode"] == "paper_spot"
    assert payload["trade"]["reconcile_account_positions"] is False
    assert payload["api"]["is_simulated"] is True
    assert payload["strategy"]["per_coin_params_path"] == "configs/per_coin_params_default.csv"
    assert payload["api"]["okx_api_key"] == ""
    assert payload["api"]["okx_secret_key"] == ""
    assert payload["api"]["okx_passphrase"] == ""
    assert payload["news"]["ccdata_api_key"] == ""

    params = pd.read_csv(root / "configs" / "per_coin_params_default.csv")
    assert set(params["symbol"]) == set(payload["trade"]["coins"])
    parameter_columns = [column for column in params.columns if column != "symbol"]
    assert len(params[parameter_columns].drop_duplicates()) == 1
    assert not {"sharpe", "total_return", "final_equity", "objective"}.intersection(params.columns)


def test_news_html_cleaner_has_no_optional_dependency_and_ignores_scripts() -> None:
    raw = "<article><h1>BTC &amp; ETH</h1><script>secret()</script><p>Daily update</p></article>"
    assert clean_html_text(raw) == "BTC & ETH Daily update"
