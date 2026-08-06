from __future__ import annotations

import argparse
import json
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml
from platformdirs import user_data_path

PACKAGE_ROOT = resources.files("quant_bench")


def _load_resource_yaml(*parts: str) -> dict:
    resource = PACKAGE_ROOT.joinpath(*parts)
    return yaml.safe_load(resource.read_text(encoding="utf-8")) or {}


def check_configs() -> None:
    required = [
        ("1h", "config_mlp.yaml"),
        ("4h", "config_xgboost.yaml"),
        ("4h", "config_gats.yaml"),
        ("4h", "config_lstm.yaml"),
        ("4h", "config_tra.yaml"),
        ("15m", "config_double_ensemble.yaml"),
        ("15m", "config_tabnet.yaml"),
        ("1h", "config_tcn.yaml"),
        ("5m", "config_macrohft_v1.yaml"),
        ("15m", "config_macrohft_v1.yaml"),
        ("1h", "config_macrohft_v1.yaml"),
        ("4h", "config_macrohft_v1.yaml"),
    ]
    for frequency, name in required:
        cfg = _load_resource_yaml("resources", "runtime", "timeframes", frequency, name)
        label = f"{frequency}/{name}"
        assert cfg, f"empty config: {label}"
        assert cfg["strategy"].get("capital_allocation_mode") == "full_investment", label
        assert float(cfg["strategy"].get("risk_degree", 0)) == 0.95, label
        assert cfg["api"].get("is_simulated") is True, label
        assert not cfg["api"].get("api_key"), label
        assert not cfg["api"].get("secret_key"), label
        assert not cfg["api"].get("passphrase"), label

    fingpt = _load_resource_yaml(
        "methods", "fingpt_news", "configs", "sentiment_sft_live.yaml"
    )
    assert fingpt["runtime"]["dry_run"] is True
    assert fingpt["trade"]["mode"] == "paper_spot"
    assert fingpt["api"]["is_simulated"] is True
    assert not fingpt["api"].get("okx_api_key")
    assert not fingpt["api"].get("okx_secret_key")
    assert not fingpt["api"].get("okx_passphrase")
    assert not fingpt["news"].get("ccdata_api_key")


def check_manifest_paths(workspace: Path) -> None:
    for manifest_path in workspace.rglob("run_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest.get("method_family"), manifest_path
        assert manifest.get("strategy_id"), manifest_path
        if "metrics_path" in manifest:
            assert (manifest_path.parent / manifest["metrics_path"]).exists(), manifest_path


def check_fingpt_csvs(workspace: Path) -> None:
    run_dir = workspace / "runs/fingpt_news/FinGPTSentimentSFT_1d"
    manifest = run_dir / "run_manifest.json"
    if not manifest.exists():
        return

    metrics = pd.read_csv(run_dir / "metrics/FinGPTSentimentSFT_1d_metrics.csv")
    signals = pd.read_csv(run_dir / "signals/FinGPTSentimentSFT_1d_signals.csv")
    daily = pd.read_csv(run_dir / "signals/daily_sentiment_scores.csv")
    assert {"timestamp", "strategy_equity", "method_family", "strategy_id"}.issubset(metrics.columns)
    assert {"cycle_id", "coin", "score", "method_family", "strategy_id"}.issubset(signals.columns)
    assert {"symbol", "model_label", "final_score", "threshold"}.issubset(daily.columns)


def check_capital_allocation() -> None:
    from quant_bench.runtime.adapters.base import BaseModelAdapter

    class DummyAdapter(BaseModelAdapter):
        def load_model(self, model_path: str):
            return None

        def predict(self, shared_data, config):
            return []

    cfg = SimpleNamespace(
        threshold=0.0,
        top_k=2,
        min_position_value_usdt=2.0,
        risk_degree=0.95,
        max_dropout=1,
        max_sell_per_cycle=1,
        max_position_count=2,
        trade_amount_usdt=1000.0,
        rebalance_tolerance_usdt=10.0,
        rebalance_tolerance_pct=0.001,
        capital_allocation_mode="full_investment",
        rebalance_policy="target_weight",
        enable_short=False,
        enforce_trade_coins_whitelist=False,
        coins=["BTC-USDT", "ETH-USDT", "ADA-USDT", "DOGE-USDT", "XRP-USDT"],
    )
    predictions = [
        {"coin": "BTC-USDT", "score": 0.20},
        {"coin": "ETH-USDT", "score": 0.10},
        {"coin": "XRP-USDT", "score": -0.10},
    ]
    signals = DummyAdapter().generate_signals(
        predictions,
        current_positions={},
        current_prices={"BTC-USDT": 100.0, "ETH-USDT": 50.0},
        config=cfg,
        budget_snapshot={
            "strategy_equity": 10000.0,
            "cash_available_for_strategy": 10000.0,
        },
    )
    assert len(signals) == 2
    assert {signal["side"] for signal in signals} == {"buy"}
    assert all(abs(signal["amount_usdt"] - 4750.0) < 1e-9 for signal in signals)

    no_churn = DummyAdapter().generate_signals(
        predictions,
        current_positions={"BTC-USDT": 47.54, "ETH-USDT": 94.96},
        current_prices={"BTC-USDT": 100.0, "ETH-USDT": 50.0},
        config=cfg,
        budget_snapshot={
            "strategy_equity": 10002.0,
            "cash_available_for_strategy": 0.0,
        },
    )
    assert no_churn == []

    # Safe mode: a holding outside this cycle's scored universe is treated as
    # external/manual and will not be sold.
    rollover = DummyAdapter().generate_signals(
        [
            {"coin": "BTC-USDT", "score": 0.30},
            {"coin": "ETH-USDT", "score": 0.20},
            {"coin": "ADA-USDT", "score": 0.10},
        ],
        current_positions={
            "DOGE-USDT": 1000.0,  # 100 USDT, smallest old holding -> sell
            "XRP-USDT": 1000.0,   # 500 USDT, retained, then topk cap can trim by abs(score)=0
        },
        current_prices={"DOGE-USDT": 0.1, "XRP-USDT": 0.5, "BTC-USDT": 100.0, "ETH-USDT": 50.0},
        config=cfg,
        budget_snapshot={
            "strategy_equity": 10000.0,
            "cash_available_for_strategy": 9400.0,
        },
    )
    assert not any(s["coin"] == "DOGE-USDT" and s["side"] == "sell" for s in rollover)
    assert {s["coin"] for s in rollover if s["side"] == "buy"} == {"BTC-USDT", "ETH-USDT"}

    # Dedicated strategy account mode: whitelisted holdings without a fresh
    # positive score get score=0, so custom_strategy's topk cap removes them.
    cfg.enforce_trade_coins_whitelist = True
    managed = DummyAdapter().generate_signals(
        [
            {"coin": "BTC-USDT", "score": 0.30},
            {"coin": "ETH-USDT", "score": 0.20},
            {"coin": "ADA-USDT", "score": 0.10},
        ],
        current_positions={"DOGE-USDT": 1000.0, "XRP-USDT": 1000.0},
        current_prices={"DOGE-USDT": 0.1, "XRP-USDT": 0.5, "BTC-USDT": 100.0, "ETH-USDT": 50.0},
        config=cfg,
        budget_snapshot={
            "strategy_equity": 10000.0,
            "cash_available_for_strategy": 9400.0,
        },
    )
    assert {s["coin"] for s in managed if s["side"] == "sell"} == {"DOGE-USDT", "XRP-USDT"}

    # Default live policy is intentionally quieter than strict target-weight
    # rebalance: retained target holdings are not adjusted just to hit equal
    # weights, while missing/new targets can still be bought.
    cfg.rebalance_policy = "passive_topk_dropout"
    passive = DummyAdapter().generate_signals(
        predictions,
        current_positions={"BTC-USDT": 60.0, "ETH-USDT": 20.0},
        current_prices={"BTC-USDT": 100.0, "ETH-USDT": 50.0},
        config=cfg,
        budget_snapshot={
            "strategy_equity": 10000.0,
            "cash_available_for_strategy": 3000.0,
        },
    )
    assert passive == [], passive

    passive_new = DummyAdapter().generate_signals(
        predictions,
        current_positions={"BTC-USDT": 60.0},
        current_prices={"BTC-USDT": 100.0, "ETH-USDT": 50.0},
        config=cfg,
        budget_snapshot={
            "strategy_equity": 10000.0,
            "cash_available_for_strategy": 4000.0,
        },
    )
    assert len(passive_new) == 1, passive_new
    assert passive_new[0]["side"] == "buy"
    assert passive_new[0]["coin"] == "ETH-USDT"
    assert abs(passive_new[0]["amount_usdt"] - 3500.0) < 1e-9

    passive_dropout = DummyAdapter().generate_signals(
        [
            {"coin": "BTC-USDT", "score": 0.30},
            {"coin": "ETH-USDT", "score": 0.20},
            {"coin": "ADA-USDT", "score": 0.10},
        ],
        current_positions={"DOGE-USDT": 1000.0, "BTC-USDT": 20.0},
        current_prices={"DOGE-USDT": 0.1, "BTC-USDT": 100.0, "ETH-USDT": 50.0},
        config=cfg,
        budget_snapshot={
            "strategy_equity": 10000.0,
            "cash_available_for_strategy": 7400.0,
        },
    )
    assert any(s["coin"] == "DOGE-USDT" and s["side"] == "sell" for s in passive_dropout), passive_dropout


def check_shared_target_selection() -> None:
    from quant_bench.runtime.adapters.base import BaseModelAdapter

    class DummyAdapter(BaseModelAdapter):
        def load_model(self, model_path: str):
            return None

        def predict(self, shared_data, config):
            return []

    cfg = SimpleNamespace(
        threshold=0.05,
        top_k=2,
        min_position_value_usdt=2.0,
        risk_degree=0.95,
        max_dropout=1,
        max_sell_per_cycle=1,
        max_position_count=2,
        trade_amount_usdt=1000.0,
        rebalance_tolerance_usdt=10.0,
        rebalance_tolerance_pct=0.001,
        capital_allocation_mode="full_investment",
        rebalance_policy="passive_topk_dropout",
        enable_short=False,
        enforce_trade_coins_whitelist=True,
        coins=["BTC-USDT", "ETH-USDT", "ADA-USDT", "DOGE-USDT", "XRP-USDT"],
    )
    scores = pd.Series(
        {
            "BTC-USDT": 0.30,
            "ETH-USDT": 0.20,
            "ADA-USDT": 0.10,
            "DOGE-USDT": -0.01,
            "XRP-USDT": 0.00,
        }
    )
    current_weights = {
        "DOGE-USDT": 0.01,
        "XRP-USDT": 0.05,
    }
    expected = {"BTC-USDT": 0.5, "ETH-USDT": 0.5}
    actual = DummyAdapter()._custom_strategy_target_weights(scores.to_dict(), current_weights, cfg)
    assert actual == expected, (actual, expected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline checks for packaged runtime resources.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=user_data_path("quant-bench", appauthor=False),
        help="Workspace whose existing run manifests and FinGPT CSVs should be checked.",
    )
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    check_configs()
    check_manifest_paths(workspace)
    check_fingpt_csvs(workspace)
    check_capital_allocation()
    check_shared_target_selection()
    print("framework smoke ok")


if __name__ == "__main__":
    main()
