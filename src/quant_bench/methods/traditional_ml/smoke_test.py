"""Run one-shot predictions against local runtime artifacts without exchange access."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from quant_bench.runtime.adapters.base import BaseModelAdapter
from quant_bench.runtime.core.config import ConfigManager

MODEL_SPECS: dict[str, tuple[str, str]] = {
    "mlp": ("quant_bench.runtime.adapters.mlp_adapter:MLPAdapter", "config_mlp.yaml"),
    "xgboost": (
        "quant_bench.runtime.adapters.xgboost_adapter:XGBoostAdapter",
        "config_xgboost.yaml",
    ),
    "tabnet": (
        "quant_bench.runtime.adapters.tabnet_adapter:TabNetAdapter",
        "config_tabnet.yaml",
    ),
    "tcn": ("quant_bench.runtime.adapters.tcn_adapter:TCNAdapter", "config_tcn.yaml"),
    "lgbm": ("quant_bench.runtime.adapters.lgbm_adapter:LGBMAdapter", "config_lgbm.yaml"),
    "lstm": ("quant_bench.runtime.adapters.lstm_adapter:LSTMAdapter", "config_lstm.yaml"),
    "gats": ("quant_bench.runtime.adapters.gats_adapter:GATSAdapter", "config_gats.yaml"),
    "tra": ("quant_bench.runtime.adapters.tra_adapter:TRAAdapter", "config_tra.yaml"),
    "double-ensemble": (
        "quant_bench.runtime.adapters.double_ensemble_adapter:DoubleEnsembleAdapter",
        "config_double_ensemble.yaml",
    ),
}


def synthetic_kline(rows: int = 420, frequency: str = "1h") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0002, 0.003, rows)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.001, 0.0002, rows))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(8.0, 0.25, rows)
    timestamps = pd.date_range("2026-01-01", periods=rows, freq=frequency)
    return pd.DataFrame(
        {
            "ts": (timestamps.view("int64") // 1_000_000).astype(float),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=timestamps,
    )


def _load_adapter(import_path: str) -> BaseModelAdapter:
    module_name, class_name = import_path.split(":", 1)
    adapter_type = getattr(importlib.import_module(module_name), class_name)
    return adapter_type()


def _resolve_local_path(value: str, config_path: Path, override: Path | None) -> Path:
    candidate = override.expanduser() if override else Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve()


def run_model(
    model_name: str,
    config_path: Path,
    *,
    model_path: Path | None = None,
    rows: int = 420,
    frequency: str = "1h",
) -> None:
    import_path, _ = MODEL_SPECS[model_name]
    config_path = config_path.expanduser().resolve()
    cfg = ConfigManager(str(config_path))
    artifact = _resolve_local_path(cfg.model_path, config_path, model_path)
    adapter = _load_adapter(import_path)
    adapter.load_model(str(artifact))

    frame = synthetic_kline(max(rows, int(cfg.data_limit) + 30), frequency)
    symbols = list(cfg.coins[:2]) or ["BTC-USDT", "ETH-USDT"]
    shared_data = {symbol: frame.copy() for symbol in symbols}
    current_prices = {symbol: float(frame["close"].iloc[-1]) for symbol in symbols}
    predictions = adapter.predict(shared_data, cfg)
    capital = float(cfg.initial_capital_usdt or 8_000.0)
    signals = adapter.generate_signals(
        predictions,
        current_positions={},
        current_prices=current_prices,
        config=cfg,
        budget_snapshot={"strategy_equity": capital, "cash_available_for_strategy": capital},
    )
    print(
        {
            "model": model_name,
            "config": str(config_path),
            "artifact": str(artifact),
            "predictions": predictions,
            "signals": signals,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline prediction smoke test for packaged traditional-ML runtime adapters."
    )
    parser.add_argument("--model", choices=[*MODEL_SPECS, "all"], default="all")
    parser.add_argument(
        "--config",
        type=Path,
        help="Config for a single model. For --model all, place conventional config names in --config-dir.",
    )
    parser.add_argument("--config-dir", type=Path, default=Path.cwd())
    parser.add_argument("--model-path", type=Path, help="Artifact override for a single model.")
    parser.add_argument("--rows", type=int, default=420)
    parser.add_argument("--frequency", default="1h")
    args = parser.parse_args()

    if args.model == "all" and (args.config or args.model_path):
        parser.error("--config and --model-path require a single --model")
    selected = list(MODEL_SPECS) if args.model == "all" else [args.model]
    executed = 0
    for model_name in selected:
        default_name = MODEL_SPECS[model_name][1]
        config_path = args.config or (args.config_dir / default_name)
        if args.model == "all" and not config_path.is_file():
            print({"model": model_name, "skipped": f"config not found: {config_path}"})
            continue
        run_model(
            model_name,
            config_path,
            model_path=args.model_path,
            rows=args.rows,
            frequency=args.frequency,
        )
        executed += 1
    if executed == 0:
        parser.error("no runnable configs found")


if __name__ == "__main__":
    main()
