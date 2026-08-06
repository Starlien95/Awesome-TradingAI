"""Run a MacroHFT v1 one-shot prediction without exchange access."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant_bench.runtime.adapters.macrohft_v1_adapter import MacroHFTV1Adapter
from quant_bench.runtime.core.config import ConfigManager


def synthetic_kline(rows: int = 420, frequency: str = "5min") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0001, 0.004, rows)
    close = 3_000.0 * np.cumprod(1.0 + returns)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.001, 0.0003, rows))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(10.0, 0.2, rows)
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


def _resolve_local_path(value: str, config_path: Path, override: Path | None) -> Path:
    candidate = override.expanduser() if override else Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="MacroHFT v1 offline prediction smoke test.")
    parser.add_argument("--timeframe", choices=["5m", "15m", "1h", "4h"], default="5m")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--config-dir", type=Path, default=Path.cwd())
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--subagents-dir", type=Path)
    parser.add_argument("--rows", type=int, default=420)
    args = parser.parse_args()

    config_path = (args.config or (args.config_dir / "config_macrohft_v1.yaml")).expanduser().resolve()
    cfg = ConfigManager(str(config_path))
    adapter = MacroHFTV1Adapter()
    model_file = _resolve_local_path(cfg.model_path, config_path, args.model_path)
    adapter.load_model(str(model_file))
    configured_subagents = cfg.get("model.subagents_dir", "")
    if args.subagents_dir or configured_subagents:
        subagents_dir = _resolve_local_path(
            str(configured_subagents), config_path, args.subagents_dir
        )
        adapter.set_subagents_dir(str(subagents_dir))

    pandas_frequency = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}[args.timeframe]
    frame = synthetic_kline(max(args.rows, int(cfg.data_limit) + 20), pandas_frequency)
    symbols = list(cfg.coins[:1]) or ["ETH-USDT"]
    symbol = symbols[0]
    predictions = adapter.predict({symbol: frame}, cfg)
    capital = float(cfg.initial_capital_usdt or 10_000.0)
    signals = adapter.generate_signals(
        predictions,
        current_positions={},
        current_prices={symbol: float(frame["close"].iloc[-1])},
        config=cfg,
        budget_snapshot={"strategy_equity": capital, "cash_available_for_strategy": capital},
    )
    print(
        {
            "config": str(config_path),
            "artifact": str(model_file),
            "predictions": predictions,
            "signals": signals,
        }
    )


if __name__ == "__main__":
    main()
