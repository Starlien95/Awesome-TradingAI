"""Causal forward-return label construction."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd


def forward_return(market: pd.DataFrame, horizon_bars: int, scale: float = 1.0) -> pd.Series:
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    data = market.sort_values(["symbol", "timestamp"]).copy()
    future_close = data.groupby("symbol", sort=False)["close"].shift(-horizon_bars)
    labels = (future_close / data["close"] - 1.0) * scale
    labels.index = pd.MultiIndex.from_frame(data[["timestamp", "symbol"]])
    labels.name = "label"
    return cast("pd.Series[Any]", labels.sort_index())
