"""Six small, causal OHLCV features for offline examples and baselines."""

from __future__ import annotations

import hashlib
import json
from typing import cast

import numpy as np
import pandas as pd


class CryptoOHLCV6V1:
    feature_set_id = "crypto_ohlcv6_v1"
    version = "1"
    columns = (
        "open_to_prev_close",
        "high_to_prev_close",
        "low_to_prev_close",
        "close_to_prev_close",
        "volume_to_prev_volume",
        "typical_price_to_prev_close",
    )

    @classmethod
    def schema_sha256(cls) -> str:
        payload = json.dumps(
            {"feature_set_id": cls.feature_set_id, "version": cls.version, "columns": cls.columns},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def transform(self, market: pd.DataFrame) -> pd.DataFrame:
        data = market.sort_values(["symbol", "timestamp"]).copy()
        grouped = data.groupby("symbol", sort=False)
        previous_close = grouped["close"].shift(1)
        previous_volume = grouped["volume"].shift(1)
        typical_price = (data["high"] + data["low"] + data["close"]) / 3.0

        features = pd.DataFrame(
            {
                "timestamp": data["timestamp"],
                "symbol": data["symbol"],
                "open_to_prev_close": data["open"] / previous_close - 1.0,
                "high_to_prev_close": data["high"] / previous_close - 1.0,
                "low_to_prev_close": data["low"] / previous_close - 1.0,
                "close_to_prev_close": data["close"] / previous_close - 1.0,
                "volume_to_prev_volume": data["volume"] / previous_volume.replace(0, np.nan) - 1.0,
                "typical_price_to_prev_close": typical_price / previous_close - 1.0,
            }
        )
        features = features.replace([np.inf, -np.inf], np.nan)
        return cast(
            pd.DataFrame,
            features.set_index(["timestamp", "symbol"]).sort_index(),
        )

    @staticmethod
    def qlib_config() -> tuple[list[str], list[str]]:
        fields = [
            "$open / Ref($close, 1) - 1",
            "$high / Ref($close, 1) - 1",
            "$low / Ref($close, 1) - 1",
            "$close / Ref($close, 1) - 1",
            "$volume / Ref($volume, 1) - 1",
            "(($high + $low + $close) / 3) / Ref($close, 1) - 1",
        ]
        return fields, list(CryptoOHLCV6V1.columns)
