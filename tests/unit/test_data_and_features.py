from __future__ import annotations

import pandas as pd
import pytest

from quant_bench.data import MarketDataValidationError, load_market_data, validate_market_data
from quant_bench.features import CryptoAlpha52V1, CryptoOHLCV6V1
from quant_bench.labels import forward_return


def test_packaged_fixture_and_six_features() -> None:
    frame, _ = load_market_data("fixture://crypto_ohlcv_smoke.csv")
    assert len(frame) == 40
    assert str(frame["timestamp"].dtype) == "datetime64[ns, UTC]"
    features = CryptoOHLCV6V1().transform(frame)
    assert list(features.columns) == list(CryptoOHLCV6V1.columns)
    assert len(features) == 40
    assert features.groupby(level="symbol").head(1).isna().all(axis=None)
    labels = forward_return(frame, horizon_bars=1)
    assert labels.groupby(level="symbol").tail(1).isna().all()


def test_alpha52_metadata_matches_legacy_contract() -> None:
    expressions, names = CryptoAlpha52V1.qlib_config()
    assert len(expressions) == 52
    assert len(names) == 52
    assert names[0] == "feat_00"
    assert names[-1] == "feat_51"
    assert len(CryptoAlpha52V1.schema_sha256()) == 64


def test_ohlcv_validation_rejects_duplicate_and_invalid_high() -> None:
    row = {
        "timestamp": "2024-01-01T00:00:00Z",
        "symbol": "BTC-USDT",
        "open": 10,
        "high": 9,
        "low": 8,
        "close": 10,
        "volume": 1,
    }
    frame = pd.DataFrame([row, row])
    with pytest.raises(MarketDataValidationError) as error:
        validate_market_data(frame)
    assert "duplicate" in str(error.value)
    assert "high" in str(error.value)


def test_qlib_six_feature_uses_typical_price_name() -> None:
    expressions, names = CryptoOHLCV6V1.qlib_config()
    assert len(expressions) == 6
    assert names[-1] == "typical_price_to_prev_close"
    assert "($high + $low + $close) / 3" in expressions[-1]
