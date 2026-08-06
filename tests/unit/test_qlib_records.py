from __future__ import annotations

import pandas as pd
import pytest


@pytest.mark.qlib
def test_crypto_portfolio_record_requests_benchmark_at_executor_frequency(monkeypatch) -> None:
    pytest.importorskip("qlib")
    from quant_bench.integrations.qlib import records

    index = pd.MultiIndex.from_product(
        [["BTC-USDT"], pd.date_range("2025-01-01", periods=3, freq="1h")],
        names=["instrument", "datetime"],
    )
    calls: list[str] = []

    def fake_features(*args, **kwargs):
        del args
        calls.append(kwargs["freq"])
        return pd.DataFrame({"return": [0.0, 0.01, -0.005]}, index=index)

    monkeypatch.setattr(records.D, "features", fake_features, raising=False)
    record = records.CryptoPortAnaRecord.__new__(records.CryptoPortAnaRecord)
    record.executor_config = {"kwargs": {"time_per_step": "60min"}}
    result = record._benchmark_returns("BTC-USDT", "2025-01-01", "2025-01-02")

    assert calls == ["60min"]
    assert result.index.name == "datetime"
    assert result.tolist() == [0.0, 0.01, -0.005]
