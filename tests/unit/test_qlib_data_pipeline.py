from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import quant_bench.data.qlib_pipeline as qlib_pipeline
from quant_bench.data.market import validate_market_data
from quant_bench.data.qlib_pipeline import (
    fill_vwap,
    get_qlib_freq,
    parse_symbols,
    run_dump,
    run_split,
    save_raw_csv,
)


def test_data_pipeline_helpers_use_public_defaults() -> None:
    assert get_qlib_freq("1h") == "60min"
    assert parse_symbols(" btc-usdt, ETH-USDT ") == ["BTC-USDT", "ETH-USDT"]


def test_split_adds_documented_typical_price_proxy(tmp_path: Path) -> None:
    source = tmp_path / "market.csv"
    frame = pd.DataFrame(
        {
            "date": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "symbol": ["BTC-USDT", "ETH-USDT"],
            "open": [90.0, 45.0],
            "high": [110.0, 55.0],
            "low": [80.0, 40.0],
            "close": [100.0, 50.0],
            "volume": [1.0, 2.0],
        }
    )
    frame.to_csv(source, index=False)
    output = tmp_path / "split"
    run_split("1h", str(source), str(output))

    btc = pd.read_csv(output / "BTC-USDT.csv")
    assert btc.loc[0, "typical_price"] == pytest.approx((110.0 + 80.0 + 100.0) / 3.0)
    assert btc.loc[0, "vwap"] == btc.loc[0, "typical_price"]
    assert btc.loc[0, "vwap_method"] == "typical_price_proxy"


def test_fill_vwap_does_not_mutate_input() -> None:
    source = pd.DataFrame({"high": [3.0], "low": [1.0], "close": [2.0]})
    result = fill_vwap(source)
    assert "vwap" not in source
    assert result.loc[0, "vwap"] == 2.0


def test_saved_raw_csv_is_canonical_and_qlib_compatible(tmp_path: Path) -> None:
    target = tmp_path / "market.csv"
    source = pd.DataFrame(
        {
            "datetime": ["2025-01-01T00:00:00Z"],
            "instrument": ["BTC-USDT"],
            "open": [90.0],
            "high": [110.0],
            "low": [80.0],
            "close": [100.0],
            "volume": [1.0],
        }
    )
    save_raw_csv(source, str(target), "1h")

    saved = pd.read_csv(target)
    validated = validate_market_data(saved)
    assert {"timestamp", "date", "source", "frequency", "vwap", "vwap_method"}.issubset(
        saved.columns
    )
    assert validated.loc[0, "source"] == "okx_ccxt"
    assert validated.loc[0, "frequency"] == "1h"


def test_qlib_dump_rejects_source_csv_without_rows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(qlib_pipeline, "require_dump_dependencies", lambda: None)
    source = tmp_path / "split"
    source.mkdir()
    pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]).to_csv(
        source / "BTC-USDT.csv", index=False
    )

    with pytest.raises(ValueError, match="contain no data rows"):
        run_dump(
            mode="dump_all",
            timeframe="1h",
            data_path=str(source),
            qlib_dir=str(tmp_path / "qlib"),
            max_workers=1,
        )


@pytest.mark.parametrize(
    ("output_stage", "message"),
    [
        (0, "empty calendar"),
        (1, "empty instrument registry"),
        (2, "no non-empty feature bins"),
    ],
)
def test_qlib_dump_rejects_incomplete_dumper_output(
    tmp_path: Path,
    monkeypatch,
    output_stage: int,
    message: str,
) -> None:
    monkeypatch.setattr(qlib_pipeline, "require_dump_dependencies", lambda: None)
    source = tmp_path / "split"
    source.mkdir()
    pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z"],
            "symbol": ["BTC-USDT"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10.0],
        }
    ).to_csv(source / "BTC-USDT.csv", index=False)

    class IncompleteDumper:
        def __init__(self, **kwargs) -> None:
            self.output_root = Path(kwargs["qlib_dir"])
            self.frequency = kwargs["freq"]

        def dump(self) -> None:
            if output_stage >= 1:
                calendar_dir = self.output_root / "calendars"
                calendar_dir.mkdir(parents=True)
                (calendar_dir / f"{self.frequency}.txt").write_text(
                    "2025-01-01 00:00:00\n", encoding="utf-8"
                )
            if output_stage >= 2:
                instrument_dir = self.output_root / "instruments"
                instrument_dir.mkdir(parents=True)
                (instrument_dir / "all.txt").write_text(
                    "BTC-USDT\t2025-01-01\t2025-01-01\n", encoding="utf-8"
                )

    monkeypatch.setattr(qlib_pipeline, "DumpDataAll", IncompleteDumper)
    with pytest.raises(RuntimeError, match=message):
        run_dump(
            mode="dump_all",
            timeframe="1h",
            data_path=str(source),
            qlib_dir=str(tmp_path / "qlib"),
            max_workers=1,
        )


@pytest.mark.qlib
def test_qlib_dump_auto_detects_canonical_timestamp(
    tmp_path: Path,
) -> None:
    pytest.importorskip("qlib")
    pytest.importorskip("tqdm")
    source = tmp_path / "split"
    source.mkdir()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="1h", tz="UTC"),
            "symbol": ["BTC-USDT"] * 4,
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [10.0, 11.0, 12.0, 13.0],
        }
    )
    frame.to_csv(source / "BTC-USDT.csv", index=False)
    report = run_dump(
        mode="dump_all",
        timeframe="1h",
        data_path=str(source),
        qlib_dir=str(tmp_path / "qlib"),
        max_workers=1,
    )
    assert report["date_field_name"] == "timestamp"
    assert report["calendar_rows"] == 4
    assert report["instruments"] == 1
    assert report["feature_bins"] == 7
