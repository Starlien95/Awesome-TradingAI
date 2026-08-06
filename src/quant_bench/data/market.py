"""Canonical OHLCV schema and deterministic dataset manifests."""

from __future__ import annotations

import hashlib
from importlib import resources
from pathlib import Path
from typing import Any, cast

import pandas as pd

from quant_bench.config.models import ExperimentConfig
from quant_bench.contracts.models import DataLicenseInfo, DatasetManifest, DataSourceInfo, DataTimeInfo

REQUIRED_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")
NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")


class MarketDataValidationError(ValueError):
    """Raised when canonical market data constraints are violated."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def resolve_data_path(path: str) -> Path:
    if path.startswith("fixture://"):
        fixture_name = path.removeprefix("fixture://")
        resource = resources.files("quant_bench").joinpath("resources").joinpath("fixtures").joinpath(fixture_name)
        if not resource.is_file():
            raise FileNotFoundError(f"built-in fixture not found: {fixture_name}")
        return Path(str(resource))
    return Path(path).expanduser().resolve()


def load_market_data(path: str) -> tuple[pd.DataFrame, Path]:
    resolved = resolve_data_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"market data file not found: {resolved}")
    frame = pd.read_csv(resolved)
    return validate_market_data(frame), resolved


def validate_market_data(frame: pd.DataFrame) -> pd.DataFrame:
    errors: list[str] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise MarketDataValidationError([f"missing required columns: {', '.join(missing)}"])

    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    if result["timestamp"].isna().any():
        errors.append("timestamp contains invalid or timezone-free values")

    result["symbol"] = result["symbol"].astype(str).str.strip().str.upper()
    if (result["symbol"] == "").any():
        errors.append("symbol contains empty values")

    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any():
            errors.append(f"{column} contains non-numeric or missing values")

    if result.duplicated(["timestamp", "symbol"]).any():
        errors.append("duplicate timestamp and symbol rows found")
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        errors.append("OHLC prices must be positive")
    if (result["volume"] < 0).any():
        errors.append("volume must be non-negative")
    if (result["high"] < result[["open", "close", "low"]].max(axis=1)).any():
        errors.append("high must be greater than or equal to open, close, and low")
    if (result["low"] > result[["open", "close", "high"]].min(axis=1)).any():
        errors.append("low must be less than or equal to open, close, and high")

    for symbol, group in result.groupby("symbol", sort=False):
        if not group["timestamp"].is_monotonic_increasing:
            errors.append(f"timestamps are not increasing for symbol {symbol}")

    if errors:
        raise MarketDataValidationError(errors)

    if "source" not in result.columns:
        result["source"] = "unknown"
    if "frequency" not in result.columns:
        result["frequency"] = "unknown"
    result["typical_price"] = (result["high"] + result["low"] + result["close"]) / 3.0
    return cast(
        pd.DataFrame,
        result.sort_values(["timestamp", "symbol"]).reset_index(drop=True),
    )


def build_dataset_manifest(
    config: ExperimentConfig, frame: pd.DataFrame, source_path: Path
) -> DatasetManifest:
    content_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source_name = str(frame["source"].iloc[0]) if not frame.empty else config.data.source
    return DatasetManifest(
        dataset_id=config.data.dataset_id,
        content_sha256=content_sha256,
        source=DataSourceInfo(plugin_id=source_name, source_version="fixture-v1"),
        license=DataLicenseInfo(redistribution=config.data.license),
        time=DataTimeInfo(
            frequency=config.frequency.id,
            start=frame["timestamp"].min().to_pydatetime(),
            end=frame["timestamp"].max().to_pydatetime(),
        ),
        universe_id=config.universe.universe_id,
        symbols=sorted(frame["symbol"].unique().tolist()),
        rows=len(frame),
        validation_report="validation.json",
        transformations=["canonical_ohlcv_v1", "typical_price_v1"],
    )


def validation_report(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "status": "passed",
        "rows": len(frame),
        "symbols": sorted(frame["symbol"].unique().tolist()),
        "start": frame["timestamp"].min().isoformat(),
        "end": frame["timestamp"].max().isoformat(),
        "duplicate_rows": int(frame.duplicated(["timestamp", "symbol"]).sum()),
        "missing_values": {column: int(value) for column, value in frame.isna().sum().items()},
    }
