# Canonical market data contract

## Required columns

| Column | Type | Constraint |
| --- | --- | --- |
| `timestamp` | timezone-aware datetime | UTC, increasing per symbol |
| `symbol` | string | canonical, non-empty |
| `open` | float | positive |
| `high` | float | at least open, close, and low |
| `low` | float | at most open, close, and high |
| `close` | float | positive |
| `volume` | float | non-negative with documented units |

`source` and `frequency` are added as `unknown` when absent. Published datasets must provide accurate values.

## Validation

The loader rejects duplicate timestamp and symbol rows, invalid timestamps, non-numeric values, non-positive prices, negative volume, and invalid OHLC ranges. Every accepted dataset receives a source-byte SHA-256 and `DatasetManifest`.

## Typical price and VWAP

`typical_price` is `(high + low + close) / 3`. It is not true VWAP. The legacy Qlib exporter mirrors this value into `vwap` only for compatibility and records `vwap_method=typical_price_proxy`.

## Publication

Git may contain only small synthetic or license-cleared fixtures. Larger datasets use an external downloader or versioned data store. A data card must state source, terms, redistribution status, coverage, missing intervals, symbol mapping, and checksum.
