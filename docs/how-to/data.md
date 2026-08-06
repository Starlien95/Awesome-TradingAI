# Prepare market data

## Canonical CSV

The research loader requires:

```text
timestamp,symbol,open,high,low,close,volume
```

`timestamp` must parse as UTC, `(timestamp, symbol)` must be unique, prices must be positive, `high` and `low` must contain the other OHLC values, and volume must be nonnegative.

```bash
quant-bench data validate ./market.csv
```

The validator prints row count, symbols, time range, duplicate count, and per-column missing counts.

## Public OKX OHLCV

Install `data-ccxt` or `data-qlib`, then run:

```bash
quant-bench data pull \
  --timeframe 15m \
  --start 2022-01-01 \
  --symbols BTC-USDT,ETH-USDT,SOL-USDT \
  --output ./qb-workspace/datasets/raw/okx_15m.csv
```

This command explicitly accesses OKX through CCXT. It retries a bounded number of consecutive failures, uses a 20-second request timeout, honors CCXT rate limiting, and writes UTC timestamps.

Review exchange terms and dataset redistribution rights before publishing downloaded rows. Public benchmark releases should provide a collector and dataset card when redistribution is unavailable.

## Split files and Qlib bin data

```bash
quant-bench data split \
  --timeframe 15m \
  --input ./qb-workspace/datasets/raw/okx_15m.csv \
  --output-dir ./qb-workspace/datasets/split/15m

quant-bench data qlib-dump \
  --mode dump_all \
  --timeframe 15m \
  --data-path ./qb-workspace/datasets/split/15m \
  --qlib-dir ./qb-workspace/datasets/qlib/15m
```

`qlib-dump` auto-detects a common `date` or canonical `timestamp` column,
normalizes it to naive UTC for Qlib, and fails if calendar, instrument, or
feature output is empty. Use `--date-field-name` only for a reviewed custom
schema.

Use `dump_fix` to add instruments to an existing store and `dump_update` for an intentional update. Supply `--backup-dir` during updates. Generated Qlib bins and source data are ignored by Git.

## One-command pipeline

```bash
quant-bench data pipeline \
  --timeframe 1h \
  --start 2021-01-01 \
  --symbols BTC-USDT,ETH-USDT \
  --output ./qb-workspace/datasets/raw/okx_1h.csv \
  --split-dir ./qb-workspace/datasets/split/1h \
  --qlib-dir ./qb-workspace/datasets/qlib/1h
```

Use the separate commands in production automation so validation and data-review gates can run between stages.
