# First Qlib run

This path uses Qlib 0.9.7 and a packaged workflow template. Training cost depends on the selected model. Start with LightGBM on a small symbol set.

## 1. Install the backend

```bash
python -m pip install -e ".[data-qlib,lgbm]"
quant-bench doctor
```

`data-qlib` installs Qlib, CCXT, and the local bin conversion dependencies. The download step below accesses the public OKX market-data endpoint. No account credential is required.

## 2. Prepare hourly data

```bash
mkdir -p ./qb-workspace/datasets/{raw,split,qlib}

quant-bench data pull \
  --timeframe 1h \
  --start 2021-01-01 \
  --symbols BTC-USDT,ETH-USDT \
  --output ./qb-workspace/datasets/raw/okx_1h.csv

quant-bench data validate ./qb-workspace/datasets/raw/okx_1h.csv

quant-bench data split \
  --timeframe 1h \
  --input ./qb-workspace/datasets/raw/okx_1h.csv \
  --output-dir ./qb-workspace/datasets/split/1h

quant-bench data qlib-dump \
  --timeframe 1h \
  --data-path ./qb-workspace/datasets/split/1h \
  --qlib-dir ./qb-workspace/datasets/qlib/1h
```

OHLCV cannot produce trade-level VWAP. The converter writes `typical_price = (high + low + close) / 3`, mirrors it into the compatibility `vwap` field, and records `vwap_method: typical_price_proxy` in split CSV files.

For an existing licensed dataset, skip `data pull`. Supply a canonical CSV with `timestamp,symbol,open,high,low,close,volume`, validate it, and continue at `data split`.

## 3. Select a workflow

```bash
quant-bench workflows list --feature-set 158 --model lgb --frequency 1h
```

Inspect the returned path:

```bash
quant-bench qlib validate <workflow-path>
```

The full catalog has 202 templates across Alpha158-style crypto features and the six-field CryptoOHLCV handler. Use `quant-bench models show <model_id>` to inspect a model's dependencies, accepted dataset kinds, and default constructor parameters.

## 4. Run it

```bash
quant-bench qlib run \
  --feature-set 158 \
  --model lgb \
  --frequency 1h \
  --provider-uri ./qb-workspace/datasets/qlib/1h \
  --experiment-name lgb-crypto-1h \
  --workspace ./qb-workspace
```

The command writes an absolute `provider_uri` into `runs/<run_id>/resolved_workflow.yaml`, configures MLflow under `qb-workspace/mlruns`, and records a run manifest even when Qlib raises an error.

Intraday workflows use `CryptoPortAnaRecord` to calculate the BTC benchmark at
the workflow frequency before Qlib initializes its account. A provider built
from hourly bars therefore does not need a separate daily or one-minute store
for portfolio analysis.

## 5. Understand the defaults

Every packaged workflow has been normalized and audited:

- market `all`, benchmark `BTC-USDT`;
- train 2021 through 2023, validation 2024, test 2025;
- one-bar forward return `Ref($close, -1) / $close - 1`;
- seed 42 where the model exposes a seed;
- `topk: 5`, `n_drop: 1`, `min_score: 0.0`, `risk_degree: 0.95`;
- account 100000 and open/close cost 0.001;
- no developer-machine absolute paths.

These values are starting points. Match the periods to the actual data and preserve a purged boundary when the label horizon or feature lookback requires it.

## Saved-artifact commands

The migrated command implementations remain available for experiments that depend on their artifact naming and PyTorch extraction behavior:

```bash
quant-bench qlib train --workspace ./qb-workspace -- --freq 1h --model lgb --feature-set 158
quant-bench qlib backtest --workspace ./qb-workspace -- --freq 1h --model lgb --feature-set 158
quant-bench qlib tune --workspace ./qb-workspace -- --freq 1h --model lgb --feature-set 158 --sweep-param learning_rate=0.5
```

Arguments after `--` belong to the migrated command. Run the subcommand with `-- --help` to view its complete option list.
