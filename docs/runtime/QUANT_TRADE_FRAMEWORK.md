# Runtime operations

Use the supported CLI described in [../how-to/runtime.md](../how-to/runtime.md).

## Prepare

```bash
quant-bench runtime list
quant-bench runtime init traditional_4h \
  --output-dir ./qb-workspace/local-config/traditional_4h

quant-bench artifacts promote ./training-output/xgboost.pkl \
  --model XGBoost_4h \
  --frequency 4h \
  --workspace ./qb-workspace \
  --config ./qb-workspace/local-config/traditional_4h/config_xgboost.yaml

quant-bench runtime check traditional_4h \
  --config-dir ./qb-workspace/local-config/traditional_4h
```

`artifacts promote` does not deserialize the source. It copies bytes atomically, records a checksum and promotion manifest, and updates only the selected local config.

## Credentials

Keep credentials in environment variables. Demo uses `OKX_API_KEY_SIMU`, `OKX_SECRET_KEY_SIMU`, and `OKX_PASSPHRASE`. Live uses `OKX_API_KEY`, `OKX_SECRET_KEY`, and `OKX_PASSPHRASE`.

## Start

```bash
quant-bench runtime start traditional_4h \
  --config-dir ./qb-workspace/local-config/traditional_4h \
  --workspace ./qb-workspace \
  --mode demo \
  --confirm DEMO_ORDERS
```

This command can place demo orders. Replace demo with live only after setting `api.is_simulated: false`, reviewing all local configs, and providing `LIVE_ORDERS`.

## Dashboard

```bash
quant-bench dashboard --workspace ./qb-workspace
```

The dashboard reads local files and performs no exchange or model operation.
