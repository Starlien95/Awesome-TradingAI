# FinGPT news sentiment

This package separates three workflows that used to live in independent
experiment scripts:

1. offline sentiment research and validation-only parameter selection;
2. restart-safe paper simulation with independent per-symbol capital buckets;
3. explicitly gated OKX demo/live execution.

Start with the packaged synthetic research workflow:

```bash
quant-bench fingpt init --workspace ./qb-workspace

CONFIG=./qb-workspace/methods/fingpt_news/research/config.yaml
quant-bench fingpt validate --config "$CONFIG"
quant-bench fingpt backtest \
  --config "$CONFIG" \
  --output ./qb-workspace/runs/fingpt-smoke
```

The backtest consumes model sentiment outputs. It does not require a specific
base model or adapter. Supported score inputs are `article_score`,
`prob_positive` plus `prob_negative`, or numeric `model_score`.

The engine supports `asset_only`, `market_only`, `hybrid_fixed`, and
`hybrid_adaptive`; aggregation top-N, minimum score and score-power grids;
global and guarded per-symbol thresholds; one-day-or-longer execution lag;
fees and slippage; classification diagnostics when `true_label` is present;
and canonical metrics, signals, trades, volume, manifest, checksums, and data
quality artifacts.

Run the existing end-to-end paper fixture without network access:

```bash
quant-bench fingpt paper-dry-run --workspace ./qb-workspace
```

Production model inference additionally requires `quant-bench[news]`. RSS
collection requires `quant-bench[fingpt-live]`. Paper mode requires
`--allow-network` for news and public prices and cannot read an account.
OKX order modes require environment credentials, `--execute-orders`, and the
exact `DEMO_ORDERS` or `LIVE_ORDERS` confirmation token. Inline credentials are
rejected.

See the [complete guide](../../../../docs/methods/fingpt-news.md) and the
[Chinese guide](../../../../docs/methods/fingpt-news.zh-CN.md).
