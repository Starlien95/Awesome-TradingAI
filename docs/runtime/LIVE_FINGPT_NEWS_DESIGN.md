# FinGPT news runtime design

## Responsibilities

`quant_bench.methods.fingpt_news` has two deliberately separate paths:

- `research` validates historical price and sentiment inputs, selects parameters
  on a validation window, evaluates a frozen strategy on a disjoint test window,
  and writes canonical benchmark artifacts;
- `runtime` collects current news, performs optional local model inference, keeps
  restart-safe state, and sends targets to either a local paper broker or an
  explicitly enabled OKX executor.

Both paths use the same daily long/flat semantics and per-symbol parameters.
Research never imports an exchange client or reads credentials.

## Signal and capital semantics

Each symbol owns an independent capital bucket. Positive sentiment targets a
fully invested long bucket, negative sentiment targets cash, and neutral
sentiment retains the previous target. Capital and PnL do not move between
symbols.

The research engine supports `asset_only`, `market_only`, `hybrid_fixed`, and
`hybrid_adaptive` signals. Runtime parameters are read from a symbol-indexed CSV
with the same aggregation, alpha, threshold, and conflict-policy fields. The
public template uses uniform general starting values. Historical tuned values
remain an opt-in legacy recipe and carry no performance claim.

## Causal timing

Historical windows are half-open and validation cannot overlap test. Backtests
enforce `signal_lag_days >= 1`. Parameter selection reads validation data only;
the frozen configuration is applied once to test data. Selection metadata
records `test_metrics_used_for_selection: false`.

The scheduled runner normally scores the previous complete calendar day's
news. Collection and candidate preparation can run without loading a model.
Inference can run in an isolated process; worker exit is the GPU-memory release
boundary. A trade date is completed only after every required symbol has an
inference row. `processed_trade_dates` makes restarts idempotent.

## Capability gates

The packaged YAML starts with:

```yaml
runtime:
  dry_run: true
trade:
  mode: paper_spot
  reconcile_account_positions: false
api:
  is_simulated: true
```

Available modes have cumulative requirements:

| Mode | Network | Account client | Orders | Required confirmation |
| --- | --- | --- | --- | --- |
| fixture dry-run | no | no | no | none |
| network paper | `--allow-network` | no | no | none |
| OKX demo orders | `--allow-network` | yes | `--execute-orders` | `DEMO_ORDERS` |
| OKX live orders | `--allow-network` | yes | `--execute-orders` | `LIVE_ORDERS` |

Dry-run rejects order capability. Network paper uses unauthenticated OKX public
ticker reads and never creates an account executor. Inline OKX credentials are
rejected. The order modes load credentials only through the environment
variable names declared in the reviewed YAML.

Account-position reconciliation is separately opt-in. It remains disabled in
the public template because it performs account reads and can change the
strategy-owned quantity baseline.

## News providers

`ccdata` uses a configured environment variable for its API key. `rss`,
`rss_google`, and `google_rss` use explicit feed URLs, bounded retries, a
request timeout, local history, and record limits. Provider modules perform no
network access at import time. Raw articles and prompt/model caches remain
private runtime data.

## Offline commands

```bash
quant-bench fingpt init --workspace ./qb-workspace
quant-bench fingpt validate \
  --config ./qb-workspace/methods/fingpt_news/research/config.yaml
quant-bench fingpt backtest \
  --config ./qb-workspace/methods/fingpt_news/research/config.yaml \
  --output ./qb-workspace/runs/fingpt-research
quant-bench fingpt paper-dry-run --workspace ./qb-workspace
```

These commands use synthetic fixtures by default and require no network,
credential, model weight, transformers installation, or GPU.

## Artifacts and publication boundary

Research writes resolved configuration, input hashes, data-quality reports,
validation trials, chosen parameters, classification diagnostics when labels
exist, daily signals/backtests, trade lists, risk summaries, canonical
metrics/signals/trades/volume CSVs, checksums, and `run_manifest.json`.

Runtime always creates canonical CSV headers, including on no-trade days, and
stores mutable news, inference, ledger, and scheduler state below its run
directory. Raw news, prompts, model output, state, logs, account snapshots,
orders, fills, credentials, weights, and adapters are excluded from public
source and snapshot distributions.

See `docs/methods/fingpt-news.md`, `docs/runtime/CSV_SCHEMA.md`, and
`docs/runtime/NETWORK_ACCESS.md` for the complete user and data contracts.
