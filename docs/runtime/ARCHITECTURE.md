# Runtime architecture

The runtime is an optional backend inside `quant_bench.runtime`. Importing the base package does not create an exchange client.

## Packages

```text
quant_bench.runtime
├── core/          config, data, execution, state, metrics, audit, manifests
├── adapters/      trained artifacts to canonical scores and target signals
├── entrypoints/   explicit process wrappers
├── processes.py   process catalog, config export, readiness, safety gates
└── promotion.py   opaque artifact copy and provenance

quant_bench.methods
├── traditional_ml/
├── reinforcement_learning/macrohft_v1/
├── fingpt_news/
└── finmem/
```

External long-running systems use a separate control-plane boundary:

```text
quant_bench.integrations.ai_trade
└── live_stack.py  catalog, offline readiness, status, and guarded delegation
```

The integration imports no `ai_trade` Python modules. The external checkout
owns execution and generated state; `quant_bench` reads only its deploy
contract and normalized event stream.

Packaged configuration lives under `quant_bench.resources.runtime`. Users export editable copies with `quant-bench runtime init`; runtime code never edits package resources.

FinMem uses the same rule through `quant-bench finmem init`. The package keeps
the InvestorBench engine and safe templates, while generated configs,
checkpoints, Qdrant state, live data, account snapshots, and logs stay in the
workspace.

## Process groups

| Process | Strategies |
| --- | --- |
| `traditional_15m` | DoubleEnsemble, TabNet |
| `traditional_1h` | MLP, TCN |
| `traditional_4h` | XGBoost, GATS, LSTM, TRA |
| `macrohft_5m` | MacroHFT v1 |
| `macrohft_15m` | MacroHFT v1 |
| `macrohft_1h` | MacroHFT v1 |
| `macrohft_4h` | MacroHFT v1 |

Models within one process share market-data retrieval and maintain separate executor, metrics, signals, volume, and audit state. Different processes do not share strategy state.

## Discovery and output

New runs write `run_manifest.json` under the explicit workspace. The dashboard discovers manifests first and retains read compatibility for historical timeframe log layouts.

Every strategy emits metrics, signals, trades, and volume CSV contracts. Order-capable brokers additionally emit orders, fills, failed orders, account snapshots, rebalance plans, and JSONL order events. See [CSV_SCHEMA.md](CSV_SCHEMA.md).

## Reliability primitives

- `atomic_io`: temporary file plus atomic replacement for CSV and JSON.
- `JsonStateStore`: restart-safe local state.
- `TradeExecutionResult`: structured order, fill, fee, slippage, and error state.
- `run_manifest`: method, strategy, frequency, paths, and mode for discovery.
- `audit_log`: append and deduplicate execution records.

## Capability gates

`runtime check` performs local checks only. `runtime start` accepts `demo` and `live`; it verifies the exact confirmation token, mode/config agreement, credentials, and every model artifact before starting the process. High-risk account liquidation and production repair programs are not distributed with the public repository.
