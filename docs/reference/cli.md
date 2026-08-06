# CLI reference

Run `quant-bench <command> --help` for the authoritative options.

## Unified Methods

```text
methods list       complete Method x mode capability matrix
methods show       one Method descriptor, including mode-specific frequencies
methods check      local readiness and redacted command preview
methods run        unified backtest, paper, simulated, or live invocation
methods status     latest durable Method Run plus adapter-native status
```

`methods` is the preferred interface. Older method-specific commands remain
available for compatibility. See
[Unified Method interface](../how-to/unified-methods.md).

## Research

```text
quickstart        packaged offline benchmark
doctor            environment, workspace, dependency, and credential-presence checks
run               canonical offline experiment from a typed config
sweep             resumable Cartesian grid for canonical configs
compare           compatibility-aware completed-run inventory
report            Markdown summary for a workspace
```

## Models and plugins

```text
models list       model catalog, backend, availability, extra, dataset kinds
models show       full catalog entry
plugins list      installed quant_bench.models entry points
plugins scaffold  starter plugin source
```

## Configuration and schema

```text
config list       built-in canonical recipes
config render     resolved YAML
config validate   Pydantic validation
config init       create a model/feature recipe
schema export     public JSON schemas
```

## Data

```text
data validate     canonical OHLCV validation
data pull         public OKX OHLCV through CCXT; networked
data split        one CSV per symbol plus typical-price proxy
data qlib-dump    dump_all, dump_fix, or dump_update Qlib bin data
data pipeline     pull, split, and dump_all
```

## Qlib

```text
workflows list       filter 202 packaged templates
workflows audit      YAML, model, path, label, and default checks
workflows normalize  preview or apply normalization
qlib validate        validate one workflow structure
qlib run             run a path or select by feature/model/frequency
qlib train           migrated saved-artifact training flow
qlib backtest        migrated saved-model backtest
qlib tune            migrated workflow sweep
qlib auto-tune       migrated multi-model tuning
qlib recompute-stats prediction statistics
qlib visualize-sweep sweep plots
qlib verify-predictions Qlib/PyTorch parity check
```

The migrated commands accept their original arguments after `--`, for example:

```bash
quant-bench qlib train --workspace ./qb-workspace -- --freq 1h --model lgb --feature-set 158
```

## Artifacts and runtime

```text
artifacts inspect    typed run manifest
artifacts verify     checksum validation
artifacts promote    opaque copy into runtime model storage plus provenance
runtime list         available process groups
runtime init         export editable config templates
runtime check        local readiness check, no network
runtime fingpt-dry-run fixture news and local paper broker
runtime start        confirmed OKX demo or live process
runtime ai-trade     external four-agent 4H stack catalog and control plane
dashboard            read-only Streamlit app
```

The external stack has its own nested command surface:

```text
runtime ai-trade list    four child processes and their model/account profiles
runtime ai-trade check   checkout, simulated-mode, entrypoint, and credential-presence checks
runtime ai-trade status  PID state plus the latest normalized event; optionally one child
runtime ai-trade start   start the complete stack; requires SIMULATED_ORDERS
runtime ai-trade stop    stop the complete stack; requires STOP_LIVE_STACK
```

Pass `--repo /path/to/ai_trade` or set `AI_TRADE_ROOT`. List, check, and
status never load or print secret values. The integration requires an external
checkout and does not package its agent implementations or runtime data.

## FinGPT news sentiment

```text
fingpt init           export synthetic prices, sentiment, and editable research config
fingpt validate       validate input schemas, checksums, symbols, and disjoint windows
fingpt backtest       validation-only grid selection and frozen test evaluation
fingpt paper-dry-run  fixture-news paper cycle with static prices and no network
```

`fingpt backtest` writes global and guarded per-symbol configurations plus the
canonical dashboard CSV contract. Production paper/model execution remains in
`quant_bench.methods.fingpt_news.run_fingpt_news` and requires explicit network
and order capabilities as described in `methods/fingpt-news.md`.

## FinMem and InvestorBench

```text
finmem init           create a user-owned config and workspace; requires --data-dir
finmem doctor         optional dependencies, paths, and credential presence; offline
finmem data verify    local InvestorBench JSON contract and checksum verification
finmem run            warmup, checkpoint resume, test, and eval
finmem map-action     offline action-to-spot/SWAP mapping
finmem paper-step     deterministic long/flat/short paper ledger
finmem cycle          one symbol-configured daily data/decision cycle
finmem schedule       long-running daily scheduler
finmem dashboard      read-only multi-asset Streamlit dashboard
```

Research phases that use services and daily cycles require
`--allow-network`. Account reads additionally require `--query-account`.
Orders additionally require `--execute-orders` and the exact demo/live
confirmation token. See `methods/finmem.md` for complete examples.

Exit code 0 means the requested check or operation completed. Validation failures return 1. Compatibility or readiness warnings that require explicit user acceptance use exit code 2 where documented.
