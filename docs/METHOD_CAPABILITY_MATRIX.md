# Method and tooling capability matrix

This page is the inventory for the first public package. It distinguishes
reusable library interfaces from optional external systems and records where a
user can validate or debug each method.

| Area | Public capabilities | Analysis and diagnostics | External boundary |
| --- | --- | --- | --- |
| Unified Method interface | one capability matrix and `check/run/status` seam for canonical, Qlib, FinGPT, FinMem, packaged runtime and ai_trade Methods across backtest, local-paper, exchange-paper and live modes | interface-level safety, alias, mode-specific frequency, durable-record and command-redaction tests | adapters preserve native implementations and artifacts; no mode is inferred or silently downgraded |
| Core benchmark | Typed experiment config, canonical OHLCV, feature/label contracts, NumPy baseline, costs and risk metrics, atomic artifacts, checksums and manifests | offline quickstart, input validation, artifact verification, run comparison | none for packaged fixture |
| Qlib traditional ML | 202 normalized Alpha158/Alpha360 workflows, full model catalog, generic workflow run, train, backtest, tune, recompute, visualization and model promotion | model/workflow list, config validation, normalized workflow audit, prediction consistency checks | Qlib and model extras; user market data |
| Traditional-ML runtime | timeframe configs, model adapters, threshold/top-k/dropout targets, local-paper/exchange-paper/live process catalog and dashboard CSV contract | runtime readiness check, dry-run, framework smoke, ledger and metrics repair tools | OKX public/account/order APIs only when explicitly enabled |
| MacroHFT v1 | frequency-aware TorchScript adapter, six subagent bundle contract, runtime configs and trusted-local converter | bundle/schema validation and deterministic adapter tests | user-supplied authorized upstream source, data and checkpoints |
| FinGPT news | offline ETH or multi-symbol long/flat research, validation-only tuning, asset/market aggregation, four signal variants, global and guarded per-symbol thresholds, classification and risk evaluation, paper scheduler | synthetic research fixture, input/data-quality report, full trial table, chosen config, confusion matrix, trade list, checksums, dry-run and runtime event logs | optional local model/LoRA, user-authorized news provider, explicit OKX capability gates |
| FinMem and InvestorBench | user-owned workspace, local-data checksum verification, phase runner, paper ledger, symbol-driven live cycle/scheduler and dashboard | public-repository audit, data verifier, dry-run, plot helpers, recovery manifest | local licensed data, model endpoint and separately gated account/order access |
| Reporting | manifest-driven summaries, canonical metrics/signals/trades/volume, local Streamlit dashboard | empty/missing/legacy-schema tests and runtime checks | public showcase requires a separately reviewed sanitized snapshot |

## Verification levels

- Unit and integration tests cover offline contracts, method adapters, safety
  gates, malformed inputs, causal FinGPT lag, tuning separation, artifacts,
  Qlib workflow parsing, TorchScript loading, FinMem data handling, and
  dashboard edge cases.
- `ruff` and strict `mypy` cover the public package. The retained Guardrails
  provenance snapshot is excluded from project type checking and from the
  wheel.
- Wheel verification installs the built artifact in a clean environment and
  runs offline quickstart, FinGPT init/validate/backtest/dry-run, workflow audit,
  and release-boundary checks.
- GitHub Actions repeats dependency-light tests on Python 3.10, 3.11, and 3.12,
  then runs independent quality, wheel, and Qlib 0.9.7 contract jobs. The CI
  workflow has read-only repository permissions and no package-publish step.
- Real model downloads, licensed local model weights, private news APIs, account
  reads, demo orders, and live orders are external acceptance tests. They are
  never executed by the default test suite.

## Compatibility rule

Legacy functionality is migrated into a typed command or recorded as an
operator-only workflow. A capability that cannot be published because it
contains private data, credentials, third-party weights, or license-restricted
source remains recoverable from its documented local baseline and is excluded
from the wheel.
