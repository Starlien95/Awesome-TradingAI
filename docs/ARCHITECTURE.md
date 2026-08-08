# Architecture

## Scope

`quant-bench` separates benchmark contracts from Qlib, market-data, tracking, artifact, and runtime backends. The public package is offline by default and cannot place orders.

```text
CLI and Python API
        |
MethodRegistry + MethodRunner
        |
MethodRunSpec + capability/safety gate
        |
  Method adapters
   |      |       |       |        |
 native Qlib  FinGPT/  runtime  external
 runner       FinMem             ai_trade
        \        |       |        /
        durable Method Run record
```

## Stable boundaries

- `contracts`: versioned dataset, artifact, model capability, prediction, and run types.
- `config`: Pydantic validation, YAML loading, overrides, canonical serialization, and hashes.
- `data`: canonical OHLCV loading, validation, source manifests, and transformation lineage.
- `features` and `labels`: versioned semantic definitions and schema hashes.
- `models`: dependency-light built-ins used by offline validation.
- `registry`: complete built-in catalog plus PyPA entry-point discovery.
- `experiments`: offline lifecycle and failure recording.
- `evaluation`: signal metrics, costs, turnover, returns, and risk metrics.
- `artifacts`: atomic writes, visibility, media types, schemas, and checksums.
- `integrations.qlib`: lazy Qlib initialization, handlers, models, workflow execution, and 0.x command compatibility.
- `integrations.qlib.records`: frequency-safe benchmark preparation for Qlib intraday portfolio analysis.
- `tuning` and `reporting`: resumable canonical sweeps and manifest-driven summaries.
- `runtime`: optional execution core, model adapters, process catalog, promotion, and safety gates.
- `methods`: traditional ML, FinGPT news, FinMem/InvestorBench, and MacroHFT v1 method-specific logic.
- `methods.interface`: unified Method descriptors, Run Specs, capability and
  safety checks, durable Method Run records, and status.
- `methods.adapters`: locality-preserving adapters that delegate to native
  implementations without importing optional heavy backends during discovery.
- `trading`: versioned order, account, and execution contracts plus the common
  service used by paper, exchange-demo, and live backends.

Qlib objects, MLflow clients, CCXT exchanges, OKX clients, and Streamlit state are backend details. They are not stable public contract types.

## Lifecycle

1. Load and validate the complete `ExperimentConfig`.
2. Freeze `resolved_config.yaml` and its SHA-256.
3. Load and validate canonical market data.
4. Write the `DatasetManifest` and validation report.
5. Construct features and labels, then apply protocol splits.
6. Fit a selected plugin and generate canonical predictions.
7. Evaluate signal and portfolio metrics with explicit costs.
8. Write artifacts atomically and generate `checksums.sha256`.
9. Complete the `RunManifest` or retain a failed manifest with error context.

## Backend isolation

Optional imports occur only inside an explicit backend boundary. Importing `quant_bench` does not import Qlib, PyTorch, CCXT, Streamlit, OKX, or transformers. `QlibSession` initializes Qlib only when entered.

## Extension model

Third-party distributions register entry points under:

- `quant_bench.models`
- `quant_bench.features`
- `quant_bench.data_sources`
- `quant_bench.evaluators`

Plugins declare capabilities before execution. Unknown plugins and missing dependencies produce explicit errors without silent fallback.

## Runtime boundary

Research, local-paper, exchange-paper, and live modes are distinct capabilities. The canonical experiment runner accepts only offline research. FinGPT provides a local dry-run and paper broker. FinMem separates service access, account reads, and order writes, and writes all mutable config and state to a user workspace. Traditional ML and MacroHFT processes expose OKX demo and live modes through `runtime start`; all order paths require exported local configuration, credentials, a mode match, and an exact confirmation token. Account liquidation and production-ledger repair programs are not distributed with the public repository.

Strategy code emits a versioned `OrderRequest`; it does not own credentials or
exchange clients. `TradingService` captures account state before and after the
attempt and normalizes every backend response into `ExecutionReport`. Existing
brokers are connected through explicit adapters so their restart state and raw
audit artifacts remain compatible. See [Unified trading interface](how-to/unified-trading-interface.md).

## Unified Method seam

`MethodRunner` is the preferred public seam. A caller selects a Method and an
Execution Mode in a `MethodRunSpec`; the runner validates the capability
declaration before invoking an adapter. Backtest, local-paper, exchange-paper, and live
are not interchangeable aliases. Mode-specific frequencies remain explicit,
including FinAgent's daily historical engine and 4H runtime.

Order execution requires a separate network flag, order flag, and exact
exchange-paper or live confirmation token. External live Methods use dedicated
per-Method credential names. The runner writes a durable control-plane record
and preserves each adapter's native artifacts.
