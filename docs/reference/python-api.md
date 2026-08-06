# Python API

`quant_bench` exposes a small public surface for offline experiments, method
adapters, model plugins, artifacts, data contracts and trading backends. Import
from the modules listed here. Other modules are implementation details unless a
method guide says otherwise.

## Stability levels

| Level | Meaning |
| --- | --- |
| Public | exported through `quant_bench`, `quant_bench.contracts`, `quant_bench.methods`, `quant_bench.artifacts` or `quant_bench.trading` |
| Experimental | documented method-specific integration that can change in a minor release |
| Internal | any other module, class or function |

Removing a public field, changing time or capital semantics, or renaming a
schema requires a major version. New optional fields can be introduced in a
minor version. Deprecations remain available for at least one minor release.

## Offline experiment

The root package exports `load_config`, `ExperimentConfig` and `Experiment`.

```python
from quant_bench import Experiment, load_config

config = load_config(
    "crypto_smoke_v1",
    overrides=["experiment.workspace=./qb-workspace"],
)
result = Experiment(config).run()

print(result.run_id)
print(result.run_dir)
print(result.metrics)
```

`Experiment.run()` is offline-only. It rejects configs that enable network
access and sends Qlib models to the explicit Qlib workflow interface.

Configuration helpers are also available from `quant_bench.config`:

```python
from quant_bench.config import ExperimentConfig, config_sha256, load_config
```

Unknown fields are rejected by public Pydantic models. Dotted overrides must
refer to an existing field. Values use JSON scalar parsing first and YAML as a
fallback.

## Contracts

`quant_bench.contracts` exports:

- `DatasetManifest`, `RunManifest`, `ArtifactRef` and `ExperimentResult`
- `ModelCapabilities`, `ModelCatalogEntry` and `PredictionFrame`
- `DatasetView`, `ModelPlugin`, `ArtifactReader` and `ArtifactWriter`
- `CsvContract`, `CSV_CONTRACTS`, `CSV_SCHEMA_VERSION` and the versioned CSV contracts

All manifest models reject unknown fields. Artifact paths are relative to a run
directory and carry a media type, size, SHA-256 checksum and visibility.

`PredictionFrame` requires `timestamp`, `symbol` and `score`. CSV column and
time rules are defined in [Data contract](../DATA_CONTRACT.md) and
[Runtime CSV schemas](../runtime/CSV_SCHEMA.md).

## Artifact storage

```python
from pathlib import Path

from quant_bench.artifacts import LocalArtifactStore, verify_run

store = LocalArtifactStore(Path("./qb-workspace/runs/example"))
store.write_json("metrics.json", {"sharpe": 0.0}, schema_id="quant-bench.metrics.v1")
store.write_checksums()

errors = verify_run(store.root)
```

`LocalArtifactStore` uses atomic file replacement and rejects absolute paths or
`..` path escape. `verify_run` returns a list of validation errors; an empty
list means every recorded file matches its checksum.

## Model plugin

Third-party models implement the runtime-checkable `ModelPlugin` protocol:

```python
class ModelPlugin:
    plugin_id: str
    contract_version: str

    def capabilities(self) -> ModelCapabilities: ...
    def fit(self, dataset: DatasetView, context: object) -> object: ...
    def predict(self, model: object, dataset: DatasetView) -> PredictionFrame: ...
    def save(self, model: object, target: ArtifactWriter) -> ArtifactRef: ...
    def load(self, source: ArtifactReader) -> object: ...
```

Register a plugin through the `quant_bench.models` entry point. Feature plugins
use `quant_bench.features`. See [Add a model](../how-to/add-model.md) for the
catalog entry, scaffold and contract tests.

## Unified Method interface

The Method interface covers canonical runs, Qlib, FinGPT, FinMem, packaged
runtime processes and separately deployed `ai_trade` processes.

```python
from pathlib import Path

from quant_bench import ExecutionMode, MethodRunSpec, get_method_runner

runner = get_method_runner()
spec = MethodRunSpec(
    method_id="canonical:crypto_smoke_v1",
    mode=ExecutionMode.BACKTEST,
    workspace=Path("./qb-workspace"),
)

check = runner.check(spec)
if check.ready:
    result = runner.run(spec)
```

Public Method types are exported from `quant_bench.methods`:

- `ExecutionMode`
- `MethodDescriptor`
- `MethodRunSpec`, `MethodCheckResult` and `MethodRunResult`
- `MethodRegistry` and `MethodRunner`
- `get_method_registry()` and `get_method_runner()`

`MethodRunSpec.parameters` rejects credential-like keys. Network, account and
order permissions remain independent. Simulated and live orders require their
exact confirmation token. See [Unified Method interface](../how-to/unified-methods.md).

## Trading interface

Strategies express an order intent as `OrderRequest`. A configured backend owns
account and exchange behavior. `TradingService` records snapshots before and
after the submission and normalizes the result.

```python
from quant_bench.trading import (
    MarketType,
    OrderAction,
    OrderRequest,
    TradingBackend,
    TradingService,
)

request = OrderRequest(
    strategy_id="example",
    instrument_id="BTC-USDT",
    market_type=MarketType.SPOT,
    action=OrderAction.BUY,
    notional_usdt=10.0,
)

def execute_with_reviewed_backend(backend: TradingBackend):
    return TradingService(backend).execute(request)
```

The public module exports `OrderRequest`, `AccountSnapshot`, `ExecutionReport`,
`TradingCycleResult`, `TradingBackend`, `TradingService`, the order enums and
built-in adapter types. Creating an `OrderRequest` does not submit it. Order
submission occurs only when `TradingService` receives a backend whose
`submit_order` method performs that action.

See [Unified trading interface](../how-to/unified-trading-interface.md) before
implementing a backend.

## Error behavior

- Invalid public models raise Pydantic validation errors.
- Missing built-in configs and artifacts raise `FileNotFoundError`.
- Invalid dotted overrides raise `ValueError` or `KeyError` with the field path.
- Artifact path escape raises `ValueError` before a write.
- Method readiness issues are returned as structured `CheckIssue` records.
- Backend exceptions become a failed `ExecutionReport`; snapshot failures are recorded as warnings.

CLI commands serialize these results for shell users. Use the Python objects
when an application needs structured control flow.
