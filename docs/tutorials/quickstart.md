# Offline quickstart

This tutorial starts from a clean checkout and ends with a run that can be inspected and checksum-verified. It does not use exchange credentials or network access.

## 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Confirm the command and workspace are usable:

```bash
quant-bench doctor --workspace ./qb-workspace
```

The JSON output reports Python and package versions, installed optional backends, model catalog size, workspace writability, and whether expected credential environment variables are present. It never prints credential values.

## 2. Run the packaged benchmark

```bash
quant-bench quickstart --offline --workspace ./qb-workspace
```

The command prints an `ExperimentResult` containing `run_id`, `run_dir`, `status`, metrics, and `manifest_path`. Save the `run_id` for the following commands.

The fixture contains only a few synthetic test periods. Its return, Sharpe, and annualized metrics validate the calculation and artifact pipeline; they have no empirical or trading interpretation.

The run performs these operations in order:

1. Load and validate `crypto_smoke_v1`.
2. Load the packaged synthetic BTC/ETH hourly fixture.
3. Check timestamps, duplicate keys, OHLC ordering, volume, and symbols.
4. Build six causal OHLCV features and a one-bar forward-return label.
5. Apply fixed train, validation, and test periods.
6. Fit the deterministic `numpy_linear_v1` ridge baseline.
7. Calculate signal metrics, turnover, costs, return, volatility, Sharpe ratio, and drawdown.
8. Write the resolved configuration, manifests, tables, report, and checksums.

## 3. Inspect the result

```bash
quant-bench artifacts inspect ./qb-workspace/runs/<run_id>
quant-bench artifacts verify ./qb-workspace/runs/<run_id>
```

`inspect` prints the typed run manifest. `verify` recalculates every recorded SHA-256 value and exits nonzero if a file is missing or changed.

Open the run summary:

```bash
sed -n '1,160p' ./qb-workspace/runs/<run_id>/reports/summary.md
```

Generate a table covering every completed run in the workspace:

```bash
quant-bench report --workspace ./qb-workspace
```

The report is written to `qb-workspace/reports/benchmark_summary.md`. It marks runs as directly comparable only when their protocol id and dataset hash match.

## 4. Change one setting

Render the built-in recipe before editing it:

```bash
quant-bench config render crypto_smoke_v1 > ./qb-workspace/local-config/ridge.yaml
quant-bench config validate ./qb-workspace/local-config/ridge.yaml
```

Run an override without editing the file:

```bash
quant-bench run crypto_smoke_v1 \
  --workspace ./qb-workspace \
  --set model.parameters.ridge=0.001
```

Unknown keys are rejected. The complete resolved config and its hash are stored in the new run.

## 5. Use the Python API

```python
from pathlib import Path

from quant_bench.config import load_config
from quant_bench.experiments import Experiment

config = load_config("crypto_smoke_v1", ["model.parameters.ridge=0.001"])
config.experiment.workspace = Path("./qb-workspace")
result = Experiment(config).run()
print(result.run_dir)
```

## Common failures

- `workspace_parent_writable: false`: choose a directory owned by the current user.
- `built-in fixture not found`: reinstall the editable package; package data is missing from the environment.
- `split is empty`: inspect the configured timestamps and the input data range.
- `artifact checksum mismatch`: keep the run immutable and create a new run after changing data or files.

Continue with [the Qlib tutorial](qlib-first-run.md), [grid sweeps](../how-to/results.md), or [adding a model](../how-to/add-model.md).
