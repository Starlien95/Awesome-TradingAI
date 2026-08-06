# Troubleshooting

## The CLI is missing after installation

Activate the environment used for installation and run:

```bash
python -m pip show quant-bench
python -m quant_bench --help
```

Reinstall with `python -m pip install -e .` when the editable entry point is stale.

## Qlib reports an empty dataset

Check all three layers:

```bash
quant-bench data validate ./qb-workspace/datasets/raw/okx_1h.csv
find ./qb-workspace/datasets/qlib/1h -maxdepth 2 -type f | head
quant-bench qlib validate <workflow-path>
```

Confirm that workflow periods overlap the dataset, the frequency matches, symbols exist, and `--provider-uri` points to the bin root containing calendars, instruments, and features.

## A model is listed as unavailable

```bash
quant-bench models show <model_id>
quant-bench models list --verbose
```

Install the `extra` reported by the catalog. Some Qlib wrappers also need model-specific libraries such as LightGBM, XGBoost, CatBoost, or PyTorch.

## Runtime check is valid but not ready

`valid` confirms YAML structure. `ready` also requires every selected model path and credential environment variable. Use `artifacts promote` to place a model, then rerun `runtime check`. The check does not test OKX connectivity.

## FinGPT dry-run has no trade rows

Neutral signals can produce zero submitted trades. The current runner still creates an empty trades CSV with the documented header. Inspect signals, attribution, metrics, and the rebalance plan before interpreting the result.

## Artifact verification fails

Do not repair a completed run in place. Preserve it for audit, identify the changed or missing file from `artifacts verify`, and rerun the experiment into a new run id.

## Dashboard finds no data

Point it at the workspace that contains `runs/`:

```bash
quant-bench dashboard --workspace ./qb-workspace
```

The dashboard accepts missing files and historical schemas, but it cannot display a run with no discoverable manifest or legacy metrics path.

Open `Data Quality` when one run is missing from a curve. The page reports
invalid JSON, missing files, empty or header-only CSVs, absent time/equity
columns, path escape, timestamp order, and historical alias use without
stopping unrelated runs. Runtime manifest paths must be relative to the run
directory. Multiple runs without matching `protocol_id` and `dataset_sha256`
remain exploratory and do not form a verified ranking.
