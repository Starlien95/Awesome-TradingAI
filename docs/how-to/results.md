# Inspect, compare, and report results

## Verify one run

```bash
quant-bench artifacts inspect ./qb-workspace/runs/<run_id>
quant-bench artifacts verify ./qb-workspace/runs/<run_id>
```

Treat completed run directories as immutable. Any modification invalidates the checksum record and should result in a new run.

## Compare runs

```bash
quant-bench compare --workspace ./qb-workspace
```

The command compares only completed manifests and reports whether all rows share one `(protocol_id, dataset_sha256)` pair. It exits with code 2 when the runs are not directly comparable. `--allow-incomparable` is available for inventory work and does not make the metrics scientifically comparable.

## Run a grid sweep

```bash
quant-bench sweep crypto_smoke_v1 \
  --param model.parameters.ridge=0.0,0.000001,0.001 \
  --study-name ridge-example \
  --workspace ./qb-workspace \
  --max-trials 10
```

Trial rows are written atomically to `studies/ridge-example/trials.csv`. Use `--resume` to skip completed parameter JSON values. Each trial remains an ordinary run with its own manifest and checksums.

## Write a portable summary

```bash
quant-bench report --workspace ./qb-workspace
```

This produces `reports/benchmark_summary.md`, which is suitable for a small public snapshot after reviewing data and metric visibility.

## Open the dashboard

```bash
python -m pip install -e ".[dashboard]"
quant-bench dashboard --workspace ./qb-workspace
```

The dashboard reads local files only. It does not train, infer, query an account, or submit an order. Public deployment must read a bounded anonymized snapshot; see the runtime and showcase design documents before publishing data.

Use `Overview` for multi-run curves and common-window inspection, `Run
Explorer` for one run's performance, signals, and execution audit, `Health` for
freshness and coverage, and `Data Quality` for structured loader issues. A
multi-run comparison is scientifically verified only when every selected run
declares the same `protocol_id` and `dataset_sha256`. See the [local dashboard
design](../design/LOCAL_ANALYTICS_DASHBOARD.md).
