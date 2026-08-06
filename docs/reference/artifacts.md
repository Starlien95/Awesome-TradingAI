# Artifact reference

## Canonical run

| Path | Meaning |
| --- | --- |
| `resolved_config.yaml` | complete validated input |
| `dataset_manifest.json` | source, license field, hash, range, universe, validation link |
| `validation.json` | canonical data checks |
| `model/model.npz` | safe built-in baseline artifact |
| `model/model_card.md` | intended use and limitations |
| `predictions.csv` | timestamp, symbol, score, label |
| `backtest/returns.csv` | gross/net return, turnover, cost, positions, equity |
| `metrics.json` | signal and portfolio metrics with sample counts |
| `reports/summary.md` | one-run human-readable summary |
| `run_manifest.json` | lifecycle, config, data, code, environment, artifact references |
| `checksums.sha256` | integrity record |

Artifact paths are relative to the run root and cannot contain `..`. Writes use a temporary file and atomic replacement. Each `ArtifactRef` records media type, optional schema id, size, SHA-256, and visibility.

Visibility values are `public`, `private`, and `trusted_local_only`. Visibility metadata does not publish a file; an exporter must still enforce a public allowlist and redact sensitive fields.

## Runtime CSVs

Traditional ML, MacroHFT, and FinGPT use the per-artifact versioned contracts
in [runtime/CSV_SCHEMA.md](../runtime/CSV_SCHEMA.md) and
`quant_bench.contracts.csv`. A runtime manifest declares
`manifest_schema_version`, `csv_schema_version`, and relative paths for
dashboard discovery. FinGPT initializes empty metrics, signals, trades, and
volume CSV contracts so no-trade periods remain machine-readable.
