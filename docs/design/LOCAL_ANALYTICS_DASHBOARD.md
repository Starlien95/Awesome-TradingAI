# Local analytics dashboard design

## Product boundary

The local dashboard helps a library user inspect benchmark and paper/runtime
artifacts already present in a chosen workspace. It is read only, works without
network access, and never trains a model, performs inference, queries an
exchange, reads credentials, reconciles positions, or submits orders.

The public Community Cloud showcase is a separate product. It consumes only a
validated, delayed, redacted public snapshot. Local runtime artifacts, account
snapshots, order identifiers, raw news, current positions, prompts, and model
state are never public dashboard inputs.

## User needs

The dashboard answers four ordinary questions:

1. Which runs exist, and which results are scientifically comparable?
2. What do equity, benchmark, alpha, and drawdown curves look like?
3. What signals and execution records explain a selected run?
4. Is the run healthy, current, complete, degraded, or structurally invalid?

The pages are `Overview`, `Run Explorer`, `Health`, and `Data Quality`.
Overview loads only metrics or canonical returns. Detailed signals and audit
tables are loaded after a user selects a run.

## Discovery

The preferred discovery contract is `runs/**/run_manifest.json`.

- canonical benchmark manifests refer to `backtest/returns.csv` and
  `predictions.csv` through their artifact list;
- runtime manifests declare relative artifact paths and
  `csv_schema_version`;
- `timeframes/*/logs/*/*_metrics.csv` is an explicit legacy fallback.

Resolved paths must remain inside the run directory. Invalid JSON, path escape,
symlink escape, missing files, empty CSVs, and malformed columns are isolated to
the affected run and reported with stable issue codes.

## Internal views

External files retain their per-artifact contracts. Adapters map them into four
internal views:

- performance: UTC time, normalized equity, period return, benchmark, alpha,
  and drawdown;
- signals: UTC time, symbol, score, label, and signal label;
- execution: trades, orders, fills, failures, rebalance plans, volume, and
  local-only account snapshots;
- health: coverage, duplicate timestamps, gaps, invalid equity, freshness, and
  structured loader issues.

Alias handling is explicit and row-wise. Epoch bar time is preferred over
legacy naive datetime text. Input order is recorded as a quality warning and
the analysis view is then stably sorted.

## Comparability

Two runs are verified comparable only when both declare equal `protocol_id` and
`dataset_sha256`. Frequency, method family, and mode labels are metadata, not
sufficient scientific evidence. Runs lacking evidence can be plotted for
exploration and never produce an implied unified ranking.

Comparison curves can use the common valid window and are rebased at the
window start. Downsampling preserves endpoints and extrema so large local logs
remain responsive.

## Validation matrix

Tests cover canonical and runtime discovery, ISO and epoch time, mixed schema
aliases, first-period returns, completed versus stale health, missing and empty
files, invalid manifest fields, path and symlink escape, reverse ordering,
duplicate time, NaN and Inf equity, common-window comparisons, and FinGPT daily
timezone semantics. Streamlit AppTest and browser QA cover empty and populated
workspaces at desktop and narrow viewport widths.
