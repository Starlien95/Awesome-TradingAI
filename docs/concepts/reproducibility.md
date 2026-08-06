# Reproducibility and fair comparison

A useful benchmark result identifies the data bytes, feature semantics, label semantics, split, strategy, costs, model parameters, seed, code, environment, and output artifacts.

Every canonical run records:

- resolved config and SHA-256;
- dataset source, license field, content SHA-256, symbols, range, and row count;
- feature schema SHA-256;
- protocol id, costs, seed list, and initial cash;
- package and Python versions plus Git commit when available;
- predictions, returns, metrics, model card, artifact references, and checksums;
- failed status, exception type, and message when execution stops.

Two headline metrics are directly comparable only when protocol id and dataset hash match. Matching names, symbols, or date labels do not establish identical data bytes. Changes to feature formulas, label horizon, purge, costs, annualization, or strategy rules require a new semantic version or protocol id.

Publication results should report all declared seeds, aggregate statistics, drawdown, turnover, costs, coverage, failed intervals, missing data, and method limitations. A public dashboard must not select only profitable runs.
