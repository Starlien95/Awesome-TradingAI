# Qlib workflow public defaults

All YAML files under `src/quant_bench/resources/qlib_workflows` are package resources normalized by `quant_bench.config.legacy_workflows`.

## Shared values

| Field | Public template value |
| --- | --- |
| Market | `all` |
| Benchmark | `BTC-USDT` |
| Data path | `./qlib_data/<frequency>` |
| Train | `2021-01-01` through `2023-12-31` |
| Validation | `2024-01-01` through `2024-12-31` |
| Test | `2025-01-01` through `2025-12-31` |
| Label | `Ref($close, -1) / $close - 1` |
| Model hyperparameters | Qlib constructor defaults |
| Seed | `42` where the constructor supports a seed |
| Strategy | `ThresholdTopkDropoutStrategy` |
| `topk` | `5` |
| `min_score` | `0.0` |
| `max_dropout` | `1` |
| Open and close cost | `0.001` each |
| Initial capital | `100000` |
| Portfolio executor | `SimulatorExecutor` at the workflow's own frequency |
| Portfolio record | `CryptoPortAnaRecord` with a frequency-matched benchmark Series |

Structural parameters remain where a constructor cannot infer them, including feature dimensions, TRA `model_config`, TRA `tra_config`, and HIST metadata paths.

## Rebuild and audit

Preview changes:

```bash
quant-bench workflows normalize src/quant_bench/resources/qlib_workflows
```

Write and verify:

```bash
quant-bench workflows normalize src/quant_bench/resources/qlib_workflows --write
quant-bench workflows audit src/quant_bench/resources/qlib_workflows --require-normalized
```

The audit rejects YAML errors, unknown models, developer-home paths, CSI300 or `cn_data` defaults, malformed label parentheses, and non-idempotent templates. Frequency-specific executor defaults and `CryptoPortAnaRecord` prevent intraday portfolio analysis from falling back to Qlib's daily or one-minute data lookup.
