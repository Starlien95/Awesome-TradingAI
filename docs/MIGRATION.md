# Migration from qlib_model_trade

## Repository address

The public project is hosted at
`https://github.com/Starlien95/Awesome-TradingAI`. Existing Python imports,
the `quant-bench` distribution, the `quant-bench` CLI, entry point groups and
schema identifiers keep their current names. Update Git remotes and repository
links; application code does not need a package rename.

## Command mapping

| Legacy command | New command |
| --- | --- |
| Local environment inspection | `quant-bench doctor` |
| Find model workflow | `quant-bench workflows list` and `quant-bench models list` |
| Parse a workflow | `quant-bench qlib validate <workflow>` |
| Run a workflow | `quant-bench qlib run <workflow> --workspace <path>` |
| Audit workflow matrix | `quant-bench workflows audit src/quant_bench/resources/qlib_workflows --require-normalized` |
| Download, split, and dump data | `quant-bench data pull`, `data split`, `data qlib-dump` |
| Train or backtest saved Qlib artifacts | `quant-bench qlib train` and `quant-bench qlib backtest` |
| Run a parameter sweep | `quant-bench sweep` or `quant-bench qlib tune` |
| Promote a runtime model | `quant-bench artifacts promote` |
| Inspect runtime readiness | `quant-bench runtime init` and `runtime check` |
| Verify outputs | `quant-bench artifacts verify <run_dir>` |
| Lightweight end-to-end test | `quant-bench quickstart --offline` |

## Compatibility window

- `CustomHandler158` maps to `CryptoAlpha52Handler` inside `quant_bench.integrations.qlib.handlers`.
- `CustomHandler360` maps to `CryptoOHLCV6Handler` in the same module.
- `ThresholdTopkDropoutStrategy` lives in `quant_bench.integrations.qlib.strategy`.
- Source-tree wrappers under `tools/legacy_qlib` call packaged 0.x compatibility modules.

The former top-level directory has been removed. Packaged workflow YAML uses canonical module paths. New integrations should not import source-tree wrappers.

## Artifact safety

Legacy pickle files are trusted-local inputs. The new public model API does not load them. Keep old environments available for controlled conversion to native formats and model manifests.

MacroHFT has a dedicated migration boundary because the audited upstream
repository does not provide a software license. The public adapter loads a
local `*_hyperagent.pt` TorchScript artifact and six local subagent `.pt`
artifacts. Authorized users can run `tools/convert_macrohft_external.py` against
their reviewed upstream checkout and checkpoints. The source tree, original
pickle files, converted artifacts, and conversion manifest remain local and
are excluded from Git and release archives.

## Workflow semantic change

Legacy workflow files now contain neutral integration defaults. Historical tuned values were intentionally absent from the former `opensource` branch. Use a private reproducibility record or a versioned public benchmark protocol for published experiments.

## FinGPT research and runtime migration

| Legacy area | Public interface |
| --- | --- |
| ETH pretrained and sentiment-SFT threshold scripts | `quant-bench fingpt backtest` with a one-symbol config |
| Ten-coin aggregation and strategy sweeps | typed `aggregation` and `signals` grids in the FinGPT research config |
| Validation and test scripts | disjoint half-open `validation` and `test` windows with validation-only selection |
| Per-coin parameter CSV generation | guarded `enable_per_symbol_thresholds` tuning output |
| Plot input CSVs and metric summaries | `global_tuned/`, `per_symbol_tuned/`, and canonical result CSVs |
| FinGPT daily paper runner | `quant-bench fingpt paper-dry-run` or the capability-gated runtime module |
| Historical rebuild scripts | repeatable offline research runs over user-supplied immutable CSVs |
| Account alignment and liquidation helpers | operator-only workflow; excluded from ordinary library commands |

The source baseline for this consolidation is quant-bench commit
`6d9ed3cae5382d4ca1d8c910aa287cfcf79162f0`. Raw news, results, model weights,
adapters, prompt caches, credentials, and account state are not migrated into
the public repository.
