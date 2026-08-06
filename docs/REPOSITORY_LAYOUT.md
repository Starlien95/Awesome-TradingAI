# Repository layout

The repository contains one installable Python package and the material needed
to test, document and audit it. User data and runtime state belong outside the
checkout.

## Root

| Path | Purpose |
| --- | --- |
| `pyproject.toml` | package metadata, dependencies, entry points and test/tool configuration |
| `README.md`, `README.zh-CN.md` | short user entry points |
| `LICENSE`, `LICENSES/` | project and retained upstream license texts |
| `THIRD_PARTY_NOTICES.md` | distributed and external dependency attribution |
| `CITATION.cff` | software citation metadata |
| `CONTRIBUTING.md`, `SECURITY.md` | contribution and private security-reporting rules |
| `MANIFEST.in` | source-distribution allowlist |
| `mkdocs.yml` | documentation navigation and site configuration |

## Package source

`src/quant_bench/` is the only installable package root.

| Subdirectory | Purpose |
| --- | --- |
| `artifacts/` | atomic local storage and checksum verification |
| `cli/` | the `quant-bench` command line interface |
| `config/` | typed config loading, overrides and hashing |
| `contracts/` | versioned manifests, CSV contracts and plugin protocols |
| `dashboard/` | read-only local run discovery, analytics, charts and Streamlit pages |
| `data/` | canonical OHLCV loading, validation and Qlib data preparation |
| `evaluation/` | prediction, portfolio and risk metrics |
| `experiments/` | offline canonical experiment lifecycle |
| `features/`, `labels/` | causal feature and label implementations |
| `integrations/qlib/` | Qlib session, models, records, workflows and compatibility modules |
| `integrations/ai_trade/` | process boundary for a separately deployed external stack |
| `methods/` | FinGPT, FinMem, traditional ML and reinforcement-learning adapters |
| `models/`, `registry/` | built-in models and model discovery |
| `reporting/` | benchmark comparison and report generation |
| `resources/` | packaged synthetic fixtures, configs and normalized Qlib workflows |
| `runtime/` | guarded paper, demo and live process adapters |
| `strategies/` | shared portfolio-selection semantics |
| `trading/` | order, account, backend and execution-service contracts |
| `tuning/` | dependency-light grid search |

Method directories may include small deterministic fixtures and configuration
templates. They do not contain model weights, raw market/news datasets,
credentials, accounts or production logs.

## Tests

`tests/` contains the regression suite used by contributors and CI.

| Subdirectory | Purpose |
| --- | --- |
| `contract/` | public schema and trading-boundary tests |
| `unit/` | isolated config, model, method, artifact, release and analytics tests |
| `integration/` | offline workflows, Qlib integration, CLI and Streamlit AppTest |

Optional dependency tests skip when the dependency is absent and run in a
dedicated CI job with the relevant extra installed. These are maintained tests,
not end-user commands.

## Tools and release material

| Path | Purpose |
| --- | --- |
| `tools/check_release.py` | source-tree privacy, license and safety audit |
| `tools/check_wheel.py`, `tools/check_sdist.py` | package-content audits |
| `tools/check_runtime.py`, `tools/smoke_runtime.py` | dependency and offline runtime checks |
| `tools/check_finmem_public.py` | FinMem public-boundary audit |
| `tools/generate_release_metadata.py` | SBOM and dependency-license metadata generation |
| `tools/normalize_workflows.py` | Qlib workflow maintenance |
| `tools/convert_macrohft_external.py` | trusted-local converter for separately obtained MacroHFT material |
| `tools/legacy_qlib/` | small 0.x compatibility wrappers that call packaged Qlib modules |
| `release/` | generated SBOM, dependency-license report and checksums |
| `.github/workflows/ci.yml` | Python matrix, quality, packaging, Qlib and dashboard jobs |

Production ledger repair, account liquidation and ad hoc order smoke scripts
are not distributed. They depend on private operator context and are outside
the public library boundary.

## Documentation

`docs/` follows the tutorials, how-to, concepts, reference and maintainer design
split. Current behavior comes from source and current reference documents.
Historical design documents retain the decision context that led to the public
package.

## User workspace

Commands write generated state under a path supplied with `--workspace` or a
documented platform data directory. A typical workspace contains:

```text
qb-workspace/
  runs/
    <run_id>/
      run_manifest.json
      checksums.sha256
      predictions.csv
      backtest/returns.csv
  reports/
  studies/
  methods/
```

Do not commit a workspace. The repository ignore and release checks reject
common run, data, model and credential paths.
