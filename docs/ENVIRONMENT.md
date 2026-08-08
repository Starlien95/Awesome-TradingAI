# Environment and dependency requirements

`pyproject.toml` is the authoritative dependency specification for the
installable `quant-bench` library. The repository intentionally does not ship a
single root `requirements.txt`: core research, dashboards, Qlib, model
backends, FinGPT, FinMem, and exchange runtime support have different sizes,
licenses, hardware needs, and network boundaries.

## Supported base environment

| Requirement | Supported or required value |
| --- | --- |
| Python | 64-bit CPython 3.10, 3.11, or 3.12 |
| Build frontend | `pip`, with `setuptools>=77` and `wheel` resolved from `pyproject.toml` |
| Tested CI platform | GitHub-hosted Ubuntu on Python 3.10 through 3.12 |
| Local reference platform | Linux or WSL with Python 3.12 |
| Core hardware | CPU only; no GPU is required for the offline quickstart |
| Core network access | Required for installation unless dependencies are already cached; not required at runtime for the offline quickstart |
| Workspace | A writable user-selected directory outside the installed package |

Native Windows and macOS are not part of the current CI matrix. Use WSL on
Windows when you need the same path and process behavior as the validated Linux
environment. Platform-specific model wheels can impose narrower constraints.

Create an isolated source environment:

```bash
git clone https://github.com/Starlien95/Awesome-TradingAI.git
cd Awesome-TradingAI
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check
quant-bench doctor
```

For PowerShell activation, use `.\.venv\Scripts\Activate.ps1`. An offline
installation needs a pre-populated wheelhouse containing the build frontend,
the selected extras, and all transitive dependencies.

## Optional dependency groups

Install only the capabilities you plan to use.

| Extra | Capability and important boundary |
| --- | --- |
| `dashboard` | Local read-only Streamlit and Plotly analysis |
| `qlib` | Qlib 0.9.7 integration |
| `data-qlib` | Qlib data download, conversion, and audit tools, including CCXT |
| `lgbm` | LightGBM model backend |
| `xgboost` | XGBoost model backend |
| `catboost` | CatBoost model backend |
| `torch` | PyTorch models and trusted-local TorchScript adapters |
| `tuning` | Optuna parameter studies |
| `tracking` | MLflow experiment tracking |
| `data-ccxt` | CCXT market-data access without the full Qlib data stack |
| `visualization` | Matplotlib and Seaborn plots, including InvestorBench plot helpers |
| `runtime` | Shared SciPy, Joblib, Requests, and runtime support |
| `runtime-okx` | OKX account and order client for explicitly enabled runtime paths |
| `news` | Local PyTorch, Transformers, PEFT, and FinGPT-family model inference |
| `fingpt-live` | Live news feeds, public prices, and FinGPT OKX adapters |
| `finmem` | InvestorBench engine, Qdrant client, LLM endpoints, and local FinMem workflows |
| `finmem-live` | Live news and OKX adapters for FinMem |
| `finmem-guardrails` | Optional Guardrails AI endpoint support |
| `docs` | MkDocs Material documentation build |
| `dev` | pytest, ruff, mypy, build, and typing stubs for contributors |
| `release` | SBOM and dependency-license generation |
| `all` | Broad user environment covering all research, runtime, dashboard, visualization, and documentation capabilities; excludes `dev` and `release` |

Common installation combinations:

```bash
# Local analytics
python -m pip install -e ".[dashboard]"

# Qlib data pipeline with a LightGBM model
python -m pip install -e ".[data-qlib,lgbm]"

# Qlib neural models
python -m pip install -e ".[data-qlib,torch]"

# MacroHFT OKX runtime
python -m pip install -e ".[runtime,runtime-okx,torch]"

# LightGBM and XGBoost OKX runtime
python -m pip install -e ".[runtime,runtime-okx,lgbm,xgboost]"

# Qlib TCN and GATS OKX runtime
python -m pip install -e ".[runtime,runtime-okx,qlib,torch]"

# FinGPT local inference and guarded runtime adapters
python -m pip install -e ".[news,fingpt-live,runtime]"

# FinMem research, live adapters, dashboard, and plot helpers
python -m pip install -e ".[finmem,finmem-live,dashboard,visualization]"

# Contributor quality gates
python -m pip install -e ".[dev,docs,dashboard]"

# Maintainer-only broad environment
python -m pip install -e ".[all,dev,release]"
```

PyTorch publishes platform-specific CPU and accelerator wheels. Install the
wheel appropriate for the host before installing `.[torch]` when the default
index does not provide the required build. Qlib model requirements are listed
per model in the [model support matrix](MODEL_SUPPORT.md).

Some compiled packages, including Qlib model backends, CVXPY solvers, and
gradient-boosting libraries, may require a platform compiler or system library
when a compatible wheel is unavailable. The validated Ubuntu and WSL path uses
binary wheels where available. Native builds must follow the selected
dependency's platform documentation and remain outside the package contract.

## External data, models, and services

Package installation alone does not provide every method's runtime inputs.

| Capability | Additional requirement |
| --- | --- |
| Qlib research | User-provided or downloaded OHLCV data and a writable Qlib store |
| FinGPT local inference | User-supplied model and adapter weights with compatible licenses; GPU and memory requirements depend on the selected model |
| FinGPT news runtime | An authorized news source or configured public feed; model endpoint or local model when inference is enabled |
| FinMem research | User-owned InvestorBench-format data, Qdrant, an embedding endpoint, and an OpenAI-compatible chat endpoint |
| OKX demo | Network access, demo credentials in environment variables, `api.is_simulated: true`, and the exact demo confirmation token |
| OKX live | Separately scoped live credentials, reviewed configuration, explicit order permission, and `LIVE_ORDERS` confirmation |
| External `ai_trade` stack | A separately reviewed checkout with its own environment, model artifacts, credentials, and process controls |

Model weights, private datasets, credentials, account data, and production logs
are not distributed. See [network access](runtime/NETWORK_ACCESS.md),
[runtime modes](how-to/runtime.md), and
[security boundaries](concepts/security-boundaries.md) before enabling any
network, account, or order capability.

## Validation and reproducibility

After changing an environment, run:

```bash
python -m pip check
quant-bench doctor
quant-bench quickstart --offline --workspace ./qb-workspace
```

Contributors should additionally run the commands in
[`CONTRIBUTING.md`](https://github.com/Starlien95/Awesome-TradingAI/blob/main/CONTRIBUTING.md).
CI installs the core package on Python
3.10, 3.11, and 3.12, builds documentation, tests the dashboard, exercises the
Qlib contract, builds wheel and source distributions, and runs an installed
wheel outside the checkout. The current CI does not install every heavy or
service-backed optional profile. `news`, FinMem service integrations, every
model backend, GPU wheels, and OKX clients remain explicitly selected
environments and must pass `python -m pip check` plus the relevant `doctor`
command before use.

The project uses bounded library dependencies instead of a cross-platform
transitive lockfile. Every completed benchmark run records Python,
`quant-bench`, core package, and Qlib versions in its manifest. For a deployment
or paper/runtime process, retain a complete environment lock or container
definition outside the public repository together with the run manifest.
