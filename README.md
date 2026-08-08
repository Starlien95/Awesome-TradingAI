<div align="center">

<img src="assets/banner.png" alt="Can AI Make Money in Crypto?" width="650">

[简体中文](README.zh-CN.md)

[Documentation](docs/index.md) | [CLI reference](docs/reference/cli.md)

*A unified benchmark for evaluating AI trading methods from historical backtesting to real-time paper trading and live execution.*

[![Exchange paper trading](https://img.shields.io/badge/Exchange%20Paper%20Trading-Visualization-brightgreen.svg)](https://quant-bench-showcase.streamlit.app/)
[![CI](https://github.com/Starlien95/Awesome-TradingAI/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Starlien95/Awesome-TradingAI/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
<br><br>

</div>


**Can AI Make Money in Crypto?** is an open-source benchmark and codebase for evaluating AI trading methods, including machine learning, reinforcement learning, LLM-based methods, and trading agents. It provides unified interfaces for historical backtesting, real-time paper trading on cryptocurrency exchanges, and live trading, while continuously publishing backtest and paper-trading results. Real-money trading results will be added in future evaluations.

## Real-time Paper-Trading Results

[Open the public results dashboard](https://quant-bench-showcase.streamlit.app/) to explore normalized return curves, comparisons with BTC Buy & Hold, drawdowns, trading signals, execution summaries, method details, and data-quality status across 16 public runs covering 15 distinct methods.

The public results currently use the OKX Demo Trading environment, corresponding to the library's `exchange-paper` mode. This mode is distinct from both local `local-paper` execution and real-money `live` trading. The dashboard operates in read-only mode using delayed and sanitized result snapshots; it cannot access exchange accounts, read trading credentials, or submit orders.

The evaluation period begins at `2026-06-01 00:00 UTC+8` for all methods except the agent-based approaches, which begin at `2026-06-04 00:00 UTC+8`. Results are reported only from each method's actual start time, with no synthetic backfilling applied to earlier periods.



## 🔥 News

- **[Coming Soon]** Real-money live trading evaluation.
- **[Aug. 8, 2026]** Public OKX demo visualization is online.
- **[Aug. 6, 2026]** Full benchmark codebase released.

## Why This Benchmark?

Most AI trading methods are evaluated only through **historical backtesting**. However, strong backtest performance does not necessarily translate into profits in unseen and continuously evolving markets.

We evaluate trading methods across three stages:

**Historical Backtesting → Real-Time Paper Trading → Live Trading**

Our goal is to answer a simple question:

> **Can AI actually make money in crypto?**

## Supported Methods & Features

The repository provides a unified framework for integrating, running, and evaluating AI trading methods across different methodological paradigms, from traditional machine learning to LLM-based trading agents.

### Method Integrations

| Category | What we provide |
| --- | --- |
| **Traditional ML** | Interfaces to classical baselines and Qlib-based trading workflows, including training, tuning, backtesting, and saved-model execution |
| **Reinforcement Learning** | Interfaces to RL-based trading methods such as MacroHFT |
| **LLM-based Trading** | Interfaces to LLM-based trading methods such as FinGPT |
| **Trading Agents** | Interfaces to agent-based trading methods such as FinMem |

### Core Capabilities

| Capability | What we provide |
| --- | --- |
| **Historical Backtesting** | A consistent backtesting environment with configurable transaction costs, portfolio settings, and reproducible experiment configurations |
| **Real-Time Paper Trading** | Run supported methods on live cryptocurrency market data through paper-trading environments and continuously track their trading performance |
| **Unified Method Interface** | A common interface for configuring, validating, running, and comparing ML, RL, LLM, and agent-based trading methods |
| **Unified Analysis** | A common analysis pipeline for all supported methods, covering returns, alpha, drawdown, trading signals, execution summaries, and data quality |

The default quickstart runs entirely offline and does not require exchange credentials or submit any real orders.

## 🚀 Quick Start

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Starlien95/Awesome-TradingAI.git
cd Awesome-TradingAI
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

See [Environment and dependency requirements](docs/ENVIRONMENT.md) for the
supported Python and platform matrix, optional extras, external services, GPU
boundaries, and offline installation requirements.

Install only the integrations you need:

```bash
python -m pip install -e ".[dashboard]"
python -m pip install -e ".[qlib,lgbm]"
python -m pip install -e ".[data-qlib]"
python -m pip install -e ".[finmem,finmem-live]"
python -m pip install -e ".[news,fingpt-live,runtime]"
```

### Offline Quickstart

Run the benchmark locally without network access or exchange credentials:

```bash
quant-bench doctor
quant-bench quickstart --offline --workspace ./qb-workspace
quant-bench report --workspace ./qb-workspace
```

The offline quickstart uses bundled synthetic BTC and ETH OHLCV data to verify the complete evaluation pipeline. It validates the input data, constructs leakage-safe features and labels, trains a lightweight NumPy baseline, performs cost-aware historical backtesting, and stores the resulting artifacts and checksums under `qb-workspace/runs/`.

### Parameter Sweep

A parameter sweep can be launched through the same interface:

```bash
quant-bench sweep crypto_smoke_v1 \
  --param model.parameters.ridge=0.0,0.000001,0.001 \
  --study-name ridge-example \
  --workspace ./qb-workspace

quant-bench compare --workspace ./qb-workspace
```
The resulting runs can then be compared under a consistent evaluation protocol using the compare command.

### Local Dashboard

Install the optional dashboard dependencies and launch the local analytics dashboard:

```bash
python -m pip install -e ".[dashboard]"
quant-bench dashboard --workspace ./qb-workspace
```

The dashboard provides an interactive view of results and artifacts stored in the selected workspace. It operates in read-only mode: it does not train models, access exchange accounts, read trading credentials, submit orders, or control any running trading process. See [Inspect results](docs/how-to/results.md) and the [dashboard design](docs/design/LOCAL_ANALYTICS_DASHBOARD.md).

### Methods and Execution Modes

Use the following commands to inspect available methods, review their supported capabilities, and verify whether a method is compatible with a given execution mode:


```bash
quant-bench methods list
quant-bench methods show canonical:crypto_smoke_v1
quant-bench methods check canonical:crypto_smoke_v1 \
  --mode backtest \
  --workspace ./qb-workspace
```

The framework supports four execution modes:

| Mode | Description |
| --- | --- |
| `backtest` | Evaluate a method on historical or synthetic market data without connecting to an exchange account |
| `local-paper` | Run a method with local portfolio accounting and simulated order execution, without submitting orders to an exchange |
| `exchange-paper` | Run a method in an exchange-provided paper-trading environment using live market data and simulated funds; network access and exchange credentials are required |
| `live` | Run a method in the live exchange environment with real funds; separate trading permissions and explicit `LIVE_ORDERS` confirmation are required |

Execution capabilities vary across methods. Use `methods show` to inspect the interfaces supported by a method and `methods check` to verify that all requirements for the selected execution mode are satisfied before starting a run.
External `ai_trade` integrations are executed as separate processes and require an independently reviewed local checkout.


## Public interfaces

- [Python API](docs/reference/python-api.md)
- [CLI](docs/reference/cli.md)
- [Unified Method interface](docs/how-to/unified-methods.md)
- [Model plugin guide](docs/how-to/add-model.md)
- [Unified trading interface](docs/how-to/unified-trading-interface.md)
- [Data contract](docs/DATA_CONTRACT.md)
- [Runtime CSV schemas](docs/runtime/CSV_SCHEMA.md)
- [Artifacts and manifests](docs/reference/artifacts.md)
- [Repository layout](docs/REPOSITORY_LAYOUT.md)
- [Environment requirements](docs/ENVIRONMENT.md)

The repository name is `Awesome TradingAI`. The stable distribution, import,
CLI and plugin namespaces remain `quant-bench` and `quant_bench` for API
compatibility.

## Repository map

```text
src/quant_bench/    installable library, CLI, methods, runtime and dashboard
tests/              contract, unit and integration tests
tools/              release checks, workflow maintenance and compatibility tools
docs/               tutorials, guides, concepts, reference and design records
release/            checked SBOM and dependency-license metadata
.github/workflows/  test, package, documentation and boundary checks
```

Runtime data belongs in a user-selected workspace. Model weights, exchange
credentials, account snapshots, orders, private news, Qlib stores and complete
production logs are excluded from this repository.

## Safety and scope

- Research and dashboard commands do not place orders.
- Credentials come from environment variables or an untracked local provider.
- Account reads and order writes use separate permissions.
- Demo and live orders require different exact confirmation tokens.
- Account liquidation and production-ledger repair scripts are not distributed.
- Public benchmark comparison requires matching protocol and dataset hashes.

Read [Security boundaries](docs/concepts/security-boundaries.md) before enabling
network, account or order capabilities.

## Development and Quality Checks

Install the development dependencies and run the same linting, static type checking, testing, documentation build, and release-boundary checks used by CI:

```bash
python -m pip install -e ".[dev,docs,dashboard]"
ruff check .
mypy --strict src/quant_bench
pytest -q
mkdocs build --strict
python tools/check_release.py
```

These commands are intended for local development and validation only. They do not publish the package, deploy the public Streamlit site, or start any paper-trading or live-trading process.
For contribution guidelines, security policies, and third-party provenance, see [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the [Provenance Register](docs/legal/PROVENANCE.md).
The project is licensed under the MIT License. Bundled third-party components remain subject to their respective notices and license terms.
