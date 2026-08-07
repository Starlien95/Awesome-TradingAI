<div align="center">

<img src="assets/banner.png" alt="Can AI Make Money in Crypto?" width="650">

[简体中文](README.zh-CN.md)
<br><br>

</div>

**A unified benchmark for evaluating AI trading methods from historical backtesting to real-time and live trading.**

[![Live Paper Trading](https://img.shields.io/badge/Live%20Paper%20Trading-View%20Results-brightgreen.svg)](https://quant-bench-showcase.streamlit.app/)
[![CI](https://github.com/Starlien95/Awesome-TradingAI/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Starlien95/Awesome-TradingAI/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Can AI Make Money in Crypto?** is an open-source benchmark and codebase for evaluating various AI for trading methods, including machine learning, reinforcement learning, LLM, and agent-based trading methods. It provides unified interfaces for historical backtesting and real-time paper trading on cryptocurrency exchanges, while continuously presenting backtest, live paper-trading results, and real-money trading results.

[Documentation](docs/index.md) | [CLI reference](docs/reference/cli.md)

## 🔥 News

- **[Coming Soon]** Real-money live trading evaluation.
- **[Aug. 8, 2026]** Live paper-trading leaderboard is online.
- **[Aug. 6, 2026]** Full benchmark codebase released.

## Why This Benchmark?

Most AI trading methods are evaluated only through **historical backtesting**. However, strong backtest performance does not necessarily translate into profits in unseen and continuously evolving markets.

We evaluate trading methods across three stages:

**Historical Backtesting → Real-Time Paper Trading → Live Trading**

Our goal is to answer a simple question:

> **Can AI actually make money in crypto?**

## Supported Methods & Features

The repository provides a unified framework for running and evaluating different generations of AI trading methods, from traditional machine learning to LLM-based trading agents.

| Category | What we provide |
| --- | --- |
| **Traditional ML** | Classical baselines and Qlib-based trading workflows, with support for training, tuning, backtesting, and saved models |
| **Reinforcement Learning** | Integration with RL-based trading methods such as MacroHFT, with a unified evaluation interface |
| **LLM-based Trading** | Support for FinGPT-style sentiment and trading pipelines, including backtesting and paper trading |
| **Trading Agents** | Integration with agent-based methods such as FinMem for research evaluation and real-time paper trading |
| **Historical Backtesting** | A consistent backtesting environment with transaction costs, trading metrics, and reproducible configurations |
| **Real-Time Paper Trading** | Run supported methods on live cryptocurrency market data and track their trading performance over time |
| **Unified Interface** | Discover, configure, run, and compare ML, RL, LLM, and agent-based methods through a common workflow |
| **Analysis & Reporting** | Portfolio value, returns, alpha, drawdown, trading signals, executions, and data-quality analysis |

The default quickstart runs entirely offline and does not require exchange credentials or submit any real orders.

## 🚀 Quick Start

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
pip install -r requirements.txt
```

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

Run a small parameter sweep:

```bash
quant-bench sweep crypto_smoke_v1 \
  --param model.parameters.ridge=0.0,0.000001,0.001 \
  --study-name ridge-example \
  --workspace ./qb-workspace

quant-bench compare --workspace ./qb-workspace
```

### Local dashboard

```bash
python -m pip install -e ".[dashboard]"
quant-bench dashboard --workspace ./qb-workspace
```

The dashboard reads artifacts already present in the selected workspace. It
does not train models, call an exchange, read credentials, or control a trading
process. See [Inspect results](docs/how-to/results.md) and the
[dashboard design](docs/design/LOCAL_ANALYTICS_DASHBOARD.md).

### Methods and execution modes

```bash
quant-bench methods list
quant-bench methods show canonical:crypto_smoke_v1
quant-bench methods check canonical:crypto_smoke_v1 \
  --mode backtest \
  --workspace ./qb-workspace
```

| Mode | Meaning |
| --- | --- |
| `backtest` | historical or synthetic research without exchange-account access |
| `paper` | local accounting with no exchange orders |
| `simulated` | exchange demo environment with explicit network, credential and confirmation gates |
| `live` | live exchange environment with separate order permission and `LIVE_ORDERS` confirmation |

Method capabilities differ. Use `methods show` and `methods check` before a
run. External `ai_trade` methods remain separate processes and require their
own reviewed checkout.

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

## Development

```bash
python -m pip install -e ".[dev,docs,dashboard]"
ruff check .
mypy --strict src/quant_bench
pytest -q
mkdocs build --strict
python tools/check_release.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the
[provenance register](docs/legal/PROVENANCE.md). The project is licensed under
MIT. Bundled third-party portions retain their own notices and licenses.
