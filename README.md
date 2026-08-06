# Awesome TradingAI

[![CI](https://github.com/Starlien95/Awesome-TradingAI/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Starlien95/Awesome-TradingAI/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Awesome TradingAI is an open-source benchmark and local analysis toolkit for
cryptocurrency trading methods. Its installable engine is named `quant-bench`.
It provides reproducible backtests, method adapters, versioned artifacts,
paper/runtime safety gates, and a read-only Streamlit dashboard.

[简体中文](README.zh-CN.md) | [Documentation](docs/index.md) | [CLI reference](docs/reference/cli.md)

## What is included

| Area | Available functionality |
| --- | --- |
| Benchmark core | typed configs, deterministic runs, explicit costs, metrics, checksums, comparison, reports |
| Traditional ML | NumPy baseline, Qlib 0.9.7 workflows, model catalog, tuning and saved-artifact adapters |
| Reinforcement learning | MacroHFT runtime adapter and trusted-local TorchScript conversion boundary |
| FinGPT | synthetic news dry-run, sentiment research, backtest, tuning and guarded paper/runtime adapters |
| FinMem | InvestorBench research integration, local data verification, paper ledger and guarded service adapters |
| Unified methods | one discovery, check, run and status interface across built-in and external methods |
| Data contracts | versioned market, prediction, return, metric, signal, trade and manifest contracts |
| Analysis | local equity, benchmark, alpha, drawdown, signal, execution, health and data-quality views |

The default install and offline quickstart do not use network access, read
credentials, query an account, or submit an order.

## Install

Python 3.10 through 3.12 is supported.

```bash
git clone https://github.com/Starlien95/Awesome-TradingAI.git
cd Awesome-TradingAI
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install only the integrations you need:

```bash
python -m pip install -e ".[dashboard]"
python -m pip install -e ".[qlib,lgbm]"
python -m pip install -e ".[data-qlib]"
python -m pip install -e ".[finmem,finmem-live]"
python -m pip install -e ".[news,fingpt-live,runtime]"
```

## Five-minute offline run

```bash
quant-bench doctor
quant-bench quickstart --offline --workspace ./qb-workspace
quant-bench report --workspace ./qb-workspace
```

The quickstart uses bundled synthetic BTC and ETH OHLCV data. It validates the
data, builds causal features and labels, trains a NumPy model, runs a cost-aware
backtest, and writes a checksummed run under `qb-workspace/runs/`.

Run a small parameter sweep:

```bash
quant-bench sweep crypto_smoke_v1 \
  --param model.parameters.ridge=0.0,0.000001,0.001 \
  --study-name ridge-example \
  --workspace ./qb-workspace

quant-bench compare --workspace ./qb-workspace
```

## Local dashboard

```bash
python -m pip install -e ".[dashboard]"
quant-bench dashboard --workspace ./qb-workspace
```

The dashboard reads artifacts already present in the selected workspace. It
does not train models, call an exchange, read credentials, or control a trading
process. See [Inspect results](docs/how-to/results.md) and the
[dashboard design](docs/design/LOCAL_ANALYTICS_DASHBOARD.md).

## Methods and execution modes

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
