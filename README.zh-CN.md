# Awesome TradingAI

[![CI](https://github.com/Starlien95/Awesome-TradingAI/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Starlien95/Awesome-TradingAI/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Awesome TradingAI 是面向加密货币交易方法的开源 benchmark 和本地分析工具。
可安装的核心包名为 `quant-bench`，提供可复现回测、方法适配、版本化 artifact、
paper/runtime 安全门禁和只读 Streamlit 仪表盘。

[English](README.md) | [文档](docs/index.md) | [CLI 参考](docs/reference/cli.md)

## 公开模拟盘结果

[打开只读可视化网站](https://quant-bench-showcase.streamlit.app/)，可以查看 16 个公开运行、15 个方法身份的归一化收益曲线、BTC 买入并持有对照、回撤、信号、执行聚合、方法说明和数据质量状态。

当前公开数据全部来自 OKX demo，对应库中的 `simulated` 模式。它与本地 `paper` 账本和实盘交易是三个独立边界。网站只读取延迟、脱敏后的公共快照，不能查询账户或提交订单。所有时间序列从 `2026-06-01 00:00 UTC` 展示，agent 方法从 6 月 4 日的首个真实观测点开始，起点前不生成补线。

## 功能

| 范围 | 已实现功能 |
| --- | --- |
| Benchmark 核心 | 类型化配置、确定性运行、显式成本、指标、checksum、比较和报告 |
| 传统机器学习 | NumPy baseline、Qlib 0.9.7 workflow、模型目录、调参和已保存 artifact 适配 |
| 强化学习 | MacroHFT runtime adapter 和受信本地 TorchScript 转换边界 |
| FinGPT | 合成新闻 dry-run、情绪研究、回测、调参和带门禁的 paper/runtime adapter |
| FinMem | InvestorBench 研究集成、本地数据校验、paper 账本和服务 adapter |
| 统一方法接口 | 对内置和外部方法统一执行发现、检查、运行和状态查询 |
| 数据契约 | 版本化 market、prediction、return、metric、signal、trade 和 manifest 契约 |
| 本地分析 | equity、benchmark、alpha、drawdown、signal、execution、health 和数据质量页面 |

默认安装和离线 quickstart 不访问网络，不读取凭据，不查询账户，也不提交订单。

## 安装

支持 Python 3.10 至 3.12。

```bash
git clone https://github.com/Starlien95/Awesome-TradingAI.git
cd Awesome-TradingAI
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

按需安装扩展：

```bash
python -m pip install -e ".[dashboard]"
python -m pip install -e ".[qlib,lgbm]"
python -m pip install -e ".[data-qlib]"
python -m pip install -e ".[finmem,finmem-live]"
python -m pip install -e ".[news,fingpt-live,runtime]"
```

完整的 Python 与平台范围、可选依赖组合、外部服务、GPU 边界和离线安装要求见
[环境与依赖要求](docs/ENVIRONMENT.md)。项目以 `pyproject.toml` 作为权威依赖声明，
不使用会把全部重依赖强制安装给所有用户的根目录 `requirements.txt`。

## 五分钟离线运行

```bash
quant-bench doctor
quant-bench quickstart --offline --workspace ./qb-workspace
quant-bench report --workspace ./qb-workspace
```

quickstart 使用包内合成 BTC、ETH OHLCV 数据，依次完成数据校验、因果特征和标签生成、
NumPy 模型训练、显式成本回测，并在 `qb-workspace/runs/` 写入带 checksum 的结果。

运行一个小型参数 sweep：

```bash
quant-bench sweep crypto_smoke_v1 \
  --param model.parameters.ridge=0.0,0.000001,0.001 \
  --study-name ridge-example \
  --workspace ./qb-workspace

quant-bench compare --workspace ./qb-workspace
```

## 本地仪表盘

```bash
python -m pip install -e ".[dashboard]"
quant-bench dashboard --workspace ./qb-workspace
```

仪表盘只读取指定 workspace 中已有的 artifact。它不会训练模型、访问交易所、读取凭据或
控制交易进程。页面和数据边界见[结果分析](docs/how-to/results.md)与
[仪表盘设计](docs/design/LOCAL_ANALYTICS_DASHBOARD.md)。

## 方法和执行模式

```bash
quant-bench methods list
quant-bench methods show canonical:crypto_smoke_v1
quant-bench methods check canonical:crypto_smoke_v1 \
  --mode backtest \
  --workspace ./qb-workspace
```

| 模式 | 含义 |
| --- | --- |
| `backtest` | 不访问交易账户的历史或合成数据研究 |
| `paper` | 只使用本地账本，不向交易所提交订单 |
| `simulated` | 交易所模拟环境，需要显式网络、凭据和确认门禁 |
| `live` | 实盘环境，订单权限独立，并要求 `LIVE_ORDERS` 确认 |

每种方法支持的模式不同，运行前使用 `methods show` 和 `methods check` 核对。
外部 `ai_trade` 方法保持独立进程边界，需要用户提供经过审查的本地 checkout。

## 公开接口

- [Python API](docs/reference/python-api.md)
- [CLI](docs/reference/cli.md)
- [统一 Method 接口](docs/how-to/unified-methods.md)
- [模型 plugin](docs/how-to/add-model.md)
- [统一交易接口](docs/how-to/unified-trading-interface.md)
- [数据契约](docs/DATA_CONTRACT.md)
- [Runtime CSV schema](docs/runtime/CSV_SCHEMA.md)
- [Artifact 和 manifest](docs/reference/artifacts.md)
- [目录说明](docs/REPOSITORY_LAYOUT.md)
- [环境与依赖要求](docs/ENVIRONMENT.md)

仓库名为 `Awesome TradingAI`。为保持 API 兼容，distribution、import、CLI 和 plugin
namespace 继续使用 `quant-bench` 与 `quant_bench`。

## 目录

```text
src/quant_bench/    安装包、CLI、方法、runtime 和 dashboard
tests/              contract、unit 和 integration 测试
tools/              发布检查、workflow 维护和兼容工具
docs/               教程、指南、概念、参考和设计记录
release/            已校验的 SBOM 与依赖许可证元数据
.github/workflows/  测试、打包、文档和公开边界检查
```

运行数据应写入用户选择的 workspace。模型权重、交易所凭据、账户快照、订单、私有新闻、
Qlib 数据库和完整生产日志不进入本仓库。

## 安全边界

- 研究和仪表盘命令不会下单。
- 凭据只来自环境变量或未跟踪的本地 secret provider。
- 账户读取和订单写入使用独立权限。
- 模拟盘和实盘使用不同的精确确认 token。
- 本仓库不分发清仓、生产账本修复和临时下单 smoke 脚本。
- 公共 benchmark 比较要求 `protocol_id` 和 `dataset_sha256` 一致。

启用网络、账户或订单能力前先阅读[安全边界](docs/concepts/security-boundaries.md)。

## 开发和质量检查

以下命令用于准备贡献者环境，并运行源码、类型、测试、文档和发布边界检查。它们不会部署 Python 包、公开 Streamlit 网站或交易进程。

```bash
python -m pip install -e ".[dev,docs,dashboard]"
ruff check .
mypy --strict src/quant_bench
pytest -q
mkdocs build --strict
python tools/check_release.py
```

贡献、安全和第三方来源分别见 [CONTRIBUTING.md](CONTRIBUTING.md)、
[SECURITY.md](SECURITY.md) 和 [provenance register](docs/legal/PROVENANCE.md)。
项目采用 MIT。第三方部分保留其原许可证和版权声明。
