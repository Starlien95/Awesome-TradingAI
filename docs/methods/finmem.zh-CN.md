# FinMem 与 InvestorBench 使用指南

FinMem 已作为 `quant_bench.methods.finmem` 方法族接入统一 CLI。支持任意具备
InvestorBench JSON 数据的标的，支持单资产、多资产、warmup、test、checkpoint
续跑、eval、paper 账本、每日数据循环、调度器和只读 dashboard。

## 安装

```bash
pip install -e ".[finmem]"
```

需要实时行情、新闻和 OKX 适配时安装：

```bash
pip install -e ".[finmem,finmem-live]"
```

Dashboard 使用 `.[dashboard]`，Guardrails endpoint 使用
`.[finmem-guardrails]`。

## 数据校验

恢复的数据保存在本地发布边界外。先执行完整校验：

```bash
quant-bench finmem data verify \
  --data-dir local_data/finmem/investorbench \
  --full
```

15 个文件的 SHA-256、大小和日期范围记录在
[数据说明](../data/FINMEM_INVESTORBENCH_DATA.md)对应的机器可读 manifest 中。

## 创建工作区

单资产：

```bash
quant-bench finmem init \
  --workspace ./qb-workspace \
  --data-dir local_data/finmem/investorbench \
  --symbol BTC
```

多资产通过重复 `--symbol` 指定：

```bash
quant-bench finmem init \
  --workspace ./qb-workspace-multi \
  --data-dir local_data/finmem/investorbench \
  --symbol BTC --symbol ETH --symbol LINK
```

命令会计算公共日期交集，生成有效的 70/30 warmup/test 时间段，并写入可编辑的
`methods/finmem/configs/investorbench.json`。安装包内的模板不会被修改。

## 环境检查与实验流程

```bash
quant-bench finmem doctor \
  --workspace ./qb-workspace \
  --data-dir local_data/finmem/investorbench

CONFIG=./qb-workspace/methods/finmem/configs/investorbench.json
quant-bench finmem run warmup --config "$CONFIG" --allow-network
quant-bench finmem run test --config "$CONFIG" --allow-network
quant-bench finmem run eval --config "$CONFIG"
```

`doctor` 不访问网络。`warmup`、`test`、`warmup-checkpoint`、
`test-checkpoint` 会使用 LLM、embedding 和 Qdrant，所以必须显式传入
`--allow-network`。`eval` 只读取本地 checkpoint。

## 离线决策与 paper 账本

```bash
quant-bench finmem map-action \
  --input /tmp/finmem-action.json \
  --trade-mode swap \
  --notional-usdt 10000

quant-bench finmem paper-step \
  --state ./qb-workspace/methods/finmem/state/btc/paper.json \
  --price 50000 \
  --target-position -1 \
  --fee-bps 5
```

这两个命令不访问网络。paper 账本支持 long、flat、short，并使用原子写入。

## 每日循环与安全边界

默认 paper 循环只使用公开行情、新闻、LLM 和 Qdrant，不读取账户，不下单：

```bash
quant-bench finmem cycle \
  --workspace ./qb-workspace --symbol BTC --config "$CONFIG" \
  --mode paper --allow-network
```

OKX demo 账户只读检查：

```bash
quant-bench finmem cycle \
  --workspace ./qb-workspace --symbol BTC --config "$CONFIG" \
  --mode demo --allow-network --query-account
```

OKX demo 下单必须增加：

```text
--execute-orders --confirm DEMO_ORDERS
```

live 下单必须使用 `--mode live --confirm LIVE_ORDERS`。demo 与 live 的凭据变量、
调度命令、dashboard 命令和输出文件说明见[英文完整指南](finmem.md)。

所有 checkpoint、Qdrant state、账户快照、订单、原始新闻和运行日志都写入本地
workspace，不进入公开仓库或 Python 包。
