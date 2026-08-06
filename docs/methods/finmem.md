# FinMem and InvestorBench

The FinMem integration provides a complete method workflow without requiring
users to edit files inside an installed package. It supports every symbol with
an InvestorBench-format JSON dataset, single-asset and multi-asset backtests,
checkpoint resume, deterministic paper accounting, daily live-data cycles,
OKX demo/live execution gates, and a read-only Streamlit dashboard.

## Install

Install the research engine:

```bash
pip install -e ".[finmem]"
```

Add live market/news and OKX adapters only when needed:

```bash
pip install -e ".[finmem,finmem-live]"
```

The dashboard and optional Guardrails endpoints have separate extras:

```bash
pip install -e ".[finmem,dashboard]"
pip install -e ".[finmem,finmem-guardrails]"
```

## 1. Verify local data

InvestorBench datasets use this daily JSON contract:

```json
{
  "2023-01-01": {
    "prices": 16625.08,
    "news": ["A news summary available on that date"]
  }
}
```

Verify checksums and date ranges without network access:

```bash
quant-bench finmem data verify \
  --data-dir local_data/finmem/investorbench \
  --full
```

Use `--symbol BTC --symbol ETH` to check only selected assets. The recovered
files and their release boundary are documented in
[the data card](../data/FINMEM_INVESTORBENCH_DATA.md).

## 2. Create an editable workspace

```bash
quant-bench finmem init \
  --workspace ./qb-workspace \
  --data-dir local_data/finmem/investorbench \
  --symbol BTC
```

For a multi-asset run, repeat `--symbol`:

```bash
quant-bench finmem init \
  --workspace ./qb-workspace-multi \
  --data-dir local_data/finmem/investorbench \
  --symbol BTC \
  --symbol ETH \
  --symbol LINK
```

The command computes a valid common date range, uses a 70/30 warmup/test
split, and writes:

- `methods/finmem/configs/investorbench.json`
- `methods/finmem/finmem.env.example`
- `methods/finmem/workspace.json`

Package resources remain unchanged. Review the generated config before using
any service.

## 3. Check the environment

```bash
quant-bench finmem doctor \
  --workspace ./qb-workspace \
  --data-dir local_data/finmem/investorbench
```

`doctor` checks paths, optional imports, and whether credential variables are
present. It does not contact Qdrant, an LLM endpoint, OKX, or any news source.

## 4. Run InvestorBench phases

Start Qdrant and an OpenAI-compatible chat endpoint, then export the required
LLM and embedding key. The generated config defaults to local vLLM at
`http://127.0.0.1:8000`, Qdrant at `http://127.0.0.1:6333`, and the
`text-embedding-3-small` API.

```bash
CONFIG=./qb-workspace/methods/finmem/configs/investorbench.json

quant-bench finmem run warmup --config "$CONFIG" --allow-network
quant-bench finmem run test --config "$CONFIG" --allow-network
quant-bench finmem run eval --config "$CONFIG"
```

Resume interrupted phases with `warmup-checkpoint` or `test-checkpoint`.
`warmup`, `test`, and the resume commands require `--allow-network` because
they call LLM, embedding, and Qdrant services. `eval` reads local checkpoints.

## 5. Test decision and accounting logic offline

Map an InvestorBench action to an execution-neutral decision:

```bash
cat > /tmp/finmem-action.json <<'JSON'
{"symbol": "BTC", "date": "2025-01-01", "position": -1}
JSON

quant-bench finmem map-action \
  --input /tmp/finmem-action.json \
  --trade-mode swap \
  --notional-usdt 10000
```

Exercise the long/flat/short paper ledger:

```bash
quant-bench finmem paper-step \
  --state ./qb-workspace/methods/finmem/state/btc/paper.json \
  --price 50000 \
  --target-position 1 \
  --fee-bps 5
```

Both commands are offline. The paper ledger writes atomically and applies the
previous position to the next observed return before changing the target.

## 6. Run a daily live-data cycle

A cycle reads public OKX candles, collects configured RSS/Google News sources,
runs the prepared InvestorBench agent, and updates the local paper ledger.
Paper mode does not query an account or submit an order:

```bash
quant-bench finmem cycle \
  --workspace ./qb-workspace \
  --symbol BTC \
  --config "$CONFIG" \
  --mode paper \
  --allow-network
```

Account reads and order writes are separate capabilities:

```bash
# OKX demo account read, no order
quant-bench finmem cycle \
  --workspace ./qb-workspace --symbol BTC --config "$CONFIG" \
  --mode demo --allow-network --query-account

# OKX demo order
quant-bench finmem cycle \
  --workspace ./qb-workspace --symbol BTC --config "$CONFIG" \
  --mode demo --allow-network --query-account --execute-orders \
  --confirm DEMO_ORDERS
```

Live orders require `--mode live --confirm LIVE_ORDERS` and the standard
`OKX_API_KEY`, `OKX_SECRET_KEY`, and `OKX_PASSPHRASE` variables. Demo orders
use `OKX_API_KEY_SIMU`, `OKX_SECRET_KEY_SIMU`, and `OKX_PASSPHRASE`.

Use `finmem.env.example` as a local template and pass it with `--env-file`.
Keep the completed env file outside Git.

## 7. Scheduler and dashboard

The same options work with the daily scheduler:

```bash
quant-bench finmem schedule \
  --workspace ./qb-workspace --symbol BTC --config "$CONFIG" \
  --mode paper --allow-network
```

BTC defaults to 00:05 and ETH to 00:15 in `Asia/Shanghai`. Override the minute
with `FINMEM_DAILY_RUN_MINUTE`.

Launch the read-only multi-asset dashboard:

```bash
quant-bench finmem dashboard --workspace ./qb-workspace
```

The dashboard discovers all symbol-scoped logs. Logger output includes
metrics, signals, trades, volume, metadata, and `run_manifest.json`.

## Architecture

| Path | Responsibility |
| --- | --- |
| `methods/finmem/config.py` | mode, path, and confirmation validation |
| `methods/finmem/data.py` | dataset contract and checksum verification |
| `methods/finmem/workspace.py` | editable config/workspace generation |
| `methods/finmem/paper.py` | atomic long/flat/short paper ledger |
| `methods/finmem/investorbench/` | modified InvestorBench engine and license |
| `methods/finmem/live/` | optional data, news, cycle, scheduler, OKX, and dashboard modules |

Generated checkpoints, Qdrant state, account snapshots, orders, raw news, and
cycle logs stay under the local workspace. They are excluded from the public
repository and package.
