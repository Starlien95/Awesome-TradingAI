# Unified Method interface

The `methods` command is the preferred control plane for selecting an
implementation and an execution mode. Existing `run`, `qlib`, `fingpt`,
`finmem`, and `runtime` commands remain compatibility interfaces.

## Discover capabilities

```bash
quant-bench methods list
quant-bench methods list --mode backtest
quant-bench methods list --mode simulated
quant-bench methods list --mode live
quant-bench methods show finagent
```

Each Method declares:

- supported modes;
- frequencies, including mode-specific frequencies;
- network, account, and order capabilities;
- whether order execution is optional or required;
- whether historical execution can resume;
- the adapter that owns its implementation.

The current catalog covers:

- canonical `ModelPlugin` experiments;
- Qlib workflows;
- FinGPT news;
- FinMem/InvestorBench;
- all packaged traditional-ML and MacroHFT runtime processes;
- ai_trade benchmark DeepSeek, benchmark Qwen, FinAgent, and TradingAgents.

FinAgent intentionally declares `backtest=1d` and
`paper/simulated/live=4h`. Its existing historical engine is daily; the
unified interface does not relabel it as a 4H backtest.

## Check before running

`check` performs local validation and prints a redacted command preview. It
does not access the network, query an account, or place an order.

```bash
quant-bench methods check tradingagents \
  --mode backtest \
  --repo /srv/ai_trade \
  --workspace ./qb-workspace \
  --start 2025-01-01 \
  --end 2025-07-01 \
  --allow-network
```

Method-specific values use repeatable `--param KEY=VALUE` arguments. JSON
values are decoded; other values remain strings.

## Run a backtest

```bash
quant-bench methods run canonical \
  --mode backtest \
  --workspace ./qb-workspace \
  --param recipe=crypto_smoke_v1

quant-bench methods run benchmark_deepseek \
  --mode backtest \
  --repo /srv/ai_trade \
  --workspace ./qb-workspace \
  --start "2025-01-01 00:00:00" \
  --end "2025-07-01 00:00:00" \
  --resume \
  --allow-network
```

`--resume` reuses the previous unified run's `native_run_dir`. Use
`--resume-from /path/to/native-run` to select an older checkpoint explicitly.
The command is rejected when no valid native resume source exists.

Every invocation writes:

```text
<workspace>/method_runs/<method>/<run-id>/
  run_spec.json
  method_run.json
  checksums.sha256
  native/...
```

`native/` preserves the Method's own artifacts. The unified record references
those artifacts instead of rewriting their scientific contents. A
`latest.json` pointer record is maintained per Method.

## Simulated and live modes

Order-capable Methods require all of the following:

- the Method declares the selected mode;
- `--allow-network`;
- `--execute-orders`;
- the exact mode token:
  - simulated: `SIMULATED_ORDERS`;
  - live: `LIVE_ORDERS`.

Example single-cycle simulated check:

```bash
quant-bench methods check benchmark_deepseek \
  --mode simulated \
  --repo /srv/ai_trade \
  --workspace ./qb-workspace \
  --once \
  --allow-network \
  --execute-orders \
  --confirm SIMULATED_ORDERS
```

Replace `check` with `run` only after reviewing the command and account
profile. `--once` exits after at most one eligible cycle. Without `--once`,
runtime Methods may remain attached as long-running processes.

## ai_trade credentials

Backtests require only the selected LLM credential. Simulated execution uses
the existing ai_trade profiles (`ZI1`, `ZI2`, `ZI3`, and the explicit
TradingAgents profile).

Live execution deliberately does not reuse those simulated profiles. Configure
three dedicated variables per Method:

```text
AI_TRADE_BENCHMARK_DEEPSEEK_LIVE_API_KEY
AI_TRADE_BENCHMARK_DEEPSEEK_LIVE_SECRET_KEY
AI_TRADE_BENCHMARK_DEEPSEEK_LIVE_PASSPHRASE

AI_TRADE_BENCHMARK_QWEN_LIVE_API_KEY
AI_TRADE_BENCHMARK_QWEN_LIVE_SECRET_KEY
AI_TRADE_BENCHMARK_QWEN_LIVE_PASSPHRASE

AI_TRADE_FINAGENT_LIVE_API_KEY
AI_TRADE_FINAGENT_LIVE_SECRET_KEY
AI_TRADE_FINAGENT_LIVE_PASSPHRASE

AI_TRADE_TRADINGAGENTS_LIVE_API_KEY
AI_TRADE_TRADINGAGENTS_LIVE_SECRET_KEY
AI_TRADE_TRADINGAGENTS_LIVE_PASSPHRASE
```

The adapter maps only the selected Method's dedicated variables into the
external process environment. Values are never included in `check`, run
records, or command previews.

## Python interface

```python
from pathlib import Path

from quant_bench import ExecutionMode, MethodRunSpec, get_method_runner

runner = get_method_runner()
spec = MethodRunSpec(
    method_id="tradingagents",
    mode=ExecutionMode.BACKTEST,
    workspace=Path("./qb-workspace"),
    repo=Path("/srv/ai_trade"),
    start="2025-01-01",
    end="2025-07-01",
    allow_network=True,
)

readiness = runner.check(spec)
if readiness.ready:
    result = runner.run(spec)
```
