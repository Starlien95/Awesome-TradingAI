# Runtime modes

The runtime is optional. Complete offline research and artifact review before configuring an exchange process.

## Capability table

| Mode | Entry point | Account/network behavior |
| --- | --- | --- |
| Research | `quickstart`, `run`, `sweep` | Local files only |
| FinGPT dry-run | `runtime fingpt-dry-run` | Fixture news, static prices, local paper broker |
| OKX demo | `runtime start --mode demo` | Demo account queries and demo orders |
| OKX live | `runtime start --mode live` | Live account queries and live orders |

The traditional ML and MacroHFT process runner currently supports OKX demo and live modes. The local paper path is implemented for FinGPT. Do not label OKX demo execution as local paper trading.

## 1. Export configuration

```bash
quant-bench runtime list
quant-bench runtime init traditional_1h \
  --output-dir ./qb-workspace/local-config/traditional_1h
```

Packaged templates contain blank credentials, `api.is_simulated: true`, threshold 0.0, `risk_degree: 0.95`, small operational tolerances, and workspace-relative model paths. Edit only the exported copy.

## 2. Promote model artifacts

Promotion copies bytes without unpickling them, records SHA-256 provenance, and updates the selected local config:

```bash
quant-bench artifacts promote ./training-output/mlp_1h_model.pth \
  --model MLP_1h \
  --frequency 1h \
  --workspace ./qb-workspace \
  --config ./qb-workspace/local-config/traditional_1h/config_mlp.yaml
```

Run once with `--dry-run` to inspect paths. Existing targets require `--force`.

## 3. Set credentials in the environment

Demo mode reads:

```text
OKX_API_KEY_SIMU
OKX_SECRET_KEY_SIMU
OKX_PASSPHRASE
```

Live mode reads:

```text
OKX_API_KEY
OKX_SECRET_KEY
OKX_PASSPHRASE
```

Do not write values into YAML, examples, test output, shell history, or Git.

## 4. Check without network access

```bash
quant-bench runtime check traditional_1h \
  --config-dir ./qb-workspace/local-config/traditional_1h
```

This checks YAML parsing, mode, credential presence, and model paths. It does not contact OKX. `valid` describes configuration parsing. `ready` requires every credential and model file.

## 5. Start a demo process

```bash
quant-bench runtime start traditional_1h \
  --config-dir ./qb-workspace/local-config/traditional_1h \
  --workspace ./qb-workspace \
  --mode demo \
  --confirm DEMO_ORDERS
```

The exact token is required. Demo configuration must retain `api.is_simulated: true`.

Live mode requires `api.is_simulated: false` in every selected config and the exact token `LIVE_ORDERS`. Review symbol lists, account mode, model files, capital, threshold, top-k, dropout, minimum order value, tolerance, and environment variables before using it.

## FinGPT no-network check

```bash
quant-bench runtime fingpt-dry-run --workspace ./qb-workspace
```

The command moves packaged sample news into the current trade window, runs deterministic mock sentiment inference, writes signals and metrics, emits empty CSV contracts when there is no trade, and does not require OKX, CCData, transformers, torch, or a GPU.

## External ai_trade four-agent stack

`quant_bench` can act as a control plane for a separate `ai_trade` checkout.
The checkout remains the source of its deploy scripts, Python environment,
credentials, checkpoints, logs, and normalized events.

```bash
export AI_TRADE_ROOT=/srv/ai_trade

quant-bench runtime ai-trade list
quant-bench runtime ai-trade check
quant-bench runtime ai-trade status
quant-bench runtime ai-trade status --process tradingagents_live
```

The four child processes are:

| Process | Project | Model | Credential profile |
| --- | --- | --- | --- |
| `benchmark_deepseek` | benchmark | DeepSeek V4 Pro thinking | `zi1` |
| `benchmark_qwen` | benchmark | Qwen 3.6 Max thinking | `zi2` |
| `finagent_live` | FinAgent | DeepSeek V4 Pro thinking | `zi3` |
| `tradingagents_live` | TradingAgents | DeepSeek V4 Pro thinking | explicit `TRADINGAGENTS_OKX_*` |

`check` is offline. It checks file presence, the simulated-mode contract, and
whether each required environment-variable group has a non-empty value. It
never returns the value. The explicit TradingAgents variables are mandatory at
this boundary so that it cannot silently fall back to another process's
account.

The adopted stack currently supports only the reviewed OKX simulated contract:

```bash
quant-bench runtime ai-trade start --confirm SIMULATED_ORDERS
quant-bench runtime ai-trade stop --confirm STOP_LIVE_STACK
```

Start and stop operate on the complete stack because dependency preparation
and FinAgent data refresh are aggregate deploy steps in `ai_trade`. Use
`--process` with `list`, `check`, or `status` to select one child.

## High-risk operations

The public repository does not distribute account liquidation, production-ledger repair, or ad hoc order smoke programs. Runtime commands keep account reads and order writes behind separate flags and confirmation tokens.
