# Network access

## Commands

| Command or process | Network behavior |
| --- | --- |
| `quickstart`, `run`, `sweep`, `compare`, `report` | None |
| `data validate`, `data split`, `data qlib-dump` | None |
| `data pull`, `data pipeline` | Public OKX OHLCV through CCXT |
| `runtime check`, `artifacts promote`, `dashboard` | None |
| `runtime fingpt-dry-run` | None |
| `fingpt init`, `fingpt validate`, `fingpt backtest`, `fingpt paper-dry-run` | None |
| `finmem doctor`, `finmem data verify`, `finmem init`, `finmem map-action`, `finmem paper-step` | None |
| `finmem run warmup|test|*-checkpoint` | Configured LLM, embedding, and Qdrant endpoints; requires `--allow-network` |
| `finmem run eval` | None |
| `finmem cycle|schedule --mode paper` | Public OKX candles, configured RSS/Google News, LLM, embedding, and Qdrant; requires `--allow-network`; no account access |
| `finmem cycle|schedule --query-account` | Adds OKX demo/live account reads; requires demo/live mode |
| `finmem cycle|schedule --execute-orders` | Adds confirmed OKX order operations; requires `DEMO_ORDERS` or `LIVE_ORDERS` |
| FinGPT network paper mode | Configured CCData or RSS/Google News provider, model endpoint/model hub when configured, and unauthenticated OKX ticker; requires `--allow-network`; no account client |
| FinGPT OKX order mode | Adds account and order access; requires environment credentials, `--execute-orders`, and `DEMO_ORDERS` or `LIVE_ORDERS` |
| `runtime start` | OKX market, account, and order REST APIs |

## OKX runtime endpoints

The current execution core uses REST calls for candles, ticker, account balance, swap positions when configured, order submission, order status and fills, leverage in swap mode, and explicit close-position operations. It does not use a WebSocket client.

Market data is shared within a process. Account requests remain per strategy executor. Transient OKX codes and connection errors use bounded retries and account state is invalidated after an acknowledged order.

No network test runs in the default test suite. Public connectivity checks require an explicit network marker or command. Account smoke tests and liquidation tools are never invoked automatically.

FinGPT rejects inline YAML credentials. Its safe template uses paper mode,
uniform parameters, empty credential fields, and disabled account
reconciliation. Public-data access and order execution are separate flags.

FinMem keeps public-data access, account reads, and order writes as independent
capabilities. Enabling public data does not enable an account client. Paper
mode returns a disabled account snapshot and cannot submit orders.
