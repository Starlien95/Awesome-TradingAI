# Migrating from `ai_trade_FinMem`

The standalone `ai_trade_FinMem` tree was consolidated into the installable
package. Existing capabilities map as follows:

| Previous entry | Current entry |
| --- | --- |
| `INVESTOR-BENCH/run.py warmup|test|eval` | `quant-bench finmem run ...` |
| `prepare_live_warmup.py` | generated workspace config plus `finmem run warmup` |
| `investor_live_test_runner.py` | `quant_bench.methods.finmem.live.investor_runner` |
| `run_one_live_cycle.py` / `_eth.py` | `quant-bench finmem cycle --symbol SYMBOL` |
| `run_live_loop.py` / `_eth.py` | `quant-bench finmem schedule --symbol SYMBOL` |
| BTC/ETH dashboard loggers | one symbol-configured `live.dashboard_log` |
| multi-asset dashboard script | `quant-bench finmem dashboard` |
| `test_swap_short_stage_b.py` | removed from the public repository; validate demo orders in a separately reviewed operator environment |
| nested `.env` | user-owned `--env-file` based on `finmem.env.example` |

Behavioral changes are intentional:

- installed configs are read-only templates; commands write runtime configs to
  the workspace;
- paper mode does not query OKX accounts;
- account reads and orders require separate flags;
- demo and live orders use exact confirmation tokens;
- JSON/CSV/state writes use atomic replacement;
- BTC and ETH use the same implementation, and any dataset symbol is accepted;
- generated logs include a method-family `run_manifest.json`.

The restored dataset files remain available in the local-data area described
in `docs/data/FINMEM_INVESTORBENCH_DATA.md`. Their machine-independent
checksums are distributed with the package.
