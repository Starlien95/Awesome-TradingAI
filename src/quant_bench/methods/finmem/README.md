# FinMem

FinMem is integrated as a first-class quant-bench method family. The public API
keeps optional dependencies lazy, while the `quant-bench finmem` commands cover
data validation, workspace creation, InvestorBench warmup/test/evaluation,
paper-ledger checks, live-data cycles, scheduling, and dashboards.

Start with the [English guide](../../../../docs/methods/finmem.md) or the
[中文指南](../../../../docs/methods/finmem.zh-CN.md). InvestorBench-derived code
and its original license are retained under `investorbench/`.

Safety defaults:

- importing `quant_bench.methods.finmem` performs no network access;
- `doctor`, `data verify`, `map-action`, and `paper-step` are offline;
- LLM, embedding, Qdrant, news, and public market calls require
  `--allow-network`;
- account reads require `--query-account` plus `mode=demo|live`;
- orders require `--execute-orders` and an exact confirmation token.
