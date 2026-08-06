# Security and data boundaries

## Default boundary

Importing `quant_bench` does not initialize Qlib, PyTorch, CCXT, Streamlit, transformers, or OKX. The offline quickstart loads package data and writes to an explicit workspace.

## Network boundary

Network access occurs only in explicitly selected capabilities:

- `data pull` and `data pipeline` use public OKX market data through CCXT;
- FinGPT production news uses the configured news provider;
- runtime `start` uses OKX account and order APIs;
- account liquidation, production-ledger repair, and ad hoc order smoke programs are not distributed.

`doctor`, `data validate`, `runtime check`, artifact commands, reports, and the dashboard use local state only.

## Credential boundary

Credentials come from environment variables or untracked local secret management. Config templates store environment variable names and blank values. Diagnostics report presence as booleans and never return the secret.

## Public-data boundary

Production runs, orders, fills, account snapshots, raw news, model weights, caches, and unredacted logs remain private. A public result snapshot contains bounded anonymized metrics, signals, manifests, freshness information, schema versions, and limitations. It must not contain account ids, exchange order ids, credential values, unreleased signals, or raw production data.

## Artifact boundary

Public loaders do not unpickle arbitrary files. Runtime promotion copies opaque bytes and records checksums. Loading a pickle remains an explicit trusted-local operation inside a selected backend.
