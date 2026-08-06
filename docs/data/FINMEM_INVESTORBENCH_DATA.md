# FinMem and InvestorBench local data card

## Status

Fifteen InvestorBench-format JSON files were restored byte-for-byte from the
former source repository's `origin/opensource` tree. Their SHA-256 hashes,
sizes, and date ranges are in
`src/quant_bench/methods/finmem/resources/investorbench-data-manifest.json`.

The working copy is stored under `local_data/finmem/investorbench/`. A second
recovery copy is maintained outside the repository. The Python wheel, source
release, and public Git boundary exclude the JSON contents.

## Why the files are local-only

The files combine prices, news text, and locally derived fields. Repository
history does not contain source-specific redistribution grants for every field.
InvestorBench's MIT software license covers its code and does not establish
rights for all third-party market or news content. Preserving the files locally
and publishing only checksums keeps the restored research inputs available
without asserting an unverified public-data license.

## Contract

Each file is a JSON object keyed by ISO date. Every daily item requires a
numeric `prices` field and accepts a list or null `news` field. Stock records may
also contain `10k` and `10q` lists. The loader validates only information in the
selected date window and never downloads data implicitly.

## Verification

```bash
quant-bench finmem data verify \
  --data-dir local_data/finmem/investorbench \
  --full
```

The command fails on a missing file, checksum mismatch, size mismatch, invalid
date, nonnumeric price, or invalid news field. It performs no network access.

## Public replacement

A future public dataset release needs a per-field source register, terms for
market observations and news, a documented transformation pipeline, and a
license compatible with redistribution. Until that audit is complete, examples
and automated tests use synthetic fixtures.
