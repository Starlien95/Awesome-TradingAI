# Provenance register

Audit date: 2026-07-18

## Distributed code

| Local material | Provenance evidence | Decision |
| --- | --- | --- |
| `src/quant_bench/data/qlib_pipeline.py` | File comparison against Qlib v0.9.7 `scripts/dump_bin.py`; Qlib commit `da920b7f954f48ab1bb64117c976710de198373e`; upstream file and repository carry Microsoft MIT notices | Retained with Microsoft copyright, Qlib MIT text, and modification notice |
| `src/quant_bench/integrations/qlib/models/tft.py` | Comparison against Qlib v0.9.7 `examples/benchmarks/TFT/tft.py`; migrated local source first recorded in quant-bench commit `81fc447aa0f169112c0e246242c466de9d15e6b0` | Retained as an MIT Qlib-derived and substantially modified implementation with upstream notice |
| Qlib workflow YAML resources | Migrated from the same Qlib-oriented local history and normalized in quant-bench | Retained under the project MIT license with Qlib attribution |
| FinGPT runtime under `src/quant_bench/methods/fingpt_news` | Local history from quant-bench commits beginning with `cdd1d7c9f0896aff886305a92d40ae9f31e0b693`; comparison against FinGPT audit commit found no matching local service/function names | Retained as local integration code; method attribution and upstream MIT text included |
| FinGPT research engine under `src/quant_bench/methods/fingpt_news/research` | Consolidated from local quant-bench `fingpt_crypto_repro` ETH and ten-coin experiment semantics at source baseline `6d9ed3cae5382d4ca1d8c910aa287cfcf79162f0`; implementation was rewritten around typed configuration, common contracts, causal lag, atomic artifacts, and validation-only selection | Retained as project MIT code; raw news, model weights, adapters, and private experiment outputs are excluded |
| `src/quant_bench/methods/finmem/investorbench` | File comparison against `felis33/INVESTOR-BENCH` commit `3509aeca668824cf2970e3950781d4d09f6c2adb`; nested MIT license names Haohang Li, Yupeng Cao, and Yangyang Yu | Retained as a modified and package-integrated MIT subtree with its original license; market/news JSON contents remain local-only |
| Canonical benchmark library, contracts, CLI, tests, and docs | Added during the 2026-07-18 package refactor in this worktree | Project MIT license |

## Removed or externalized code

| Material | Evidence | Action |
| --- | --- | --- |
| Guardrails AI Valid Choices source snapshot | Compared with `guardrails-ai/valid_choice` commit `02593feb2506db3e4290134076bf50c5ecc9920a`; the local snapshot was unused by package imports | Removed from the repository. Users install the optional upstream validator under its Apache-2.0 terms. |
| Inline MacroHFT `_Subagent`, `_Hyperagent`, and Q-combination implementation | Quant-bench history begins at commit `e4f2d2f392e2a58bc6ac49f8ae37af260bdabae1`; comparison shows derivation from `ZONG0004/MacroHFT` `model/net.py`; upstream audit commit `31e5ef41f93b2aea6e63e6ae2267675c997a8e54` contains no LICENSE | Removed from the public package. Replaced by a TorchScript-only loader and trusted-local conversion tool that imports user-supplied source. |
| MacroHFT checkpoints, feature arrays, and datasets | Present in the upstream repository without an accompanying software/data license | Never bundled or copied into quant-bench |
| Llama 2 base model and tokenizer | Model card requires acceptance of Meta's Llama 2 terms | Kept external; only the model identifier is configurable |
| FinGPT/LoRA weights | Terms vary by model repository and base model | Kept external; users record source and terms locally |
| CCData/CryptoCompare raw news | Governed by the CryptoCompare API Licence Agreement | Local runtime data only; excluded from Git, wheel, source archive, and public snapshot |
| InvestorBench and locally extended JSON data | Upstream/local files combine prices, news text, and generated sentiment; this repository did not record source-specific redistribution grants | Restored byte-for-byte to a local-only data area and a separate recovery copy. The public package distributes checksums and a data card, not JSON contents. |

## Bundled data

The OHLCV fixture, FinGPT dry-run news, and FinGPT research price/sentiment
fixtures are deterministic synthetic records created for integration testing.
Their values do not reproduce observations from OKX, CCData, an account, or a
news article. They are released under MIT with the surrounding test software.

## Audit method

The audit used Git history, immutable upstream revisions, direct file comparison, official repository license files, official model cards, and API terms. Package-license metadata was checked against PyPI on the audit date. Search results and package metadata are evidence inputs; the full upstream license text and immutable revision remain authoritative.
