# Third-party notices

Quant-bench is licensed under the MIT License. That license applies to original quant-bench material and modifications whose files do not state different terms. Dependencies, external models, downloaded data, and third-party portions retain their respective terms.

## Distributed third-party source

| Component | Upstream source and revision | License | Local use and modifications |
| --- | --- | --- | --- |
| Microsoft Qlib | `microsoft/qlib`, tag `v0.9.7`, commit `da920b7f954f48ab1bb64117c976710de198373e` | MIT, full text in `LICENSES/QLIB-MIT.txt` | The Qlib bin dumper in `src/quant_bench/data/qlib_pipeline.py` is based on `scripts/dump_bin.py`. Quant-bench added lazy optional imports, CCXT/OKX acquisition, crypto frequency mapping, bounded retries, canonical CSV fields, typical-price proxy metadata, and CLI integration. |
| Microsoft Qlib TFT example | Same Qlib revision | MIT, full text in `LICENSES/QLIB-MIT.txt` | `src/quant_bench/integrations/qlib/models/tft.py` retains Qlib model/dataset integration concepts and has been substantially modified for the local PyTorch crypto workflow. The file carries the upstream copyright notice. |
| InvestorBench | `felis33/INVESTOR-BENCH`, baseline commit `3509aeca668824cf2970e3950781d4d09f6c2adb` | MIT, full text retained at `src/quant_bench/methods/finmem/investorbench/LICENSE` | The modified framework is integrated under `quant_bench.methods.finmem.investorbench` with user-owned workspace configs, lazy optional endpoints, CLI orchestration, plotting, and multi-asset support. Market/news JSON contents remain local-only. Files ending in `.pkl` under its `configs/` directory are source files for the Pkl configuration language, not Python pickle objects. |
| Guardrails AI Valid Choices | `guardrails-ai/valid_choice`, audited commit `02593feb2506db3e4290134076bf50c5ecc9920a` | Apache-2.0, installed separately by the user | No source snapshot is distributed. The default FinMem path does not import Guardrails. |

The Qlib workflow YAML files are modified benchmark configurations derived from Qlib examples and use the same Qlib MIT attribution.

## Referenced projects whose source is not redistributed

| Component | Audited source | Terms and repository policy |
| --- | --- | --- |
| FinGPT | `AI4Finance-Foundation/FinGPT`, audited commit `e5e516470e7a25ed3690889b7b54d9946dd17520`, MIT | Quant-bench's scheduler, news store, paper broker, attribution, label-logprob service, and research engine are local integration code. The research behavior was consolidated from local quant-bench ETH and ten-coin experiments at baseline `6d9ed3cae5382d4ca1d8c910aa287cfcf79162f0`. FinGPT is credited as the method and model ecosystem. Its MIT text is retained in `LICENSES/FINGPT-MIT.txt`. No upstream weights or datasets are bundled. |
| FinMem | `pipiku915/FinMem-LLM-StockTrading`, audited HEAD `be814aa47970de9bf2fdd6a1d5a60ae5cf361b46`, MIT | FinMem is credited as the method underlying the integrated live workflow and InvestorBench lineage. Quant-bench does not redistribute the separate FinMem repository, model output, checkpoints, or datasets. |
| MacroHFT | `ZONG0004/MacroHFT`, audited commit `31e5ef41f93b2aea6e63e6ae2267675c997a8e54` | The upstream repository contained no LICENSE. Quant-bench does not distribute its network implementation, Python source, checkpoints, feature arrays, or data. The runtime accepts a user-created TorchScript bundle. `tools/convert_macrohft_external.py` imports a user-supplied source tree only after the explicit `TRUSTED_MACROHFT_SOURCE` confirmation. |
| Llama 2 13B | `NousResearch/Llama-2-13b-hf` | External model governed by the Meta Llama 2 Community License. Users must obtain and accept the model terms. Quant-bench contains only a configurable model identifier and does not bundle weights or tokenizer files. |
| FinGPT/LoRA adapters | User-selected external artifact | Adapter terms depend on the selected Hugging Face repository and base model. Quant-bench does not bundle adapters. Users must record the source revision and license in their run/model manifest. |
| CCData/CryptoCompare news | CryptoCompare API Licence Agreement | The client downloads data for the user's local run. Raw news, derived private artifacts, and API keys are excluded from source distributions and public snapshots. Redistribution requires separate authorization under the applicable API agreement. |
| OKX market/account APIs | OKX API and SDK terms | Quant-bench distributes integration code and configuration templates only. Credentials, account data, orders, and fills are not distributed. |

## Bundled fixtures

`src/quant_bench/resources/fixtures/crypto_ohlcv_smoke.csv`, the FinGPT dry-run news records, and `src/quant_bench/methods/fingpt_news/resources/research_fixture/` are synthetic test data created for quant-bench. They contain no exchange observations, news articles, account data, or personal data and are distributed under the project MIT License.

## Python dependencies

Python packages are installed separately by the user or package installer and are not vendored into the quant-bench wheel. Their direct-license inventory is maintained in `docs/legal/DEPENDENCY_LICENSES.md`. Release-environment transitive packages are recorded in:

- `release/sbom.cdx.json`
- `release/dependency-licenses.json`

## Release boundary

The following are excluded from source and binary releases:

- model weights, LoRA adapters, TorchScript bundles, pickle files, and checkpoints;
- raw or derived CCData news content;
- InvestorBench and FinMem market/news JSON contents; only a checksum manifest and data card are distributed;
- full exchange datasets and Qlib bin stores;
- account snapshots, credentials, orders, fills, logs, and production run state;
- any MacroHFT upstream source or checkpoint.

See `docs/legal/PROVENANCE.md` for the audit evidence and `docs/legal/LICENSE_POLICY.md` for contribution and release rules.
