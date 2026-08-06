# Changelog

All notable changes use semantic versioning once the first public release is published.

## 0.1.0 - Unreleased

### Added

- Awesome TradingAI repository identity, concise English and Chinese entry
  documentation, Python API reference, repository layout and local analytics
  dashboard.

- Installable `quant_bench` package and `quant-bench` CLI.
- Typed experiment, dataset, artifact, and run-manifest contracts.
- Offline synthetic quickstart and deterministic NumPy baseline.
- Complete Qlib 0.9.7 model catalog and generic Qlib model adapter.
- Versioned six-feature and 52-feature crypto contracts.
- Atomic local artifact store and SHA-256 verification.
- Shared threshold/top-k/dropout strategy semantics.
- Workflow audit and normalization tools.
- English and Simplified Chinese READMEs.
- MIT project license, retained Qlib and FinGPT license texts, third-party
  notices, an audited provenance register, and a direct-dependency license
  inventory.
- Reproducible CycloneDX SBOM and dependency-license report generation with
  checksum verification.
- Unified FinGPT research commands with typed input contracts, causal signal
  lag, validation-only parameter selection, four asset/market signal variants,
  guarded per-symbol thresholds, classification diagnostics, and canonical
  benchmark artifacts.
- Synthetic FinGPT research fixtures, optional RSS/Google News collection, and
  a method capability matrix covering analysis and debugging interfaces.
- Python 3.10 through 3.12 CI, strict static and documentation checks, a Qlib
  0.9.7 contract job, and clean-installed wheel smoke coverage.
- Explicit wheel and source-distribution content audits for unsafe paths,
  credentials, private data, runtime state, model artifacts, licenses, and
  packaged workflow completeness.
- External `ai_trade` four-agent 4H catalog with offline readiness checks,
  normalized-event status, explicit account-profile isolation, and guarded
  simulated-stack delegation.
- Versioned unified Method interface and CLI across canonical, Qlib, FinGPT,
  FinMem, packaged runtime, and external ai_trade adapters. The interface
  declares mode-specific frequencies, centralizes network/account/order
  safety gates, supports native checkpoint resume, and preserves native
  artifacts behind checksummed Method Run records.
- Versioned trading contracts for order requests, account snapshots, execution
  reports, and complete execution cycles. Traditional runtime, FinGPT, and
  FinMem paper/demo/live paths now use the same trading service while retaining
  their native state and audit payloads.

### Changed

- Migrated repository metadata and CI branch targeting from the former
  `jianyingzhihe/quant_bench` development repository to
  `Starlien95/Awesome-TradingAI` while preserving the `quant-bench`
  distribution and `quant_bench` API namespace.
- Removed unreferenced production repair, liquidation and ad hoc order smoke
  programs from the public tree.
- Removed an unused vendored Guardrails Valid Choices source snapshot; users
  install the optional upstream validator under its own license.

- Normalized all 202 legacy workflows to public integration defaults.
- Replaced legacy handler and strategy implementations with compatibility aliases.
- Documented the OHLCV `vwap` compatibility field as a typical-price proxy.
- Added bounded retries, request timeout, and UTC timestamps to the legacy CCXT downloader.
- Split the former monolithic dependency list into `pyproject.toml` optional extras.
- Replaced the inline MacroHFT network implementation with a loader for
  trusted-local TorchScript exports. The upstream source and checkpoints stay
  outside the public distribution because its audited repository has no
  software license.
- Changed the FinGPT public runtime recipe to uniform general starting values;
  the dated experiment recipe remains available only as an opt-in historical
  configuration.
- Qlib bin creation now auto-detects canonical `timestamp` or legacy `date`,
  normalizes timestamps to UTC, and rejects empty calendar, instrument, or
  feature output.
- Intraday Qlib workflows now bind the simulator to their own frequency and
  precompute benchmark returns at that frequency, removing hidden daily and
  one-minute provider requirements.

### Security

- Public model loading rejects arbitrary Qlib pickle artifacts.
- Offline quickstart performs no network access and reads no credentials.
- Trading runtime commands remain capability-gated outside the research runner.
- Release policy checks reject bundled model artifacts, unresolved provenance
  markers, copied MacroHFT implementation symbols, developer absolute paths,
  and stale release metadata checksums.
- Release policy checks also reject automated PyPI or GitHub publishing from
  the repository CI; publishing remains a separate maintainer decision.
- FinGPT paper mode no longer creates an account executor. Network access,
  order writes, and account reconciliation have separate opt-in gates; inline
  YAML credentials are rejected.
