# License policy

## Project decision

The project license is MIT, effective for the first library release prepared on 2026-07-18. The copyright line is `Copyright (c) 2026 quant-bench contributors`.

MIT was selected because quant-bench is an integration and benchmark library intended for broad research and engineering reuse, and its distributed Qlib-derived portions already use MIT. The license is compatible with the permissive direct dependencies currently declared by the package. Dependency compatibility does not replace compliance with each dependency's notices and distribution conditions.

## Scope

Unless a file contains another notice, the root `LICENSE` covers:

- original Python source, tests, tools, configuration, and documentation;
- modifications made by quant-bench contributors to permissively licensed upstream source;
- bundled synthetic OHLCV and dry-run news fixtures.

The vendored `ai_trade_FinMem/INVESTOR-BENCH` subtree remains under its nested
MIT license. Its copyright, license, and provenance records take precedence for
that path. Guardrails validators are installed from their upstream packages and
are not vendored in this repository.

The project license does not relicense:

- third-party portions carrying retained copyright notices;
- installed Python dependencies;
- downloaded market or news data;
- external model weights, tokenizers, LoRA adapters, or checkpoints;
- user-generated run artifacts;
- source obtained from a repository with no explicit license.

## Contributions

Contributions are provided under MIT. A contributor must have the right to submit the work under those terms. Derived code must preserve the upstream notice and record the immutable source revision, license identifier, local path, and modification summary in `THIRD_PARTY_NOTICES.md` or `docs/legal/PROVENANCE.md`.

Code from a repository without an explicit compatible license cannot enter the source distribution. A paper or public GitHub repository alone does not grant redistribution rights.

## Models and data

Model and dataset licenses are evaluated separately from software licenses. A configurable model identifier does not grant access or redistribution rights. Public benchmark results must link to a dataset card and may include only data that is synthetic, explicitly redistributable, or sufficiently aggregated and authorized.

The Llama 2 and CCData terms remain external conditions. Quant-bench does not call either resource during its default installation, tests, or offline quickstart.

## Release checks

Before publishing a source archive or wheel:

1. Run `python tools/check_release.py`.
2. Run the test, lint, type, workflow, documentation, and wheel-install checks.
3. Build in a clean environment.
4. Generate the CycloneDX SBOM and dependency-license report for that environment.
5. Confirm the archive does not contain ignored model, data, credential, account, or runtime files.
6. Review every `LicenseRef`, `UNKNOWN`, copyleft, custom model, and data-source term manually.

This document is an engineering compliance record. Project owners should obtain legal review when organizational ownership, contributor authorization, patents, trained-model terms, or commercial data rights require it.
