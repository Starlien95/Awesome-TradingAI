# Contributing

Contributions are accepted under the repository's MIT License. By submitting a contribution, you represent that you have the right to provide it under those terms. Preserve upstream copyright and license notices, and do not copy code from a repository that lacks an explicit compatible license.

## Development environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the local quality gates before opening a change:

```bash
ruff check src tests tools
mypy --strict src/quant_bench
pytest
python -m build
python tools/check_wheel.py dist/quant_bench-*.whl
python tools/check_sdist.py dist/quant_bench-*.tar.gz
python tools/check_release.py
```

The repository CI runs the core test suite on Python 3.10, 3.11, and 3.12,
builds documentation, audits public-release boundaries, installs the wheel in
a clean environment, exercises offline FinGPT workflows, and checks Qlib 0.9.7
contracts. It contains no package-publishing credentials or publish job.

## Adding a model

1. Implement the public `ModelPlugin` contract or use `QlibModelPlugin` with a catalog entry.
2. Declare task types, feature shapes, dataset kinds, optional dependencies, metadata requirements, determinism, and artifact formats.
3. Add a minimal typed config and contract tests.
4. Add a model card with intended use, training resources, limitations, and license provenance.
5. Validate the plugin from an installed wheel outside the source directory.

Generate a starting directory with:

```bash
quant-bench plugins scaffold my_model --target ./plugins
```

## Adding data or features

- Data must use UTC timestamps and the canonical OHLCV schema.
- Add a `DatasetManifest`, data card, source terms URL, redistribution status, and checksum.
- Feature sets must declare input fields, lookback, output order, missing-value rules, and a schema hash.
- Label changes require a new label id and leakage tests.
- Do not commit production data, exchange identifiers, raw account records, or restricted news text.

## Benchmark results

Results are comparable only when protocol id, dataset hash, universe, split, cost model, metric version, and seed policy match. Include failed seeds, drawdowns, missing-data intervals, and sample counts.

## Pull requests

Every pull request should state:

- Changed public contracts and compatibility impact.
- Tests and exact commands run.
- Network, credential, account, or order impact.
- New dependencies and licenses.
- Generated artifacts that must stay outside Git.
- Source URL, immutable revision, license identifier, and modification summary for any derived code.

Do not mix unrelated runtime logs or model artifacts into a source change.

Open changes against the `main` branch of
`https://github.com/Starlien95/Awesome-TradingAI`. Keep generated workspaces,
credentials, model weights and account records outside the checkout.
