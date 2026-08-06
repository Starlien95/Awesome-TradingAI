# Release compliance artifacts

This directory contains machine-readable legal metadata generated from a clean
base installation of the release candidate:

- `sbom.cdx.json`: CycloneDX JSON software bill of materials;
- `dependency-licenses.json`: resolved package license metadata from
  `pip-licenses`;
- `metadata.sha256`: SHA-256 checksums for both generated JSON files.

Regenerate the files for every release candidate. Build and install the wheel
before the release tools so the report records the packaged project:

```bash
python -m venv /tmp/quant-bench-release
/tmp/quant-bench-release/bin/python -m pip install --upgrade pip
/tmp/quant-bench-release/bin/python -m pip install dist/quant_bench-*.whl
/tmp/quant-bench-release/bin/python -m pip install 'cyclonedx-bom>=7,<8' 'pip-licenses>=5,<6'
/tmp/quant-bench-release/bin/python tools/generate_release_metadata.py
python tools/check_release.py
```

Audit the wheel itself before installing it:

```bash
python tools/check_wheel.py dist/quant_bench-*.whl
python tools/check_sdist.py dist/quant_bench-*.tar.gz
```

The wheel audit rejects private data directories, runtime state, credentials,
model artifacts, unsafe archive paths, unexpected pickle files, missing
licenses, missing CLI metadata, missing synthetic fixtures, and an incomplete
202-workflow catalog.

The source-distribution audit applies the same safety boundary and also
requires the public documentation, tests, maintenance checks, and committed
SBOM needed to reproduce release validation from the archive alone.

The committed report covers the base wheel and release tooling on its recorded
platform. Optional dependency profiles need separate environment-specific
SBOMs when they are included in a published image or managed environment.
