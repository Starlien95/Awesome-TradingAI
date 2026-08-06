"""Generate release SBOM, dependency-license records, and checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _command(name: str) -> str:
    # Keep the virtual-environment path. Resolving the Python symlink would
    # incorrectly search /usr/bin instead of the venv's bin directory.
    adjacent = Path(sys.executable).parent / name
    resolved = adjacent if adjacent.is_file() else shutil.which(name)
    if not resolved:
        raise RuntimeError(
            f"{name} is unavailable; install the release tools with "
            "python -m pip install '.[release]'"
        )
    return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(output_dir: Path) -> dict[str, str]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sbom = output_dir / "sbom.cdx.json"
    licenses = output_dir / "dependency-licenses.json"
    checksums = output_dir / "metadata.sha256"

    subprocess.run(
        [
            _command("cyclonedx-py"),
            "environment",
            "--output-format",
            "JSON",
            "--output-file",
            str(sbom),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "piplicenses",
            "--format=json",
            "--with-authors",
            "--with-urls",
            f"--output-file={licenses}",
        ],
        check=True,
    )

    license_rows = json.loads(licenses.read_text(encoding="utf-8"))
    if not isinstance(license_rows, list) or not license_rows:
        raise RuntimeError("pip-licenses produced an empty or invalid report")
    sbom_document = json.loads(sbom.read_text(encoding="utf-8"))
    if sbom_document.get("bomFormat") != "CycloneDX":
        raise RuntimeError("cyclonedx-py did not produce a CycloneDX document")

    checksum_lines = [
        f"{_sha256(licenses)}  {licenses.name}",
        f"{_sha256(sbom)}  {sbom.name}",
    ]
    checksums.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return {
        "sbom": str(sbom),
        "dependency_licenses": str(licenses),
        "checksums": str(checksums),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "release")
    result = generate(parser.parse_args().output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
