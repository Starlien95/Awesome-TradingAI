"""Audit a built quant-bench source distribution before release testing."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

MAX_UNCOMPRESSED_MIB = 30
FORBIDDEN_SUFFIXES = {".pickle", ".pt", ".pth", ".safetensors", ".sqlite"}
ALLOWED_PKL_SOURCES = {
    "src/quant_bench/methods/finmem/investorbench/configs/character_string_catalog.pkl",
    "src/quant_bench/methods/finmem/investorbench/configs/chat_models.pkl",
    "src/quant_bench/methods/finmem/investorbench/configs/data.pkl",
    "src/quant_bench/methods/finmem/investorbench/configs/embedding.pkl",
    "src/quant_bench/methods/finmem/investorbench/configs/main.pkl",
    "src/quant_bench/methods/finmem/investorbench/configs/memory.pkl",
    "src/quant_bench/methods/finmem/investorbench/configs/meta.pkl",
}
REQUIRED_MEMBERS = {
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "README.zh-CN.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/ARCHITECTURE.md",
    "mkdocs.yml",
    "pyproject.toml",
    "release/sbom.cdx.json",
    "src/quant_bench/__init__.py",
    "tests/integration/test_offline_quickstart.py",
    "tools/check_release.py",
    "tools/check_wheel.py",
}
FORBIDDEN_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "local_data",
    "mlruns",
    "runs",
    "venv",
}
FORBIDDEN_FRAGMENTS = {
    "account_snapshots",
    "raw_news_master",
    "valid_choice-main/test/",
    "valid_choice-main/validator/",
}


def audit_sdist(path: Path, *, max_uncompressed_mib: int = MAX_UNCOMPRESSED_MIB) -> list[str]:
    path = path.expanduser().resolve()
    errors: list[str] = []
    if not path.is_file():
        return [f"source distribution does not exist: {path}"]
    if not path.name.endswith(".tar.gz"):
        return [f"expected a .tar.gz source distribution: {path}"]
    try:
        archive = tarfile.open(path, mode="r:gz")  # noqa: SIM115
    except (tarfile.TarError, OSError) as exc:
        return [f"invalid source distribution: {exc}"]
    with archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            errors.append("source distribution contains duplicate member names")
        roots = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
        if len(roots) != 1:
            errors.append(f"source distribution requires one root directory; found {sorted(roots)}")
            return errors
        root = next(iter(roots))
        if not re.fullmatch(r"quant_bench-\d+\.\d+\.\d+(?:[a-zA-Z0-9.+-]*)?", root):
            errors.append(f"unexpected source distribution root: {root}")
        relative_names: set[str] = set()
        total_size = 0
        for member in members:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in member.name:
                errors.append(f"unsafe source distribution path: {member.name}")
                continue
            if member.issym() or member.islnk():
                errors.append(f"source distribution contains a link: {member.name}")
            elif not member.isfile() and not member.isdir():
                errors.append(f"source distribution contains a special file: {member.name}")
            total_size += max(member.size, 0)
            relative = PurePosixPath(*pure.parts[1:]).as_posix() if len(pure.parts) > 1 else ""
            if not relative:
                continue
            relative_names.add(relative)
            relative_path = PurePosixPath(relative)
            if any(part in FORBIDDEN_PARTS for part in relative_path.parts):
                errors.append(f"private or generated directory in source distribution: {relative}")
            if relative_path.name.startswith(".env"):
                errors.append(f"environment file in source distribution: {relative}")
            if any(fragment in relative for fragment in FORBIDDEN_FRAGMENTS):
                errors.append(f"excluded source or private artifact in source distribution: {relative}")
            suffix = relative_path.suffix.lower()
            if suffix == ".pkl" and relative not in ALLOWED_PKL_SOURCES:
                errors.append(f"unapproved pickle artifact in source distribution: {relative}")
            elif suffix in FORBIDDEN_SUFFIXES:
                errors.append(f"model or runtime artifact in source distribution: {relative}")
        limit = int(max_uncompressed_mib) * 1024 * 1024
        if total_size > limit:
            errors.append(
                f"source distribution uncompressed size {total_size} exceeds {max_uncompressed_mib} MiB"
            )
        missing = sorted(REQUIRED_MEMBERS - relative_names)
        if missing:
            errors.append(f"required source distribution members are missing: {missing}")
        workflows = [
            name
            for name in relative_names
            if name.startswith("src/quant_bench/resources/qlib_workflows/")
            and name.endswith((".yaml", ".yml"))
        ]
        if len(workflows) != 202:
            errors.append(f"expected 202 source Qlib workflows; found {len(workflows)}")
        pkg_info_name = f"{root}/PKG-INFO"
        if pkg_info_name not in names:
            errors.append("PKG-INFO is missing from source distribution")
        else:
            try:
                pkg_info = archive.extractfile(pkg_info_name)
            except (KeyError, OSError, tarfile.TarError) as exc:
                errors.append(f"PKG-INFO is unreadable: {exc}")
                return errors
            if pkg_info is None:
                errors.append("PKG-INFO is not a regular file in source distribution")
                return errors
            try:
                metadata_text = pkg_info.read().decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                errors.append(f"PKG-INFO is unreadable: {exc}")
                return errors
            metadata = Parser().parsestr(metadata_text)
            if metadata.get("Name") != "quant-bench":
                errors.append(f"unexpected distribution name: {metadata.get('Name')!r}")
            if metadata.get("License-Expression") != "MIT":
                errors.append("source metadata License-Expression must be MIT")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path)
    parser.add_argument("--max-uncompressed-mib", type=int, default=MAX_UNCOMPRESSED_MIB)
    args = parser.parse_args()
    errors = audit_sdist(args.sdist, max_uncompressed_mib=args.max_uncompressed_mib)
    print(json.dumps({"valid": not errors, "errors": errors}, indent=2, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
