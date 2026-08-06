"""Audit a built quant-bench wheel before clean-install or release testing."""

from __future__ import annotations

import argparse
import json
import re
import stat
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

MAX_UNCOMPRESSED_MIB = 20
FORBIDDEN_SUFFIXES = {".pickle", ".pt", ".pth", ".safetensors", ".sqlite"}
ALLOWED_PKL_SOURCES = {
    "quant_bench/methods/finmem/investorbench/configs/character_string_catalog.pkl",
    "quant_bench/methods/finmem/investorbench/configs/chat_models.pkl",
    "quant_bench/methods/finmem/investorbench/configs/data.pkl",
    "quant_bench/methods/finmem/investorbench/configs/embedding.pkl",
    "quant_bench/methods/finmem/investorbench/configs/main.pkl",
    "quant_bench/methods/finmem/investorbench/configs/memory.pkl",
    "quant_bench/methods/finmem/investorbench/configs/meta.pkl",
}
REQUIRED_MEMBERS = {
    "quant_bench/__init__.py",
    "quant_bench/cli/app.py",
    "quant_bench/integrations/qlib/records.py",
    "quant_bench/methods/fingpt_news/research/pipeline.py",
    "quant_bench/methods/fingpt_news/resources/research_fixture/config.yaml",
    "quant_bench/resources/fixtures/crypto_ohlcv_smoke.csv",
}
FORBIDDEN_PARTS = {
    "__pycache__",
    "ai_trade_FinMem",
    "local_data",
    "mlruns",
    "qlib_model_trade",
    "runs",
}
FORBIDDEN_FRAGMENTS = {
    "account_snapshots",
    "raw_news_master",
    "valid_choice-main/test/",
    "valid_choice-main/validator/",
}


def _wheel_metadata(archive: zipfile.ZipFile, names: set[str], errors: list[str]) -> None:
    metadata_files = sorted(name for name in names if name.endswith(".dist-info/METADATA"))
    entry_point_files = sorted(name for name in names if name.endswith(".dist-info/entry_points.txt"))
    if len(metadata_files) != 1:
        errors.append(f"wheel requires exactly one METADATA file; found {len(metadata_files)}")
        return
    try:
        metadata_text = archive.read(metadata_files[0]).decode("utf-8")
    except (KeyError, RuntimeError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        errors.append(f"wheel METADATA is unreadable: {exc}")
        return
    metadata = Parser().parsestr(metadata_text)
    if metadata.get("Name") != "quant-bench":
        errors.append(f"unexpected distribution name: {metadata.get('Name')!r}")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[a-zA-Z0-9.+-]*)?", metadata.get("Version", "")):
        errors.append(f"invalid distribution version: {metadata.get('Version')!r}")
    if metadata.get("License-Expression") != "MIT":
        errors.append("wheel metadata License-Expression must be MIT")
    if len(entry_point_files) != 1:
        errors.append(f"wheel requires exactly one entry_points.txt; found {len(entry_point_files)}")
    else:
        try:
            entry_points = archive.read(entry_point_files[0]).decode("utf-8")
        except (KeyError, RuntimeError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
            errors.append(f"wheel entry_points.txt is unreadable: {exc}")
        else:
            if "quant-bench = quant_bench.cli.app:main" not in entry_points:
                errors.append("quant-bench console entry point is missing")
    license_names = {name for name in names if ".dist-info/licenses/" in name}
    if not any(name.endswith("/LICENSE") for name in license_names):
        errors.append("project LICENSE is missing from wheel metadata")
    if not any(name.endswith("/THIRD_PARTY_NOTICES.md") for name in license_names):
        errors.append("THIRD_PARTY_NOTICES.md is missing from wheel metadata")


def audit_wheel(path: Path, *, max_uncompressed_mib: int = MAX_UNCOMPRESSED_MIB) -> list[str]:
    path = path.expanduser().resolve()
    errors: list[str] = []
    if not path.is_file():
        return [f"wheel does not exist: {path}"]
    if path.suffix != ".whl":
        return [f"expected a .whl file: {path}"]
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        return [f"invalid wheel ZIP: {exc}"]
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        name_set = set(names)
        if len(names) != len(name_set):
            errors.append("wheel contains duplicate member names")
        total_size = sum(info.file_size for info in infos)
        limit = int(max_uncompressed_mib) * 1024 * 1024
        if total_size > limit:
            errors.append(
                f"wheel uncompressed size {total_size} exceeds {max_uncompressed_mib} MiB"
            )
        for info in infos:
            name = info.filename
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                errors.append(f"unsafe wheel member path: {name}")
                continue
            if stat.S_ISLNK(info.external_attr >> 16):
                errors.append(f"wheel contains a symbolic link: {name}")
            if any(part in FORBIDDEN_PARTS for part in pure.parts):
                errors.append(f"private or generated directory in wheel: {name}")
            if pure.name.startswith(".env"):
                errors.append(f"environment file in wheel: {name}")
            if any(fragment in name for fragment in FORBIDDEN_FRAGMENTS):
                errors.append(f"excluded source or private artifact in wheel: {name}")
            suffix = pure.suffix.lower()
            if suffix == ".pkl" and name not in ALLOWED_PKL_SOURCES:
                errors.append(f"unapproved pickle artifact in wheel: {name}")
            elif suffix in FORBIDDEN_SUFFIXES:
                errors.append(f"model or runtime artifact in wheel: {name}")
        missing = sorted(REQUIRED_MEMBERS - name_set)
        if missing:
            errors.append(f"required package members are missing: {missing}")
        workflows = [
            name
            for name in names
            if name.startswith("quant_bench/resources/qlib_workflows/")
            and name.endswith((".yaml", ".yml"))
        ]
        if len(workflows) != 202:
            errors.append(f"expected 202 packaged Qlib workflows; found {len(workflows)}")
        _wheel_metadata(archive, name_set, errors)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--max-uncompressed-mib", type=int, default=MAX_UNCOMPRESSED_MIB)
    args = parser.parse_args()
    errors = audit_wheel(args.wheel, max_uncompressed_mib=args.max_uncompressed_mib)
    print(json.dumps({"valid": not errors, "errors": errors}, indent=2, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
