"""Fail when the source tree violates quant-bench release and license policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 release environment
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".pkl", ".pickle", ".pth", ".pt", ".safetensors", ".sqlite"}
IGNORED_PARTS = {".git", ".venv", "venv", "__pycache__", "build", "dist", "site"}
FINMEM_ROOT = Path("src/quant_bench/methods/finmem")
INVESTORBENCH_CONFIGS = FINMEM_ROOT / "investorbench" / "configs"
INVESTORBENCH_PKL_SOURCES = {
    "character_string_catalog.pkl",
    "chat_models.pkl",
    "data.pkl",
    "embedding.pkl",
    "main.pkl",
    "memory.pkl",
    "meta.pkl",
}


def _error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON: {path}: {exc}")
        return {}
    if not isinstance(loaded, dict):
        errors.append(f"JSON root must be an object: {path}")
        return {}
    return loaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_investorbench_pkl_source(path: Path, root: Path) -> bool:
    config_root = root / INVESTORBENCH_CONFIGS
    if path.parent != config_root or path.name not in INVESTORBENCH_PKL_SOURCES:
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return bool(text.strip()) and "\x00" not in text


def audit_repository(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    package_root = root / "src" / "quant_bench"
    pyproject_path = root / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject["project"]
    _error(errors, project.get("license") == "MIT", "pyproject project.license must be MIT")
    license_files = project.get("license-files")
    _error(
        errors,
        isinstance(license_files, list) and "THIRD_PARTY_NOTICES.md" in license_files,
        "THIRD_PARTY_NOTICES.md must be included in distribution license-files",
    )
    dependencies = project.get("dependencies")
    _error(errors, isinstance(dependencies, list) and bool(dependencies), "project dependencies are missing")
    urls = project.get("urls")
    _error(
        errors,
        isinstance(urls, dict) and all(isinstance(value, str) for value in urls.values()),
        "project.urls must contain URL strings only",
    )
    ci_path = root / ".github" / "workflows" / "ci.yml"
    _error(errors, ci_path.is_file(), "main CI workflow is missing")
    _error(
        errors,
        not (root / ".github" / "workflows" / "ai-trade-finmem-public-check.yml").exists(),
        "obsolete ai_trade_FinMem workflow remains",
    )
    if ci_path.is_file():
        ci_text = ci_path.read_text(encoding="utf-8")
        for required in (
            'python-version: ["3.10", "3.11", "3.12"]',
            "ruff check .",
            "mypy --strict src/quant_bench",
            "python -m build",
            "python tools/check_wheel.py",
            "python tools/check_sdist.py",
            "fingpt paper-dry-run",
            "qlib-contract:",
        ):
            _error(errors, required in ci_text, f"CI workflow is missing gate: {required}")
        for forbidden in (
            "gh-action-pypi-publish",
            "twine upload",
            "id-token: write",
            "gh release create",
        ):
            _error(errors, forbidden not in ci_text, f"unapproved publish capability in CI: {forbidden}")
    _error(errors, (root / "LICENSE").is_file(), "root LICENSE is missing")
    _error(errors, (root / "MANIFEST.in").is_file(), "source distribution manifest is missing")
    _error(errors, (root / "LICENSES" / "QLIB-MIT.txt").is_file(), "Qlib MIT text is missing")
    _error(errors, (root / "LICENSES" / "FINGPT-MIT.txt").is_file(), "FinGPT MIT text is missing")
    investorbench_license = root / FINMEM_ROOT / "investorbench" / "LICENSE"
    _error(errors, investorbench_license.is_file(), "InvestorBench MIT license is missing")
    _error(
        errors,
        not any(path.is_file() for path in (root / "ops").rglob("*")),
        "production account or ledger operation scripts must not be distributed",
    )
    _error(
        errors,
        not any(
            path.is_file()
            for path in (root / FINMEM_ROOT / "investorbench" / "valid_choice-main").rglob("*")
        ),
        "unused Guardrails Valid Choices source snapshot must not be distributed",
    )
    if investorbench_license.is_file():
        _error(
            errors,
            "Copyright (c) 2024 Haohang Li, Yupeng Cao, Yangyang Yu"
            in investorbench_license.read_text(encoding="utf-8"),
            "InvestorBench copyright notice is incomplete",
        )
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8")) or {}
    _error(errors, citation.get("license") == "MIT", "CITATION.cff license must be MIT")

    notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for required in (
        "da920b7f954f48ab1bb64117c976710de198373e",
        "e5e516470e7a25ed3690889b7b54d9946dd17520",
        "31e5ef41f93b2aea6e63e6ae2267675c997a8e54",
        "3509aeca668824cf2970e3950781d4d09f6c2adb",
        "02593feb2506db3e4290134076bf50c5ecc9920a",
        "be814aa47970de9bf2fdd6a1d5a60ae5cf361b46",
        "LICENSES/QLIB-MIT.txt",
        "release/sbom.cdx.json",
    ):
        _error(errors, required in notices, f"third-party notices missing: {required}")
    _error(errors, "Pending local provenance review" not in notices, "notices still contain a pending review")
    _error(errors, "Blocks public release" not in notices, "notices still contain a release blocker")

    pipeline = (package_root / "data" / "qlib_pipeline.py").read_text(encoding="utf-8")
    tft = (package_root / "integrations" / "qlib" / "models" / "tft.py").read_text(encoding="utf-8")
    _error(errors, "Copyright (c) Microsoft Corporation" in pipeline, "Qlib dumper notice is missing")
    _error(errors, "LICENSES/QLIB-MIT.txt" in pipeline, "Qlib dumper license pointer is missing")
    _error(errors, "Copyright (c) Microsoft Corporation" in tft, "Qlib TFT notice is missing")

    macro_path = package_root / "runtime" / "adapters" / "macrohft_v1_adapter.py"
    macro = macro_path.read_text(encoding="utf-8")
    package_python = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(package_root.rglob("*.py"))
    )
    for forbidden in ("class _Subagent", "class _Hyperagent", "adaLN_modulation", "_MAX_PUNISH"):
        _error(
            errors,
            forbidden not in package_python,
            f"unlicensed inline MacroHFT implementation remains: {forbidden}",
        )
    _error(errors, "torch.jit.load" in macro, "MacroHFT adapter must load external TorchScript")

    runtime_configs = package_root / "resources" / "runtime" / "timeframes"
    for path in runtime_configs.glob("*/config_macrohft_v1.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        model = config.get("model", {})
        _error(errors, str(model.get("path", "")).endswith("_hyperagent.pt"), f"unsafe MacroHFT path: {path}")
        _error(errors, bool(model.get("subagents_dir")), f"MacroHFT subagents_dir missing: {path}")

    for path in root.rglob("*"):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            if _is_investorbench_pkl_source(path, root):
                continue
            errors.append(f"forbidden generated/model artifact in release tree: {path.relative_to(root)}")

    investorbench_data = root / FINMEM_ROOT / "investorbench" / "data"
    _error(
        errors,
        not list(investorbench_data.glob("*.json")),
        "InvestorBench market/news JSON contents must not be redistributed",
    )
    _error(
        errors,
        (root / "docs" / "data" / "FINMEM_INVESTORBENCH_DATA.md").is_file(),
        "InvestorBench local data card is missing",
    )
    _error(
        errors,
        (root / FINMEM_ROOT / "resources" / "investorbench-data-manifest.json").is_file(),
        "InvestorBench checksum manifest is missing",
    )
    _error(errors, not (root / "ai_trade_FinMem").exists(), "obsolete ai_trade_FinMem tree remains")

    finmem_config = (root / FINMEM_ROOT / "config.py").read_text(encoding="utf-8")
    finmem_cycle = (root / FINMEM_ROOT / "live" / "cycle.py").read_text(encoding="utf-8")
    for token in ("DEMO_ORDERS", "LIVE_ORDERS"):
        _error(errors, token in finmem_config, f"FinMem safety token is missing: {token}")
    _error(errors, "FINMEM_QUERY_ACCOUNT" in finmem_cycle, "FinMem account-read gate is missing")

    fingpt_root = package_root / "methods" / "fingpt_news"
    fingpt_config = yaml.safe_load(
        (fingpt_root / "configs" / "sentiment_sft_live.yaml").read_text(encoding="utf-8")
    ) or {}
    runtime = fingpt_config.get("runtime", {})
    trade = fingpt_config.get("trade", {})
    api = fingpt_config.get("api", {})
    news = fingpt_config.get("news", {})
    strategy = fingpt_config.get("strategy", {})
    _error(errors, runtime.get("dry_run") is True, "FinGPT public runtime must default to dry-run")
    _error(errors, trade.get("mode") == "paper_spot", "FinGPT public runtime must default to paper_spot")
    _error(
        errors,
        trade.get("reconcile_account_positions") is False,
        "FinGPT account reconciliation must default to false",
    )
    _error(errors, api.get("is_simulated") is True, "FinGPT public config must use OKX demo mode")
    for key in ("okx_api_key", "okx_secret_key", "okx_passphrase"):
        _error(errors, api.get(key, "") == "", f"FinGPT public config contains inline {key}")
    _error(errors, news.get("ccdata_api_key", "") == "", "FinGPT public config contains a news key")
    _error(
        errors,
        strategy.get("per_coin_params_path") == "configs/per_coin_params_default.csv",
        "FinGPT public config must use the general default parameter file",
    )
    params_path = fingpt_root / "configs" / "per_coin_params_default.csv"
    with params_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _error(errors, bool(rows), "FinGPT default parameter file is empty")
    if rows:
        parameter_fields = [field for field in rows[0] if field != "symbol"]
        signatures = {tuple(row.get(field, "") for field in parameter_fields) for row in rows}
        _error(errors, len(signatures) == 1, "FinGPT public defaults must be uniform across symbols")
        _error(
            errors,
            {row.get("symbol") for row in rows} == set(trade.get("coins", [])),
            "FinGPT public defaults and configured coins do not match",
        )
        performance_fields = {"sharpe", "total_return", "final_equity", "objective"}
        _error(
            errors,
            not performance_fields.intersection(rows[0]),
            "FinGPT public default parameters contain experiment performance fields",
        )

    text_paths = [
        path
        for path in package_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml", ".md", ".sh"}
    ]
    absolute_path_pattern = re.compile(r"/(?:home|Users)/[^/\s]+/")
    for path in text_paths:
        text = path.read_text(encoding="utf-8")
        if absolute_path_pattern.search(text):
            errors.append(f"developer absolute path in package: {path.relative_to(root)}")

    recipe = yaml.safe_load(
        (package_root / "resources" / "configs" / "crypto_smoke_v1.yaml").read_text(encoding="utf-8")
    )
    _error(
        errors,
        recipe["data"].get("license") == "synthetic",
        "synthetic fixture policy must be synthetic",
    )

    sbom = _read_json(root / "release" / "sbom.cdx.json", errors)
    licenses = root / "release" / "dependency-licenses.json"
    checksums = root / "release" / "metadata.sha256"
    _error(errors, sbom.get("bomFormat") == "CycloneDX", "release SBOM must be CycloneDX JSON")
    _error(errors, licenses.is_file(), "release dependency license report is missing")
    if licenses.is_file():
        try:
            report = json.loads(licenses.read_text(encoding="utf-8"))
            _error(errors, isinstance(report, list) and bool(report), "dependency license report is empty")
        except json.JSONDecodeError as exc:
            errors.append(f"invalid dependency license report: {exc}")
    _error(errors, checksums.is_file(), "release metadata checksum file is missing")
    if checksums.is_file() and licenses.is_file() and (root / "release" / "sbom.cdx.json").is_file():
        expected = {
            "dependency-licenses.json": _sha256(licenses),
            "sbom.cdx.json": _sha256(root / "release" / "sbom.cdx.json"),
        }
        recorded: dict[str, str] = {}
        for line in checksums.read_text(encoding="utf-8").splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                recorded[parts[1].lstrip("* ")] = parts[0]
        _error(errors, recorded == expected, "release metadata checksums do not match")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    errors = audit_repository(args.root.expanduser().resolve())
    print(json.dumps({"valid": not errors, "errors": errors}, indent=2, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
