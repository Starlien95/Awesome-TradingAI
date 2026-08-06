#!/usr/bin/env python3
"""Check the FinMem public subtree for private data and unsafe defaults."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINMEM_ROOT = ROOT / "src" / "quant_bench" / "methods" / "finmem"
MAX_FILE_SIZE = 95 * 1024 * 1024
SECRET_PATTERNS = {
    "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "API token": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    "Linux home path": re.compile(r"/home/[^/\s]+/"),
    "Windows user path": re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\s]+\\\\"),
    "assigned secret": re.compile(
        r"(?i)\b(?:api[_-]?key|secret[_-]?key|passphrase)\b\s*[=:]\s*"
        r"[\"'`]?\s*(?!your_|replace_|xxxx|\$\{|[\"'`]?\s*$)[A-Za-z0-9_+/=-]{12,}"
    ),
}


def audit() -> list[str]:
    errors: list[str] = []
    if (ROOT / "ai_trade_FinMem").exists():
        errors.append("obsolete ai_trade_FinMem directory remains")
    if list((FINMEM_ROOT / "investorbench" / "data").glob("*.json")):
        errors.append("InvestorBench market/news JSON is inside the public package")

    forbidden_runtime_names = {
        "live_cycle_records",
        "live_loop_records",
        "live_data",
        "live_decision",
        "results",
        "storage",
        "qdrant",
    }
    files = [path for path in FINMEM_ROOT.rglob("*") if path.is_file()]
    for path in files:
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_FILE_SIZE:
            errors.append(f"file exceeds 95 MiB: {relative}")
        if any(part in forbidden_runtime_names for part in path.parts):
            errors.append(f"runtime output inside package: {relative}")
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(relative))
            except Exception as exc:
                errors.append(f"invalid Python: {relative}: {exc}")
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception as exc:
                errors.append(f"invalid JSON: {relative}: {exc}")
        if path.suffix.lower() not in {".json", ".md", ".pkl", ".py", ".toml", ".yaml", ".yml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            errors.append(f"unexpected binary content: {relative}")
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"possible {label}: {relative}")

    config_text = (FINMEM_ROOT / "config.py").read_text(encoding="utf-8")
    cycle_text = (FINMEM_ROOT / "live" / "cycle.py").read_text(encoding="utf-8")
    exchange_text = (FINMEM_ROOT / "live" / "exchange.py").read_text(encoding="utf-8")
    for token in ("DEMO_ORDERS", "LIVE_ORDERS"):
        if token not in config_text or token not in exchange_text:
            errors.append(f"missing order confirmation gate: {token}")
    if "FINMEM_QUERY_ACCOUNT" not in cycle_text:
        errors.append("missing separate account-query gate")
    return sorted(set(errors))


def main() -> int:
    errors = audit()
    print(json.dumps({"valid": not errors, "errors": errors}, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
