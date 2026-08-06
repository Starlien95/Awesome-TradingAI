"""Validation helpers for InvestorBench JSON datasets."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).with_name("resources") / "investorbench-data-manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def inspect_dataset(path: str | Path, *, verify_entries: bool = True) -> dict[str, Any]:
    """Validate one InvestorBench JSON file without altering it."""

    target = Path(path).expanduser().resolve()
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"dataset must be a non-empty JSON object: {target}")

    dates = sorted(payload)
    # Ordering is not part of JSON semantics, so report it without rejecting
    # otherwise valid data.
    ordered = dates == list(payload)

    for value in (dates[0], dates[-1]):
        date.fromisoformat(value)

    news_rows = 0
    if verify_entries:
        for day, row in payload.items():
            date.fromisoformat(day)
            if not isinstance(row, dict):
                raise ValueError(f"{target}: {day} must map to an object")
            price = row.get("prices")
            if not isinstance(price, (int, float)) or isinstance(price, bool):
                raise ValueError(f"{target}: {day}.prices must be numeric")
            news = row.get("news", [])
            if news is not None and not isinstance(news, list):
                raise ValueError(f"{target}: {day}.news must be a list or null")
            news_rows += len(news or [])

    return {
        "path": str(target),
        "symbol": target.stem.upper(),
        "days": len(dates),
        "start_date": dates[0],
        "end_date": dates[-1],
        "news_rows": news_rows,
        "keys_in_file_order": ordered,
        "size_bytes": target.stat().st_size,
        "sha256": _sha256(target),
    }


def verify_data_dir(
    data_dir: str | Path,
    *,
    symbols: list[str] | None = None,
    verify_entries: bool = False,
) -> dict[str, Any]:
    """Verify local datasets against the recovery manifest."""

    root = Path(data_dir).expanduser().resolve()
    manifest = load_data_manifest()
    expected = manifest["files"]
    selected = [symbol.lower() for symbol in symbols] if symbols else sorted(expected)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for symbol in selected:
        if symbol not in expected:
            errors.append(f"unknown symbol in manifest: {symbol.upper()}")
            continue
        path = root / f"{symbol}.json"
        if not path.is_file():
            errors.append(f"missing dataset: {path}")
            continue
        row = inspect_dataset(path, verify_entries=verify_entries)
        reference = expected[symbol]
        row["checksum_matches"] = row["sha256"] == reference["sha256"]
        row["size_matches"] = row["size_bytes"] == reference["size_bytes"]
        if not row["checksum_matches"]:
            errors.append(f"checksum mismatch: {path}")
        if not row["size_matches"]:
            errors.append(f"size mismatch: {path}")
        rows.append(row)
    return {"valid": not errors, "data_dir": str(root), "files": rows, "errors": errors}
