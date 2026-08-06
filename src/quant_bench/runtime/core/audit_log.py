from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from quant_bench.runtime.core.atomic_io import atomic_write_csv, ensure_parent, safe_read_csv


def append_csv_rows(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    dedupe_cols: list[str] | None = None,
) -> None:
    row_list = [row for row in rows if row]
    if not row_list:
        return
    old = safe_read_csv(path)
    new = pd.DataFrame(row_list)
    combined = pd.concat([old, new], ignore_index=True) if not old.empty else new
    existing_dedupe = [col for col in (dedupe_cols or []) if col in combined.columns]
    if existing_dedupe:
        combined = combined.drop_duplicates(existing_dedupe, keep="last")
    atomic_write_csv(combined, path)


def append_jsonl_events(path: str | Path, events: Iterable[dict[str, Any]]) -> None:
    event_list = [event for event in events if event]
    if not event_list:
        return
    target = ensure_parent(path)
    with target.open("a", encoding="utf-8") as handle:
        for event in event_list:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True, default=str) + "\n")
