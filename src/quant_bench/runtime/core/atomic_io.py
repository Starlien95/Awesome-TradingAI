import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    target = ensure_parent(path)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, target)


def atomic_write_json(path: str | Path, data: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    target = ensure_parent(path)
    tmp = target.with_name(target.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, target)


def safe_read_json(path: str | Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return dict(default or {})
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        backup = target.with_name(f"{target.name}.corrupted.{int(time.time())}")
        with contextlib.suppress(OSError):
            os.replace(target, backup)
        return dict(default or {})


def safe_read_csv(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if not target.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(target)
    except Exception:
        backup = target.with_name(f"{target.name}.corrupted.{int(time.time())}")
        with contextlib.suppress(OSError):
            os.replace(target, backup)
        return pd.DataFrame()
