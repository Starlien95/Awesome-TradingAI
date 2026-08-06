"""Discovery for packaged Qlib workflow templates."""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import Any


def workflow_root() -> Path:
    root = resources.files("quant_bench").joinpath("resources").joinpath("qlib_workflows")
    path = Path(str(root))
    if not path.is_dir():
        raise FileNotFoundError("packaged Qlib workflow resources are unavailable")
    return path


def _frequency(path: Path) -> str:
    match = re.search(r"_(1m|5m|15m|1h|4h|1d|Alpha158|Alpha360)$", path.stem, re.IGNORECASE)
    return match.group(1) if match else "unknown"


def list_workflows(
    *,
    feature_set: str | None = None,
    model: str | None = None,
    frequency: str | None = None,
) -> list[dict[str, Any]]:
    root = workflow_root()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/*.yaml")):
        relative = path.relative_to(root)
        row = {
            "feature_set": relative.parts[0],
            "model": relative.parts[1],
            "frequency": _frequency(path),
            "path": str(path),
            "resource": relative.as_posix(),
        }
        if feature_set and row["feature_set"].lower() != feature_set.lower():
            continue
        if model and row["model"].lower() != model.lower():
            continue
        if frequency and row["frequency"].lower() != frequency.lower():
            continue
        rows.append(row)
    return rows


def resolve_workflow(feature_set: str, model: str, frequency: str) -> Path:
    matches = list_workflows(feature_set=feature_set, model=model, frequency=frequency)
    if not matches:
        raise FileNotFoundError(
            f"no packaged workflow for feature_set={feature_set}, model={model}, frequency={frequency}"
        )
    if len(matches) > 1:
        resources_found = [row["resource"] for row in matches]
        raise ValueError(f"workflow selection is ambiguous: {resources_found}")
    return Path(matches[0]["path"])
