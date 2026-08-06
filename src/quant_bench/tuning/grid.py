"""Small, dependency-light grid search with resumable CSV results."""

from __future__ import annotations

import itertools
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_bench.config import load_config
from quant_bench.experiments import Experiment


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_grid_parameter(value: str) -> tuple[str, list[Any]]:
    """Parse ``dotted.path=value1,value2`` while accepting JSON scalars."""

    if "=" not in value:
        raise ValueError(f"invalid grid parameter {value!r}; expected dotted.path=value1,value2")
    path, raw_values = value.split("=", 1)
    path = path.strip()
    values = [_parse_scalar(item) for item in raw_values.split(",") if item.strip()]
    if not path or not values:
        raise ValueError(f"invalid grid parameter {value!r}")
    return path, values


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temp = Path(temp_name)
    try:
        frame.to_csv(temp, index=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def run_grid_sweep(
    config: str | Path,
    workspace: Path,
    parameters: list[str],
    *,
    study_name: str,
    max_trials: int = 100,
    resume: bool = False,
) -> Path:
    """Run a Cartesian grid and return the resumable trial table path."""

    if not parameters:
        raise ValueError("at least one --param is required")
    if max_trials < 1:
        raise ValueError("max_trials must be positive")
    parsed = [parse_grid_parameter(parameter) for parameter in parameters]
    paths = [item[0] for item in parsed]
    combinations = list(itertools.product(*(item[1] for item in parsed)))
    if len(combinations) > max_trials:
        raise ValueError(f"grid has {len(combinations)} trials, above --max-trials={max_trials}")

    workspace_path = workspace.expanduser().resolve()
    study_dir = workspace_path / "studies" / study_name
    results_path = study_dir / "trials.csv"
    if results_path.exists() and not resume:
        raise FileExistsError(f"study already exists; pass --resume or choose another name: {study_dir}")
    existing = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    completed = set(existing.get("parameters_json", pd.Series(dtype=str)).astype(str))
    rows: list[dict[str, Any]] = [
        {str(key): value for key, value in record.items()}
        for record in existing.to_dict(orient="records")
    ]

    for trial_index, values in enumerate(combinations, start=1):
        assignment = dict(zip(paths, values, strict=True))
        canonical = json.dumps(assignment, sort_keys=True, separators=(",", ":"))
        if canonical in completed:
            continue
        overrides = [f"{path}={json.dumps(value)}" for path, value in assignment.items()]
        resolved = load_config(config, overrides)
        resolved.experiment.workspace = workspace_path
        started = datetime.now(timezone.utc)
        result = Experiment(resolved).run()
        row: dict[str, Any] = {
            "trial": trial_index,
            "parameters_json": canonical,
            "run_id": result.run_id,
            "run_dir": str(result.run_dir),
            "status": result.status,
            "started_at": started.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        row.update({f"param.{key}": value for key, value in assignment.items()})
        row.update({f"metric.{key}": value for key, value in result.metrics.items()})
        rows.append(row)
        completed.add(canonical)
        _atomic_csv(pd.DataFrame(rows), results_path)
    return results_path
