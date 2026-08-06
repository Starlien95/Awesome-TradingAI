"""Validation and execution for legacy Qlib workflow YAML files."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_workflow(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"workflow root must be a mapping: {path}")
    return raw


def validate_workflow(path: Path) -> list[str]:
    config = load_workflow(path)
    errors: list[str] = []
    for key in ("qlib_init", "task"):
        if not isinstance(config.get(key), dict):
            errors.append(f"missing mapping: {key}")
    task = config.get("task", {})
    for key in ("model", "dataset"):
        if not isinstance(task.get(key), dict):
            errors.append(f"missing mapping: task.{key}")
    model = task.get("model", {})
    for key in ("class", "module_path"):
        if not model.get(key):
            errors.append(f"missing value: task.model.{key}")
    return errors


def run_workflow(
    path: Path,
    workspace: Path,
    experiment_name: str = "quant-bench",
    provider_uri: Path | None = None,
) -> Path:
    errors = validate_workflow(path)
    if errors:
        raise ValueError("invalid workflow: " + "; ".join(errors))
    try:
        from qlib.cli.run import workflow
    except ModuleNotFoundError as exc:
        raise RuntimeError('Qlib workflow support requires: pip install "quant-bench[qlib]"') from exc

    workspace = workspace.expanduser().resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-qlib"
    run_dir = workspace / "runs" / run_id
    config_bytes = path.read_bytes()
    resolved = load_workflow(path)
    original_uri = resolved["qlib_init"].get("provider_uri")
    if provider_uri is not None:
        resolved_uri = provider_uri.expanduser().resolve()
    elif isinstance(original_uri, str) and not Path(original_uri).expanduser().is_absolute():
        frequency = Path(original_uri).name
        resolved_uri = workspace / "datasets" / "qlib" / frequency
    else:
        resolved_uri = Path(str(original_uri)).expanduser().resolve()
    if not resolved_uri.is_dir():
        raise FileNotFoundError(
            f"Qlib provider_uri does not exist: {resolved_uri}; build it with quant-bench data qlib-dump"
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved["qlib_init"]["provider_uri"] = str(resolved_uri)
    resolved_path = run_dir / "resolved_workflow.yaml"
    resolved_path.write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1",
        "run_id": run_id,
        "status": "running",
        "backend": "qlib",
        "workflow": str(path.resolve()),
        "resolved_workflow": str(resolved_path),
        "provider_uri": str(resolved_uri),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        workflow(str(resolved_path), experiment_name=experiment_name, uri_folder=str(workspace / "mlruns"))
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error_message"] = str(exc)
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        temporary = run_dir / ".run_manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(run_dir / "run_manifest.json")
    return run_dir
