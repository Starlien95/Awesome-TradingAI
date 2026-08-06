"""Promote a trained artifact into a reviewed runtime workspace."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ALLOWED_SUFFIXES = {".json", ".pkl", ".pt", ".pth", ".safetensors", ".yaml", ".yml"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        shutil.copy2(source, temp_path)
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)


def _updated_config(config_path: Path, artifact_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"runtime config root must be a mapping: {config_path}")
    model = payload.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError(f"runtime config model must be a mapping: {config_path}")
    model["path"] = os.path.relpath(artifact_path, config_path.parent).replace(os.sep, "/")
    return payload


def promote_artifact(
    source: Path,
    workspace: Path,
    *,
    model_id: str,
    frequency: str,
    config_path: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Copy one artifact, optionally update a local config, and write provenance metadata.

    This function treats the source as opaque bytes and never unpickles it.
    """

    source_path = source.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"artifact not found: {source_path}")
    if source_path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"unsupported artifact suffix: {source_path.suffix}")
    if not model_id.strip() or not frequency.strip():
        raise ValueError("model_id and frequency are required")

    workspace_path = workspace.expanduser().resolve()
    target_dir = workspace_path / "models" / "runtime" / frequency
    target = target_dir / source_path.name
    if target.exists() and not force:
        raise FileExistsError(f"refusing to replace existing artifact without --force: {target}")

    local_config = config_path.expanduser().resolve() if config_path else None
    updated_config = None
    if local_config:
        if not local_config.is_file():
            raise FileNotFoundError(f"runtime config not found: {local_config}")
        updated_config = _updated_config(local_config, target)

    manifest = {
        "schema_version": "1",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "frequency": frequency,
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "target": str(target),
        "config": str(local_config) if local_config else None,
        "dry_run": dry_run,
    }
    manifest_path = target_dir / f"{model_id}_promotion.json"
    manifest["manifest"] = str(manifest_path)
    if dry_run:
        return manifest

    _atomic_copy(source_path, target)
    if local_config and updated_config is not None:
        local_config.write_text(
            yaml.safe_dump(updated_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
