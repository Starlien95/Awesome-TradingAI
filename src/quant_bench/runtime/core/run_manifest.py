from pathlib import Path
from typing import Any

from quant_bench.runtime.core.atomic_io import atomic_write_json

RUNTIME_MANIFEST_SCHEMA_VERSION = "quant-bench.runtime-manifest.v1"


def _relative_artifact_path(run_path: Path, raw_path: str) -> str:
    root = run_path.resolve()
    candidate = Path(raw_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"runtime artifact path escapes run directory: {raw_path}") from exc
    return relative.as_posix()


def write_run_manifest(
    run_dir: str | Path,
    strategy_id: str,
    method_family: str,
    frequency: str,
    mode: str,
    paths: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> None:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    normalized_paths = {
        key: _relative_artifact_path(run_path, value)
        for key, value in paths.items()
    }
    manifest = {
        "manifest_schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "csv_schema_version": "quant-bench.csv.v1",
        "strategy_id": strategy_id,
        "method_family": method_family,
        "frequency": frequency,
        "mode": mode,
        "status": "ACTIVE",
        **normalized_paths,
    }
    if extra:
        manifest.update(extra)
    atomic_write_json(run_path / "run_manifest.json", manifest)
