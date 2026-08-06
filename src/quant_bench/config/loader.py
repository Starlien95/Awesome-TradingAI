"""Configuration loading, overrides, and deterministic hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from quant_bench.config.models import ExperimentConfig


def builtin_config_path(recipe_id: str) -> Path:
    name = recipe_id if recipe_id.endswith(".yaml") else f"{recipe_id}.yaml"
    resource = resources.files("quant_bench").joinpath("resources").joinpath("configs").joinpath(name)
    if not resource.is_file():
        raise FileNotFoundError(f"built-in recipe not found: {recipe_id}")
    return Path(str(resource))


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ValueError(f"override must use dotted.path=value syntax: {raw}")
    key, value = raw.split("=", 1)
    path = [part for part in key.split(".") if part]
    if not path:
        raise ValueError(f"override path cannot be empty: {raw}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = yaml.safe_load(value)
    return path, parsed


def _apply_override(config: dict[str, Any], raw: str) -> None:
    path, value = _parse_override(raw)
    cursor: dict[str, Any] = config
    for part in path[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            raise KeyError(f"override path does not exist: {'.'.join(path)}")
        cursor = child
    if path[-1] not in cursor:
        raise KeyError(f"override key does not exist: {'.'.join(path)}")
    cursor[path[-1]] = value


def load_config(source: str | Path, overrides: Iterable[str] = ()) -> ExperimentConfig:
    candidate = Path(source).expanduser()
    path = candidate if candidate.is_file() else builtin_config_path(str(source))
    raw = _load_yaml(path)
    for override in overrides:
        _apply_override(raw, override)
    return ExperimentConfig.model_validate(raw)


def canonical_config_json(config: ExperimentConfig) -> str:
    value = config.model_dump(mode="json", exclude_none=False)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_sha256(config: ExperimentConfig) -> str:
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def render_config(config: ExperimentConfig) -> str:
    return yaml.safe_dump(
        config.model_dump(mode="json", exclude_none=False),
        sort_keys=False,
        allow_unicode=True,
    )
