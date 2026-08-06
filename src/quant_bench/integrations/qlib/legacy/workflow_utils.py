import sys
from pathlib import Path

from quant_bench.integrations.qlib.catalog import workflow_root

FEATURE_SETS = ("158", "360")


def _normalize_model_name(name: str) -> str:
    return name.lower()


def _model_aliases(model: str) -> list[str]:
    aliases = [model]
    aliases.extend(alias.lower() for alias in aliases[:])
    return list(dict.fromkeys(aliases))


def _groups(feature_set: str) -> tuple[str, ...]:
    if feature_set in FEATURE_SETS:
        return (feature_set,)
    raise ValueError("feature_set must be one of: 158, 360")


def _matches_model(path: Path, model: str, freq: str) -> bool:
    stem = path.stem
    prefix = stem.removeprefix("workflow_config_").removesuffix(f"_{freq}")
    expected = _normalize_model_name(model)
    return _normalize_model_name(path.parent.name) == expected or _normalize_model_name(prefix) == expected


def resolve_workflow_config(model: str, freq: str, feature_set: str = "158") -> str:
    root = workflow_root()
    aliases = _model_aliases(model)
    checked: list[Path] = []

    for group in _groups(feature_set):
        group_root = root / group
        for directory in aliases:
            for filename_model in aliases:
                candidate = group_root / directory / f"workflow_config_{filename_model}_{freq}.yaml"
                checked.append(candidate)
                if candidate.exists():
                    return str(candidate)

        matches = sorted(
            path
            for path in group_root.glob(f"*/workflow_config_*_{freq}.yaml")
            if _matches_model(path, model, freq)
        )
        if matches:
            return str(matches[0])

    tried = "\n  - ".join(str(path) for path in checked)
    raise FileNotFoundError(f"Config not found for model={model}, freq={freq}, feature_set={feature_set}. Tried:\n  - {tried}")


def apply_config_sys_path(config: dict, config_path: str) -> None:
    rel_path = (config.get("sys") or {}).get("rel_path") if isinstance(config, dict) else None
    if not rel_path:
        return
    path = (Path(config_path).resolve().parent / rel_path).resolve()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
