#!/usr/bin/env python3
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path.cwd()

MODEL_ALIASES = {
    "xgb": "xgboost",
    "lightgbm": "lgb",
    "double_ensemble": "doubleensemble",
    "doubleensemble": "doubleensemble",
    "tabnet": "tabnet",
}

TORCH_PRIMARY_MODELS = {"mlp", "gats", "lstm"}

DEFAULT_FRAMEWORK_CONFIGS = {
    ("mlp", "1h"): "local-config/traditional_1h/config_mlp.yaml",
    ("tcn", "1h"): "local-config/traditional_1h/config_tcn.yaml",
    ("xgboost", "4h"): "local-config/traditional_4h/config_xgboost.yaml",
    ("gats", "4h"): "local-config/traditional_4h/config_gats.yaml",
    ("lstm", "4h"): "local-config/traditional_4h/config_lstm.yaml",
    ("tra", "4h"): "local-config/traditional_4h/config_tra.yaml",
    ("doubleensemble", "15m"): "local-config/traditional_15m/config_double_ensemble.yaml",
    ("tabnet", "15m"): "local-config/traditional_15m/config_tabnet.yaml",
}

COPY_SUFFIXES = {".pkl", ".pth", ".json", ".yaml", ".yml"}


def canonical_model(model: str) -> str:
    key = model.strip().lower()
    return MODEL_ALIASES.get(key, key)


def resolve_project_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def resolve_framework_config(framework_dir: Path, model: str, freq: str, framework_config: str | None) -> Path | None:
    if framework_config:
        candidate = Path(framework_config)
        if candidate.is_absolute():
            return candidate
        framework_candidate = framework_dir / candidate
        return framework_candidate if framework_candidate.exists() else ROOT / candidate

    rel_path = DEFAULT_FRAMEWORK_CONFIGS.get((canonical_model(model), freq))
    if not rel_path:
        return None
    candidate = framework_dir / rel_path
    return candidate if candidate.exists() else None


def find_artifact(source_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = source_dir / name
        if candidate.exists():
            return candidate

    lower_map = {path.name.lower(): path for path in source_dir.iterdir() if path.is_file()}
    for name in names:
        candidate = lower_map.get(name.lower())
        if candidate is not None:
            return candidate
    return None


def primary_artifact(source_dir: Path, model: str, freq: str) -> Path:
    canonical = canonical_model(model)
    torch_names = [
        f"{model}_{freq}_model.pth",
        f"{canonical}_{freq}_model.pth",
    ]
    qlib_names = [
        f"qlib_{model}_{freq}_model.pkl",
        f"qlib_{canonical}_{freq}_model.pkl",
    ]
    raw_names = [
        f"{model}_{freq}_model.pkl",
        f"{canonical}_{freq}_model.pkl",
    ]

    preferred = torch_names + qlib_names + raw_names if canonical in TORCH_PRIMARY_MODELS else qlib_names + raw_names + torch_names
    artifact = find_artifact(source_dir, preferred)
    if artifact is None:
        tried = ", ".join(preferred)
        raise FileNotFoundError(f"No deployable model artifact found in {source_dir}. Tried: {tried}")
    return artifact


def deployment_name(source: Path, model: str, freq: str) -> str:
    if source.name == "train_metrics.json":
        return f"{model}_{freq}_train_metrics.json"
    if source.name == f"data_handler_{freq}.pkl":
        return f"{model}_{freq}_data_handler.pkl"
    return source.name


def copy_artifacts(
    source_dir: Path,
    weights_dir: Path,
    model: str,
    freq: str,
    workflow_config: str | Path | None,
    dry_run: bool,
) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    if not dry_run:
        weights_dir.mkdir(parents=True, exist_ok=True)

    for source in sorted(source_dir.iterdir()):
        if not source.is_file() or source.suffix not in COPY_SUFFIXES:
            continue
        target = weights_dir / deployment_name(source, model, freq)
        copied.append({"source": str(source), "target": str(target)})
        if not dry_run:
            shutil.copy2(source, target)

    if workflow_config:
        config_source = resolve_project_path(workflow_config, ROOT / str(workflow_config))
        if config_source.exists():
            target = weights_dir / config_source.name
            copied.append({"source": str(config_source), "target": str(target)})
            if not dry_run:
                shutil.copy2(config_source, target)

    return copied


def update_framework_config(
    config_path: Path | None,
    model_rel_path: str,
    dry_run: bool,
) -> bool:
    if config_path is None:
        return False
    if not config_path.exists():
        raise FileNotFoundError(f"Framework config not found: {config_path}")

    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    output: list[str] = []
    in_model = False
    model_indent = 0
    path_updated = False

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if stripped.startswith("model:"):
            in_model = True
            model_indent = indent
            output.append(line)
            continue

        if in_model:
            starts_next_top_level = stripped and not stripped.startswith("#") and indent <= model_indent
            if starts_next_top_level:
                if not path_updated:
                    output.append(" " * (model_indent + 2) + f'path: "{model_rel_path}"\n')
                    path_updated = True
                in_model = False
            elif stripped.startswith("path:"):
                output.append(" " * indent + f'path: "{model_rel_path}"\n')
                path_updated = True
                continue

        output.append(line)

    if in_model and not path_updated:
        output.append(" " * (model_indent + 2) + f'path: "{model_rel_path}"\n')
        path_updated = True

    if not path_updated:
        if output and not output[-1].endswith("\n"):
            output[-1] += "\n"
        output.extend(["model:\n", f'  path: "{model_rel_path}"\n'])

    if not dry_run:
        config_path.write_text("".join(output), encoding="utf-8")
    return True


def deploy_model_to_framework(
    model: str,
    freq: str,
    feature_set: str = "158",
    source_dir: str | Path | None = None,
    framework_dir: str | Path = ".",
    framework_config: str | None = None,
    workflow_config: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    source = resolve_project_path(source_dir, ROOT / "qlib_models" / feature_set / model / freq)
    if not source.exists():
        raise FileNotFoundError(f"Source model directory not found: {source}")

    framework = resolve_project_path(framework_dir, ROOT)
    if not framework.exists():
        raise FileNotFoundError(f"Framework directory not found: {framework}")

    weights_dir = framework / "models" / "runtime" / freq
    primary = primary_artifact(source, model, freq)
    copied = copy_artifacts(source, weights_dir, model, freq, workflow_config, dry_run)
    primary_target = weights_dir / primary.name

    if not dry_run and not primary_target.exists():
        shutil.copy2(primary, primary_target)
        copied.append({"source": str(primary), "target": str(primary_target)})

    model_rel_path = primary_target.relative_to(framework).as_posix()
    config_path = resolve_framework_config(framework, model, freq, framework_config)

    manifest = {
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "feature_set": feature_set,
        "model": model,
        "canonical_model": canonical_model(model),
        "freq": freq,
        "source_dir": str(source),
        "framework_dir": str(framework),
        "weights_dir": str(weights_dir),
        "primary_artifact": str(primary),
        "framework_model_path": model_rel_path,
        "framework_config": str(config_path) if config_path else None,
        "config_updated": False,
        "copied": copied,
    }

    manifest["config_updated"] = update_framework_config(config_path, model_rel_path, dry_run)

    manifest_path = weights_dir / f"{model}_{freq}_deployment_manifest.json"
    if not dry_run:
        weights_dir.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote Qlib artifacts into the optional trading runtime.")
    parser.add_argument("--model", "-m", required=True, help="Model tag, e.g. mlp, lstm, tra, xgboost.")
    parser.add_argument("--freq", "-f", required=True, help="Timeframe tag, e.g. 15m, 1h, 4h.")
    parser.add_argument("--feature-set", "--feature_set", choices=("158", "360"), default="158")
    parser.add_argument("--source-dir", default=None, help="Override source artifact directory.")
    parser.add_argument("--framework-dir", default=".", help="Workspace or repository root for runtime artifacts.")
    parser.add_argument("--framework-config", default=None, help="Optional framework YAML config to update.")
    parser.add_argument("--workflow-config", default=None, help="Optional Qlib workflow YAML to copy into model_weights.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned deployment without copying files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = deploy_model_to_framework(
        model=args.model,
        freq=args.freq,
        feature_set=args.feature_set,
        source_dir=args.source_dir,
        framework_dir=args.framework_dir,
        framework_config=args.framework_config,
        workflow_config=args.workflow_config,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
