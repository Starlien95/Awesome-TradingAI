"""Local, read-only discovery and loading for dashboard workspaces."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

IssueSeverity = Literal["info", "warning", "error"]
RunKind = Literal["canonical", "runtime", "legacy"]


@dataclass(frozen=True)
class DashboardIssue:
    code: str
    message: str
    severity: IssueSeverity = "warning"
    run_key: str = ""
    path: Path | None = None


@dataclass(frozen=True)
class RunSource:
    key: str
    run_id: str
    display_name: str
    kind: RunKind
    method_family: str
    frequency: str
    mode: str
    status: str
    run_dir: Path
    manifest_path: Path | None
    manifest: dict[str, Any]
    files: dict[str, Path]
    protocol_id: str = ""
    dataset_sha256: str = ""


@dataclass
class RunData:
    source: RunSource
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    orders: pd.DataFrame = field(default_factory=pd.DataFrame)
    fills: pd.DataFrame = field(default_factory=pd.DataFrame)
    failed_orders: pd.DataFrame = field(default_factory=pd.DataFrame)
    account_snapshots: pd.DataFrame = field(default_factory=pd.DataFrame)
    rebalance_plan: pd.DataFrame = field(default_factory=pd.DataFrame)
    ic: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_sentiment: pd.DataFrame = field(default_factory=pd.DataFrame)
    news_sentiment: pd.DataFrame = field(default_factory=pd.DataFrame)
    volume: pd.DataFrame = field(default_factory=pd.DataFrame)
    issues: list[DashboardIssue] = field(default_factory=list)


@dataclass(frozen=True)
class WorkspaceInventory:
    root: Path
    runs: tuple[RunSource, ...]
    issues: tuple[DashboardIssue, ...]


RUNTIME_FILE_FIELDS = {
    "metrics": "metrics_path",
    "signals": "signals_path",
    "trades": "trades_path",
    "orders": "orders_path",
    "volume": "volume_path",
    "fills": "fills_path",
    "failed_orders": "failed_orders_path",
    "account_snapshots": "account_snapshots_path",
    "rebalance_plan": "rebalance_plan_path",
    "ic": "ic_path",
    "daily_sentiment": "daily_sentiment_path",
    "news_sentiment": "news_sentiment_path",
}

RUN_DATA_FIELDS = tuple(RUNTIME_FILE_FIELDS)


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_json(path: Path) -> tuple[dict[str, Any], DashboardIssue | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, DashboardIssue(
            code="INVALID_MANIFEST",
            message=f"无法读取 JSON: {exc}",
            severity="error",
            path=path,
        )
    if not isinstance(value, dict):
        return {}, DashboardIssue(
            code="INVALID_MANIFEST_ROOT",
            message="JSON 根节点必须是 object。",
            severity="error",
            path=path,
        )
    return value, None


def _read_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "resolved_config.yaml"
    if not path.is_file():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _safe_relative_file(
    run_dir: Path,
    raw_path: Any,
    *,
    run_key: str,
    field_name: str,
) -> tuple[Path | None, DashboardIssue | None]:
    if raw_path is None or raw_path == "":
        return None, None
    if not isinstance(raw_path, (str, os.PathLike)):
        return None, DashboardIssue(
            code="INVALID_PATH_FIELD",
            message=f"manifest 字段 {field_name} 必须是字符串路径。",
            severity="error",
            run_key=run_key,
        )
    candidate = (run_dir / str(raw_path)).resolve()
    if not _is_inside(candidate, run_dir):
        return None, DashboardIssue(
            code="PATH_ESCAPE",
            message=f"manifest 字段 {field_name} 指向 run 目录之外，已拒绝读取。",
            severity="error",
            run_key=run_key,
            path=candidate,
        )
    if not candidate.is_file():
        return None, DashboardIssue(
            code="MISSING_DECLARED_FILE",
            message=f"manifest 声明的 {field_name} 文件不存在。",
            severity="warning",
            run_key=run_key,
            path=candidate,
        )
    return candidate, None


def _runtime_source(
    workspace: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> tuple[RunSource, list[DashboardIssue]]:
    run_dir = manifest_path.parent.resolve()
    relative_dir = run_dir.relative_to(workspace).as_posix()
    strategy_id = str(manifest.get("strategy_id") or run_dir.name)
    key = f"runtime:{relative_dir}"
    files: dict[str, Path] = {}
    issues: list[DashboardIssue] = []
    for file_key, manifest_field in RUNTIME_FILE_FIELDS.items():
        path, issue = _safe_relative_file(
            run_dir,
            manifest.get(manifest_field),
            run_key=key,
            field_name=manifest_field,
        )
        if path is not None:
            files[file_key] = path
        if issue is not None:
            issues.append(issue)

    fallback_names = {
        "metrics": ("metrics.csv", f"{strategy_id}_metrics.csv"),
        "signals": ("signals.csv", f"{strategy_id}_signals.csv"),
        "trades": ("trades.csv", f"{strategy_id}_trades.csv"),
        "orders": ("orders.csv", f"{strategy_id}_orders.csv"),
        "volume": ("volume.csv", f"{strategy_id}_volume.csv"),
    }
    for file_key, names in fallback_names.items():
        if file_key in files:
            continue
        for filename in names:
            candidate, issue = _safe_relative_file(
                run_dir,
                filename,
                run_key=key,
                field_name=f"legacy fallback {filename}",
            )
            if candidate is not None:
                files[file_key] = candidate
                break
            if issue is not None and issue.code == "PATH_ESCAPE":
                issues.append(issue)

    source = RunSource(
        key=key,
        run_id=strategy_id,
        display_name=str(manifest.get("display_name") or strategy_id),
        kind="runtime",
        method_family=str(manifest.get("method_family") or "runtime"),
        frequency=str(manifest.get("frequency") or "unknown"),
        mode=str(manifest.get("mode") or "unknown").lower(),
        status=str(manifest.get("status") or "unknown").lower(),
        run_dir=run_dir,
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        files=files,
        protocol_id=str(manifest.get("protocol_id") or ""),
        dataset_sha256=str(manifest.get("dataset_sha256") or ""),
    )
    return source, issues


def _canonical_source(
    workspace: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> tuple[RunSource, list[DashboardIssue]]:
    run_dir = manifest_path.parent.resolve()
    run_id = str(manifest.get("run_id") or run_dir.name)
    key = f"canonical:{run_dir.relative_to(workspace).as_posix()}"
    config = _read_config(run_dir)
    frequency_cfg = config.get("frequency", {}) if isinstance(config, dict) else {}
    runtime_cfg = config.get("runtime", {}) if isinstance(config, dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    frequency_cfg = frequency_cfg if isinstance(frequency_cfg, dict) else {}
    runtime_cfg = runtime_cfg if isinstance(runtime_cfg, dict) else {}
    model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
    files: dict[str, Path] = {}
    issues: list[DashboardIssue] = []

    raw_artifacts = manifest.get("artifacts", [])
    artifacts = raw_artifacts if isinstance(raw_artifacts, list) else []
    if not isinstance(raw_artifacts, list):
        issues.append(
            DashboardIssue(
                code="INVALID_ARTIFACT_LIST",
                message="canonical manifest 的 artifacts 必须是 list。",
                severity="error",
                run_key=key,
                path=manifest_path,
            )
        )
    artifact_paths = {
        str(item.get("path")): item
        for item in artifacts
        if isinstance(item, dict) and item.get("path")
    }
    canonical_files = {
        "metrics": "backtest/returns.csv",
        "signals": "predictions.csv",
        "metrics_json": "metrics.json",
        "config": "resolved_config.yaml",
    }
    for file_key, relative_path in canonical_files.items():
        if relative_path not in artifact_paths and not (run_dir / relative_path).is_file():
            continue
        path, issue = _safe_relative_file(
            run_dir,
            relative_path,
            run_key=key,
            field_name=relative_path,
        )
        if path is not None:
            files[file_key] = path
        if issue is not None:
            issues.append(issue)

    experiment_cfg = config.get("experiment", {})
    if not isinstance(experiment_cfg, dict):
        experiment_cfg = {}
    source = RunSource(
        key=key,
        run_id=run_id,
        display_name=str(experiment_cfg.get("name") or run_id),
        kind="canonical",
        method_family=str(model_cfg.get("model_id") or "canonical_benchmark"),
        frequency=str(frequency_cfg.get("id") or "unknown"),
        mode=str(runtime_cfg.get("mode") or "research").lower(),
        status=str(manifest.get("status") or "unknown").lower(),
        run_dir=run_dir,
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        files=files,
        protocol_id=str(manifest.get("protocol_id") or ""),
        dataset_sha256=str(manifest.get("dataset_sha256") or ""),
    )
    return source, issues


def _legacy_sources(workspace: Path, known_metrics: set[Path]) -> list[RunSource]:
    sources: list[RunSource] = []
    timeframes_root = workspace / "timeframes"
    if not timeframes_root.is_dir():
        return sources
    for metrics_path in sorted(timeframes_root.glob("*/logs/*/*_metrics.csv")):
        resolved_metrics = metrics_path.resolve()
        if resolved_metrics in known_metrics:
            continue
        model_dir = metrics_path.parent.resolve()
        timeframe = metrics_path.relative_to(timeframes_root).parts[0]
        model_name = model_dir.name
        prefix = metrics_path.name.removesuffix("_metrics.csv")
        files = {"metrics": resolved_metrics}
        for file_key, suffix in {
            "signals": "_signals.csv",
            "trades": "_trades.csv",
            "orders": "_orders.csv",
            "volume": "_volume.csv",
            "ic": "_ic_report.csv",
        }.items():
            candidate = model_dir / f"{prefix}{suffix}"
            if candidate.is_file():
                files[file_key] = candidate.resolve()
        relative_dir = model_dir.relative_to(workspace).as_posix()
        sources.append(
            RunSource(
                key=f"legacy:{relative_dir}",
                run_id=model_name,
                display_name=model_name,
                kind="legacy",
                method_family="traditional_ml",
                frequency=timeframe,
                mode="unknown",
                status="historical",
                run_dir=model_dir,
                manifest_path=None,
                manifest={},
                files=files,
            )
        )
    return sources


def discover_workspace(workspace: str | Path) -> WorkspaceInventory:
    root = Path(workspace).expanduser().resolve()
    issues: list[DashboardIssue] = []
    runs: list[RunSource] = []
    runs_root = root / "runs"
    if not root.exists():
        issues.append(
            DashboardIssue(
                code="WORKSPACE_NOT_FOUND",
                message="workspace 目录不存在。",
                severity="error",
                path=root,
            )
        )
        return WorkspaceInventory(root=root, runs=(), issues=tuple(issues))

    if runs_root.is_dir():
        for manifest_path in sorted(runs_root.glob("**/run_manifest.json")):
            resolved_manifest = manifest_path.resolve()
            if not _is_inside(resolved_manifest, runs_root.resolve()):
                issues.append(
                    DashboardIssue(
                        code="MANIFEST_PATH_ESCAPE",
                        message="run manifest 通过符号链接离开 runs 目录，已忽略。",
                        severity="error",
                        path=resolved_manifest,
                    )
                )
                continue
            manifest, issue = _read_json(resolved_manifest)
            if issue is not None:
                issues.append(issue)
                continue
            if "strategy_id" in manifest or "method_family" in manifest:
                source, source_issues = _runtime_source(root, resolved_manifest, manifest)
            else:
                source, source_issues = _canonical_source(root, resolved_manifest, manifest)
            runs.append(source)
            issues.extend(source_issues)

    known_metrics = {
        source.files["metrics"].resolve()
        for source in runs
        if "metrics" in source.files
    }
    runs.extend(_legacy_sources(root, known_metrics))
    runs.sort(key=lambda source: (source.kind, source.method_family, source.frequency, source.key))
    if not runs:
        issues.append(
            DashboardIssue(
                code="NO_RUNS",
                message="未发现 canonical、runtime 或 legacy run。",
                severity="warning",
                path=root,
            )
        )
    return WorkspaceInventory(root=root, runs=tuple(runs), issues=tuple(issues))


def read_csv_file(path: Path, *, run_key: str, file_key: str) -> tuple[pd.DataFrame, list[DashboardIssue]]:
    try:
        frame = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), [
            DashboardIssue(
                code="EMPTY_CSV",
                message=f"{file_key} CSV 没有表头或数据。",
                severity="warning",
                run_key=run_key,
                path=path,
            )
        ]
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        return pd.DataFrame(), [
            DashboardIssue(
                code="INVALID_CSV",
                message=f"无法解析 {file_key} CSV: {exc}",
                severity="error",
                run_key=run_key,
                path=path,
            )
        ]
    if frame.empty:
        return frame, [
            DashboardIssue(
                code="NO_ROWS",
                message=f"{file_key} CSV 只有表头，没有数据行。",
                severity="info",
                run_key=run_key,
                path=path,
            )
        ]
    return frame, []


def load_run(
    source: RunSource,
    *,
    file_keys: tuple[str, ...] = RUN_DATA_FIELDS,
) -> RunData:
    data = RunData(source=source)
    targets = {
        "metrics": "metrics",
        "signals": "signals",
        "trades": "trades",
        "orders": "orders",
        "volume": "volume",
        "fills": "fills",
        "failed_orders": "failed_orders",
        "account_snapshots": "account_snapshots",
        "rebalance_plan": "rebalance_plan",
        "ic": "ic",
        "daily_sentiment": "daily_sentiment",
        "news_sentiment": "news_sentiment",
    }
    unknown = sorted(set(file_keys) - set(targets))
    if unknown:
        raise ValueError(f"unknown run file keys: {', '.join(unknown)}")
    for file_key in file_keys:
        attribute = targets[file_key]
        path = source.files.get(file_key)
        if path is None:
            continue
        frame, issues = read_csv_file(path, run_key=source.key, file_key=file_key)
        setattr(data, attribute, frame)
        data.issues.extend(issues)
    if "metrics" in file_keys and "metrics" not in source.files:
        data.issues.append(
            DashboardIssue(
                code="MISSING_METRICS",
                message="run 没有可发现的 metrics 或 returns CSV。",
                severity="error",
                run_key=source.key,
                path=source.run_dir,
            )
        )
    return data
