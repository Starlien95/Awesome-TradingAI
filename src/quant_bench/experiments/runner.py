"""Offline-first experiment lifecycle."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import cast, overload

import pandas as pd
from platformdirs import user_data_path

from quant_bench import __version__
from quant_bench.artifacts import LocalArtifactStore
from quant_bench.config.loader import config_sha256, render_config
from quant_bench.config.models import ExperimentConfig, TimeRange
from quant_bench.contracts import DatasetView, ExperimentResult, RunManifest
from quant_bench.contracts.models import CodeVersion, EnvironmentInfo
from quant_bench.data.market import build_dataset_manifest, load_market_data, validation_report
from quant_bench.evaluation import evaluate_predictions
from quant_bench.features import CryptoOHLCV6V1
from quant_bench.labels import forward_return
from quant_bench.registry import ModelRegistry


def _new_run_id() -> str:
    prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _git_version() -> CodeVersion:
    try:
        root = Path(__file__).resolve().parents[3]
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).strip()
        )
        return CodeVersion(git_commit=commit, dirty=dirty)
    except (OSError, subprocess.SubprocessError):
        return CodeVersion()


def _environment() -> EnvironmentInfo:
    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "pydantic", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        pyqlib = importlib.metadata.version("pyqlib")
    except importlib.metadata.PackageNotFoundError:
        pyqlib = None
    return EnvironmentInfo(
        python=platform.python_version(),
        quant_bench=__version__,
        pyqlib=pyqlib,
        packages=packages,
    )


@overload
def _slice(frame: pd.DataFrame, time_range: TimeRange) -> pd.DataFrame: ...


@overload
def _slice(frame: pd.Series, time_range: TimeRange) -> pd.Series: ...


def _slice(frame: pd.DataFrame | pd.Series, time_range: TimeRange) -> pd.DataFrame | pd.Series:
    timestamps = frame.index.get_level_values("timestamp")
    start = pd.Timestamp(time_range.start)
    end = pd.Timestamp(time_range.end)
    sliced = frame.loc[(timestamps >= start) & (timestamps <= end)]
    if isinstance(frame, pd.DataFrame):
        return cast(pd.DataFrame, sliced)
    return cast(pd.Series, sliced)


def _model_card(config: ExperimentConfig) -> str:
    return (
        f"# Model card: {config.model.model_id}\n\n"
        f"- Backend: `{config.model.backend}`\n"
        f"- Plugin: `{config.model.plugin}`\n"
        f"- Feature set: `{config.features.feature_set_id}`\n"
        f"- Label: `{config.label.label_id}`\n"
        f"- Protocol: `{config.protocol.protocol_id}`\n"
        "- Intended use: offline benchmark and integration testing.\n"
        "- Limitation: this run is not evidence of live trading performance.\n"
    )


class Experiment:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> ExperimentResult:
        if self.config.runtime.allow_network:
            raise ValueError("Experiment.run is offline-only")
        if self.config.model.backend != "builtin":
            raise ValueError("Qlib models must use the explicit qlib workflow command")
        if self.config.features.plugin != "crypto_ohlcv6_v1":
            raise ValueError(
                "the canonical runner currently supports crypto_ohlcv6_v1; Qlib feature sets use qlib run"
            )

        workspace = self.config.experiment.workspace or user_data_path("quant-bench", appauthor=False)
        workspace = Path(workspace).expanduser().resolve()
        run_id = _new_run_id()
        run_dir = workspace / "runs" / run_id
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        store = LocalArtifactStore(run_dir)
        started_at = datetime.now(timezone.utc)
        config_hash = config_sha256(self.config)
        manifest = RunManifest(
            run_id=run_id,
            status="validating",
            recipe_id=self.config.recipe_id,
            protocol_id=self.config.protocol.protocol_id,
            config_sha256=config_hash,
            code=_git_version(),
            environment=_environment(),
            seeds=self.config.protocol.seeds,
            started_at=started_at,
        )

        try:
            store.write_text("resolved_config.yaml", render_config(self.config), "application/yaml")
            market, source_path = load_market_data(self.config.data.path)
            requested_symbols = set(self.config.universe.symbols)
            available_symbols = set(market["symbol"].unique())
            missing_symbols = requested_symbols - available_symbols
            if missing_symbols:
                raise ValueError(
                    f"configured symbols are missing from the dataset: {sorted(missing_symbols)}"
                )
            market = market[market["symbol"].isin(requested_symbols)].copy()
            dataset_manifest = build_dataset_manifest(self.config, market, source_path)
            manifest.dataset_sha256 = dataset_manifest.content_sha256
            store.write_json(
                "dataset_manifest.json", dataset_manifest, schema_id="quant-bench.dataset-manifest.v1"
            )
            store.write_json(
                "validation.json", validation_report(market), schema_id="quant-bench.data-validation.v1"
            )

            feature_plugin = CryptoOHLCV6V1()
            features = feature_plugin.transform(market)
            manifest.feature_schema_sha256 = feature_plugin.schema_sha256()
            labels = forward_return(market, self.config.label.horizon_bars, self.config.label.scale)
            labels = labels.reindex(features.index)

            split = self.config.protocol.split
            dataset = DatasetView(
                train_features=_slice(features, split.train),
                train_labels=_slice(labels, split.train),
                valid_features=_slice(features, split.valid),
                valid_labels=_slice(labels, split.valid),
                test_features=_slice(features, split.test),
                test_labels=_slice(labels, split.test),
            )
            for name, value in {
                "train": dataset.train_features,
                "valid": dataset.valid_features,
                "test": dataset.test_features,
            }.items():
                if value.empty:
                    raise ValueError(f"{name} split is empty after feature construction")

            manifest.status = "running"
            registry = ModelRegistry()
            entry = registry.get(self.config.model.model_id)
            if entry.backend != "builtin":
                raise ValueError(f"model {entry.model_id} is not supported by the canonical runner")
            module = __import__(entry.module_path, fromlist=[entry.class_name])
            plugin_class = getattr(module, entry.class_name)
            plugin = plugin_class(**self.config.model.parameters)
            fitted = plugin.fit(dataset, context={"run_id": run_id, "seed": self.config.experiment.seed})
            plugin.save(fitted, store)
            prediction_frame = plugin.predict(fitted, dataset).frame
            metrics, predictions, returns = evaluate_predictions(
                prediction_frame,
                dataset.test_labels,
                self.config.protocol,
                self.config.frequency,
            )
            store.write_csv("predictions.csv", predictions, schema_id="quant-bench.predictions.v1")
            store.write_csv("backtest/returns.csv", returns, schema_id="quant-bench.returns.v1")
            store.write_json("metrics.json", metrics, schema_id="quant-bench.metrics.v1")
            store.write_text("model/model_card.md", _model_card(self.config), "text/markdown; charset=utf-8")
            store.write_text(
                "reports/summary.md",
                "# Offline benchmark summary\n\n```json\n"
                + json.dumps(metrics, indent=2, sort_keys=True)
                + "\n```\n",
                "text/markdown; charset=utf-8",
            )

            manifest.status = "completed"
            manifest.completed_at = datetime.now(timezone.utc)
            manifest.artifacts = list(store.artifacts)
            store.write_checksums()
            store.write_json(
                "run_manifest.json",
                manifest,
                schema_id="quant-bench.run-manifest.v1",
                track=False,
            )
            return ExperimentResult(
                run_id=run_id,
                run_dir=run_dir,
                status="completed",
                metrics=metrics,
                manifest_path=run_dir / "run_manifest.json",
            )
        except Exception as exc:
            manifest.status = "failed"
            manifest.completed_at = datetime.now(timezone.utc)
            manifest.error_type = type(exc).__name__
            manifest.error_message = str(exc)
            manifest.artifacts = list(store.artifacts)
            store.write_checksums()
            store.write_json(
                "run_manifest.json",
                manifest,
                schema_id="quant-bench.run-manifest.v1",
                visibility="private",
                track=False,
            )
            raise
