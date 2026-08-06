"""Audit and normalize legacy Qlib workflows into public template defaults."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from quant_bench.registry import ModelRegistry
from quant_bench.registry.models import apply_feature_dimension

PUBLIC_WINDOWS = {
    "start": "2021-01-01",
    "train_end": "2023-12-31",
    "valid_start": "2024-01-01",
    "valid_end": "2024-12-31",
    "test_start": "2025-01-01",
    "end": "2025-12-31",
}

FREQUENCIES = {
    "1m": ("1min", 525_600),
    "5m": ("5min", 105_120),
    "15m": ("15min", 35_040),
    "30m": ("30min", 17_520),
    "1h": ("60min", 8_760),
    "4h": ("240min", 2_190),
    "1d": ("day", 365),
}

SEED_MODELS = {
    "qlib_lightgbm",
    "qlib_highfreq_lightgbm",
    "qlib_xgboost",
    "qlib_catboost",
    "qlib_double_ensemble",
    "qlib_mlp",
    "qlib_adarnn",
    "qlib_add",
    "qlib_alstm",
    "qlib_alstm_ts",
    "qlib_gats",
    "qlib_gats_ts",
    "qlib_general_ptnn",
    "qlib_gru",
    "qlib_gru_ts",
    "qlib_hist",
    "qlib_igmtf",
    "qlib_krnn",
    "qlib_localformer",
    "qlib_localformer_ts",
    "qlib_lstm",
    "qlib_lstm_ts",
    "qlib_sandwich",
    "qlib_sfm",
    "qlib_tabnet",
    "qlib_tcn",
    "qlib_tcn_ts",
    "qlib_tcts",
    "qlib_tra",
    "qlib_transformer",
    "qlib_transformer_ts",
    "qlib_tft",
}


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


@dataclass
class WorkflowAudit:
    files: int = 0
    yaml_errors: list[str] = field(default_factory=list)
    unsupported_models: list[str] = field(default_factory=list)
    absolute_paths: list[str] = field(default_factory=list)
    stock_defaults: list[str] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)
    non_normalized: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(
            (
                self.yaml_errors,
                self.unsupported_models,
                self.absolute_paths,
                self.stock_defaults,
                self.invalid_labels,
                self.non_normalized,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "passed": self.passed,
            "yaml_errors": self.yaml_errors,
            "unsupported_models": self.unsupported_models,
            "absolute_paths": self.absolute_paths,
            "stock_defaults": self.stock_defaults,
            "invalid_labels": self.invalid_labels,
            "non_normalized": self.non_normalized,
        }


def _infer_frequency(path: Path, config: dict[str, Any]) -> tuple[str, str, int]:
    match = re.search(r"_(1m|5m|15m|30m|1h|4h|1d)\.ya?ml$", path.name, re.IGNORECASE)
    if match:
        key = match.group(1).lower()
    else:
        handler = config.get("data_handler_config", {})
        raw = str(handler.get("freq", "60min"))
        reverse = {qlib: name for name, (qlib, _) in FREQUENCIES.items()}
        key = reverse.get(raw, "1h")
    qlib_frequency, annualization = FREQUENCIES[key]
    return key, qlib_frequency, annualization


def _resolve_model(config: dict[str, Any], registry: ModelRegistry) -> Any:
    model = config.get("task", {}).get("model", {})
    module_path = str(model.get("module_path", ""))
    class_name = str(model.get("class", ""))
    for entry in registry.list():
        if entry.module_path == module_path and entry.class_name == class_name:
            return entry
    return registry.get(class_name)


def _feature_dimensions(handler_class: str) -> tuple[int, int]:
    if handler_class in {"CustomHandler158", "CryptoAlpha52Handler"}:
        return 52, 52
    if handler_class in {"CustomHandler360", "CryptoOHLCV6Handler"}:
        return 6, 6
    if handler_class == "Alpha158":
        return 158, 158
    if handler_class == "Alpha360":
        return 360, 6
    raise ValueError(f"unsupported handler class: {handler_class}")


def _normalize_model_kwargs(entry: Any, flat_dim: int, sequence_dim: int) -> dict[str, Any]:
    kwargs = copy.deepcopy(entry.default_kwargs)
    structural_dim = flat_dim if entry.model_id in {"qlib_mlp", "qlib_tabnet", "qlib_tcts"} else sequence_dim
    kwargs = apply_feature_dimension(entry.model_id, kwargs, structural_dim)
    if entry.model_id == "qlib_tcts":
        kwargs["input_dim"] = flat_dim
    if entry.model_id in SEED_MODELS:
        if entry.model_id == "qlib_catboost":
            kwargs["random_seed"] = 42
        else:
            kwargs["seed"] = 42
    return kwargs


def _normalize_handler(handler: dict[str, Any], qlib_frequency: str) -> tuple[int, int]:
    class_name = str(handler.get("class", "CustomHandler158"))
    flat_dim, sequence_dim = _feature_dimensions(class_name)
    if class_name in {"CustomHandler158", "CryptoAlpha52Handler"}:
        handler["class"] = "CryptoAlpha52Handler"
        handler["module_path"] = "quant_bench.integrations.qlib.handlers"
    elif class_name in {"CustomHandler360", "CryptoOHLCV6Handler"}:
        handler["class"] = "CryptoOHLCV6Handler"
        handler["module_path"] = "quant_bench.integrations.qlib.handlers"
    elif class_name in {"Alpha158", "Alpha360"}:
        handler["module_path"] = "qlib.contrib.data.handler"

    kwargs = handler.setdefault("kwargs", {})
    kwargs.clear()
    kwargs.update(
        {
            "start_time": PUBLIC_WINDOWS["start"],
            "end_time": PUBLIC_WINDOWS["end"],
            "fit_start_time": PUBLIC_WINDOWS["start"],
            "fit_end_time": PUBLIC_WINDOWS["train_end"],
            "freq": qlib_frequency,
            "instruments": "all",
            "infer_processors": [
                {
                    "class": "RobustZScoreNorm",
                    "module_path": "qlib.data.dataset.processor",
                    "kwargs": {
                        "fields_group": "feature",
                        "clip_outlier": True,
                        "fit_start_time": PUBLIC_WINDOWS["start"],
                        "fit_end_time": PUBLIC_WINDOWS["train_end"],
                    },
                },
                {
                    "class": "Fillna",
                    "module_path": "qlib.data.dataset.processor",
                    "kwargs": {"fields_group": "feature", "fill_value": 0},
                },
            ],
            "learn_processors": [
                {
                    "class": "DropnaLabel",
                    "module_path": "qlib.data.dataset.processor",
                }
            ],
            "label": ["Ref($close, -1) / $close - 1"],
        }
    )
    return flat_dim, sequence_dim


def _normalize_dataset(dataset: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    dataset_class = str(dataset.get("class", "DatasetH"))
    if dataset_class == "MTSDatasetH":
        dataset["module_path"] = "qlib.contrib.data.dataset"
    else:
        dataset["module_path"] = "qlib.data.dataset"
    old_kwargs = dataset.get("kwargs", {})
    handler = copy.deepcopy(old_kwargs.get("handler", {}))
    flat_dim, sequence_dim = _normalize_handler(handler, "60min")
    dataset_kwargs: dict[str, Any] = {
        "handler": handler,
        "segments": {
            "train": [PUBLIC_WINDOWS["start"], PUBLIC_WINDOWS["train_end"]],
            "valid": [PUBLIC_WINDOWS["valid_start"], PUBLIC_WINDOWS["valid_end"]],
            "test": [PUBLIC_WINDOWS["test_start"], PUBLIC_WINDOWS["end"]],
        },
    }
    if dataset_class == "TSDatasetH":
        dataset_kwargs["step_len"] = 60
    elif dataset_class == "MTSDatasetH":
        dataset_kwargs.update(
            {
                "seq_len": 60,
                "num_states": 1,
                "batch_size": 256,
                "n_samples": None,
                "memory_mode": "sample",
                "drop_last": True,
            }
        )
    dataset["kwargs"] = dataset_kwargs
    return handler, flat_dim, sequence_dim


def normalize_workflow(
    config: dict[str, Any], path: Path, registry: ModelRegistry | None = None
) -> dict[str, Any]:
    registry = registry or ModelRegistry()
    result = copy.deepcopy(config)
    frequency_id, qlib_frequency, annualization = _infer_frequency(path, result)
    entry = _resolve_model(result, registry)

    task = result.setdefault("task", {})
    dataset = task.setdefault("dataset", {})
    handler, flat_dim, sequence_dim = _normalize_dataset(dataset)
    handler["kwargs"]["freq"] = qlib_frequency

    model = task.setdefault("model", {})
    model.clear()
    model.update(
        {
            "class": entry.class_name,
            "module_path": entry.module_path,
            "kwargs": _normalize_model_kwargs(entry, flat_dim, sequence_dim),
        }
    )

    strategy_config = {
        "strategy": {
            "class": "ThresholdTopkDropoutStrategy",
            "module_path": "quant_bench.integrations.qlib.strategy",
            "kwargs": {
                "signal": "<PRED>",
                "topk": 5,
                "min_score": 0.0,
                "max_dropout": 1,
                "enable_short": False,
            },
        },
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": qlib_frequency,
                "generate_portfolio_metrics": True,
            },
        },
        "backtest": {
            "start_time": PUBLIC_WINDOWS["test_start"],
            "end_time": PUBLIC_WINDOWS["end"],
            "account": 100_000,
            "benchmark": "BTC-USDT",
            "exchange_kwargs": {
                "freq": qlib_frequency,
                "limit_threshold": None,
                "deal_price": "open",
                "open_cost": 0.001,
                "close_cost": 0.001,
                "min_cost": 0,
                "trade_unit": None,
            },
        },
    }
    task["record"] = [
        {
            "class": "SignalRecord",
            "module_path": "qlib.workflow.record_temp",
            "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"},
        },
        {
            "class": "SigAnaRecord",
            "module_path": "qlib.workflow.record_temp",
            "kwargs": {"ana_long_short": False, "ann_scaler": annualization},
        },
        {
            "class": "CryptoPortAnaRecord",
            "module_path": "quant_bench.integrations.qlib.records",
            "kwargs": {"config": strategy_config},
        },
    ]

    normalized = {
        "template_version": "quant-bench-public-v1",
        "experiment_name": f"quant-bench-{entry.model_id}-{frequency_id}",
        "qlib_init": {"provider_uri": f"./qlib_data/{frequency_id}", "region": "cn"},
        "market": "all",
        "benchmark": "BTC-USDT",
        "model_id": entry.model_id,
        "feature_set_id": {
            "CryptoAlpha52Handler": "crypto_alpha52_v1",
            "CryptoOHLCV6Handler": "crypto_ohlcv6_v1",
            "Alpha158": "qlib_alpha158_v1",
            "Alpha360": "qlib_alpha360_v1",
        }[handler["class"]],
        "task": task,
    }
    return normalized


def dump_workflow(config: dict[str, Any]) -> str:
    return yaml.dump(
        config,
        Dumper=NoAliasDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=110,
    )


def normalize_workflow_file(path: Path, *, write: bool = False) -> tuple[bool, str]:
    original = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(original)
    if not isinstance(parsed, dict):
        raise ValueError(f"workflow root must be a mapping: {path}")
    rendered = dump_workflow(normalize_workflow(parsed, path))
    changed = rendered != original
    if write and changed:
        path.write_text(rendered, encoding="utf-8", newline="\n")
    return changed, rendered


def audit_workflows(root: Path, *, require_normalized: bool = True) -> WorkflowAudit:
    audit = WorkflowAudit()
    registry = ModelRegistry()
    for path in sorted(root.glob("**/*.yaml")):
        audit.files += 1
        relative = str(path.relative_to(root))
        try:
            text = path.read_text(encoding="utf-8")
            config = yaml.safe_load(text)
            if not isinstance(config, dict):
                raise ValueError("workflow root is not a mapping")
        except Exception as exc:
            audit.yaml_errors.append(f"{relative}: {type(exc).__name__}: {exc}")
            continue
        try:
            _resolve_model(config, registry)
        except Exception as exc:
            audit.unsupported_models.append(f"{relative}: {exc}")
        if re.search(r"(?:^|[\s:'\"])/home/|[A-Za-z]:\\", text):
            audit.absolute_paths.append(relative)
        if "csi300" in text.lower() or "cn_data" in text.lower():
            audit.stock_defaults.append(relative)
        handler = config.get("task", {}).get("dataset", {}).get("kwargs", {}).get("handler", {})
        labels = handler.get("kwargs", {}).get("label", []) if isinstance(handler, dict) else []
        for label in labels if isinstance(labels, list) else []:
            depth = 0
            for character in str(label):
                depth += character == "("
                depth -= character == ")"
                if depth < 0:
                    break
            if depth != 0:
                audit.invalid_labels.append(f"{relative}: {label}")
        if require_normalized:
            try:
                rendered = dump_workflow(normalize_workflow(config, path, registry))
                if rendered != text:
                    audit.non_normalized.append(relative)
            except Exception as exc:
                audit.non_normalized.append(f"{relative}: {exc}")
    return audit
