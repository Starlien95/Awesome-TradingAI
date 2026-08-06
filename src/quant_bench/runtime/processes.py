"""Catalog and launch helpers for the optional trading runtime.

The research package never calls :func:`start_process`.  A caller must export
and review local configuration, provide model artifacts, select demo or live
mode, and pass the corresponding confirmation token.
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RuntimeModel:
    strategy_id: str
    config_name: str
    adapter: str


@dataclass(frozen=True)
class RuntimeProcess:
    process_id: str
    frequency: str
    interval_seconds: int
    method_family: str
    models: tuple[RuntimeModel, ...]
    macrohft_data: bool = False


PROCESSES: dict[str, RuntimeProcess] = {
    "traditional_15m": RuntimeProcess(
        "traditional_15m",
        "15m",
        900,
        "traditional_ml",
        (
            RuntimeModel(
                "DoubleEnsemble_15m",
                "config_double_ensemble.yaml",
                "quant_bench.runtime.adapters.double_ensemble_adapter:DoubleEnsembleAdapter",
            ),
            RuntimeModel(
                "TabNet_15m",
                "config_tabnet.yaml",
                "quant_bench.runtime.adapters.tabnet_adapter:TabNetAdapter",
            ),
        ),
    ),
    "traditional_1h": RuntimeProcess(
        "traditional_1h",
        "1h",
        3600,
        "traditional_ml",
        (
            RuntimeModel(
                "MLP_1h", "config_mlp.yaml", "quant_bench.runtime.adapters.mlp_adapter:MLPAdapter"
            ),
            RuntimeModel(
                "TCN_1h", "config_tcn.yaml", "quant_bench.runtime.adapters.tcn_adapter:TCNAdapter"
            ),
        ),
    ),
    "traditional_4h": RuntimeProcess(
        "traditional_4h",
        "4h",
        14_400,
        "traditional_ml",
        (
            RuntimeModel(
                "XGBoost_4h",
                "config_xgboost.yaml",
                "quant_bench.runtime.adapters.xgboost_adapter:XGBoostAdapter",
            ),
            RuntimeModel(
                "GATS_4h", "config_gats.yaml", "quant_bench.runtime.adapters.gats_adapter:GATSAdapter"
            ),
            RuntimeModel(
                "LSTM_4h", "config_lstm.yaml", "quant_bench.runtime.adapters.lstm_adapter:LSTMAdapter"
            ),
            RuntimeModel(
                "TRA_4h", "config_tra.yaml", "quant_bench.runtime.adapters.tra_adapter:TRAAdapter"
            ),
        ),
    ),
    **{
        f"macrohft_{frequency}": RuntimeProcess(
            f"macrohft_{frequency}",
            frequency,
            interval,
            "reinforcement_learning",
            (
                RuntimeModel(
                    f"MacroHFTv1_{frequency}",
                    "config_macrohft_v1.yaml",
                    "quant_bench.runtime.adapters.macrohft_v1_adapter:MacroHFTV1Adapter",
                ),
            ),
            macrohft_data=True,
        )
        for frequency, interval in (("5m", 300), ("15m", 900), ("1h", 3600), ("4h", 14_400))
    },
}


def list_processes() -> list[dict[str, Any]]:
    """Return serializable runtime process metadata."""

    return [
        {
            "process_id": process.process_id,
            "frequency": process.frequency,
            "method_family": process.method_family,
            "strategies": [model.strategy_id for model in process.models],
            "config_files": [model.config_name for model in process.models],
        }
        for process in PROCESSES.values()
    ]


def _resource_config(process: RuntimeProcess, name: str) -> resources.abc.Traversable:
    return (
        resources.files("quant_bench")
        .joinpath("resources")
        .joinpath("runtime")
        .joinpath("timeframes")
        .joinpath(process.frequency)
        .joinpath(name)
    )


def export_process_configs(process_id: str, output_dir: Path, *, force: bool = False) -> list[Path]:
    """Export editable runtime templates for one process."""

    process = PROCESSES[process_id]
    target = output_dir.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model in process.models:
        destination = target / model.config_name
        if destination.exists() and not force:
            raise FileExistsError(f"refusing to replace existing config without --force: {destination}")
        destination.write_text(_resource_config(process, model.config_name).read_text(encoding="utf-8"), encoding="utf-8")
        written.append(destination)
    return written


def inspect_process_configs(process_id: str, config_dir: Path) -> list[dict[str, Any]]:
    """Validate configuration presence, mode, credentials, and model paths without network access."""

    from quant_bench.runtime.core.config import ConfigManager

    process = PROCESSES[process_id]
    root = config_dir.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for model in process.models:
        path = root / model.config_name
        if not path.is_file():
            rows.append({"strategy_id": model.strategy_id, "config": str(path), "valid": False, "error": "missing"})
            continue
        cfg = ConfigManager(str(path))
        model_path = Path(cfg.model_path).expanduser()
        if not model_path.is_absolute():
            model_path = root / model_path
        credentials_present = bool(cfg.api_key and cfg.secret_key and cfg.passphrase)
        model_exists = model_path.is_file()
        rows.append(
            {
                "strategy_id": model.strategy_id,
                "config": str(path),
                "valid": True,
                "is_simulated": cfg.is_simulated,
                "credentials_present": credentials_present,
                "model_path": str(model_path.resolve()),
                "model_exists": model_exists,
                "ready": credentials_present and model_exists,
            }
        )
    return rows


def _import_symbol(value: str) -> type[Any]:
    module_name, symbol_name = value.split(":", 1)
    return getattr(importlib.import_module(module_name), symbol_name)


def _confirmation_token(mode: str) -> str:
    return "DEMO_ORDERS" if mode == "demo" else "LIVE_ORDERS"


def start_process(
    process_id: str,
    config_dir: Path,
    workspace: Path,
    *,
    mode: str,
    confirm: str,
) -> None:
    """Start a long-running runtime process after all local safety checks pass."""

    if mode not in {"demo", "live"}:
        raise ValueError("runtime start supports only demo or live; use research commands for offline work")
    expected = _confirmation_token(mode)
    if confirm != expected:
        raise PermissionError(f"refusing to start {mode} runtime; pass --confirm {expected}")

    from quant_bench.runtime.core.config import ConfigManager
    from quant_bench.runtime.core.runner import TimeframeRunner

    process = PROCESSES[process_id]
    config_root = config_dir.expanduser().resolve()
    workspace_root = workspace.expanduser().resolve()
    run_root = workspace_root / "runtime" / "runs" / process.method_family
    log_root = workspace_root / "runtime" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_root / f"{process_id}.log", encoding="utf-8")],
    )

    loaded: list[tuple[RuntimeModel, ConfigManager, Any]] = []
    for model in process.models:
        config_path = config_root / model.config_name
        if not config_path.is_file():
            raise FileNotFoundError(
                f"missing runtime config: {config_path}; run quant-bench runtime init {process_id} first"
            )
        cfg = ConfigManager(str(config_path))
        expected_simulated = mode == "demo"
        if cfg.is_simulated != expected_simulated:
            raise ValueError(
                f"{config_path}: api.is_simulated must be {str(expected_simulated).lower()} for mode={mode}"
            )
        if not (cfg.api_key and cfg.secret_key and cfg.passphrase):
            raise RuntimeError(f"{model.strategy_id}: required OKX environment variables are not set")
        artifact = Path(cfg.model_path).expanduser()
        if not artifact.is_absolute():
            artifact = config_root / artifact
        if not artifact.is_file():
            raise FileNotFoundError(f"model artifact not found: {artifact}")
        adapter = _import_symbol(model.adapter)()
        adapter.load_model(str(artifact.resolve()))
        subagents_dir = cfg.get("model.subagents_dir")
        if subagents_dir and hasattr(adapter, "set_subagents_dir"):
            candidate = Path(str(subagents_dir)).expanduser()
            if not candidate.is_absolute():
                candidate = config_root / candidate
            adapter.set_subagents_dir(str(candidate.resolve()))
        loaded.append((model, cfg, adapter))

    runner = TimeframeRunner(
        timeframe=process.frequency,
        update_interval_sec=process.interval_seconds,
        method_family=process.method_family,
        run_root=str(run_root),
    )
    if process.macrohft_data:
        data_manager_type = _import_symbol(
            "quant_bench.runtime.adapters.macrohft_data_manager:MacroHFTDataManager"
        )
        runner.data_manager = data_manager_type(
            api_client=runner.shared_api_client,
            timeframe_str=runner.timeframe,
        )
    for model, cfg, adapter in loaded:
        runner.register_model(model.strategy_id, cfg, adapter)
    runner.start()


def runtime_registry_json() -> str:
    """Return the packaged strategy registry as formatted JSON for diagnostics."""

    source = (
        resources.files("quant_bench")
        .joinpath("resources")
        .joinpath("runtime")
        .joinpath("strategies.yaml")
    )
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    return json.dumps(payload, indent=2, ensure_ascii=False)
