"""Single command surface for research, configuration, plugins, and artifacts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from platformdirs import user_data_path

from quant_bench import __version__
from quant_bench.artifacts import verify_run
from quant_bench.config.legacy_workflows import audit_workflows, normalize_workflow_file
from quant_bench.config.loader import builtin_config_path, load_config, render_config
from quant_bench.contracts import DatasetManifest, RunManifest
from quant_bench.data import load_market_data
from quant_bench.data.market import validation_report
from quant_bench.experiments import Experiment
from quant_bench.integrations.qlib.workflow import run_workflow, validate_workflow
from quant_bench.registry import ModelRegistry


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def _workspace(value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else user_data_path("quant-bench", appauthor=False)


def _cmd_quickstart(args: argparse.Namespace) -> int:
    config = load_config("crypto_smoke_v1")
    config.experiment.workspace = _workspace(args.workspace)
    result = Experiment(config).run()
    print(_json(result.model_dump(mode="json")))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set or [])
    if args.workspace:
        config.experiment.workspace = _workspace(args.workspace)
    result = Experiment(config).run()
    print(_json(result.model_dump(mode="json")))
    return 0


def _doctor_payload(workspace: Path) -> dict[str, Any]:
    registry = ModelRegistry()
    dependencies: dict[str, str | None] = {}
    for distribution in ("quant-bench", "pyqlib", "lightgbm", "xgboost", "catboost", "torch"):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution] = None
    workspace.parent.mkdir(parents=True, exist_ok=True)
    credentials = {
        name: bool(os.environ.get(name))
        for name in (
            "OKX_API_KEY_SIMU",
            "OKX_SECRET_KEY_SIMU",
            "OKX_PASSPHRASE",
            "OKX_API_KEY",
            "OKX_SECRET_KEY",
        )
    }
    available = sum(1 for entry in registry.list() if registry.availability(entry.model_id)[0])
    return {
        "python": platform.python_version(),
        "quant_bench": __version__,
        "platform": platform.platform(),
        "workspace": str(workspace),
        "workspace_parent_writable": os.access(workspace.parent, os.W_OK),
        "dependencies": dependencies,
        "model_catalog_entries": len(registry.list()),
        "models_with_importable_modules": available,
        "credentials_present": credentials,
        "network_checked": False,
    }


def _cmd_doctor(args: argparse.Namespace) -> int:
    payload = _doctor_payload(_workspace(args.workspace))
    print(_json(payload))
    return 0 if payload["workspace_parent_writable"] else 1


def _cmd_models_list(args: argparse.Namespace) -> int:
    registry = ModelRegistry()
    rows = []
    for entry in registry.list():
        available, detail = registry.availability(entry.model_id)
        rows.append(
            {
                "model_id": entry.model_id,
                "family": entry.family,
                "backend": entry.backend,
                "status": entry.status,
                "available": available,
                "detail": detail if args.verbose else None,
                "extra": entry.capabilities.optional_extra,
                "dataset_kinds": entry.capabilities.dataset_kinds,
            }
        )
    print(_json(rows))
    return 0


def _cmd_models_show(args: argparse.Namespace) -> int:
    entry = ModelRegistry().get(args.model)
    print(entry.model_dump_json(indent=2))
    return 0


def _cmd_config_list(args: argparse.Namespace) -> int:
    del args
    path = builtin_config_path("crypto_smoke_v1")
    print(_json([{"recipe_id": "crypto_smoke_v1", "path": str(path)}]))
    return 0


def _cmd_config_render(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set or [])
    print(render_config(config), end="")
    return 0


def _cmd_config_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set or [])
    print(_json({"valid": True, "recipe_id": config.recipe_id, "model_id": config.model.model_id}))
    return 0


def _cmd_config_init(args: argparse.Namespace) -> int:
    registry = ModelRegistry()
    config = load_config("crypto_smoke_v1")
    feature_dimensions = {
        "crypto_ohlcv6_v1": 6,
        "crypto_alpha52_v1": 52,
        "qlib_alpha158_v1": 158,
        "qlib_alpha360_v1": 360,
    }
    feature_dim = feature_dimensions[args.feature_set]
    model = registry.to_model_spec(args.model, feature_dim=feature_dim)
    config.recipe_id = f"{model.model_id}_{args.feature_set}_v1"
    config.experiment.name = config.recipe_id
    config.model = model
    config.features.feature_set_id = args.feature_set
    config.features.plugin = args.feature_set
    target = Path(args.output).expanduser().resolve()
    if target.exists() and not args.force:
        raise FileExistsError(f"refusing to replace existing file without --force: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_config(config), encoding="utf-8", newline="\n")
    print(_json({"written": str(target), "model_id": model.model_id}))
    return 0


def _cmd_data_validate(args: argparse.Namespace) -> int:
    frame, path = load_market_data(args.path)
    print(_json({"path": str(path), **validation_report(frame)}))
    return 0


def _cmd_data_pull(args: argparse.Namespace) -> int:
    from quant_bench.data.qlib_pipeline import run_download

    output = run_download(args.timeframe, args.start, args.symbols, args.output)
    if output is None:
        return 1
    print(_json({"output": str(Path(output).expanduser().resolve()), "network": "OKX via CCXT"}))
    return 0


def _cmd_data_split(args: argparse.Namespace) -> int:
    from quant_bench.data.qlib_pipeline import run_split

    output = run_split(args.timeframe, args.input, args.output_dir)
    print(_json({"output_dir": str(Path(output).expanduser().resolve())}))
    return 0


def _cmd_data_qlib_dump(args: argparse.Namespace) -> int:
    from quant_bench.data.qlib_pipeline import run_dump

    report = run_dump(
        mode=args.mode,
        timeframe=args.timeframe,
        data_path=args.data_path,
        qlib_dir=args.qlib_dir,
        backup_dir=args.backup_dir,
        freq=args.frequency,
        max_workers=args.max_workers,
        date_field_name=args.date_field_name,
        symbol_field_name=args.symbol_field_name,
        include_fields=args.include_fields,
    )
    print(_json(report))
    return 0


def _cmd_data_pipeline(args: argparse.Namespace) -> int:
    from quant_bench.data.qlib_pipeline import run_download, run_dump, run_split

    raw = run_download(args.timeframe, args.start, args.symbols, args.output)
    if raw is None:
        return 1
    split_dir = run_split(args.timeframe, raw, args.split_dir)
    dump_report = run_dump(
        mode="dump_all",
        timeframe=args.timeframe,
        data_path=split_dir,
        qlib_dir=args.qlib_dir,
        freq=args.frequency,
        max_workers=args.max_workers,
        date_field_name=args.date_field_name,
        symbol_field_name=args.symbol_field_name,
        include_fields=args.include_fields,
    )
    print(
        _json(
            {
                "raw_csv": str(Path(raw).expanduser().resolve()),
                "split_dir": str(Path(split_dir).expanduser().resolve()),
                "qlib_dir": str(Path(args.qlib_dir).expanduser().resolve()),
                "dump": dump_report,
            }
        )
    )
    return 0


def _cmd_artifacts_verify(args: argparse.Namespace) -> int:
    errors = verify_run(Path(args.run_dir))
    print(_json({"valid": not errors, "errors": errors}))
    return 0 if not errors else 1


def _cmd_artifacts_inspect(args: argparse.Namespace) -> int:
    path = Path(args.run_dir).expanduser().resolve() / "run_manifest.json"
    manifest = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    print(manifest.model_dump_json(indent=2))
    return 0


def _cmd_artifacts_promote(args: argparse.Namespace) -> int:
    from quant_bench.runtime.promotion import promote_artifact

    manifest = promote_artifact(
        Path(args.source),
        _workspace(args.workspace),
        model_id=args.model,
        frequency=args.frequency,
        config_path=Path(args.config) if args.config else None,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(_json(manifest))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    runs_root = _workspace(args.workspace) / "runs"
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(runs_root.glob("*/run_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") != "completed":
            continue
        metrics_path = manifest_path.parent / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        rows.append(
            {
                "run_id": manifest.get("run_id"),
                "protocol_id": manifest.get("protocol_id"),
                "dataset_sha256": manifest.get("dataset_sha256"),
                "total_return": metrics.get("total_return"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
            }
        )
    comparable = len({(row["protocol_id"], row["dataset_sha256"]) for row in rows}) <= 1
    print(_json({"comparable": comparable, "runs": rows}))
    return 0 if comparable or args.allow_incomparable else 2


def _cmd_sweep(args: argparse.Namespace) -> int:
    from quant_bench.tuning import run_grid_sweep

    results = run_grid_sweep(
        args.config,
        _workspace(args.workspace),
        args.param,
        study_name=args.study_name,
        max_trials=args.max_trials,
        resume=args.resume,
    )
    print(_json({"study_name": args.study_name, "trials": str(results)}))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from quant_bench.reporting import write_workspace_summary

    target = write_workspace_summary(
        _workspace(args.workspace),
        Path(args.output) if args.output else None,
    )
    print(_json({"report": str(target)}))
    return 0


def _cmd_qlib_validate(args: argparse.Namespace) -> int:
    errors = validate_workflow(Path(args.workflow))
    print(_json({"valid": not errors, "errors": errors}))
    return 0 if not errors else 1


def _cmd_qlib_run(args: argparse.Namespace) -> int:
    if args.workflow:
        workflow_path = Path(args.workflow)
    else:
        if not (args.feature_set and args.model and args.frequency):
            raise ValueError("provide WORKFLOW or all of --feature-set, --model, and --frequency")
        from quant_bench.integrations.qlib.catalog import resolve_workflow

        workflow_path = resolve_workflow(args.feature_set, args.model, args.frequency)
    run_dir = run_workflow(
        workflow_path,
        _workspace(args.workspace),
        args.experiment_name,
        Path(args.provider_uri) if args.provider_uri else None,
    )
    print(_json({"run_dir": str(run_dir)}))
    return 0


def _cmd_workflows_list(args: argparse.Namespace) -> int:
    from quant_bench.integrations.qlib.catalog import list_workflows

    rows = list_workflows(
        feature_set=args.feature_set,
        model=args.model,
        frequency=args.frequency,
    )
    print(_json(rows))
    return 0


def _cmd_qlib_compat(args: argparse.Namespace) -> int:
    modules = {
        "train": "quant_bench.integrations.qlib.legacy.train_model",
        "backtest": "quant_bench.integrations.qlib.legacy.backtest_model",
        "tune": "quant_bench.integrations.qlib.legacy.hyperparam_sweep",
        "auto-tune": "quant_bench.integrations.qlib.legacy.auto_tune_trade_models",
        "recompute-stats": "quant_bench.integrations.qlib.legacy.recompute_prediction_stats",
        "visualize-sweep": "quant_bench.integrations.qlib.legacy.visualize_results",
        "verify-predictions": "quant_bench.integrations.qlib.legacy.verify_prediction_consistency",
    }
    workspace = _workspace(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    forwarded = list(args.forwarded)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    completed = subprocess.run(
        [sys.executable, "-m", modules[args.compat_command], *forwarded],
        cwd=workspace,
        check=False,
    )
    return int(completed.returncode)


def _cmd_workflows_audit(args: argparse.Namespace) -> int:
    audit = audit_workflows(Path(args.root), require_normalized=args.require_normalized)
    print(_json(audit.as_dict()))
    return 0 if audit.passed else 1


def _cmd_workflows_normalize(args: argparse.Namespace) -> int:
    root = Path(args.root)
    changed = 0
    files = sorted(root.glob("**/*.yaml"))
    for path in files:
        file_changed, _ = normalize_workflow_file(path, write=args.write)
        changed += int(file_changed)
    print(_json({"files": len(files), "changed": changed, "written": bool(args.write)}))
    return 0


def _cmd_plugins_list(args: argparse.Namespace) -> int:
    del args
    points = [
        {"name": point.name, "value": point.value, "distribution": point.dist.name if point.dist else None}
        for point in ModelRegistry.external_entry_points()
    ]
    print(_json(points))
    return 0


def _cmd_plugins_scaffold(args: argparse.Namespace) -> int:
    target = Path(args.target).expanduser().resolve() / args.name
    if target.exists():
        raise FileExistsError(f"target already exists: {target}")
    target.mkdir(parents=True)
    class_name = "".join(part.capitalize() for part in args.name.split("_")) + "Plugin"
    (target / "plugin.py").write_text(
        "from quant_bench.contracts import ModelCapabilities\n\n\n"
        f"class {class_name}:\n"
        f"    plugin_id = {args.name!r}\n"
        "    contract_version = '1'\n\n"
        "    def capabilities(self):\n"
        "        return ModelCapabilities(task_types=['regression'], feature_shapes=['tabular'], dataset_kinds=['canonical'])\n",
        encoding="utf-8",
        newline="\n",
    )
    (target / "README.md").write_text(
        f"# {args.name}\n\nImplement the ModelPlugin contract and add contract tests before publishing.\n",
        encoding="utf-8",
        newline="\n",
    )
    print(_json({"written": str(target)}))
    return 0


def _cmd_schema_export(args: argparse.Namespace) -> int:
    from quant_bench.methods import MethodDescriptor, MethodRunResult, MethodRunSpec

    target = Path(args.output).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    schemas = {
        "dataset-manifest-v1.json": DatasetManifest.model_json_schema(),
        "method-descriptor-v1.json": MethodDescriptor.model_json_schema(),
        "method-run-result-v1.json": MethodRunResult.model_json_schema(),
        "method-run-spec-v1.json": MethodRunSpec.model_json_schema(),
        "run-manifest-v1.json": RunManifest.model_json_schema(),
    }
    for name, schema in schemas.items():
        (target / name).write_text(_json(schema) + "\n", encoding="utf-8", newline="\n")
    print(_json({"written": [str(target / name) for name in schemas]}))
    return 0


def _cmd_runtime_list(args: argparse.Namespace) -> int:
    del args
    from quant_bench.runtime.processes import list_processes

    print(_json(list_processes()))
    return 0


def _cmd_runtime_init(args: argparse.Namespace) -> int:
    from quant_bench.runtime.processes import export_process_configs

    written = export_process_configs(args.process, Path(args.output_dir), force=args.force)
    print(_json({"process_id": args.process, "written": [str(path) for path in written]}))
    return 0


def _cmd_runtime_check(args: argparse.Namespace) -> int:
    from quant_bench.runtime.processes import inspect_process_configs

    rows = inspect_process_configs(args.process, Path(args.config_dir))
    valid = bool(rows) and all(row.get("valid") for row in rows)
    ready = valid and all(row.get("ready") for row in rows)
    print(_json({"process_id": args.process, "valid": valid, "ready": ready, "configs": rows}))
    return 0 if ready else 2


def _cmd_runtime_start(args: argparse.Namespace) -> int:
    from quant_bench.runtime.processes import start_process

    start_process(
        args.process,
        Path(args.config_dir),
        _workspace(args.workspace),
        mode=args.mode,
        confirm=args.confirm,
    )
    return 0


def _cmd_runtime_fingpt_dry_run(args: argparse.Namespace) -> int:
    from quant_bench.methods.fingpt_news import run_fingpt_news

    workspace = _workspace(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    run_fingpt_news.WORKSPACE_ROOT = workspace
    config = args.config or run_fingpt_news.CURRENT_DIR / "configs" / "sentiment_sft_live.yaml"
    runner = run_fingpt_news.FinGPTNewsPaperRunner(config, dry_run=True)
    runner.run_once(trade=True, fetch=True, force_trade=True, trade_date=args.trade_date)
    print(_json({"dry_run": True, "run_dir": str(runner.run_dir)}))
    return 0


def _cmd_runtime_ai_trade_list(args: argparse.Namespace) -> int:
    from quant_bench.integrations.ai_trade import list_live_stack_processes

    print(_json(list_live_stack_processes(args.process)))
    return 0


def _cmd_runtime_ai_trade_check(args: argparse.Namespace) -> int:
    from quant_bench.integrations.ai_trade import inspect_live_stack

    result = inspect_live_stack(args.repo, args.process)
    print(_json(result))
    return 0 if result["ready"] else 2


def _cmd_runtime_ai_trade_status(args: argparse.Namespace) -> int:
    from quant_bench.integrations.ai_trade import status_live_stack

    result = status_live_stack(args.repo, args.process)
    print(_json(result))
    return 0 if result["healthy"] else 2


def _cmd_runtime_ai_trade_start(args: argparse.Namespace) -> int:
    from quant_bench.integrations.ai_trade import start_live_stack

    result = start_live_stack(args.repo, confirm=args.confirm)
    print(_json(result))
    return int(result["returncode"])


def _cmd_runtime_ai_trade_stop(args: argparse.Namespace) -> int:
    from quant_bench.integrations.ai_trade import stop_live_stack

    result = stop_live_stack(args.repo, confirm=args.confirm)
    print(_json(result))
    return int(result["returncode"])


def _cmd_fingpt_init(args: argparse.Namespace) -> int:
    from quant_bench.methods.fingpt_news.research import initialize_research_workspace

    result = initialize_research_workspace(_workspace(args.workspace), force=args.force)
    print(_json(result))
    return 0


def _cmd_fingpt_validate(args: argparse.Namespace) -> int:
    from quant_bench.methods.fingpt_news.research import validate_research_inputs

    result = validate_research_inputs(args.config)
    print(_json(result))
    return 0 if result.get("valid") else 2


def _cmd_fingpt_backtest(args: argparse.Namespace) -> int:
    from quant_bench.methods.fingpt_news.research import run_research_pipeline

    result = run_research_pipeline(args.config, args.output, force=args.force)
    print(_json(result))
    return 0


def _cmd_finmem_init(args: argparse.Namespace) -> int:
    from quant_bench.methods.finmem.workspace import initialize_workspace

    result = initialize_workspace(
        _workspace(args.workspace) / "methods" / "finmem",
        args.data_dir,
        symbols=args.symbol,
        force=args.force,
    )
    print(_json(result))
    return 0


def _cmd_finmem_doctor(args: argparse.Namespace) -> int:
    from quant_bench.methods.finmem.operations import doctor

    print(_json(doctor(_workspace(args.workspace), args.data_dir)))
    return 0


def _cmd_finmem_data_verify(args: argparse.Namespace) -> int:
    from quant_bench.methods.finmem.data import verify_data_dir

    result = verify_data_dir(
        args.data_dir,
        symbols=args.symbol,
        verify_entries=args.full,
    )
    print(_json(result))
    return 0 if result["valid"] else 2


def _cmd_finmem_run(args: argparse.Namespace) -> int:
    from quant_bench.methods.finmem.operations import run_investorbench

    return run_investorbench(args.investor_command, args.config, allow_network=args.allow_network)


def _cmd_finmem_map_action(args: argparse.Namespace) -> int:
    from quant_bench.methods.finmem.operations import map_action, read_action

    result = map_action(
        read_action(args.input),
        trade_mode=args.trade_mode,
        notional_usdt=args.notional_usdt,
        allow_short=args.allow_short,
        leverage=args.leverage,
    )
    print(_json(result))
    return 0


def _cmd_finmem_paper_step(args: argparse.Namespace) -> int:
    from quant_bench.methods.finmem.operations import paper_step

    result = paper_step(
        args.state,
        price=args.price,
        target_position=args.target_position,
        initial_capital_usdt=args.initial_capital_usdt,
        fee_bps=args.fee_bps,
    )
    print(_json(result))
    return 0


def _cmd_finmem_live(args: argparse.Namespace) -> int:
    from quant_bench.methods.finmem.operations import run_live_process

    return run_live_process(
        args.live_process,
        workspace=_workspace(args.workspace),
        symbol=args.symbol,
        mode=args.mode,
        investor_config=args.config,
        allow_network=args.allow_network,
        query_account=args.query_account,
        execute_orders=args.execute_orders,
        confirmation=args.confirm,
        env_file=args.env_file,
    )


def _cmd_finmem_dashboard(args: argparse.Namespace) -> int:
    from quant_bench.methods.finmem.operations import run_dashboard

    return run_dashboard(_workspace(args.workspace), args.streamlit_arg)


def _cmd_dashboard(args: argparse.Namespace) -> int:
    try:
        import streamlit  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError('dashboard support requires: pip install "quant-bench[dashboard]"') from exc
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    env = os.environ.copy()
    env["QUANT_BENCH_DASHBOARD_ROOT"] = str(_workspace(args.workspace))
    completed = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), *args.streamlit_arg],
        env=env,
        check=False,
    )
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-bench", description="Reproducible cryptocurrency benchmark tooling"
    )
    parser.add_argument("--version", action="version", version=f"quant-bench {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    quickstart = commands.add_parser(
        "quickstart", help="Run the packaged synthetic benchmark without network access"
    )
    quickstart.add_argument(
        "--offline", action="store_true", help="Compatibility flag; quickstart is always offline"
    )
    quickstart.add_argument("--workspace")
    quickstart.set_defaults(handler=_cmd_quickstart)

    doctor = commands.add_parser("doctor", help="Inspect the local package and optional capabilities")
    doctor.add_argument("--workspace")
    doctor.set_defaults(handler=_cmd_doctor)

    run = commands.add_parser("run", help="Run a canonical offline experiment")
    run.add_argument("config")
    run.add_argument("--workspace")
    run.add_argument("--set", action="append")
    run.set_defaults(handler=_cmd_run)

    models = commands.add_parser("models", help="Inspect the complete model catalog")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    models_list = model_commands.add_parser("list")
    models_list.add_argument("--verbose", action="store_true")
    models_list.set_defaults(handler=_cmd_models_list)
    models_show = model_commands.add_parser("show")
    models_show.add_argument("model")
    models_show.set_defaults(handler=_cmd_models_show)

    config = commands.add_parser("config", help="List, render, validate, or create typed configs")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_list = config_commands.add_parser("list")
    config_list.set_defaults(handler=_cmd_config_list)
    for name, handler in (("render", _cmd_config_render), ("validate", _cmd_config_validate)):
        subcommand = config_commands.add_parser(name)
        subcommand.add_argument("config")
        subcommand.add_argument("--set", action="append")
        subcommand.set_defaults(handler=handler)
    config_init = config_commands.add_parser("init")
    config_init.add_argument("--model", required=True)
    config_init.add_argument(
        "--feature-set",
        choices=["crypto_ohlcv6_v1", "crypto_alpha52_v1", "qlib_alpha158_v1", "qlib_alpha360_v1"],
        default="crypto_ohlcv6_v1",
    )
    config_init.add_argument("--output", required=True)
    config_init.add_argument("--force", action="store_true")
    config_init.set_defaults(handler=_cmd_config_init)

    data = commands.add_parser("data", help="Pull, split, validate, or build Qlib market data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_validate = data_commands.add_parser("validate")
    data_validate.add_argument("path")
    data_validate.set_defaults(handler=_cmd_data_validate)
    data_pull = data_commands.add_parser("pull", help="Download public OKX OHLCV through CCXT")
    data_pull.add_argument("--timeframe", default="1h")
    data_pull.add_argument("--start", default="2021-01-01")
    data_pull.add_argument("--symbols", default="BTC-USDT,ETH-USDT")
    data_pull.add_argument("--output", required=True)
    data_pull.set_defaults(handler=_cmd_data_pull)
    data_split = data_commands.add_parser("split", help="Split a canonical CSV into one file per symbol")
    data_split.add_argument("--timeframe", default="1h")
    data_split.add_argument("--input", required=True)
    data_split.add_argument("--output-dir", required=True)
    data_split.set_defaults(handler=_cmd_data_split)
    data_dump = data_commands.add_parser("qlib-dump", help="Build or update Qlib bin data")
    data_dump.add_argument("--mode", choices=["dump_all", "dump_fix", "dump_update"], default="dump_all")
    data_dump.add_argument("--timeframe", default="1h")
    data_dump.add_argument("--data-path", required=True)
    data_dump.add_argument("--qlib-dir", required=True)
    data_dump.add_argument("--backup-dir")
    data_dump.add_argument("--frequency")
    data_dump.add_argument("--max-workers", type=int, default=4)
    data_dump.add_argument("--date-field-name", default="auto")
    data_dump.add_argument("--symbol-field-name", default="symbol")
    data_dump.add_argument(
        "--include-fields", default="open,high,low,close,volume,typical_price,vwap"
    )
    data_dump.set_defaults(handler=_cmd_data_qlib_dump)
    data_pipeline = data_commands.add_parser(
        "pipeline", help="Run explicit OKX pull, per-symbol split, and Qlib bin build"
    )
    data_pipeline.add_argument("--timeframe", default="1h")
    data_pipeline.add_argument("--start", default="2021-01-01")
    data_pipeline.add_argument("--symbols", default="BTC-USDT,ETH-USDT")
    data_pipeline.add_argument("--output", required=True)
    data_pipeline.add_argument("--split-dir", required=True)
    data_pipeline.add_argument("--qlib-dir", required=True)
    data_pipeline.add_argument("--frequency")
    data_pipeline.add_argument("--max-workers", type=int, default=4)
    data_pipeline.add_argument("--date-field-name", default="auto")
    data_pipeline.add_argument("--symbol-field-name", default="symbol")
    data_pipeline.add_argument(
        "--include-fields", default="open,high,low,close,volume,typical_price,vwap"
    )
    data_pipeline.set_defaults(handler=_cmd_data_pipeline)

    artifacts = commands.add_parser("artifacts", help="Inspect and verify run artifacts")
    artifact_commands = artifacts.add_subparsers(dest="artifact_command", required=True)
    artifact_inspect = artifact_commands.add_parser("inspect")
    artifact_inspect.add_argument("run_dir")
    artifact_inspect.set_defaults(handler=_cmd_artifacts_inspect)
    artifact_verify = artifact_commands.add_parser("verify")
    artifact_verify.add_argument("run_dir")
    artifact_verify.set_defaults(handler=_cmd_artifacts_verify)
    artifact_promote = artifact_commands.add_parser(
        "promote", help="Copy an opaque trained artifact into a runtime workspace"
    )
    artifact_promote.add_argument("source")
    artifact_promote.add_argument("--model", required=True)
    artifact_promote.add_argument("--frequency", required=True)
    artifact_promote.add_argument("--workspace")
    artifact_promote.add_argument("--config")
    artifact_promote.add_argument("--dry-run", action="store_true")
    artifact_promote.add_argument("--force", action="store_true")
    artifact_promote.set_defaults(handler=_cmd_artifacts_promote)

    compare = commands.add_parser("compare", help="Compare completed runs with compatibility checks")
    compare.add_argument("--workspace")
    compare.add_argument("--allow-incomparable", action="store_true")
    compare.set_defaults(handler=_cmd_compare)

    sweep = commands.add_parser("sweep", help="Run a resumable grid over canonical config fields")
    sweep.add_argument("config")
    sweep.add_argument("--param", action="append", required=True, help="dotted.path=value1,value2")
    sweep.add_argument("--study-name", required=True)
    sweep.add_argument("--workspace")
    sweep.add_argument("--max-trials", type=int, default=100)
    sweep.add_argument("--resume", action="store_true")
    sweep.set_defaults(handler=_cmd_sweep)

    report = commands.add_parser("report", help="Write a Markdown summary from completed run manifests")
    report.add_argument("--workspace")
    report.add_argument("--output")
    report.set_defaults(handler=_cmd_report)

    qlib = commands.add_parser("qlib", help="Validate or run a legacy Qlib workflow")
    qlib_commands = qlib.add_subparsers(dest="qlib_command", required=True)
    qlib_validate = qlib_commands.add_parser("validate")
    qlib_validate.add_argument("workflow")
    qlib_validate.set_defaults(handler=_cmd_qlib_validate)
    qlib_run = qlib_commands.add_parser("run")
    qlib_run.add_argument("workflow", nargs="?")
    qlib_run.add_argument("--feature-set", choices=["158", "360"])
    qlib_run.add_argument("--model")
    qlib_run.add_argument("--frequency")
    qlib_run.add_argument("--workspace")
    qlib_run.add_argument("--provider-uri")
    qlib_run.add_argument("--experiment-name", default="quant-bench")
    qlib_run.set_defaults(handler=_cmd_qlib_run)
    for command_name, command_help in (
        ("train", "Run the saved-artifact Qlib training flow"),
        ("backtest", "Backtest a saved Qlib artifact"),
        ("tune", "Run the Qlib workflow parameter sweep"),
        ("auto-tune", "Run the migrated multi-model tuning flow"),
        ("recompute-stats", "Recompute prediction distribution statistics"),
        ("visualize-sweep", "Render plots from a Qlib sweep table"),
        ("verify-predictions", "Compare Qlib and standalone PyTorch predictions"),
    ):
        compat = qlib_commands.add_parser(command_name, help=command_help)
        compat.add_argument("--workspace")
        compat.add_argument("forwarded", nargs=argparse.REMAINDER, help="arguments after --")
        compat.set_defaults(handler=_cmd_qlib_compat, compat_command=command_name)

    workflows = commands.add_parser("workflows", help="Audit or normalize legacy workflow templates")
    workflow_commands = workflows.add_subparsers(dest="workflow_command", required=True)
    workflow_list = workflow_commands.add_parser("list", help="Filter the packaged workflow catalog")
    workflow_list.add_argument("--feature-set", choices=["158", "360"])
    workflow_list.add_argument("--model")
    workflow_list.add_argument("--frequency")
    workflow_list.set_defaults(handler=_cmd_workflows_list)
    workflow_audit = workflow_commands.add_parser("audit")
    workflow_audit.add_argument("root")
    workflow_audit.add_argument("--require-normalized", action="store_true")
    workflow_audit.set_defaults(handler=_cmd_workflows_audit)
    workflow_normalize = workflow_commands.add_parser("normalize")
    workflow_normalize.add_argument("root")
    workflow_normalize.add_argument("--write", action="store_true")
    workflow_normalize.set_defaults(handler=_cmd_workflows_normalize)

    plugins = commands.add_parser("plugins", help="List or scaffold extension packages")
    plugin_commands = plugins.add_subparsers(dest="plugin_command", required=True)
    plugins_list = plugin_commands.add_parser("list")
    plugins_list.set_defaults(handler=_cmd_plugins_list)
    plugins_scaffold = plugin_commands.add_parser("scaffold")
    plugins_scaffold.add_argument("name")
    plugins_scaffold.add_argument("--target", default="plugins")
    plugins_scaffold.set_defaults(handler=_cmd_plugins_scaffold)

    schema = commands.add_parser("schema", help="Export public JSON schemas")
    schema_commands = schema.add_subparsers(dest="schema_command", required=True)
    schema_export = schema_commands.add_parser("export")
    schema_export.add_argument("--output", default="schemas")
    schema_export.set_defaults(handler=_cmd_schema_export)

    runtime = commands.add_parser("runtime", help="Inspect, configure, or explicitly start runtime processes")
    runtime_commands = runtime.add_subparsers(dest="runtime_command", required=True)
    runtime_list = runtime_commands.add_parser("list", help="List packaged runtime processes")
    runtime_list.set_defaults(handler=_cmd_runtime_list)
    runtime_init = runtime_commands.add_parser("init", help="Export editable safe-default configs")
    runtime_init.add_argument("process")
    runtime_init.add_argument("--output-dir", required=True)
    runtime_init.add_argument("--force", action="store_true")
    runtime_init.set_defaults(handler=_cmd_runtime_init)
    runtime_check = runtime_commands.add_parser("check", help="Check configs and model paths without network access")
    runtime_check.add_argument("process")
    runtime_check.add_argument("--config-dir", required=True)
    runtime_check.set_defaults(handler=_cmd_runtime_check)
    runtime_start = runtime_commands.add_parser("start", help="Start a confirmed OKX demo or live process")
    runtime_start.add_argument("process")
    runtime_start.add_argument("--config-dir", required=True)
    runtime_start.add_argument("--workspace")
    runtime_start.add_argument("--mode", choices=["demo", "live"], required=True)
    runtime_start.add_argument("--confirm", required=True)
    runtime_start.set_defaults(handler=_cmd_runtime_start)
    fingpt_dry = runtime_commands.add_parser("fingpt-dry-run", help="Run FinGPT with fixture news and no network")
    fingpt_dry.add_argument("--workspace")
    fingpt_dry.add_argument("--config", type=Path)
    fingpt_dry.add_argument("--trade-date")
    fingpt_dry.set_defaults(handler=_cmd_runtime_fingpt_dry_run)
    ai_trade = runtime_commands.add_parser(
        "ai-trade",
        help="Inspect or control a separate ai_trade 4H simulated agent stack",
    )
    ai_trade_commands = ai_trade.add_subparsers(dest="ai_trade_command", required=True)
    from quant_bench.integrations.ai_trade import AI_TRADE_PROCESSES

    ai_trade_choices = [process.process_id for process in AI_TRADE_PROCESSES]
    ai_trade_list = ai_trade_commands.add_parser("list", help="List the four agent processes")
    ai_trade_list.add_argument("--process", choices=ai_trade_choices)
    ai_trade_list.set_defaults(handler=_cmd_runtime_ai_trade_list)
    ai_trade_check = ai_trade_commands.add_parser(
        "check",
        help="Check the external checkout and credentials without network access",
    )
    ai_trade_check.add_argument("--repo", help="ai_trade checkout; defaults to AI_TRADE_ROOT")
    ai_trade_check.add_argument("--process", choices=ai_trade_choices)
    ai_trade_check.set_defaults(handler=_cmd_runtime_ai_trade_check)
    ai_trade_status = ai_trade_commands.add_parser(
        "status",
        help="Read process state and latest normalized events",
    )
    ai_trade_status.add_argument("--repo", help="ai_trade checkout; defaults to AI_TRADE_ROOT")
    ai_trade_status.add_argument("--process", choices=ai_trade_choices)
    ai_trade_status.set_defaults(handler=_cmd_runtime_ai_trade_status)
    ai_trade_start = ai_trade_commands.add_parser(
        "start",
        help="Start the complete reviewed simulated stack",
    )
    ai_trade_start.add_argument("--repo", help="ai_trade checkout; defaults to AI_TRADE_ROOT")
    ai_trade_start.add_argument("--confirm", required=True)
    ai_trade_start.set_defaults(handler=_cmd_runtime_ai_trade_start)
    ai_trade_stop = ai_trade_commands.add_parser(
        "stop",
        help="Stop the complete external stack",
    )
    ai_trade_stop.add_argument("--repo", help="ai_trade checkout; defaults to AI_TRADE_ROOT")
    ai_trade_stop.add_argument("--confirm", required=True)
    ai_trade_stop.set_defaults(handler=_cmd_runtime_ai_trade_stop)

    fingpt = commands.add_parser(
        "fingpt", help="Initialize, validate, tune, and backtest FinGPT sentiment strategies"
    )
    fingpt_commands = fingpt.add_subparsers(dest="fingpt_command", required=True)
    fingpt_init = fingpt_commands.add_parser(
        "init", help="Create an editable offline research workspace with synthetic fixtures"
    )
    fingpt_init.add_argument("--workspace")
    fingpt_init.add_argument("--force", action="store_true")
    fingpt_init.set_defaults(handler=_cmd_fingpt_init)
    fingpt_validate = fingpt_commands.add_parser(
        "validate", help="Validate price/sentiment contracts and time splits without network access"
    )
    fingpt_validate.add_argument("--config", type=Path, required=True)
    fingpt_validate.set_defaults(handler=_cmd_fingpt_validate)
    fingpt_backtest = fingpt_commands.add_parser(
        "backtest", help="Tune only on validation data and evaluate the frozen test configuration"
    )
    fingpt_backtest.add_argument("--config", type=Path, required=True)
    fingpt_backtest.add_argument("--output", type=Path, required=True)
    fingpt_backtest.add_argument("--force", action="store_true")
    fingpt_backtest.set_defaults(handler=_cmd_fingpt_backtest)
    fingpt_paper = fingpt_commands.add_parser(
        "paper-dry-run", help="Run the existing fixture-news paper cycle without network or orders"
    )
    fingpt_paper.add_argument("--workspace")
    fingpt_paper.add_argument("--config", type=Path)
    fingpt_paper.add_argument("--trade-date")
    fingpt_paper.set_defaults(handler=_cmd_runtime_fingpt_dry_run)

    finmem = commands.add_parser(
        "finmem", help="Initialize, validate, backtest, and paper-test the FinMem method"
    )
    finmem_commands = finmem.add_subparsers(dest="finmem_command", required=True)
    finmem_init = finmem_commands.add_parser("init", help="Create an editable FinMem workspace")
    finmem_init.add_argument("--workspace")
    finmem_init.add_argument("--data-dir", required=True)
    finmem_init.add_argument("--symbol", action="append", help="repeat for multi-asset runs")
    finmem_init.add_argument("--force", action="store_true")
    finmem_init.set_defaults(handler=_cmd_finmem_init)
    finmem_doctor = finmem_commands.add_parser(
        "doctor", help="Check local dependencies and paths without network access"
    )
    finmem_doctor.add_argument("--workspace")
    finmem_doctor.add_argument("--data-dir")
    finmem_doctor.set_defaults(handler=_cmd_finmem_doctor)
    finmem_data = finmem_commands.add_parser("data", help="Validate local InvestorBench data")
    finmem_data_commands = finmem_data.add_subparsers(dest="finmem_data_command", required=True)
    finmem_verify = finmem_data_commands.add_parser("verify")
    finmem_verify.add_argument("--data-dir", required=True)
    finmem_verify.add_argument("--symbol", action="append")
    finmem_verify.add_argument("--full", action="store_true", help="validate every daily record")
    finmem_verify.set_defaults(handler=_cmd_finmem_data_verify)
    finmem_run = finmem_commands.add_parser(
        "run", help="Run InvestorBench; service-using phases require --allow-network"
    )
    finmem_run.add_argument(
        "investor_command",
        choices=["warmup", "warmup-checkpoint", "test", "test-checkpoint", "eval"],
    )
    finmem_run.add_argument("--config", required=True)
    finmem_run.add_argument("--allow-network", action="store_true")
    finmem_run.set_defaults(handler=_cmd_finmem_run)
    finmem_map = finmem_commands.add_parser(
        "map-action", help="Convert an InvestorBench action without network or account access"
    )
    finmem_map.add_argument("--input", required=True)
    finmem_map.add_argument("--trade-mode", choices=["spot", "swap"], default="swap")
    finmem_map.add_argument("--notional-usdt", type=float, default=10_000.0)
    finmem_map.add_argument("--leverage", type=int, default=1)
    finmem_map.add_argument("--allow-short", action=argparse.BooleanOptionalAction, default=True)
    finmem_map.set_defaults(handler=_cmd_finmem_map_action)
    finmem_paper = finmem_commands.add_parser(
        "paper-step", help="Apply one deterministic long/flat/short paper-ledger step"
    )
    finmem_paper.add_argument("--state", required=True)
    finmem_paper.add_argument("--price", type=float, required=True)
    finmem_paper.add_argument("--target-position", type=float, required=True)
    finmem_paper.add_argument("--initial-capital-usdt", type=float, default=10_000.0)
    finmem_paper.add_argument("--fee-bps", type=float, default=0.0)
    finmem_paper.set_defaults(handler=_cmd_finmem_paper_step)
    for process, help_text in (
        ("cycle", "Run one FinMem market/news/LLM cycle"),
        ("schedule", "Run the symbol-scoped daily FinMem scheduler"),
    ):
        finmem_live = finmem_commands.add_parser(process, help=help_text)
        finmem_live.add_argument("--workspace")
        finmem_live.add_argument("--symbol", default="BTC")
        finmem_live.add_argument("--config", required=True)
        finmem_live.add_argument("--env-file")
        finmem_live.add_argument("--mode", choices=["paper", "demo", "live"], default="paper")
        finmem_live.add_argument("--allow-network", action="store_true")
        finmem_live.add_argument("--query-account", action="store_true")
        finmem_live.add_argument("--execute-orders", action="store_true")
        finmem_live.add_argument("--confirm", default="")
        finmem_live.set_defaults(handler=_cmd_finmem_live, live_process=process)
    finmem_dashboard = finmem_commands.add_parser(
        "dashboard", help="Open the read-only multi-asset FinMem dashboard"
    )
    finmem_dashboard.add_argument("--workspace")
    finmem_dashboard.add_argument("streamlit_arg", nargs=argparse.REMAINDER)
    finmem_dashboard.set_defaults(handler=_cmd_finmem_dashboard)

    dashboard = commands.add_parser("dashboard", help="Open the read-only local results dashboard")
    dashboard.add_argument("--workspace")
    dashboard.add_argument("streamlit_arg", nargs=argparse.REMAINDER)
    dashboard.set_defaults(handler=_cmd_dashboard)

    from quant_bench.cli.methods import register_method_commands

    register_method_commands(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        print(_json({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
