"""CLI projection of the unified Method interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from platformdirs import user_data_path

from quant_bench.methods import (
    ExecutionMode,
    MethodRunSpec,
    get_method_registry,
    get_method_runner,
)


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def _parameters(values: list[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for value in values or []:
        key, separator, raw = value.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"invalid --param {value!r}; expected KEY=VALUE")
        try:
            parsed[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[key.strip()] = raw
    return parsed


def _spec(args: argparse.Namespace) -> MethodRunSpec:
    workspace = (
        Path(args.workspace).expanduser().resolve()
        if args.workspace
        else user_data_path("quant-bench", appauthor=False)
    )
    return MethodRunSpec(
        method_id=args.method,
        mode=ExecutionMode(args.mode),
        workspace=workspace,
        config=Path(args.config) if args.config else None,
        repo=Path(args.repo) if args.repo else None,
        frequency=args.frequency,
        start=args.start,
        end=args.end,
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        resume=args.resume,
        resume_from=Path(args.resume_from) if args.resume_from else None,
        allow_network=args.allow_network,
        query_account=args.query_account,
        execute_orders=args.execute_orders,
        confirm=args.confirm,
        once=args.once,
        dry_run=getattr(args, "dry_run", False),
        parameters=_parameters(args.param),
    )


def _cmd_list(args: argparse.Namespace) -> int:
    mode = ExecutionMode(args.mode) if args.mode else None
    rows = [
        descriptor.model_dump(mode="json")
        for descriptor in get_method_registry().list(mode)
    ]
    print(_json(rows))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    descriptor = get_method_registry().get(args.method).descriptor
    print(_json(descriptor))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    result = get_method_runner().check(_spec(args))
    print(_json(result))
    return 0 if result.ready else 2


def _cmd_run(args: argparse.Namespace) -> int:
    result = get_method_runner().run(_spec(args))
    print(_json(result))
    return 0 if result.status in {"planned", "running", "completed"} else 1


def _cmd_status(args: argparse.Namespace) -> int:
    result = get_method_runner().status(_spec(args))
    print(_json(result))
    return 0


def _add_run_spec_arguments(parser: argparse.ArgumentParser, *, include_dry_run: bool) -> None:
    parser.add_argument("method", help="Method id or alias from `quant-bench methods list`")
    parser.add_argument("--mode", choices=[mode.value for mode in ExecutionMode], required=True)
    parser.add_argument("--workspace")
    parser.add_argument("--config", help="Method config, workflow, or runtime config directory")
    parser.add_argument("--repo", help="External Method checkout, such as ai_trade")
    parser.add_argument("--frequency")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--fee-rate", type=float, default=1.5e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", help="native run directory to continue")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--query-account", action="store_true")
    parser.add_argument("--execute-orders", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--param",
        action="append",
        help="Method-specific KEY=VALUE; JSON values are decoded",
    )
    if include_dry_run:
        parser.add_argument("--dry-run", action="store_true", help="write a planned run without executing")


def register_method_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the unified command surface on the root parser."""

    methods = commands.add_parser(
        "methods",
        help="List, check, run, or inspect any Method in any supported mode",
    )
    method_commands = methods.add_subparsers(dest="method_command", required=True)

    list_command = method_commands.add_parser("list", help="List the Method capability matrix")
    list_command.add_argument("--mode", choices=[mode.value for mode in ExecutionMode])
    list_command.set_defaults(handler=_cmd_list)

    show_command = method_commands.add_parser("show", help="Show one Method capability declaration")
    show_command.add_argument("method")
    show_command.set_defaults(handler=_cmd_show)

    check_command = method_commands.add_parser(
        "check",
        help="Validate capability, paths, credentials, and safety flags without running",
    )
    _add_run_spec_arguments(check_command, include_dry_run=False)
    check_command.set_defaults(handler=_cmd_check)

    run_command = method_commands.add_parser(
        "run",
        help="Run one Method through the unified safety and artifact interface",
    )
    _add_run_spec_arguments(run_command, include_dry_run=True)
    run_command.set_defaults(handler=_cmd_run)

    status_command = method_commands.add_parser(
        "status",
        help="Read the latest unified run record and adapter-native status",
    )
    _add_run_spec_arguments(status_command, include_dry_run=False)
    status_command.set_defaults(handler=_cmd_status)
