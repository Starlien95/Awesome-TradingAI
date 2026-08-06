"""Shared parser for ``python -m quant_bench.runtime.entrypoints.*`` commands."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from platformdirs import user_data_path

from quant_bench.runtime.processes import start_process


def run(process_id: str, argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=f"Start the {process_id} runtime process.")
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=user_data_path("quant-bench", appauthor=False))
    parser.add_argument("--mode", choices=("demo", "live"), required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args(argv)
    start_process(process_id, args.config_dir, args.workspace, mode=args.mode, confirm=args.confirm)
