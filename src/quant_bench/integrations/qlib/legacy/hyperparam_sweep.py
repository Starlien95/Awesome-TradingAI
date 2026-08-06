#!/usr/bin/env python3
"""
Hyperparameter sweep for Qlib-based LightGBM models.

Key improvements:
- Hyperparameter ranges auto-derived from the base workflow config; we create
    +/- scale factors to avoid unreasonable values.
- Intelligent sampling via Latin Hypercube / random sampling centered around
    the base config.
- Parallel execution using ProcessPoolExecutor (default workers = min(cpu, 4)).
- Metrics read from `hyper_param/{freq}` JSON files written by train_model.py.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

from quant_bench.integrations.qlib.legacy.workflow_utils import resolve_workflow_config

EXCLUDED_PARAMS = {"n_estimators"}


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for Qlib LGB models")
    parser.add_argument("--freq", "-f", required=True, help="frequency tag, e.g. 1m,5m,15m,1h")
    parser.add_argument("--model", "-m", default="lgb", help="model tag used in config filename (default: lgb)")
    parser.add_argument(
        "--base-config",
        "-c",
        default=None,
        help="base workflow yaml (default: workflow/{158,360}/{model}/workflow_config_{model}_{freq}.yaml)",
    )
    parser.add_argument(
        "--feature-set",
        "--feature_set",
        choices=("158", "360"),
        default="158",
        help="Feature set used for default base config lookup (default: 158)",
    )
    parser.add_argument("--out-dir", default="hyper_runs", help="directory to store temp configs, logs and results")
    parser.add_argument("--sweep-param", "-p", action="append", dest="sweep_params", required=True,
                        help="Parameter=ratio pairs (e.g. learning_rate=0.1). Multiple values allowed.")
    parser.add_argument("--spread-count", type=int, default=5,
                        help="Odd number of points per parameter axis (e.g. 5 -> center±2 steps)")
    parser.add_argument("--max-workers", type=int, default=1, help="max parallel workers (default 1)")
    parser.add_argument("--resume", action="store_true", help="skip trials already present in results.csv")
    return parser.parse_args()


def extract_model_kwargs(base_cfg: Path) -> dict:
    with open(base_cfg) as f:
        cfg = yaml.safe_load(f)
    try:
        return cfg["task"]["model"].get("kwargs", {})
    except KeyError:
        raise RuntimeError("Invalid config: missing task.model.kwargs")


def build_ranges_from_base(base_params: dict) -> dict:
    ranges = {}
    for key, val in base_params.items():
        if key in EXCLUDED_PARAMS:
            continue
        if isinstance(val, (int, float)) and key not in ("random_state",):
            if key == "learning_rate":
                ranges[key] = (max(val * 0.5, 0.005), min(val * 1.5, 0.3))
            elif key in ("subsample", "colsample_bytree"):
                ranges[key] = (max(0.5, val - 0.2), min(1.0, val + 0.2))
            elif key in ("reg_alpha", "reg_lambda"):
                ranges[key] = (max(0.0, val * 0.5), val * 2 + 1e-6)
            elif key == "n_estimators":
                ranges[key] = (int(val * 0.5), int(val * 1.5))
            elif key == "num_leaves":
                ranges[key] = (max(8, int(val * 0.5)), int(val * 2))
            else:
                ranges[key] = (val * 0.8, val * 1.2)
    return ranges


def parse_param_ratios(items: list[str]) -> dict:
    ratios: dict[str, float] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"Invalid --sweep-param value '{raw}'. Expected format param=ratio")
        name, value = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid parameter name in '{raw}'")
        try:
            ratio = float(value)
        except ValueError:
            raise ValueError(f"Invalid ratio for '{name}' in '{raw}'") from None
        if ratio <= 0:
            raise ValueError(f"Ratio for '{name}' must be positive")
        ratios[name] = ratio
    if not ratios:
        raise ValueError("At least one --sweep-param must be provided")
    return ratios


def _generate_axis_values(param: str, base_val: float | int, ratio: float, count: int, bounds: tuple[float, float] | None):
    if count < 1 or count % 2 == 0:
        raise ValueError("spread-count must be an odd positive integer")
    half = count // 2
    if half == 0:
        return [base_val]

    low = bounds[0] if bounds else None
    high = bounds[1] if bounds else None
    scale = base_val if base_val not in (0, 0.0) else max(
        abs(low) if low is not None else 0,
        abs(high) if high is not None else 0,
        1.0,
    )
    step_fraction = ratio / half
    values = []
    for pos in range(-half, half + 1):
        candidate = base_val + scale * step_fraction * pos
        if low is not None:
            candidate = max(candidate, low)
        if high is not None:
            candidate = min(candidate, high)
        values.append(candidate)

    is_int = isinstance(base_val, int)
    cleaned = []
    for val in values:
        cleaned_val = int(round(val)) if is_int else round(val, 6)
        if cleaned and cleaned[-1] == cleaned_val:
            continue
        cleaned.append(cleaned_val)
    return cleaned or [base_val]


def build_param_grid(base_params: dict, ranges: dict, ratios: dict, spread_count: int):
    sweep_values = {}
    for param, ratio in ratios.items():
        if param not in base_params:
            raise ValueError(f"Parameter '{param}' not found in base config")
        base_val = base_params[param]
        if not isinstance(base_val, (int, float)):
            raise ValueError(f"Parameter '{param}' is not numeric and cannot be swept")
        bounds = ranges.get(param)
        sweep_values[param] = _generate_axis_values(param, base_val, ratio, spread_count, bounds)

    keys = list(sweep_values.keys())
    combos = []
    for values in product(*(sweep_values[k] for k in keys)):
        combos.append(dict(zip(keys, values)))
    return keys, combos


def write_temp_config(base_cfg_path: Path, dest_path: Path, params: dict):
    with open(base_cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg_model = cfg["task"]["model"]
    kwargs = cfg_model.get("kwargs", {}).copy()
    kwargs.update(params)
    cfg_model["kwargs"] = kwargs
    with open(dest_path, "w") as f:
        yaml.safe_dump(cfg, f)


def read_latest_json(freq: str) -> dict:
    freq_dir = Path("hyper_param") / freq
    if not freq_dir.exists():
        return {}
    files = sorted(freq_dir.glob("metrics_*.json"))
    if not files:
        return {}
    latest = files[-1]
    with open(latest) as f:
        return json.load(f)


def run_trial(args_tuple):
    (idx, params, base_cfg, args, tmp_cfg_dir, logs_dir, out_dir) = args_tuple
    cfg_path = tmp_cfg_dir / f"cfg_{idx}.yaml"
    write_temp_config(base_cfg, cfg_path, params)
    log_path = logs_dir / f"run_{idx}.log"
    save_dir = str(out_dir / f"artifacts_run_{idx}")

    cmd = [
        sys.executable,
        "train_model.py",
        "--freq",
        args.freq,
        "--model",
        args.model,
        "--config",
        str(cfg_path),
        "--save-dir",
        save_dir,
    ]
    with open(log_path, "wb") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

    metrics = read_latest_json(args.freq)
    record = {**params}
    record["cfg_path"] = str(cfg_path)
    record["log_path"] = str(log_path)
    record["return_code"] = proc.returncode
    record.update({f"metric_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
    return idx, record


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    tmp_cfg_dir = out_dir / "configs"
    logs_dir = out_dir / "logs"
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.base_config:
        base_cfg = Path(args.base_config)
    else:
        try:
            base_cfg = Path(resolve_workflow_config(args.model, args.freq, args.feature_set))
        except FileNotFoundError as exc:
            print(exc)
            sys.exit(1)
    if not base_cfg.exists():
        print(f"Base config not found: {base_cfg}")
        sys.exit(1)

    base_params = extract_model_kwargs(base_cfg)
    ranges = build_ranges_from_base(base_params)
    ratios = parse_param_ratios(args.sweep_params)
    sweep_keys, combos = build_param_grid(base_params, ranges, ratios, args.spread_count)
    total_trials = len(combos)
    results_file = out_dir / "results.csv"

    # init results file
    if results_file.exists():
        df_existing = pd.read_csv(results_file)
    else:
        df_existing = pd.DataFrame()

    worker_args = []
    for idx, params in enumerate(combos, start=1):
        worker_args.append((idx, params, base_cfg, args, tmp_cfg_dir, logs_dir, out_dir))

    max_workers = min(args.max_workers, os.cpu_count() or 2)
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for arg in worker_args:
            futures.append(executor.submit(run_trial, arg))

        for future in as_completed(futures):
            idx, record = future.result()
            df_existing = pd.concat([df_existing, pd.DataFrame([record])], ignore_index=True)
            df_existing.to_csv(results_file, index=False)
            print(f"Trial {idx} completed. Return={record.get('return_code')} Metrics keys: {[k for k in record if k.startswith('metric_')]}")

    print(f"All trials finished. Total={total_trials}. Results saved to {results_file}")


if __name__ == "__main__":
    main()
