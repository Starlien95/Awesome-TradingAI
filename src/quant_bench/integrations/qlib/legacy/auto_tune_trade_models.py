#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import yaml

from quant_bench.integrations.qlib.legacy.workflow_utils import resolve_workflow_config

ROOT = Path.cwd()

MODEL_SPECS = {
    "mlp1h": {
        "model": "mlp",
        "freq": "1h",
    },
    "lstm4h": {
        "model": "lstm",
        "freq": "4h",
    },
    "tra4h": {
        "model": "tra",
        "freq": "4h",
    },
}

TIME_KEYS = ("start_time", "end_time", "fit_start_time", "fit_end_time")


def parse_args():
    parser = argparse.ArgumentParser(description="Automatic isolated tuning for MLP1h/LSTM4h/TRA4h.")
    parser.add_argument(
        "--feature-set",
        "--feature_set",
        choices=("158", "360"),
        default="158",
        help="Feature set used for base configs (default: 158)",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--mlp-trials", type=int, default=24)
    parser.add_argument("--lstm-trials", type=int, default=18)
    parser.add_argument("--tra-trials", type=int, default=12)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--long-run",
        action="store_true",
        help="Broaden ranges and use longer training schedules for a multi-hour search.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def capture_time_contract(cfg: dict) -> dict:
    handler = cfg["task"]["dataset"]["kwargs"]["handler"]["kwargs"]
    segments = cfg["task"]["dataset"]["kwargs"]["segments"]
    backtest = cfg["port_analysis_config"]["backtest"]
    return {
        "handler": {k: handler.get(k) for k in TIME_KEYS if k in handler},
        "segments": deepcopy(segments),
        "backtest_start": backtest.get("start_time"),
        "backtest_end": backtest.get("end_time"),
    }


def assert_time_contract_unchanged(base: dict, trial: dict):
    if capture_time_contract(base) != capture_time_contract(trial):
        raise RuntimeError("Refusing to run trial: data/backtest time range changed.")


def choose(rng, values):
    return values[rng.randrange(len(values))]


def log_uniform(rng, low, high):
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def base_config_path(tag: str, feature_set: str) -> Path:
    spec = MODEL_SPECS[tag]
    cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        return ROOT / resolve_workflow_config(spec["model"], spec["freq"], feature_set)
    finally:
        os.chdir(cwd)


def set_label_and_strategy(cfg: dict, rng: random.Random, label_choices, topk_choices, threshold_choices, dropout_choices):
    cfg["data_handler_config"]["label"] = [choose(rng, label_choices)]
    handler_kwargs = cfg["task"]["dataset"]["kwargs"]["handler"]["kwargs"]
    handler_kwargs["label"] = cfg["data_handler_config"]["label"]
    strategy = cfg["port_analysis_config"]["strategy"]["kwargs"]
    strategy["topk"] = choose(rng, topk_choices)
    strategy["min_score"] = choose(rng, threshold_choices)
    if "max_dropout" in strategy or dropout_choices:
        strategy["max_dropout"] = choose(rng, dropout_choices)


def mutate_mlp(cfg: dict, rng: random.Random, trial_idx: int, long_run: bool = False) -> dict:
    kwargs = cfg["task"]["model"]["kwargs"]
    kwargs["seed"] = 1000 + trial_idx
    kwargs["optimizer"] = "adam"
    kwargs["loss"] = "mse"
    kwargs["lr"] = round(log_uniform(rng, 5e-5 if long_run else 1e-4, 5e-3 if long_run else 2.5e-3), 7)
    kwargs["max_steps"] = choose(rng, [6000, 8000, 12000, 16000, 20000] if long_run else [4000, 6000, 8000, 10000, 12000])
    kwargs["batch_size"] = choose(rng, [1024, 2048, 4096, 8192] if long_run else [2048, 4096, 8192])
    kwargs["weight_decay"] = 0.0 if long_run and rng.random() < 0.10 else round(log_uniform(rng, 1e-7 if long_run else 1e-6, 1e-2 if long_run else 3e-3), 8)
    kwargs["early_stop_rounds"] = choose(rng, [60, 90, 120, 160, 220] if long_run else [40, 60, 80, 120])
    kwargs["eval_steps"] = choose(rng, [250, 500])
    kwargs["GPU"] = 0
    kwargs["pt_model_uri"] = kwargs.get("pt_model_uri", "adapter_compatible_models.MLPStrategy")
    pt_model_kwargs = dict(kwargs.get("pt_model_kwargs") or {})
    pt_model_kwargs.update({
        "input_dim": pt_model_kwargs.get("input_dim", 52),
        "hidden_dim": choose(rng, [32, 48, 64, 96, 128, 192, 256] if long_run else [48, 64, 96, 128]),
        "output_dim": pt_model_kwargs.get("output_dim", 1),
        "dropout": choose(rng, [0.0, 0.03, 0.05, 0.1, 0.2, 0.3, 0.45] if long_run else [0.05, 0.1, 0.2, 0.3, 0.4]),
    })
    kwargs["pt_model_kwargs"] = pt_model_kwargs
    label_choices = [
        "(Ref($close, -1) / $close - 1) * 10",
        "(Ref($close, -1) / $close - 1) * 15",
        "(Ref($close, -1) / $close - 1) * 20",
        "(Mean(Ref($close, -1), 2) / $close - 1) * 10",
    ]
    if long_run:
        label_choices += [
            "(Ref($close, -2) / $close - 1) * 10",
            "(Mean(Ref($close, -1), 3) / $close - 1) * 8",
            "(Ref($close, -1) / $close - 1) * 5",
        ]
    set_label_and_strategy(
        cfg,
        rng,
        label_choices,
        [2, 3, 4, 5] if long_run else [2, 3, 4],
        [0.0, 0.0005, 0.001, 0.002, 0.005, 0.010, 0.015, 0.025] if long_run else [0.001, 0.002, 0.005, 0.010, 0.015],
        [1, 2, 3] if long_run else [1, 2],
    )
    return cfg


def mutate_lstm(cfg: dict, rng: random.Random, trial_idx: int, long_run: bool = False) -> dict:
    kwargs = cfg["task"]["model"]["kwargs"]
    kwargs["seed"] = 2000 + trial_idx
    kwargs["hidden_size"] = choose(rng, [48, 64, 96, 128, 192, 256, 320] if long_run else [64, 96, 128, 192])
    kwargs["num_layers"] = choose(rng, [1, 2, 3, 4] if long_run else [1, 2, 3])
    kwargs["dropout"] = 0.0 if kwargs["num_layers"] == 1 else choose(rng, [0.0, 0.05, 0.1, 0.2, 0.35, 0.5] if long_run else [0.05, 0.1, 0.2, 0.3])
    kwargs["n_epochs"] = choose(rng, [120, 160, 220, 300, 380] if long_run else [60, 80, 100, 120, 150])
    kwargs["lr"] = round(log_uniform(rng, 5e-5 if long_run else 2e-4, 3e-3 if long_run else 2e-3), 7)
    kwargs["early_stop"] = choose(rng, [15, 25, 40, 60] if long_run else [8, 10, 15, 20])
    kwargs["batch_size"] = choose(rng, [256, 512, 1024, 2048] if long_run else [512, 1024, 2048])
    kwargs["weight_decay"] = 0.0 if long_run and rng.random() < 0.10 else round(log_uniform(rng, 1e-7 if long_run else 1e-6, 3e-3 if long_run else 1e-3), 8)
    kwargs["GPU"] = 0
    cfg["task"]["dataset"]["kwargs"]["step_len"] = choose(rng, [8, 12, 18, 24, 36, 48] if long_run else [12, 24, 36])
    label_choices = [
        "(Mean(Ref($close, -1), 1) / $close - 1) * 10",
        "(Ref($close, -1) / $close - 1) * 10",
        "(Ref($close, -1) / $close - 1) * 20",
        "(Mean(Ref($close, -1), 2) / $close - 1) * 10",
    ]
    if long_run:
        label_choices += [
            "(Ref($close, -2) / $close - 1) * 8",
            "(Mean(Ref($close, -1), 3) / $close - 1) * 8",
        ]
    set_label_and_strategy(
        cfg,
        rng,
        label_choices,
        [1, 2, 3, 4] if long_run else [2, 3],
        [0.0, 0.005, 0.010, 0.020, 0.030, 0.050, 0.080] if long_run else [0.010, 0.020, 0.030, 0.050],
        [1, 2, 3] if long_run else [1, 2],
    )
    return cfg


def mutate_tra(cfg: dict, rng: random.Random, trial_idx: int, logdir: Path, long_run: bool = False) -> dict:
    kwargs = cfg["task"]["model"]["kwargs"]
    kwargs["seed"] = 3000 + trial_idx
    kwargs["lr"] = round(log_uniform(rng, 5e-5 if long_run else 1e-4, 1.2e-3 if long_run else 8e-4), 7)
    kwargs["n_epochs"] = choose(rng, [220, 300, 420, 560, 700] if long_run else [120, 160, 200, 260])
    kwargs["early_stop"] = choose(rng, [40, 70, 100, 140] if long_run else [25, 40, 60])
    kwargs["lamb"] = choose(rng, [0.1, 0.3, 0.5, 1.0, 2.0, 3.0] if long_run else [0.5, 1.0, 2.0])
    kwargs["rho"] = choose(rng, [0.90, 0.95, 0.98, 0.99, 0.995] if long_run else [0.95, 0.98, 0.99])
    kwargs["alpha"] = choose(rng, [0.1, 0.25, 0.5, 1.0, 1.5] if long_run else [0.25, 0.5, 1.0])
    kwargs["logdir"] = str(logdir)
    model_config = kwargs["model_config"]
    model_config["hidden_size"] = choose(rng, [24, 32, 48, 64, 96, 128] if long_run else [32, 48, 64, 96])
    model_config["num_layers"] = choose(rng, [1, 2, 3] if long_run else [1, 2])
    model_config["use_attn"] = choose(rng, [True, False])
    model_config["dropout"] = choose(rng, [0.0, 0.05, 0.1, 0.2, 0.35] if long_run else [0.0, 0.05, 0.1, 0.2])
    tra_config = kwargs["tra_config"]
    num_states = choose(rng, [2, 3, 4, 5] if long_run else [2, 3, 4])
    tra_config["num_states"] = num_states
    tra_config["hidden_size"] = choose(rng, [8, 16, 24, 32, 48] if long_run else [8, 16, 32])
    tra_config["num_layers"] = choose(rng, [1, 2, 3] if long_run else [1, 2])
    tra_config["dropout"] = choose(rng, [0.0, 0.05, 0.1, 0.2] if long_run else [0.0, 0.05, 0.1])
    dataset_kwargs = cfg["task"]["dataset"]["kwargs"]
    dataset_kwargs["seq_len"] = choose(rng, [8, 12, 18, 24, 36, 48] if long_run else [12, 24, 36])
    dataset_kwargs["batch_size"] = choose(rng, [256, 512, 1024] if long_run else [512, 1024])
    dataset_kwargs["num_states"] = num_states
    label_choices = [
        "(Ref($close, -1) / $close - 1) * 10",
        "(Ref($close, -1) / $close - 1) * 20",
        "(Mean(Ref($close, -1), 2) / $close - 1) * 10",
    ]
    if long_run:
        label_choices += [
            "(Ref($close, -2) / $close - 1) * 8",
            "(Mean(Ref($close, -1), 3) / $close - 1) * 8",
        ]
    set_label_and_strategy(
        cfg,
        rng,
        label_choices,
        [1, 2, 3, 4] if long_run else [2, 3],
        [0.0, 0.005, 0.010, 0.020, 0.030, 0.050, 0.080] if long_run else [0.010, 0.020, 0.030, 0.050],
        [1, 2, 3] if long_run else [1, 2],
    )
    return cfg


def build_trial(tag: str, idx: int, base_cfg: dict, out_dir: Path, seed: int, long_run: bool = False) -> dict:
    rng = random.Random(seed + idx * 1009 + sum(ord(c) for c in tag))
    cfg = deepcopy(base_cfg)
    logdir = out_dir / "tra_tensorboard" / f"{tag}_{idx:03d}"
    if tag == "mlp1h":
        cfg = mutate_mlp(cfg, rng, idx, long_run)
    elif tag == "lstm4h":
        cfg = mutate_lstm(cfg, rng, idx, long_run)
    elif tag == "tra4h":
        cfg = mutate_tra(cfg, rng, idx, logdir, long_run)
    else:
        raise ValueError(tag)
    assert_time_contract_unchanged(base_cfg, cfg)
    return cfg


def extract_annual_return(metrics: dict):
    port_metrics = metrics.get("port_metrics", {}) or {}
    preferred = [
        key for key in port_metrics
        if "annualized_return" in key and "with_cost" in key
    ]
    keys = preferred or [key for key in port_metrics if "annualized_return" in key]
    if not keys:
        return None, None
    key = sorted(keys)[0]
    return key, port_metrics.get(key)


def prediction_is_normal(stats: dict) -> bool:
    if not stats:
        return False
    pos = stats.get("positive_ratio") or 0.0
    neg = stats.get("negative_ratio") or 0.0
    std = stats.get("std") or 0.0
    p01 = abs(stats.get("p01") or 0.0)
    p99 = abs(stats.get("p99") or 0.0)
    return pos >= 0.02 and neg >= 0.02 and std > 1e-7 and max(p01, p99) < 10.0


def run_trial(task: dict) -> dict:
    cfg_path = task["cfg_path"]
    log_path = task["log_path"]
    save_dir = task["save_dir"]
    gpu = task["gpu"]
    spec = MODEL_SPECS[task["tag"]]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "train_model.py",
        "--freq",
        spec["freq"],
        "--model",
        spec["model"],
        "--feature-set",
        task["feature_set"],
        "--config",
        str(cfg_path),
        "--save-dir",
        str(save_dir),
    ]
    start = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    metrics_path = save_dir / "train_metrics.json"
    metrics = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
    metric_key, annual_return = extract_annual_return(metrics)
    pred_stats = metrics.get("prediction_stats", {}) or {}
    return {
        "tag": task["tag"],
        "feature_set": task["feature_set"],
        "trial": task["trial"],
        "return_code": proc.returncode,
        "elapsed_sec": round(elapsed, 2),
        "gpu": gpu,
        "annual_return_key": metric_key,
        "annual_return": annual_return,
        "prediction_normal": prediction_is_normal(pred_stats),
        "prediction_mean": pred_stats.get("mean"),
        "prediction_std": pred_stats.get("std"),
        "prediction_positive_ratio": pred_stats.get("positive_ratio"),
        "prediction_negative_ratio": pred_stats.get("negative_ratio"),
        "save_dir": str(save_dir),
        "config_path": str(cfg_path),
        "log_path": str(log_path),
    }


def write_leaderboard(rows: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r.get("return_code") == 0,
            r.get("prediction_normal") is True,
            r.get("annual_return") if r.get("annual_return") is not None else -1e99,
        ),
        reverse=True,
    )
    json_path = out_dir / "leaderboard.json"
    csv_path = out_dir / "leaderboard.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, indent=2, ensure_ascii=False)
    if rows_sorted:
        fields = list(rows_sorted[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows_sorted)


def iter_interleaved_trials(trial_counts: dict[str, int]):
    order = ["mlp1h", "lstm4h", "tra4h"]
    max_count = max(trial_counts.values()) if trial_counts else 0
    for idx in range(1, max_count + 1):
        for tag in order:
            if idx <= trial_counts.get(tag, 0):
                yield tag, idx


def main():
    args = parse_args()
    run_id = time.strftime("auto_tune_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "auto_runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("configs", "logs", "artifacts"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    shutil.copy2(__file__, out_dir / "auto_tune_trade_models.py")
    trial_counts = {"mlp1h": args.mlp_trials, "lstm4h": args.lstm_trials, "tra4h": args.tra_trials}
    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        gpu_ids = ["0"]

    tasks = []
    base_paths = {tag: base_config_path(tag, args.feature_set) for tag in trial_counts}
    base_cfgs = {tag: load_yaml(path) for tag, path in base_paths.items()}
    for tag, idx in iter_interleaved_trials(trial_counts):
        base_cfg = base_cfgs[tag]
        cfg = build_trial(tag, idx, base_cfg, out_dir, args.seed, args.long_run)
        cfg_path = out_dir / "configs" / f"{tag}_trial_{idx:03d}.yaml"
        dump_yaml(cfg, cfg_path)
        tasks.append({
            "tag": tag,
            "feature_set": args.feature_set,
            "trial": idx,
            "cfg_path": cfg_path,
            "save_dir": out_dir / "artifacts" / tag / f"trial_{idx:03d}",
            "log_path": out_dir / "logs" / f"{tag}_trial_{idx:03d}.log",
            "gpu": gpu_ids[(len(tasks)) % len(gpu_ids)],
        })

    manifest = {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "trial_counts": trial_counts,
        "feature_set": args.feature_set,
        "base_configs": {tag: str(path) for tag, path in base_paths.items()},
        "max_workers": args.max_workers,
        "gpus": gpu_ids,
        "seed": args.seed,
        "long_run": args.long_run,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "All artifacts are isolated here; qlib_models and runtime model_weights are not overwritten.",
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = [executor.submit(run_trial, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_leaderboard(rows, out_dir)
            print(
                f"[{len(rows)}/{len(tasks)}] {row['tag']} trial={row['trial']:03d} "
                f"rc={row['return_code']} ann={row['annual_return']} "
                f"normal={row['prediction_normal']} elapsed={row['elapsed_sec']}s",
                flush=True,
            )

    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    write_leaderboard(rows, out_dir)
    print(f"Done. Leaderboard: {out_dir / 'leaderboard.csv'}")


if __name__ == "__main__":
    main()
