#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import qlib
import yaml
from qlib.utils import init_instance_by_config

from quant_bench.integrations.qlib.legacy.auto_tune_trade_models import (
    prediction_is_normal,
    write_leaderboard,
)
from quant_bench.integrations.qlib.legacy.prediction_stats_utils import (
    object_to_current_frame,
    prediction_distribution_stats,
)

ROOT = Path.cwd()


def parse_args():
    parser = argparse.ArgumentParser(description="Recompute prediction stats for finished auto-tune trials.")
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--tag", default="lstm4h")
    parser.add_argument("--trial", type=int, default=None)
    return parser.parse_args()


def update_trial(stage_dir: Path, tag: str, trial: int) -> dict:
    if tag == "lstm4h":
        model_name, freq = "lstm", "4h"
    elif tag == "mlp1h":
        model_name, freq = "mlp", "1h"
    elif tag == "tra4h":
        model_name, freq = "tra", "4h"
    else:
        raise ValueError(f"unsupported tag: {tag}")

    cfg_path = stage_dir / "configs" / f"{tag}_trial_{trial:03d}.yaml"
    save_dir = stage_dir / "artifacts" / tag / f"trial_{trial:03d}"
    metrics_path = save_dir / "train_metrics.json"
    model_path = save_dir / f"qlib_{model_name}_{freq}_model.pkl"
    if not cfg_path.exists() or not metrics_path.exists() or not model_path.exists():
        return {"trial": trial, "updated": False, "reason": "missing files"}

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    qlib.init(provider_uri=cfg["qlib_init"]["provider_uri"], mount_path=cfg["qlib_init"]["provider_uri"])
    with model_path.open("rb") as f:
        model = pickle.load(f)
    dataset = init_instance_by_config(cfg["task"]["dataset"])

    pred = object_to_current_frame(model.predict(dataset), "score")
    label = object_to_current_frame(dataset.prepare("test", col_set="label"), "label")
    eval_df = pred.join(label, how="inner").dropna()
    stats = prediction_distribution_stats(eval_df["score"])

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    metrics["prediction_stats"] = stats
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {"trial": trial, "updated": True, "stats": stats}


def main():
    args = parse_args()
    stage_dir = Path(args.stage_dir)
    leaderboard_path = stage_dir / "leaderboard.json"
    rows = json.loads(leaderboard_path.read_text(encoding="utf-8")) if leaderboard_path.exists() else []
    trials = [args.trial] if args.trial is not None else sorted({int(r["trial"]) for r in rows if r.get("tag") == args.tag})

    for trial in trials:
        result = update_trial(stage_dir, args.tag, trial)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not result.get("updated"):
            continue
        stats = result["stats"]
        for row in rows:
            if row.get("tag") == args.tag and int(row.get("trial")) == trial:
                row["prediction_normal"] = prediction_is_normal(stats)
                row["prediction_mean"] = stats.get("mean")
                row["prediction_std"] = stats.get("std")
                row["prediction_positive_ratio"] = stats.get("positive_ratio")
                row["prediction_negative_ratio"] = stats.get("negative_ratio")
                break
        write_leaderboard(rows, stage_dir)


if __name__ == "__main__":
    main()
