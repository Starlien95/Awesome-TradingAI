import argparse
import json
import os
import pickle
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ====== 修复 PyTorch 2.6+ 与 Qlib 加载旧版模型冲突的补丁 ======
original_torch_load = torch.load
torch.load = lambda *args, **kwargs: original_torch_load(*args, **{**kwargs, 'weights_only': False})
# ==============================================================

import qlib
import yaml
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SignalRecord

from quant_bench.integrations.qlib.legacy.prediction_stats_utils import (
    object_to_current_frame,
    prediction_distribution_stats,
)
from quant_bench.integrations.qlib.legacy.workflow_utils import apply_config_sys_path, resolve_workflow_config


def patch_dnn_model() -> None:
    try:
        from qlib.contrib.model.pytorch_nn import DNNModelPytorch

        def patched_get_metric(self, pred, target, index):
            pred_tensor = pred.float() if isinstance(pred, torch.Tensor) else torch.tensor(pred, dtype=torch.float32)
            target_tensor = (
                target.float() if isinstance(target, torch.Tensor) else torch.tensor(target, dtype=torch.float32)
            )
            device = getattr(self, "device", None)
            if device is not None:
                pred_tensor = pred_tensor.to(device)
                target_tensor = target_tensor.to(device)
            mask = ~(torch.isnan(pred_tensor) | torch.isnan(target_tensor))
            if mask.sum() < 2:
                return torch.tensor(-999.0, device=pred_tensor.device)
            pred_clean = pred_tensor[mask]
            target_clean = target_tensor[mask]
            return -torch.mean((pred_clean - target_clean) ** 2)

        DNNModelPytorch.get_metric = patched_get_metric
        print("  ✓ DNNModelPytorch metric patched.")
    except ImportError:
        pass


patch_dnn_model()


# === [新增] 全局补丁函数 (必须定义在 main 之外以支持 Pickle) ===
def global_tcn_forward_patch(self, x):
    # 自动检测维度错位: [Batch, Time, Feat] -> [Batch, Feat, Time]
    # 这里 52 是你的 d_feat (根据配置调整)，如果最后一维是 52 且倒数第二维不是，说明需要转置
    if x.shape[-1] == 52 and x.shape[-2] != 52:
        x = x.permute(0, 2, 1)

    # 调用备份的原始 forward 方法
    return self._original_forward(x)
# ========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qlib model with configurable freq/model")
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to workflow config YAML. Defaults to workflow/{158,360}/{model}/workflow_config_{model}_{freq}.yaml",
    )
    parser.add_argument(
        "--feature-set",
        "--feature_set",
        choices=("158", "360"),
        default="158",
        help="Feature set used for default config lookup and model output paths (default: 158)",
    )
    parser.add_argument(
        "--freq",
        "-f",
        required=True,
        help="Timeframe string (1m, 5m, 15m, 1h, etc.). Used for default config and output paths.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="lgb",
        help="Model name tag used in config filename and saved artifacts (default: lgb)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for numpy/torch/python (default: 42)",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Override directory to save trained models. Defaults to qlib_models/{feature_set}/{model}/{freq}",
    )
    parser.add_argument(
        "--deploy-framework",
        action="store_true",
        help="Deploy saved artifacts into the integrated trading runtime after training.",
    )
    parser.add_argument(
        "--framework-dir",
        default=".",
        help="Project root used with --deploy-framework.",
    )
    parser.add_argument(
        "--framework-config",
        default=None,
        help="Optional framework YAML config to update during deployment.",
    )

    args = parser.parse_args()
    if args.config is None:
        try:
            args.config = resolve_workflow_config(args.model, args.freq, args.feature_set)
        except FileNotFoundError as exc:
            print(f"✗ {exc}")
            sys.exit(1)
    return args


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to: {seed}")


def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
        apply_config_sys_path(config, path)
        return config
    except FileNotFoundError:
        print(f"✗ Config not found: {path}")
        sys.exit(1)


def build_save_paths(model_tag: str, freq: str, feature_set: str, override: str | None) -> tuple[Path, Path, Path]:
    base_dir = Path(override) if override else Path("qlib_models") / feature_set / model_tag / freq
    base_dir.mkdir(parents=True, exist_ok=True)
    qlib_model_path = base_dir / f"qlib_{model_tag}_{freq}_model.pkl"
    raw_model_path = base_dir / f"{model_tag}_{freq}_model.pkl"
    return base_dir, qlib_model_path, raw_model_path


def prepare_benchmark(config: dict, freq: str):
    from qlib.data import D

    freqs_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "60min",
        "4h": "240min",
    }
    freq_str = freqs_map.get(freq, freq)
    print(f"  Preparing BTC-USDT benchmark data ({freq_str})...")
    # 注意: Crypto 数据通常使用 $close
    btc_close_df = D.features(instruments=["btc-usdt"], fields=["$close"], freq=freq_str)

    if btc_close_df.empty:
        print("  ! Warning: Benchmark data is empty. Check your data provider.")
        return

    btc_price = btc_close_df.loc["btc-usdt", "$close"]
    benchmark_series = btc_price / btc_price.shift(1) - 1
    benchmark_series = benchmark_series.dropna()
    config["port_analysis_config"]["backtest"]["benchmark"] = benchmark_series
    print(f"  ✓ Benchmark series ready. Length={len(benchmark_series)}")


def configure_local_mlflow() -> None:
    try:
        mlflow_dir = Path.cwd() / "mlruns"
        mlflow_dir.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:{mlflow_dir}"
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
        except Exception:
            pass
    except Exception as exc:
        print(f"  ! Warning: failed to set local MLflow dir: {exc}")


def json_float(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def collect_recorder_metrics(recorder) -> dict:
    try:
        raw_metrics = recorder.list_metrics()
    except Exception as exc:
        print(f"  ! recorder.list_metrics failed: {exc}")
        return {}

    metrics = {}
    for key, value in (raw_metrics or {}).items():
        clean_value = json_float(value)
        if clean_value is not None:
            metrics[str(key)] = clean_value
    return metrics


def get_torch_model(qlib_model) -> nn.Module | None:
    for attr in ("model", "dnn_model", "net", "fitted_model", "_model", "nn_model", "TCN_model"):
        if hasattr(qlib_model, attr):
            inner = getattr(qlib_model, attr)
            if isinstance(inner, nn.Module):
                return inner

    for attr in dir(qlib_model):
        if attr.startswith("__"):
            continue
        try:
            inner = getattr(qlib_model, attr)
        except Exception:
            continue
        if isinstance(inner, nn.Module):
            return inner
    return None


def save_torch_artifacts(torch_model: nn.Module | None, save_dir: Path, model_tag: str, freq: str) -> None:
    if torch_model is None:
        return

    pth_path = save_dir / f"{model_tag}_{freq}_model.pth"
    torch.save(torch_model, pth_path)
    print(f"  ✓ Torch model saved to {pth_path}")

    state_dict_path = save_dir / f"{model_tag}_{freq}_state_dict.pth"
    torch.save(torch_model.state_dict(), state_dict_path)
    print(f"  ✓ Torch state_dict saved to {state_dict_path}")

    input_dim = None
    for name, param in torch_model.state_dict().items():
        if "weight" in name and len(param.shape) == 2:
            input_dim = int(param.shape[1])
            break

    config_path = save_dir / f"{model_tag}_{freq}_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "input_dim": input_dim,
                "model_class": type(torch_model).__name__,
                "state_dict_keys": list(torch_model.state_dict().keys()),
            },
            f,
            indent=2,
        )
    print(f"  ✓ Torch config saved to {config_path}")


def collect_signal_metrics(model, dataset, config: dict) -> tuple[dict, dict]:
    prediction_stats = {}
    eval_metrics = {}

    try:
        pred_series = object_to_current_frame(model.predict(dataset), "score")
        label_df = object_to_current_frame(dataset.prepare("test", col_set="label"), "label")
        df_eval = pd.concat([pred_series, label_df], axis=1).dropna()
        if df_eval.empty:
            return prediction_stats, eval_metrics

        prediction_stats = prediction_distribution_stats(df_eval["score"])
        if prediction_stats:
            print(f"  Prediction stats: {prediction_stats}")

        if df_eval.index.nlevels > 1:
            n_instruments = df_eval.index.get_level_values("instrument").nunique()
        else:
            n_instruments = 1

        if n_instruments > 1 and df_eval.index.nlevels > 1:
            datetime_level = df_eval.index.get_level_values("datetime")
            ic_per_slice = (
                df_eval.groupby(datetime_level)
                .apply(lambda g: g["score"].corr(g["label"]) if len(g) >= 2 else np.nan)
                .dropna()
            )
            ic = json_float(ic_per_slice.mean())
            ic_std = json_float(ic_per_slice.std())
            icir = json_float(ic / ic_std) if ic is not None and ic_std and ic_std > 0 else None
            eval_metrics.update({"ic": ic, "ic_std": ic_std, "icir": icir, "ic_type": "cross_section"})
        else:
            eval_metrics.update({"ic": json_float(df_eval["score"].corr(df_eval["label"])), "ic_type": "time_series"})

        label_exprs = config.get("data_handler_config", {}).get("label", [""])
        label_expr = label_exprs[0] if isinstance(label_exprs, list) and label_exprs else str(label_exprs)
        is_classification = "If(" in str(label_expr).strip()
        score_threshold = 0.5 if is_classification else 0.0
        label_threshold = 0.5 if is_classification else 0.0

        y_pred = (df_eval["score"] > score_threshold).astype(int)
        y_true = (df_eval["label"] > label_threshold).astype(int)
        eval_metrics["direction_accuracy"] = json_float((y_pred == y_true).mean())
        eval_metrics["positive_ratio"] = json_float(y_true.mean())

        try:
            from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

            eval_metrics.update(
                {
                    "precision": json_float(precision_score(y_true, y_pred, zero_division=0)),
                    "recall": json_float(recall_score(y_true, y_pred, zero_division=0)),
                    "f1_score": json_float(f1_score(y_true, y_pred, zero_division=0)),
                    "mcc": json_float(matthews_corrcoef(y_true, y_pred)),
                }
            )
        except Exception:
            pass
    except Exception as exc:
        print(f"  ! Signal metric calculation failed: {exc}")

    return prediction_stats, eval_metrics


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("Generic Train Script")
    print("=" * 60)
    config = load_config(args.config)

    set_random_seed(args.seed)
    configure_local_mlflow()

    qlib_data_path = config["qlib_init"]["provider_uri"]
    qlib.init(provider_uri=qlib_data_path, mount_path=qlib_data_path)
    print("  ✓ Qlib initialized.")

    print("Step 3: Creating model and dataset objects...")
    model = init_instance_by_config(config["task"]["model"])
    dataset = init_instance_by_config(config["task"]["dataset"])

    # === [修改 1] TCN 维度修复 (Pickle 兼容版) ===
    # if config["task"]["model"]["class"] == "TCN":
    #     print("🔧 Applying TCN dimension fix patch (Global version)...")

    #     # 获取 PyTorch 原生模型实例 (Qlib wrapper 内部叫做 TCN_model)
    #     raw_torch_model = model.TCN_model

    #     # 1. 备份原始 forward 方法
    #     if not hasattr(raw_torch_model, "_original_forward"):
    #         raw_torch_model._original_forward = raw_torch_model.forward

    #     # 2. 将全局函数绑定为实例的新 forward 方法
    #     # types.MethodType 会自动处理 'self' 参数
    #     raw_torch_model.forward = types.MethodType(global_tcn_forward_patch, raw_torch_model)

    #     print("✓ TCN patch applied successfully.")
    # ============================================

    print("Step 4: Start training model...")
    model.fit(dataset)
    print("  ...Model training finished.")

    try:
        R.save_objects(trained_qlib_model=model)
    except (PermissionError, OSError) as exc:
        print(f"  ! R.save_objects skipped: {exc}")
    except Exception as exc:
        print(f"  ! R.save_objects failed: {exc}")

    save_dir, qlib_model_path, raw_model_path = build_save_paths(args.model, args.freq, args.feature_set, args.save_dir)
    with open(qlib_model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  ✓ Qlib model saved to {qlib_model_path}")

    # === [修改 2] 更稳健的 Raw Model 保存逻辑 ===
    try:
        # 不同模型 raw model 属性名不同
        if hasattr(model, "model"):
            raw_model = model.model  # LightGBM, DNN 等
        elif hasattr(model, "TCN_model"):
            raw_model = model.TCN_model # TCN
        else:
            raw_model = model # LinearModel 等简单模型

        with open(raw_model_path, "wb") as f:
            pickle.dump(raw_model, f)
        print(f"  ✓ Raw model saved to {raw_model_path}")
    except Exception as exc:
        print(f"  ✗ Failed to save raw model: {exc}")
    # ============================================

    torch_model = get_torch_model(model)
    save_torch_artifacts(torch_model, save_dir, args.model, args.freq)

    try:
        handler_path = save_dir / f"data_handler_{args.freq}.pkl"
        with open(handler_path, "wb") as f:
            pickle.dump(dataset.handler, f)
        print(f"  ✓ DataHandler saved to {handler_path}")
    except Exception as exc:
        print(f"  ! Failed to save DataHandler: {exc}")

    recorder = R.get_recorder()
    sr = SignalRecord(model=model, dataset=dataset, recorder=recorder)
    sr.generate()

    # === [修改 3] 强制同步回测频率，防止 1h 数据跑出 1day 结果 ===
    freqs_map = {
        "1m": "1min", "5m": "5min", "15m": "15min",
        "30m": "30min", "1h": "60min", "4h": "240min"
    }
    qlib_freq = freqs_map.get(args.freq, "day")

    # 覆盖 YAML 中的频率配置
    config["port_analysis_config"]["freq"] = qlib_freq
    if "benchmark" in config["port_analysis_config"]["backtest"]:
        config["port_analysis_config"]["backtest"]["benchmark"]["freq"] = qlib_freq
    if "exchange_kwargs" in config["port_analysis_config"]["backtest"]:
        config["port_analysis_config"]["backtest"]["exchange_kwargs"]["freq"] = qlib_freq

    print(f"  ✓ Forced backtest frequency to: {qlib_freq}")
    # ========================================================

    prepare_benchmark(config, args.freq)
    par = PortAnaRecord(recorder=recorder, config=config["port_analysis_config"])
    par.generate()
    port_metrics = collect_recorder_metrics(recorder)
    prediction_stats, eval_metrics = collect_signal_metrics(model, dataset, config)

    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "freq": args.freq,
        "config": args.config,
        "pytorch_model_extracted": torch_model is not None,
        "model_kwargs": config["task"]["model"].get("kwargs", {}),
        "eval_metrics": eval_metrics,
        "prediction_stats": prediction_stats,
        "port_metrics": port_metrics,
    }

    train_metrics_path = save_dir / "train_metrics.json"
    with open(train_metrics_path, "w") as mf:
        json.dump(metrics, mf, indent=2)
    print(f"  ✓ Train metrics saved to {train_metrics_path}")

    metrics_dir = Path("hyper_param") / args.freq
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_path, "w") as mf:
        json.dump(metrics, mf, indent=2)
    print(f"  ✓ Metrics saved to {metrics_path}")

    if args.deploy_framework:
        from deploy_to_framework import deploy_model_to_framework

        manifest = deploy_model_to_framework(
            model=args.model,
            freq=args.freq,
            feature_set=args.feature_set,
            source_dir=save_dir,
            framework_dir=args.framework_dir,
            framework_config=args.framework_config,
            workflow_config=args.config,
        )
        print(f"  ✓ Deployed to quant framework: {manifest['manifest_path']}")
        if manifest.get("framework_config"):
            print(f"  ✓ Framework config updated: {manifest['framework_config']}")

    print("--- Training finished successfully ---")


if __name__ == "__main__":
    main()
