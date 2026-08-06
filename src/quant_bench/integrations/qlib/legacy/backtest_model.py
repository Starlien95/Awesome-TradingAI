import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import qlib
import torch
import yaml
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SignalRecord

from quant_bench.integrations.qlib.legacy.workflow_utils import apply_config_sys_path, resolve_workflow_config

# ====== 修复 PyTorch 2.6+ 与 Qlib 加载旧版模型冲突的补丁 ======
original_torch_load = torch.load
torch.load = lambda *args, **kwargs: original_torch_load(*args, **{**kwargs, 'weights_only': False})
# ==============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest Qlib model (No Training)")
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
        help="Feature set used for default config lookup and model paths (default: 158)",
    )
    parser.add_argument(
        "--freq",
        "-f",
        required=True,
        help="Timeframe string (1m, 5m, 15m, 1h, 4h, etc.).",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="lgb",
        help="Model name tag used to locate the saved artifacts (default: lgb)",
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
        help="Override directory to load trained models from. Defaults to qlib_models/{feature_set}/{model}/{freq}",
    )

    args = parser.parse_args()
    if args.config is None:
        try:
            args.config = resolve_workflow_config(args.model, args.freq, args.feature_set)
        except FileNotFoundError as exc:
            print(f"  ✗ {exc}")
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
    print(f"  ✓ Random seed set to: {seed}")

def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
        apply_config_sys_path(config, path)
        return config
    except FileNotFoundError:
        print(f"  ✗ Config not found: {path}")
        sys.exit(1)

def build_save_paths(model_tag: str, freq: str, feature_set: str, override: str | None) -> tuple[Path, Path, Path]:
    base_dir = Path(override) if override else Path("qlib_models") / feature_set / model_tag / freq
    qlib_model_path = base_dir / f"qlib_{model_tag}_{freq}_model.pkl"
    raw_model_path = base_dir / f"{model_tag}_{freq}_model.pkl"
    return base_dir, qlib_model_path, raw_model_path

def prepare_benchmark(config: dict, freq: str):
    from qlib.data import D

    freqs_map = {
        "1m": "1min", "5m": "5min", "15m": "15min",
        "1h": "60min", "4h": "240min",
    }
    freq_str = freqs_map.get(freq, freq)
    print(f"  Preparing BTC-USDT benchmark data ({freq_str})...")

    btc_close_df = D.features(instruments=["btc-usdt"], fields=["$close"], freq=freq_str)

    if btc_close_df.empty:
        print("  ! Warning: Benchmark data is empty. Check your data provider.")
        return

    btc_price = btc_close_df.loc["btc-usdt", "$close"]
    benchmark_series = btc_price / btc_price.shift(1) - 1
    benchmark_series = benchmark_series.dropna()
    config["port_analysis_config"]["backtest"]["benchmark"] = benchmark_series
    print(f"  ✓ Benchmark series ready. Length={len(benchmark_series)}")

def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("🚀 Qlib Backtest-Only Script 🚀")
    print("=" * 60)

    config = load_config(args.config)
    set_random_seed(args.seed)

    qlib_data_path = config["qlib_init"]["provider_uri"]
    qlib.init(provider_uri=qlib_data_path, mount_path=qlib_data_path)
    print("  ✓ Qlib initialized.")

    print("\nStep 1: Loading Dataset...")
    dataset = init_instance_by_config(config["task"]["dataset"])

    print("\nStep 2: Loading Pre-trained Model...")
    _, qlib_model_path, _ = build_save_paths(args.model, args.freq, args.feature_set, args.save_dir)

    try:
        with open(qlib_model_path, "rb") as f:
            model = pickle.load(f)
        print(f"  ✓ Pre-trained model loaded successfully from: {qlib_model_path}")
    except FileNotFoundError:
        print(f"  ✗ Error: Cannot find trained model at {qlib_model_path}.")
        print("    Please run the training script first or check your -m and -f arguments.")
        sys.exit(1)

    print("\nStep 3: Generating Signals & Running Backtest...")
    recorder = R.get_recorder()

    # 1. 使用模型和测试集生成预测信号
    sr = SignalRecord(model=model, dataset=dataset, recorder=recorder)
    sr.generate()
    print("  ✓ Prediction signals generated.")

    # 2. 强制同步回测频率，防止错位
    freqs_map = {
        "1m": "1min", "5m": "5min", "15m": "15min",
        "30m": "30min", "1h": "60min", "4h": "240min"
    }
    qlib_freq = freqs_map.get(args.freq, "day")

    config["port_analysis_config"]["freq"] = qlib_freq
    if "benchmark" in config["port_analysis_config"]["backtest"]:
        config["port_analysis_config"]["backtest"]["benchmark"]["freq"] = qlib_freq
    if "exchange_kwargs" in config["port_analysis_config"]["backtest"]:
        config["port_analysis_config"]["backtest"]["exchange_kwargs"]["freq"] = qlib_freq

    prepare_benchmark(config, args.freq)

    # 3. 运行投资组合分析 (模拟撮合交易并扣除手续费)
    par = PortAnaRecord(recorder=recorder, config=config["port_analysis_config"])
    par.generate()

    print("\n--- Backtest finished successfully ---")
    print("📊 Check the terminal output above or look in the 'mlruns' folder for your performance metrics.")

if __name__ == "__main__":
    main()
