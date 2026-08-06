import argparse
import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

from quant_bench.methods.finmem.live.env import load_project_env
from quant_bench.runtime.core.atomic_io import atomic_write_json


load_project_env()


PROJECT_ROOT = Path(os.getenv("FINMEM_WORKSPACE", "~/.local/share/quant-bench/methods/finmem")).expanduser().resolve()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    atomic_write_json(path, obj)


def check_qdrant(endpoint):
    with urllib.request.urlopen(f"{endpoint}/collections", timeout=5) as resp:
        print("[OK] Qdrant reachable:", resp.status)


def delete_collection(endpoint, collection_name):
    url = f"{endpoint}/collections/{collection_name}"
    req = urllib.request.Request(url, method="DELETE")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[OK] Deleted collection {collection_name}: {resp.status}")
    except urllib.error.HTTPError as e:
        if e.code in {400, 404}:
            print(f"[INFO] Collection {collection_name} does not exist.")
        else:
            raise


def patch_config_for_warmup(
    config,
    symbol,
    qdrant_endpoint,
    warmup_days,
):
    warmup_data_path = Path(config.pop("_data_dir")) / f"{symbol.lower()}.json"
    run_name = f"live_warmup_{symbol.lower()}"
    agent_name = f"agent_live_{symbol.lower()}"

    data = load_json(warmup_data_path)
    dates = sorted(data.keys())

    if len(dates) < warmup_days:
        raise ValueError(f"Need at least {warmup_days} days, got {len(dates)}")

    warmup_dates = dates[-warmup_days:]

    warmup_start = warmup_dates[0]
    warmup_end = warmup_dates[-1]

    # test 随便给一个合法窗口，warmup 阶段不会重点使用
    test_start = dates[-2]
    test_end = dates[-1]

    model_name = config["chat_config"]["chat_model"]
    base_dir = PROJECT_ROOT / "results" / run_name / model_name.replace("/", "_") / symbol

    config["env_config"]["trading_symbols"] = [symbol]
    config["env_config"]["env_data_path"] = {symbol: str(warmup_data_path)}
    config["env_config"]["warmup_start_time"] = warmup_start
    config["env_config"]["warmup_end_time"] = warmup_end
    config["env_config"]["test_start_time"] = test_start
    config["env_config"]["test_end_time"] = test_end

    config["portfolio_config"]["trading_symbols"] = [symbol]
    config["portfolio_config"]["type"] = "single-asset"

    config["agent_config"]["agent_name"] = agent_name
    config["agent_config"]["trading_symbols"] = [symbol]
    config["agent_config"]["memory_db_config"]["memory_db_endpoint"] = qdrant_endpoint
    config["agent_config"]["memory_db_config"]["trading_symbols"] = [symbol]

    config["meta_config"]["run_name"] = run_name
    config["meta_config"]["warmup_checkpoint_save_path"] = str(base_dir / "warmup_checkpoint")
    config["meta_config"]["warmup_output_save_path"] = str(base_dir / "warmup_output")
    config["meta_config"]["test_checkpoint_save_path"] = str(base_dir / "test_checkpoint")
    config["meta_config"]["test_output_save_path"] = str(base_dir / "test_output")
    config["meta_config"]["result_save_path"] = str(base_dir / "final_result")
    config["meta_config"]["log_save_path"] = str(base_dir / "logs")

    print("[WARMUP WINDOW]")
    print(warmup_start, "->", warmup_end)

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a BTC/ETH FinMem memory warmup for live paper trading."
    )
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--warmup-days", type=int, default=60)
    parser.add_argument(
        "--qdrant-endpoint",
        default=os.getenv("QDRANT_ENDPOINT", "http://localhost:6333"),
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Delete this asset's existing Qdrant collection before warmup.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    symbol = args.symbol.upper()
    agent_name = f"agent_live_{symbol.lower()}"

    if args.warmup_days < 4:
        raise ValueError("--warmup-days must be >= 4")
    if not args.allow_network:
        raise PermissionError("warmup uses Qdrant, LLM, and embedding services; pass --allow-network")

    check_qdrant(args.qdrant_endpoint)

    if args.reset_collection:
        delete_collection(args.qdrant_endpoint, agent_name)

    config = load_json(Path(args.config).expanduser().resolve())
    config["_data_dir"] = str(Path(args.data_dir).expanduser().resolve())
    patched = patch_config_for_warmup(
        config=config,
        symbol=symbol,
        qdrant_endpoint=args.qdrant_endpoint,
        warmup_days=args.warmup_days,
    )
    runtime_config = PROJECT_ROOT / "configs" / f"warmup_{symbol.lower()}.json"
    save_json(runtime_config, patched)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "quant_bench.methods.finmem.investorbench.run",
            "warmup",
            "--config-path",
            str(runtime_config),
        ],
        cwd=runtime_config.parent,
        check=True,
    )
    print("[DONE] warmup prepared.")
    print("warmup_output_save_path:")
    print(patched["meta_config"]["warmup_output_save_path"])


if __name__ == "__main__":
    main()
