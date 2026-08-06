import copy
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from quant_bench.methods.finmem.live.env import load_project_env
from quant_bench.runtime.core.atomic_io import atomic_write_json

load_project_env()


# ============================================================
# 1. 路径与参数配置
# ============================================================

PROJECT_ROOT = Path(os.getenv("FINMEM_WORKSPACE", "~/.local/share/quant-bench/methods/finmem")).expanduser().resolve()
INVESTOR_ROOT = Path(__file__).resolve().parents[1] / "investorbench"

# 支持 run_one_live_cycle.py 通过环境变量传入
SYMBOL = os.getenv("LIVE_SYMBOL", "BTC").upper()
INST_ID = os.getenv("LIVE_INST_ID", f"{SYMBOL}-USDT")

CONFIG_PATH = Path(
    os.getenv("FINMEM_INVESTOR_CONFIG", str(PROJECT_ROOT / "configs" / "investorbench.json"))
).expanduser().resolve()

# 最新实时 JSON，由 live_json_builder.py 生成
LIVE_DATA_PATH = Path(
    os.getenv(
        "LIVE_JSON_PATH",
        str(PROJECT_ROOT / "live_data" / f"{SYMBOL.lower()}_live.json"),
    )
)

# 为 Investor-Bench test 专门生成的临时 JSON：
# rolling_start -> latest_real_date -> dummy_future_date
# 这样 agent 会在最新真实日期做 action，而不是倒数第二天。
TEST_DATA_PATH = PROJECT_ROOT / "live_data" / f"{SYMBOL.lower()}_live_for_test.json"

LIVE_MODEL = os.getenv("LIVE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# 当前 Qdrant 实际可用端口
# 优先使用 .env 里的 QDRANT_ENDPOINT；没有则默认 localhost:6333
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT", "http://localhost:6333")

# 本次 test 输出目录
TEST_RUN_NAME = f"live_test_only_{SYMBOL.lower()}"

# 是否把 warmup checkpoint 里的 Qdrant endpoint 修正为当前 endpoint
PATCH_WARMUP_ENDPOINT = True

# run_one_live_cycle.py 传入的 rolling window 信息
ENV_LIVE_TEST_START_DATE = os.getenv("LIVE_TEST_START_DATE")
ENV_LIVE_TEST_END_DATE = os.getenv("LIVE_TEST_END_DATE")
ENV_LIVE_TARGET_DATE = os.getenv("LIVE_TARGET_DATE")


# ============================================================
# 2. 基础工具
# ============================================================

def load_json(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    path = Path(path)
    atomic_write_json(path, obj)


def check_qdrant(endpoint):
    url = f"{endpoint}/collections"
    print(f"[CHECK QDRANT] {url}")

    with urllib.request.urlopen(url, timeout=5) as resp:
        print("[OK] Qdrant reachable:", resp.status)
        text = resp.read().decode("utf-8")
        print(text[:300])


def _parse_date(date_str):
    return datetime.strptime(str(date_str), "%Y-%m-%d").date()


def _position_signal(position):
    position = float(position)

    if position > 0:
        return "long"
    if position < 0:
        return "short"
    return "flat"


def position_to_signal(position):
    """
    保留旧字段 signal，兼容后续 adapter。

    注意：
    - 这里的 signal 是从 position 映射来的兼容字段。
    - 更准确的字段是 target_position / position_signal。
    """
    position = float(position)

    if position > 0:
        return "buy"
    if position < 0:
        return "sell"
    return "hold"


def build_test_json_with_dummy_future():
    """
    为实时 test 构造一个临时 JSON：

    rolling_start ... latest_real_date -> dummy_future_date

    Investor-Bench 原始环境会在 date_t 做 action，
    用 date_t+1 作为 future price。
    如果 test window 直接到最新真实日期 D 结束，
    action 很可能落在 D-1。
    加入 dummy future 后，test window 变成 [..., D, D+1_dummy]，
    action 就会落在最新真实日期 D。

    dummy future 的价格设为最新真实价格，新闻为空。
    它只用于满足 MarketEnv 的 next-date 要求，不用于真实收益评估。
    """
    data = load_json(LIVE_DATA_PATH)
    all_dates = sorted(data.keys())

    if not all_dates:
        raise ValueError(f"No dates found in {LIVE_DATA_PATH}")

    # target_date 优先级：
    # 1. LIVE_TARGET_DATE
    # 2. LIVE_TEST_END_DATE
    # 3. live json 最后一日
    latest_real_date = ENV_LIVE_TARGET_DATE or ENV_LIVE_TEST_END_DATE or all_dates[-1]

    if latest_real_date not in data:
        raise ValueError(
            f"LIVE_TARGET_DATE/latest_real_date {latest_real_date} not found in {LIVE_DATA_PATH}. "
            f"Available range: {all_dates[0]} -> {all_dates[-1]}"
        )

    # rolling 起点优先用 LIVE_TEST_START_DATE；没有则默认只用 latest_real_date
    test_start = ENV_LIVE_TEST_START_DATE or latest_real_date

    start_date_obj = _parse_date(test_start)
    target_date_obj = _parse_date(latest_real_date)

    if start_date_obj > target_date_obj:
        raise ValueError(
            f"LIVE_TEST_START_DATE {test_start} > LIVE_TARGET_DATE {latest_real_date}"
        )

    # 只保留 rolling window 内的真实日期，避免旧数据或其他残留进入 test
    selected_data = {}
    for d in all_dates:
        d_obj = _parse_date(d)
        if start_date_obj <= d_obj <= target_date_obj:
            selected_data[d] = copy.deepcopy(data[d])

    if latest_real_date not in selected_data:
        raise ValueError(f"Target date {latest_real_date} not selected into test data.")

    latest_item = copy.deepcopy(selected_data[latest_real_date])

    if "prices" not in latest_item:
        raise ValueError(f"Latest item has no prices: {latest_real_date}")

    dummy_future_date = (
        datetime.strptime(latest_real_date, "%Y-%m-%d") + timedelta(days=1)
    ).date().isoformat()

    # 如果 dummy_future_date 已经存在于 selected_data，说明输入 JSON 已经有下一日真实数据。
    # 为了避免覆盖真实数据，这种情况下直接使用真实下一日作为 test_end。
    if dummy_future_date in selected_data:
        print("=" * 80)
        print("[DUMMY FUTURE SKIPPED]")
        print(
            f"{dummy_future_date} already exists in selected test data. "
            "Use it as real next date."
        )
        test_end = dummy_future_date

    else:
        selected_data[dummy_future_date] = {
            "prices": latest_item["prices"],
            "news": [],
            "is_dummy_future": True,
            "source_date": latest_real_date,
        }
        test_end = dummy_future_date

    save_json(TEST_DATA_PATH, selected_data)

    selected_dates = sorted(selected_data.keys())

    print("=" * 80)
    print("[DUMMY FUTURE / ROLLING TEST JSON]")
    print("source_live_json:", LIVE_DATA_PATH)
    print("test_data_path:", TEST_DATA_PATH)
    print("test_start:", test_start)
    print("target_date:", latest_real_date)
    print("test_end:", test_end)
    print("total_days:", len(selected_dates))
    print("selected_dates:", selected_dates)
    print("latest_real_price:", latest_item["prices"])
    print("=" * 80)

    return str(TEST_DATA_PATH), test_start, test_end, latest_real_date, len(selected_data)


def _resolve_config_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (CONFIG_PATH.parent / path).resolve()


def get_warmup_output_path(config=None):
    explicit = os.getenv("FINMEM_WARMUP_OUTPUT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    config = config or load_json(CONFIG_PATH)
    return _resolve_config_path(config["meta_config"]["warmup_output_save_path"])


def get_warmup_agent_state_path(config=None):
    return get_warmup_output_path(config) / "agent" / "state_dict.json"


def read_warmup_agent_name(default_name="agent_live_btc"):
    state_path = get_warmup_agent_state_path()

    if not state_path.exists():
        raise FileNotFoundError(
            f"Warmup checkpoint not found:\n{state_path}\n\n"
            f"你现在选择跳过 warmup，因此这里必须已经存在一个 warmup_output/agent/state_dict.json。"
        )

    state = load_json(state_path)
    return state.get("agent_config", {}).get("agent_name", default_name)


def patch_warmup_checkpoint_endpoint(config=None):
    """
    只修正 endpoint，不修改 agent_name。

    原因：
    - agent_name 对应 Qdrant collection 名。
    - 如果 warmup 时 collection 叫 agent，就必须继续用 agent。
    - 不能随便把 checkpoint 里的 agent_name 改成 agent_live_btc，
      否则 Qdrant 里可能没有对应 collection。
    """
    state_path = get_warmup_agent_state_path(config)
    state = load_json(state_path)

    old_endpoint = (
        state.get("agent_config", {})
        .get("memory_db_config", {})
        .get("memory_db_endpoint")
    )

    state["agent_config"]["memory_db_config"]["memory_db_endpoint"] = QDRANT_ENDPOINT

    save_json(state_path, state)

    print("=" * 80)
    print("[PATCH WARMUP ENDPOINT]")
    print("state_dict:", state_path)
    print("old endpoint:", old_endpoint)
    print("new endpoint:", QDRANT_ENDPOINT)


# ============================================================
# 3. patch configs/main.json，只运行 test
# ============================================================

def patch_config_for_test(config):
    # rolling_start ... latest_real_date -> dummy_future_date
    test_data_path, test_start, test_end, target_date, total_days = (
        build_test_json_with_dummy_future()
    )

    print("=" * 80)
    print("[LIVE TEST WINDOW]")
    print("source live json:", LIVE_DATA_PATH)
    print("test json:", test_data_path)
    print("total days:", total_days)
    print("test_start:", test_start)
    print("target_date:", target_date)
    print("test_end:  ", test_end)

    model_name = config["chat_config"]["chat_model"]

    warmup_agent_name = read_warmup_agent_name()
    test_base = PROJECT_ROOT / "results" / TEST_RUN_NAME / model_name.replace("/", "_") / SYMBOL

    # env
    config["env_config"]["trading_symbols"] = [SYMBOL]
    config["env_config"]["env_data_path"] = {
        SYMBOL: test_data_path
    }
    config["env_config"]["test_start_time"] = test_start
    config["env_config"]["test_end_time"] = test_end

    # warmup 时间在 test-only 阶段不是核心，但保留合法值
    config["env_config"]["warmup_start_time"] = test_start
    config["env_config"]["warmup_end_time"] = test_end

    # portfolio
    config["portfolio_config"]["trading_symbols"] = [SYMBOL]
    config["portfolio_config"]["type"] = "single-asset"

    # agent
    # 这里使用 warmup checkpoint 里的 agent_name，避免 Qdrant collection 不一致
    config["agent_config"]["agent_name"] = warmup_agent_name
    config["agent_config"]["trading_symbols"] = [SYMBOL]
    config["agent_config"]["memory_db_config"]["memory_db_endpoint"] = QDRANT_ENDPOINT
    config["agent_config"]["memory_db_config"]["trading_symbols"] = [SYMBOL]

    # meta
    config["meta_config"]["run_name"] = TEST_RUN_NAME

    # 关键：test 从已有 warmup_output 加载 agent
    config["meta_config"]["warmup_output_save_path"] = str(get_warmup_output_path(config))

    # test 输出单独保存，避免覆盖 warmup
    config["meta_config"]["test_checkpoint_save_path"] = str(test_base / "test_checkpoint")
    config["meta_config"]["test_output_save_path"] = str(test_base / "test_output")
    config["meta_config"]["result_save_path"] = str(test_base / "final_result")
    config["meta_config"]["log_save_path"] = str(test_base / "logs")

    print("=" * 80)
    print("[USING WARMUP]")
    print("warmup_output_save_path:", config["meta_config"]["warmup_output_save_path"])
    print("warmup agent_name:", warmup_agent_name)

    print("=" * 80)
    print("[TEST OUTPUT]")
    print("test_checkpoint_save_path:", config["meta_config"]["test_checkpoint_save_path"])
    print("test_output_save_path:    ", config["meta_config"]["test_output_save_path"])

    return config, target_date


# ============================================================
# 4. 运行 test
# ============================================================

def run_investor_test(config_path):
    print("=" * 80)
    print("[RUN] InvestorBench test")

    env = os.environ.copy()

    if not env.get("OPENAI_API_KEY") and env.get("DASHSCOPE_API_KEY"):
        env["OPENAI_API_KEY"] = env["DASHSCOPE_API_KEY"]

    subprocess.run(
        [
            sys.executable,
            "-m",
            "quant_bench.methods.finmem.investorbench.run",
            "test",
            "--config-path",
            str(config_path),
        ],
        cwd=CONFIG_PATH.parent,
        check=True,
        env=env,
    )


# ============================================================
# 5. 从 test checkpoint 里提取 target_date action
# ============================================================

def extract_latest_action(patched_config, target_date=None):
    """
    从 test_checkpoint/agent 读取 portfolio action record。

    rolling window 场景下必须提取 target_date 对应的 action，
    不能简单 tail(1)，因为 tail 可能是 dummy future 或其他日期。
    """
    # 避免 load_checkpoint 因环境变量缺失报错
    if "OPENAI_API_KEY" not in os.environ:
        if "DASHSCOPE_API_KEY" in os.environ:
            os.environ["OPENAI_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]
        else:
            os.environ["OPENAI_API_KEY"] = "dummy_key_for_loading_checkpoint_only"

    from quant_bench.methods.finmem.investorbench.engine.agent import FinMemAgent

    agent_path = _resolve_config_path(patched_config["meta_config"]["test_checkpoint_save_path"]) / "agent"

    if not agent_path.exists():
        raise FileNotFoundError(f"test checkpoint agent path not found: {agent_path}")

    agent = FinMemAgent.load_checkpoint(path=str(agent_path))

    action_df = pd.DataFrame(agent.portfolio.get_action_record())

    if action_df.empty:
        raise ValueError("Action record is empty.")

    action_df["date"] = pd.to_datetime(action_df["date"]).dt.strftime("%Y-%m-%d")

    print("=" * 80)
    print("[ACTION RECORD]")
    print(action_df.to_string(index=False))

    if target_date:
        target_rows = action_df[action_df["date"] == str(target_date)]

        if target_rows.empty:
            raise ValueError(
                f"No action found for target_date={target_date}. "
                f"Available dates: {sorted(action_df['date'].unique().tolist())}"
            )

        latest = target_rows.tail(1).iloc[0].to_dict()
    else:
        latest = action_df.tail(1).iloc[0].to_dict()

    position = latest.get("position", 0)
    signal = position_to_signal(position)
    pos_signal = _position_signal(position)

    result = {
        "symbol": SYMBOL,
        "date": latest.get("date"),
        "target_date": str(target_date) if target_date else latest.get("date"),
        "position": float(position),
        "target_position": float(position),
        "position_signal": pos_signal,
        # 兼容旧版 adapter：
        # position > 0 -> buy；position == 0 -> hold；position < 0 -> sell
        "signal": signal,
        "raw_action": latest,
    }

    out_dir = PROJECT_ROOT / "live_decision"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{SYMBOL.lower()}_latest_action.json"

    save_json(out_path, result)

    print("=" * 80)
    print("[LATEST TARGET ACTION]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("saved to:", out_path)

    return result


# ============================================================
# 6. 主流程
# ============================================================

def main():
    if not LIVE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Live JSON not found: {LIVE_DATA_PATH}\n"
            f"请先运行 live_json_builder 生成 live_data/{SYMBOL.lower()}_live.json"
        )

    print("=" * 80)
    print("[LIVE RUNNER ENV]")
    print("SYMBOL:", SYMBOL)
    print("INST_ID:", INST_ID)
    print("LIVE_DATA_PATH:", LIVE_DATA_PATH)
    print("LIVE_TEST_START_DATE:", ENV_LIVE_TEST_START_DATE)
    print("LIVE_TEST_END_DATE:", ENV_LIVE_TEST_END_DATE)
    print("LIVE_TARGET_DATE:", ENV_LIVE_TARGET_DATE)

    check_qdrant(QDRANT_ENDPOINT)

    state_path = get_warmup_agent_state_path()
    if not state_path.exists():
        raise FileNotFoundError(
            f"Warmup state_dict not found:\n{state_path}\n\n"
            f"你现在跳过 warmup，所以必须先有这个文件。"
        )

    config = load_json(CONFIG_PATH)
    if PATCH_WARMUP_ENDPOINT:
        patch_warmup_checkpoint_endpoint(config)

    patched_config, target_date = patch_config_for_test(config)
    runtime_config = PROJECT_ROOT / "configs" / f"live_test_{SYMBOL.lower()}.json"
    save_json(runtime_config, patched_config)
    print("=" * 80)
    print("[RUNTIME CONFIG WRITTEN]")
    print(runtime_config)

    run_investor_test(runtime_config)
    extract_latest_action(patched_config, target_date=target_date)
    print("=" * 80)
    print("[DONE] live test-only module completed.")


if __name__ == "__main__":
    main()
