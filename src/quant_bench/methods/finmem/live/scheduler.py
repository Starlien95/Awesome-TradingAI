# run_live_loop.py

from quant_bench.methods.finmem.live.env import load_project_env
load_project_env()

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from quant_bench.runtime.core.atomic_io import atomic_write_json


# ============================================================
# 1. 配置区
# ============================================================

PROJECT_ROOT = Path(os.getenv("FINMEM_WORKSPACE", "~/.local/share/quant-bench/methods/finmem")).expanduser().resolve()
SYMBOL = os.getenv("FINMEM_SYMBOL", "BTC").upper().strip()
ONE_CYCLE_MODULE = "quant_bench.methods.finmem.live.cycle"

LOOP_RECORD_DIR = PROJECT_ROOT / "live_loop_records"
CYCLE_RECORD_DIR = PROJECT_ROOT / "live_cycle_records"


# ============================================================
# 日频固定时间运行配置
# ============================================================

# 建议按 OKX 日线切换时间，即 Asia/Shanghai 00:05 后运行。
# 如果服务器系统时区是日本，也没关系，这里会显式使用 Asia/Shanghai。
TIMEZONE = "Asia/Shanghai"
DAILY_RUN_HOUR = 0
DAILY_RUN_MINUTE = int(os.getenv("FINMEM_DAILY_RUN_MINUTE", "15" if SYMBOL == "ETH" else "5"))

# 启动脚本后是否立刻跑一轮。
# 正式日频建议 False：启动后等待下一个 00:05。
# 调试时可以改成 True。
RUN_ON_START = False

# 是否使用固定日频时间。
# True: 每天固定 TIMEZONE 的 DAILY_RUN_HOUR:DAILY_RUN_MINUTE 运行。
# False: 使用 LOOP_INTERVAL_SECONDS 间隔运行。
USE_FIXED_DAILY_SCHEDULE = True

# 如果 USE_FIXED_DAILY_SCHEDULE=False，则使用这个间隔。
LOOP_INTERVAL_SECONDS = 60 * 60 * 24

# 连续失败多少次后停止。
MAX_CONSECUTIVE_FAILURES = 5

# 是否无限循环。
RUN_FOREVER = True

# 如果不想无限循环，可以设置最大运行次数。
# RUN_FOREVER=True 时这个值不会生效。
MAX_RUNS = 10

# 每轮之间是否打印倒计时。
SHOW_COUNTDOWN = True

# stdout/stderr 保存尾部长度，避免 loop record 过大。
TAIL_CHARS = 5000


# ============================================================
# 2. 工具函数
# ============================================================

def save_json(path: Path, obj):
    atomic_write_json(path, obj)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_latest_cycle_record_path():
    candidates = sorted(CYCLE_RECORD_DIR.glob(f"cycle_{SYMBOL.lower()}_*.json"))
    if not candidates:
        return None
    return candidates[-1]


def seconds_until_next_daily_run():
    tz = ZoneInfo(TIMEZONE)
    now = datetime.now(tz)

    next_run = now.replace(
        hour=DAILY_RUN_HOUR,
        minute=DAILY_RUN_MINUTE,
        second=0,
        microsecond=0,
    )

    if next_run <= now:
        next_run = next_run + timedelta(days=1)

    seconds = int((next_run - now).total_seconds())

    print("=" * 80)
    print("[NEXT DAILY RUN]")
    print("timezone:", TIMEZONE)
    print("now:", now.isoformat())
    print("next_run:", next_run.isoformat())
    print("sleep_seconds:", seconds)

    return seconds


def sleep_with_countdown(seconds):
    seconds = int(seconds)

    if seconds <= 0:
        return

    if not SHOW_COUNTDOWN:
        time.sleep(seconds)
        return

    remaining = seconds

    while remaining > 0:
        step = min(60, remaining)
        print(f"[LOOP] Sleeping... remaining {remaining} seconds")
        time.sleep(step)
        remaining -= step


def _extract_order_success(cycle_record, symbol):
    execution_result = cycle_record.get("execution_result", {}) or {}

    if not isinstance(execution_result, dict):
        return False

    symbol_result = execution_result.get(symbol, {}) or {}

    if isinstance(symbol_result, dict) and str(symbol_result.get("code", "")) == "0":
        data = symbol_result.get("data", [])
        if isinstance(data, list) and data:
            return any(
                str(x.get("sCode", "0")) == "0"
                for x in data
                if isinstance(x, dict)
            )
        return True

    return False


def extract_cycle_summary(cycle_record):
    if not isinstance(cycle_record, dict):
        return {}

    symbol = str(cycle_record.get("symbol", SYMBOL)).upper()

    investor_action = cycle_record.get("investor_action", {}) or {}
    live_info = cycle_record.get("live_json_info", {}) or {}
    okx_decision = cycle_record.get("okx_decision", {}) or {}
    strategy_state = cycle_record.get("strategy_account_state", {}) or {}

    symbol_decision = {}
    if isinstance(okx_decision, dict):
        symbol_decision = okx_decision.get(symbol, {}) or {}

    action_date = (
        investor_action.get("target_date")
        or investor_action.get("date")
        or live_info.get("latest_date")
        or ""
    )

    return {
        "cycle_status": cycle_record.get("status", "unknown"),
        "symbol": symbol,
        "action_date": action_date,
        "target_position": investor_action.get(
            "target_position",
            investor_action.get("position", ""),
        ),
        "position_signal": investor_action.get("position_signal", ""),
        "investor_signal": investor_action.get("signal", ""),
        "final_signal": symbol_decision.get("signal", ""),
        "risk_reason": symbol_decision.get("risk_reason", ""),
        "order_success": _extract_order_success(cycle_record, symbol),
        "strategy_account_state": strategy_state,
        "live_start_date": live_info.get("live_start_date", ""),
        "live_end_date": live_info.get("live_end_date", ""),
        "latest_date": live_info.get("latest_date", ""),
        "news_mode": live_info.get("news_mode", ""),
    }


def run_one_cycle():
    print("=" * 80)
    print("[LOOP] Run one live cycle")
    print("time:", datetime.now().isoformat())

    before_latest = get_latest_cycle_record_path()

    result = subprocess.run(
        [sys.executable, "-m", ONE_CYCLE_MODULE],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    print("=" * 80)
    print("[STDOUT]")
    print(result.stdout)

    if result.stderr:
        print("=" * 80)
        print("[STDERR]")
        print(result.stderr)

    after_latest = get_latest_cycle_record_path()

    cycle_record = {}
    cycle_record_path = ""
    cycle_status = "unknown"

    if after_latest is not None and after_latest != before_latest:
        cycle_record_path = str(after_latest)

        try:
            cycle_record = load_json(after_latest)
            cycle_status = str(cycle_record.get("status", "unknown"))
        except Exception as e:
            cycle_status = f"failed_to_read_cycle_record: {e}"

    elif after_latest is not None:
        cycle_record_path = str(after_latest)
        cycle_status = "no_new_cycle_record"
    else:
        cycle_status = "no_cycle_record_found"

    success = result.returncode == 0 and cycle_status == "success"
    cycle_summary = extract_cycle_summary(cycle_record)

    record = {
        "time": datetime.now().isoformat(),
        "success": success,
        "returncode": result.returncode,
        "cycle_status": cycle_status,
        "cycle_record_path": cycle_record_path,
        "cycle_summary": cycle_summary,
        "stdout_tail": result.stdout[-TAIL_CHARS:],
        "stderr_tail": result.stderr[-TAIL_CHARS:] if result.stderr else "",
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_path = LOOP_RECORD_DIR / f"loop_record_{SYMBOL.lower()}_{timestamp}.json"
    save_json(record_path, record)

    print("=" * 80)
    print("[LOOP RECORD SAVED]")
    print(record_path)

    if not success:
        raise RuntimeError(
            "run_one_live_cycle.py failed or cycle status is not success. "
            f"returncode={result.returncode}, cycle_status={cycle_status}, "
            f"cycle_record_path={cycle_record_path}"
        )

    return record


# ============================================================
# 3. 主循环
# ============================================================

def main():
    print("=" * 80)
    print("[LIVE LOOP STARTED]")
    print("project_root:", PROJECT_ROOT)
    print("one_cycle_module:", ONE_CYCLE_MODULE)
    print("use_fixed_daily_schedule:", USE_FIXED_DAILY_SCHEDULE)
    print("timezone:", TIMEZONE)
    print("daily_run_time:", f"{DAILY_RUN_HOUR:02d}:{DAILY_RUN_MINUTE:02d}")
    print("run_on_start:", RUN_ON_START)
    print("loop_interval_seconds:", LOOP_INTERVAL_SECONDS)
    print("run_forever:", RUN_FOREVER)
    print("max_runs:", MAX_RUNS)
    print("max_consecutive_failures:", MAX_CONSECUTIVE_FAILURES)

    run_count = 0
    consecutive_failures = 0

    if USE_FIXED_DAILY_SCHEDULE and not RUN_ON_START:
        sleep_seconds = seconds_until_next_daily_run()
        sleep_with_countdown(sleep_seconds)

    while True:
        run_count += 1

        print("=" * 80)
        print(f"[LOOP] Run #{run_count}")

        try:
            run_one_cycle()
            consecutive_failures = 0

        except KeyboardInterrupt:
            print("=" * 80)
            print("[LOOP STOPPED BY USER]")
            break

        except Exception as e:
            consecutive_failures += 1

            print("=" * 80)
            print("[LOOP ERROR]")
            print("consecutive_failures:", consecutive_failures)
            print(e)
            traceback.print_exc()

            error_record = {
                "time": datetime.now().isoformat(),
                "run_count": run_count,
                "consecutive_failures": consecutive_failures,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_path = LOOP_RECORD_DIR / f"loop_error_{SYMBOL.lower()}_{timestamp}.json"
            save_json(error_path, error_record)

            print("[LOOP ERROR RECORD SAVED]")
            print(error_path)

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print("=" * 80)
                print("[LOOP STOPPED]")
                print("Too many consecutive failures.")
                break

        if not RUN_FOREVER and run_count >= MAX_RUNS:
            print("=" * 80)
            print("[LOOP FINISHED]")
            print(f"Reached MAX_RUNS={MAX_RUNS}")
            break

        if USE_FIXED_DAILY_SCHEDULE:
            sleep_seconds = seconds_until_next_daily_run()
        else:
            sleep_seconds = LOOP_INTERVAL_SECONDS

        sleep_with_countdown(sleep_seconds)


if __name__ == "__main__":
    main()
