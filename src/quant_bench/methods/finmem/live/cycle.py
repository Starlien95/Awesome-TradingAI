import json
import os
import subprocess
import traceback
from pathlib import Path
from datetime import datetime

from quant_bench.methods.finmem.live.env import get_env_bool, load_project_env

load_project_env()

from quant_bench.methods.finmem.live.exchange import okxbot
from quant_bench.methods.finmem.live.historical_news import get_news_for_window_with_cache
from quant_bench.methods.finmem.live.live_dataset import LiveJsonBuilder
from quant_bench.methods.finmem.live.market_data import TradingDataCollector
from quant_bench.methods.finmem.live.news import NewsCollector, NewsProcessor
from quant_bench.methods.finmem.live.decision import InvestorActionAdapter
from quant_bench.runtime.core.atomic_io import atomic_write_json
from quant_bench.trading import (
    DecisionMappingBackend,
    MarketType,
    OrderAction,
    OrderRequest,
    TradingService,
)


# ============================================================
# 1. 配置区
# ============================================================

PROJECT_ROOT = Path(os.getenv("FINMEM_WORKSPACE", "~/.local/share/quant-bench/methods/finmem")).expanduser().resolve()

SYMBOL = os.getenv("FINMEM_SYMBOL", "BTC").upper().strip()
DATA_INST_ID = f"{SYMBOL}-USDT"
TRADE_INST_ID = f"{SYMBOL}-USDT-SWAP"
INST_ID = DATA_INST_ID

BAR = "1D"

# 是否使用当前未确认 K 线
# 实时交易可以 True；稳定测试可以 False
INCLUDE_UNCONFIRMED_KLINE = False

# live json 输出
LIVE_DATA_DIR = PROJECT_ROOT / "live_data"
LIVE_JSON_PATH = LIVE_DATA_DIR / f"{SYMBOL.lower()}_live.json"
NEWS_HISTORY_PATH = LIVE_DATA_DIR / "news_history.csv"

# Investor-Bench 决策输出
LATEST_ACTION_PATH = PROJECT_ROOT / "live_decision" / f"{SYMBOL.lower()}_latest_action.json"
OKX_DECISION_PATH = PROJECT_ROOT / "live_decision" / f"{SYMBOL.lower()}_okx_decision.json"

# 运行记录
RECORD_DIR = PROJECT_ROOT / "live_cycle_records"
STATE_PATH = PROJECT_ROOT / "live_decision" / f"executed_state_{SYMBOL.lower()}.json"

# 独立策略资金池：只关注策略初始资金，不使用 OKX 全账户计算策略指标
STRATEGY_INITIAL_CAPITAL_USDT = 10000.0
STRATEGY_ACCOUNT_STATE_PATH = PROJECT_ROOT / "live_decision" / f"strategy_account_state_{SYMBOL.lower()}.json"
STRATEGY_MIN_TRADE_USDT = 5.0
STRATEGY_FEE_RATE = 0.0

# 为了兼容旧字段名保留这个变量；实际买入金额由 strategy_account_state.cash_usdt 决定
TRADE_AMOUNT_USDT = STRATEGY_INITIAL_CAPITAL_USDT

# 是否执行 OKX 下单
# 阶段B：真实 OKX simulated SWAP 做多/做空，同时本地账本按论文式 position × return 记录。
RUNTIME_MODE = os.getenv("FINMEM_MODE", "paper").lower().strip()
ALLOW_NETWORK = get_env_bool("FINMEM_ALLOW_NETWORK", False)
QUERY_ACCOUNT = get_env_bool("FINMEM_QUERY_ACCOUNT", False)
EXECUTE_ORDER = get_env_bool("FINMEM_EXECUTE_ORDER", False)
VIRTUAL_PAPER_SHORT_MODE = get_env_bool("FINMEM_VIRTUAL_PAPER_SHORT_MODE", False)
ALLOW_SHORT = True
TRADE_MODE = "swap"
MARGIN_MODE = "cross"
LEVERAGE = 1
SWAP_POSITION_MODE = "net"

# 是否使用 OKX 模拟盘
IS_SIMULATED = RUNTIME_MODE != "live"

if RUNTIME_MODE not in {"paper", "demo", "live"}:
    raise ValueError("FINMEM_MODE must be paper, demo, or live")
if (QUERY_ACCOUNT or EXECUTE_ORDER) and not ALLOW_NETWORK:
    raise PermissionError("FinMem account access and orders require FINMEM_ALLOW_NETWORK=1")
if QUERY_ACCOUNT and RUNTIME_MODE not in {"demo", "live"}:
    raise PermissionError("FinMem account reads require demo or live mode")
if EXECUTE_ORDER:
    expected_confirmation = "DEMO_ORDERS" if RUNTIME_MODE == "demo" else "LIVE_ORDERS"
    if RUNTIME_MODE not in {"demo", "live"} or os.getenv("FINMEM_ORDER_CONFIRM") != expected_confirmation:
        raise PermissionError(f"FinMem {RUNTIME_MODE} orders require FINMEM_ORDER_CONFIRM={expected_confirmation}")

# 风控
MAX_BUY_USDT = STRATEGY_INITIAL_CAPITAL_USDT

# 防止同一根 K 线 / 同一日期重复执行
SKIP_IF_ALREADY_EXECUTED = True

# ============================================================
# Live rolling-window 配置
# ============================================================

# 生成 live JSON 时保留：当前日 + 前 LIVE_PREVIOUS_DAYS 天
# look_back_window_size=3 / momentum_window_size=3 时，至少要 >=3；
# live 模拟盘建议 7~10 天更稳。
LIVE_PREVIOUS_DAYS = 10

# 是否抓取 rolling window 内每一天的历史新闻
# True:
#   使用 historical_news_collector.fetch_historical_news_for_window()
#   内部为 list.md 驱动的 RSS/Google News 抓取器。
#   并以 news_mode="by_date" 构造 JSON。
# False:
#   只抓最新新闻，并以 news_mode="latest" 构造 JSON。
LIVE_USE_HISTORICAL_NEWS = True

# 历史新闻窗口最多保留多少条。抓取器会先按来源收集并去重，
# 超过上限时再按日期均衡裁剪；8000 通常不会限制 10 日 live 窗口。
HISTORICAL_NEWS_TARGET_RECORDS = 8000

# 对支持分页的网站最多回溯页数；RSS/Google News 来源本身不一定支持分页。
HISTORICAL_NEWS_MAX_PAGES = 6

# 每个日期至少缓存多少条新闻，低于该数量会尝试补拉。
NEWS_CACHE_MIN_NEWS_PER_DAY = 1

# 如果历史新闻抓取失败，是否退回到 latest news 模式。
FALLBACK_TO_LATEST_NEWS = True

# 是否通过环境变量把 rolling window 起止日期传给 investor_live_test_runner.py。
# 你的 investor_live_test_runner.py 需要读取这些环境变量才会生效：
#   LIVE_JSON_PATH
#   LIVE_TEST_START_DATE
#   LIVE_TEST_END_DATE
#   LIVE_TARGET_DATE
PASS_LIVE_WINDOW_TO_RUNNER = True


# ============================================================
# 2. 工具函数
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    atomic_write_json(path, obj)


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return default
        return float(value)
    except Exception:
        return default


def run_subprocess(module_name, env_extra=None):
    env = os.environ.copy()
    if env_extra:
        env.update({str(k): str(v) for k, v in env_extra.items()})

    print("=" * 80)
    print(f"[RUN MODULE] {module_name}")

    if env_extra:
        print("[SUBPROCESS ENV EXTRA]")
        print(json.dumps(env_extra, ensure_ascii=False, indent=2))

    subprocess.run(
        [os.sys.executable, "-m", module_name],
        cwd=PROJECT_ROOT,
        check=True,
        env=env,
    )


def load_state():
    if not STATE_PATH.exists():
        return {}

    try:
        return load_json(STATE_PATH)
    except Exception:
        return {}


def save_state(state):
    save_json(STATE_PATH, state)

def load_strategy_account_state():
    """
    阶段B：论文式SWAP long/flat/short 策略资金池。

    这个账本只用于模拟盘论文复现，不读取 OKX 全账户，也不真实开空。

    核心逻辑：
      position_sign =  1: long
      position_sign =  0: flat
      position_sign = -1: short

    每个新价格点先按上一期 position_sign 计算：
      equity_t = equity_{t-1} * (1 + position_sign_{t-1} * price_return_t)

    然后再把当前模型 target_position 映射成下一期 position_sign。
    """
    if not STRATEGY_ACCOUNT_STATE_PATH.exists():
        state = {
            "initial_capital_usdt": STRATEGY_INITIAL_CAPITAL_USDT,
            "cash_usdt": STRATEGY_INITIAL_CAPITAL_USDT,
            # 为了兼容已有 logger/dashboard，字段名仍叫 btc_qty；
            # 对 ETH 版本该字段表示 ETH 数量。
            "btc_qty": 0.0,
            "equity_usdt": STRATEGY_INITIAL_CAPITAL_USDT,
            "last_price": 0.0,
            "entry_price": 0.0,
            "position_value_usdt": 0.0,
            "position_state": "flat",
            "position_sign": 0,
            "fee_rate": STRATEGY_FEE_RATE,
            "paper_short_mode": True,
            "updated_at": datetime.now().isoformat(),
            "note": "Local paper-style long/flat/short ledger. No real OKX short orders are sent.",
        }
        save_json(STRATEGY_ACCOUNT_STATE_PATH, state)
        return state

    state = load_json(STRATEGY_ACCOUNT_STATE_PATH)

    # 向后兼容旧 long/flat 账本。
    state.setdefault("initial_capital_usdt", STRATEGY_INITIAL_CAPITAL_USDT)
    state.setdefault("cash_usdt", STRATEGY_INITIAL_CAPITAL_USDT)
    state.setdefault("btc_qty", 0.0)
    state.setdefault("equity_usdt", safe_float(state.get("cash_usdt"), STRATEGY_INITIAL_CAPITAL_USDT))
    state.setdefault("last_price", 0.0)
    state.setdefault("entry_price", 0.0)
    state.setdefault("position_value_usdt", 0.0)
    state.setdefault("position_state", "flat")
    state.setdefault("fee_rate", STRATEGY_FEE_RATE)
    state.setdefault("paper_short_mode", True)

    if "position_sign" not in state:
        old_state = str(state.get("position_state", "flat")).lower()
        if old_state == "long":
            state["position_sign"] = 1
        elif old_state == "short":
            state["position_sign"] = -1
        else:
            state["position_sign"] = 0

    return state
def save_strategy_account_state(state):
    state = dict(state)
    state["updated_at"] = datetime.now().isoformat()
    save_json(STRATEGY_ACCOUNT_STATE_PATH, state)
    return state


def sign_to_state(sign):
    sign = int(sign)
    if sign > 0:
        return "long"
    if sign < 0:
        return "short"
    return "flat"


def target_position_to_sign(target_position):
    target_position = safe_float(target_position, 0.0)
    if target_position > 0:
        return 1
    if target_position < 0 and ALLOW_SHORT:
        return -1
    return 0


def mark_to_market_strategy_account(latest_price):
    """
    按上一期 position_sign 对当前价格做盯市。

    注意：同一个 latest_price 多次调用不会重复计算收益，因为 last_price 已经更新。
    """
    state = load_strategy_account_state()
    latest_price = safe_float(latest_price, 0.0)

    prev_equity = safe_float(state.get("equity_usdt"), STRATEGY_INITIAL_CAPITAL_USDT)
    prev_price = safe_float(state.get("last_price"), 0.0)
    prev_sign = int(safe_float(state.get("position_sign"), 0.0))

    if prev_price > 0 and latest_price > 0 and latest_price != prev_price:
        price_return = latest_price / prev_price - 1.0
        equity_usdt = prev_equity * (1.0 + prev_sign * price_return)
    else:
        price_return = 0.0
        equity_usdt = prev_equity

    # 极端情况下避免负权益继续滚动。
    equity_usdt = max(equity_usdt, 0.0)

    position_state = sign_to_state(prev_sign)

    if prev_sign == 0 or latest_price <= 0 or equity_usdt < STRATEGY_MIN_TRADE_USDT:
        cash_usdt = equity_usdt
        btc_qty = 0.0
        position_value_usdt = 0.0
        if equity_usdt < STRATEGY_MIN_TRADE_USDT:
            prev_sign = 0
            position_state = "flat"
    else:
        # 论文式满仓敞口：abs(exposure) = equity。
        cash_usdt = 0.0
        position_value_usdt = equity_usdt
        btc_qty = equity_usdt / latest_price

    state.update({
        "cash_usdt": cash_usdt,
        "btc_qty": btc_qty,
        "equity_usdt": equity_usdt,
        "last_price": latest_price,
        "last_price_return": price_return,
        "position_value_usdt": position_value_usdt,
        "position_state": position_state,
        "position_sign": prev_sign,
        "paper_short_mode": True,
    })
    return save_strategy_account_state(state)
def get_action_target_position(action):
    """
    从 Investor action 中提取 target position。

    position 语义：
      >0: target long
       0: target flat
      <0: target short
    """
    if not isinstance(action, dict):
        return 0.0

    raw_action = action.get("raw_action", {}) or {}

    if "position" in raw_action:
        return safe_float(raw_action.get("position"), 0.0)

    if "target_position" in action:
        return safe_float(action.get("target_position"), 0.0)

    if "position" in action:
        return safe_float(action.get("position"), 0.0)

    return 0.0


def order_was_successful(execution_result):
    """
    只有 OKX 真正返回 code=0 / sCode=0 时，才认为订单成功。
    skipped / hold / dry-run 不算成功。
    """
    if not isinstance(execution_result, dict):
        return False

    for _, value in execution_result.items():
        if not isinstance(value, dict):
            continue

        if str(value.get("code", "")) == "0":
            data = value.get("data", [])
            if isinstance(data, list) and data:
                return any(str(x.get("sCode", "0")) == "0" for x in data if isinstance(x, dict))
            return True

    return False


# ============================================================
# 3. Step 1：构造 live JSON
# ============================================================

def fetch_news_for_window_or_latest(live_start_date, live_end_date):
    """
    按 live rolling window 读取新闻。

    按 live rolling window 读取新闻。

    优先使用本地缓存：
      live_data/news_history.csv

    逻辑：
      1. 如果 rolling window 每天都有缓存新闻，则不请求 API。
      2. 如果某些日期缺失，则只补拉缺失日期范围。
      3. 返回窗口内新闻，给 live_json_builder 以 news_mode="by_date" 挂载。
    """
    if LIVE_USE_HISTORICAL_NEWS:
        try:
            print("=" * 80)
            print("[NEWS MODE] by_date historical news with local cache")
            news_df, cache_info = get_news_for_window_with_cache(
                start_date=live_start_date,
                end_date=live_end_date,
                history_path=NEWS_HISTORY_PATH,
                target_records=HISTORICAL_NEWS_TARGET_RECORDS,
                max_pages=HISTORICAL_NEWS_MAX_PAGES,
                min_news_per_day=NEWS_CACHE_MIN_NEWS_PER_DAY,
                fetch_if_missing=True,
                verbose=True,
                return_summary=True,
            )
            return news_df, "by_date", cache_info

        except Exception as e:
            print("=" * 80)
            print("[WARNING] Historical news cache/fetch failed.")
            print(e)

            if not FALLBACK_TO_LATEST_NEWS:
                raise

            print("[FALLBACK] Use latest news only.")

    # fallback / latest mode
    print("=" * 80)
    print("[NEWS MODE] latest news only")

    try:
        news_collector = NewsCollector()
        news_df = news_collector.fetch_latest_news(limit=80)

        print("[NEWS]")
        print("rows:", len(news_df) if news_df is not None else 0)

        if news_df is not None and not news_df.empty:
            show_cols = [c for c in ["date", "subject", "title"] if c in news_df.columns]
            if show_cols:
                print(news_df[show_cols].head(10).to_string(index=False))
            else:
                print(news_df.head(10).to_string(index=False))

        return news_df, "latest", {
            "history_path": str(NEWS_HISTORY_PATH),
            "fetched": True,
            "fallback": "latest",
        }

    except Exception as e:
        print("[WARNING] Latest news fetch failed. Continue with empty news.")
        print(e)
        return None, "none", {
            "history_path": str(NEWS_HISTORY_PATH),
            "fetched": False,
            "fallback": "none",
            "error": str(e),
        }



def build_live_json():
    if not ALLOW_NETWORK:
        raise PermissionError("live data/news collection requires FINMEM_ALLOW_NETWORK=1")
    print("=" * 80)
    print("[STEP 1] Build live JSON from OKX kline + window news")

    bot = okxbot(IS_SIMULATED)

    collector = TradingDataCollector(bot, DATA_INST_ID)

    # 只用于构造 JSON，不需要很长历史
    collector.min_history_len = 300

    kline_records = collector.get_price_data(
        bar=BAR,
        limit=max(50, LIVE_PREVIOUS_DAYS + 5),
    )

    print("[KLINE]")
    print("records:", len(kline_records))
    print("last kline:", kline_records[-1])

    news_processor = NewsProcessor(
        max_coin_news=5,
        max_global_news=5,
        max_total_news=10,
    )

    builder = LiveJsonBuilder(
        output_dir=str(LIVE_DATA_DIR),
        news_processor=news_processor,
    )

    # 先用同一个 builder 计算 rolling window 起止日期，
    # 再按这个日期窗口抓取历史新闻。
    kline_df_tmp = builder.normalize_kline_records(
        kline_records,
        include_unconfirmed=INCLUDE_UNCONFIRMED_KLINE,
    )

    kline_df_tmp = builder.select_rolling_window(
        kline_df_tmp,
        previous_days=LIVE_PREVIOUS_DAYS,
    )

    if kline_df_tmp.empty:
        raise ValueError("No kline data after rolling-window selection.")

    live_start_date = kline_df_tmp["date"].iloc[0]
    live_end_date = kline_df_tmp["date"].iloc[-1]

    print("=" * 80)
    print("[LIVE KLINE WINDOW]")
    print("live_start_date:", live_start_date)
    print("live_end_date:  ", live_end_date)
    print("rows:", len(kline_df_tmp))
    print("=" * 80)

    news_df, news_mode, news_cache_info = fetch_news_for_window_or_latest(
        live_start_date=live_start_date,
        live_end_date=live_end_date,
    )

    output_path, result = builder.build_for_symbol(
        inst_id=DATA_INST_ID,
        kline_records=kline_records,
        news_df=news_df,
        max_rows=300,
        output_filename=f"{SYMBOL.lower()}_live.json",
        include_unconfirmed=INCLUDE_UNCONFIRMED_KLINE,

        # 关键：当前日 + 前 LIVE_PREVIOUS_DAYS 天
        previous_days=LIVE_PREVIOUS_DAYS,

        # historical: by_date；fallback: latest/none
        news_mode=news_mode,
    )

    keys = sorted(result.keys())
    if not keys:
        raise ValueError("Live JSON result is empty.")

    live_start_date = keys[0]
    live_end_date = keys[-1]
    latest_date = live_end_date

    # 统计每一天的新闻数量，方便检查 by_date 是否真正生效。
    news_count_by_date = {
        d: len(result[d].get("news", []))
        for d in keys
    }

    print("[LIVE JSON]")
    print("output_path:", output_path)
    print("days:", len(result))
    print("live_start_date:", live_start_date)
    print("live_end_date:", live_end_date)
    print("latest_date:", latest_date)
    print("latest_price:", result[latest_date]["prices"])
    print("latest_news_count:", len(result[latest_date]["news"]))
    print("news_mode:", news_mode)
    print("news_count_by_date:")
    print(json.dumps(news_count_by_date, ensure_ascii=False, indent=2))
    print("news_cache_info:")
    print(json.dumps(news_cache_info, ensure_ascii=False, indent=2))

    return {
        "output_path": output_path,
        "live_start_date": live_start_date,
        "live_end_date": live_end_date,
        "latest_date": latest_date,
        "all_dates": keys,
        "previous_days": LIVE_PREVIOUS_DAYS,
        "news_mode": news_mode,
        "news_count_by_date": news_count_by_date,
        "news_cache_info": news_cache_info,
        "latest_item": result[latest_date],
    }


# ============================================================
# 4. Step 2：运行 Investor-Bench test-only
# ============================================================

def run_investor_decision(live_json_info=None):
    print("=" * 80)
    print("[STEP 2] Run Investor-Bench live test runner")

    env_extra = {}

    if PASS_LIVE_WINDOW_TO_RUNNER and live_json_info:
        env_extra = {
            "LIVE_JSON_PATH": live_json_info["output_path"],
            "LIVE_TEST_START_DATE": live_json_info["live_start_date"],
            "LIVE_TEST_END_DATE": live_json_info["live_end_date"],
            "LIVE_TARGET_DATE": live_json_info["latest_date"],
            "LIVE_SYMBOL": SYMBOL,
            "LIVE_INST_ID": DATA_INST_ID,
        }

    run_subprocess("quant_bench.methods.finmem.live.investor_runner", env_extra=env_extra)

    if not LATEST_ACTION_PATH.exists():
        raise FileNotFoundError(f"Latest action not found: {LATEST_ACTION_PATH}")

    action = load_json(LATEST_ACTION_PATH)

    print("[INVESTOR ACTION]")
    print(json.dumps(action, ensure_ascii=False, indent=2))

    return action


# ============================================================
# 5. Step 3：转换成 OKX decision
# ============================================================

def convert_action_to_okx_decision(action):
    print("=" * 80)
    print("[STEP 3] Convert Investor action to OKX decision")

    adapter = InvestorActionAdapter(
        trade_amount_usdt=TRADE_AMOUNT_USDT,
        allow_short=True,
        trade_mode=TRADE_MODE,
        td_mode=MARGIN_MODE,
        leverage=LEVERAGE,
    )

    okx_decision = adapter.convert_single_action(action)

    save_json(OKX_DECISION_PATH, okx_decision)

    print("[OKX DECISION]")
    print(json.dumps(okx_decision, ensure_ascii=False, indent=2))
    print("saved to:", OKX_DECISION_PATH)

    return okx_decision


# ============================================================
# 6. Step 4：风控检查 + spot long/flat 映射
# ============================================================

def preflight_check(okx_decision):
    print("=" * 80)
    print("[STEP 4] Preflight risk check")

    if not isinstance(okx_decision, dict):
        raise ValueError("okx_decision must be a dict")

    valid_signals = {"buy", "sell", "close", "hold", "reverse_to_long", "reverse_to_short"}

    for symbol, info in okx_decision.items():
        signal = str(info.get("signal", "hold")).lower().strip()
        if signal not in valid_signals:
            raise ValueError(f"{symbol} invalid signal: {signal}")

        if signal == "hold":
            continue

        if str(info.get("trade_mode", "spot")).lower() == "swap":
            inst_id = str(info.get("instId", ""))
            if not inst_id.endswith("-SWAP"):
                raise ValueError(f"{symbol} SWAP decision must use instId ending with -SWAP, got {inst_id}")

            notional = safe_float(info.get("notional_usdt", info.get("quantity", 0.0)), 0.0)
            if notional <= 0:
                raise ValueError(f"{symbol} swap notional must be positive for signal={signal}")
            if signal in {"buy", "sell"} and notional > MAX_BUY_USDT:
                raise ValueError(f"{symbol} swap notional {notional} exceeds MAX_BUY_USDT={MAX_BUY_USDT}")
            if signal in {"reverse_to_long", "reverse_to_short"} and notional > MAX_BUY_USDT * 2.1:
                raise ValueError(f"{symbol} reverse notional {notional} exceeds safety limit")
            continue

        # spot fallback
        if signal == "buy":
            tgt_ccy = info.get("tgtCcy")
            quantity = float(info.get("quantity", 0))
            if tgt_ccy != "quote_ccy":
                raise ValueError(f"{symbol} buy must use tgtCcy='quote_ccy', got {tgt_ccy}")
            if quantity <= 0:
                raise ValueError(f"{symbol} buy quantity must be positive")
            if quantity > MAX_BUY_USDT:
                raise ValueError(f"{symbol} buy quantity {quantity} exceeds MAX_BUY_USDT={MAX_BUY_USDT}")

    print("[OK] Preflight check passed.")

def apply_strategy_account_control(okx_decision, latest_price, investor_action=None):
    """
    阶段B：真实 OKX simulated SWAP 做多/做空。

    决策价格仍来自 DATA_INST_ID=BTC-USDT 的日线；
    真实执行使用 TRADE_INST_ID=BTC-USDT-SWAP。

    本地策略账本仍按论文式 position × return 盯市，
    但只有当 OKX simulated 订单成功后才切换 position_sign。
    """
    print("=" * 80)
    print("[STAGE-B SWAP STRATEGY ACCOUNT CONTROL]")

    state = mark_to_market_strategy_account(latest_price)
    latest_price = safe_float(latest_price, 0.0)

    equity_usdt = safe_float(state.get("equity_usdt"), STRATEGY_INITIAL_CAPITAL_USDT)
    current_sign = int(safe_float(state.get("position_sign"), 0.0))
    current_state = sign_to_state(current_sign)
    asset_qty = safe_float(state.get("btc_qty"), 0.0)
    cash_usdt = safe_float(state.get("cash_usdt"), equity_usdt)
    position_value_usdt = safe_float(state.get("position_value_usdt"), 0.0)

    target_position = get_action_target_position(investor_action or {})
    target_sign = target_position_to_sign(target_position)
    target_state = sign_to_state(target_sign)

    print("[STRATEGY ACCOUNT BEFORE]")
    print(json.dumps(state, ensure_ascii=False, indent=2))
    print(f"[CURRENT SIGN] {current_sign} ({current_state})")
    print(f"[TARGET POSITION] {target_position} -> sign {target_sign} ({target_state})")
    print(f"[TRADE INST] {TRADE_INST_ID}, mode={TRADE_MODE}, margin={MARGIN_MODE}, leverage={LEVERAGE}")

    adjusted_decision = {}

    for symbol, info in okx_decision.items():
        original_signal = str(info.get("signal", "hold")).lower().strip()
        base_info = {
            "trade_mode": TRADE_MODE,
            "instId": TRADE_INST_ID,
            "tdMode": MARGIN_MODE,
            "leverage": str(LEVERAGE),
            "position_mode": SWAP_POSITION_MODE,
            "coin": symbol,
            "target_position": target_position,
            "target_sign": target_sign,
            "target_state": target_state,
            "current_sign": current_sign,
            "current_state": current_state,
            "original_signal": original_signal,
            "strategy_cash_usdt": cash_usdt,
            "strategy_btc_qty": asset_qty,
            "strategy_equity_usdt": equity_usdt,
            "strategy_position_value_usdt": position_value_usdt,
            "virtual_order": False,
            "paper_short_mode": False,
            "stage_b_swap": True,
            "price": latest_price,
        }

        # 同方向：不重复开仓。
        if target_sign == current_sign:
            adjusted_info = {
                **base_info,
                "signal": "hold",
                "quantity": "0",
                "notional_usdt": "0",
                "risk_reason": f"Already in target SWAP state: {target_state}.",
            }

        # 目标多头。
        elif target_sign > 0:
            if current_sign < 0:
                signal = "reverse_to_long"
                notional = equity_usdt * 2.0
                reason = "Reverse SWAP short -> long: close short then open long."
            else:
                signal = "buy"
                notional = equity_usdt
                reason = "Open SWAP long with strategy equity notional."

            adjusted_info = {
                **base_info,
                "signal": signal,
                "quantity": str(round(notional, 8)),
                "notional_usdt": str(round(notional, 8)),
                "open_notional_usdt": str(round(equity_usdt, 8)),
                "risk_reason": reason,
            }

        # 目标空头。
        elif target_sign < 0:
            if current_sign > 0:
                signal = "reverse_to_short"
                notional = equity_usdt * 2.0
                reason = "Reverse SWAP long -> short: close long then open short."
            else:
                signal = "sell"
                notional = equity_usdt
                reason = "Open SWAP short with strategy equity notional."

            adjusted_info = {
                **base_info,
                "signal": signal,
                "quantity": str(round(notional, 8)),
                "notional_usdt": str(round(notional, 8)),
                "open_notional_usdt": str(round(equity_usdt, 8)),
                "risk_reason": reason,
            }

        # 目标空仓。
        else:
            close_notional = position_value_usdt if position_value_usdt > 0 else equity_usdt
            adjusted_info = {
                **base_info,
                "signal": "close",
                "quantity": str(round(close_notional, 8)),
                "notional_usdt": str(round(close_notional, 8)),
                "risk_reason": "Close SWAP position and switch to flat.",
            }

        adjusted_decision[symbol] = adjusted_info

    print("[ADJUSTED SWAP DECISION]")
    print(json.dumps(adjusted_decision, ensure_ascii=False, indent=2))

    return adjusted_decision, state
def update_strategy_account_after_execution(okx_decision, latest_price, execution_result):
    """
    阶段BSWAP执行后，更新本地策略账本的 position_sign。

    收益已经在 mark_to_market_strategy_account(latest_price) 中按上一期仓位计算。
    这里仅把当前目标状态设置为下一期持仓。
    """
    state = mark_to_market_strategy_account(latest_price)
    latest_price = safe_float(latest_price, 0.0)

    if not order_was_successful(execution_result):
        return state

    equity_usdt = safe_float(state.get("equity_usdt"), STRATEGY_INITIAL_CAPITAL_USDT)

    target_sign = int(safe_float(state.get("position_sign"), 0.0))
    entry_signal = "hold"

    for symbol, info in okx_decision.items():
        target_sign = int(safe_float(info.get("target_sign"), target_position_to_sign(info.get("target_position", 0.0))))
        entry_signal = str(info.get("signal", "hold")).lower().strip()
        break

    position_state = sign_to_state(target_sign)

    if target_sign == 0 or latest_price <= 0 or equity_usdt < STRATEGY_MIN_TRADE_USDT:
        cash_usdt = equity_usdt
        btc_qty = 0.0
        position_value_usdt = 0.0
        target_sign = 0
        position_state = "flat"
        entry_price = 0.0
    else:
        cash_usdt = 0.0
        position_value_usdt = equity_usdt
        btc_qty = equity_usdt / latest_price
        entry_price = latest_price

    state.update({
        "cash_usdt": cash_usdt,
        "btc_qty": btc_qty,
        "equity_usdt": equity_usdt,
        "last_price": latest_price,
        "entry_price": entry_price,
        "position_value_usdt": position_value_usdt,
        "position_state": position_state,
        "position_sign": target_sign,
        "last_virtual_signal": entry_signal,
        "paper_short_mode": True,
        "fee_rate": STRATEGY_FEE_RATE,
        "last_execution_time": datetime.now().isoformat(),
    })

    return save_strategy_account_state(state)
def build_execution_key(action, okx_decision):
    """
    用于防止同一个 action 被重复执行。
    使用最终 OKX signal，而不是只用 investor signal。
    """
    symbol = action.get("symbol", SYMBOL)
    date = action.get("date", "unknown")

    final_signal = "hold"
    try:
        info = okx_decision.get(symbol, {})
        final_signal = str(info.get("signal", "hold")).lower().strip()
    except Exception:
        final_signal = action.get("signal", "hold")

    return f"{symbol}:{date}:{final_signal}"


def check_duplicate_execution(action, okx_decision):
    if not SKIP_IF_ALREADY_EXECUTED:
        return False

    state = load_state()
    execution_key = build_execution_key(action, okx_decision)

    if execution_key in state.get("executed_keys", []):
        print("=" * 80)
        print("[DUPLICATE SKIP]")
        print(f"Already executed key: {execution_key}")
        return True

    return False


def mark_executed(action, okx_decision, execution_result):
    state = load_state()

    if "executed_keys" not in state:
        state["executed_keys"] = []

    if "records" not in state:
        state["records"] = []

    execution_key = build_execution_key(action, okx_decision)

    if execution_key not in state["executed_keys"]:
        state["executed_keys"].append(execution_key)

    state["records"].append({
        "execution_key": execution_key,
        "time": datetime.now().isoformat(),
        "action": action,
        "okx_decision": okx_decision,
        "execution_result": execution_result,
    })

    save_state(state)


# ============================================================
# 7. Step 5：执行或 dry-run
# ============================================================

def virtual_execute_strategy_order(okx_decision):
    """
    阶段BSWAP执行：不向 OKX 发送订单，只返回类似 OKX 成功格式，
    让 logger 可以记录 volume / order_success。
    """
    print("=" * 80)
    print("[STEP 5] Virtual paper-short execution; no OKX order is sent")

    results = {}
    now_id = datetime.now().strftime("VIRTUAL_%Y%m%d_%H%M%S_%f")

    for symbol, info in okx_decision.items():
        signal = str(info.get("signal", "hold")).lower().strip()
        quantity = info.get("quantity", "0")
        target_state = info.get("target_state", "")
        msg = f"Virtual {symbol}: signal={signal}, target_state={target_state}, quantity={quantity}"
        print(msg)

        results[symbol] = {
            "code": "0",
            "msg": "virtual paper-short execution success",
            "data": [{
                "ordId": f"{now_id}_{symbol}",
                "sCode": "0",
                "sMsg": msg,
            }],
            "virtual_order": True,
            "paper_short_mode": True,
            "decision": info,
        }

    return results


def get_real_okx_snapshot(label="snapshot"):
    """
    Query real OKX simulated-account state for audit.
    This does not affect the local strategy ledger.
    """
    if not QUERY_ACCOUNT:
        return {
            "label": label,
            "account_query": "disabled",
            "time": datetime.now().isoformat(),
        }
    try:
        bot = okxbot(IS_SIMULATED)
        if hasattr(bot, "get_account_snapshot"):
            snap = bot.get_account_snapshot(instId=TRADE_INST_ID, ccy="USDT")
        else:
            snap = {
                "balance": bot.get_balance("USDT"),
                "positions": bot.get_position(inst_type="SWAP"),
            }
        snap["label"] = label
        snap["trade_inst_id"] = TRADE_INST_ID
        snap["is_simulated"] = IS_SIMULATED
        snap["time"] = datetime.now().isoformat()
        return snap
    except Exception as e:
        return {
            "label": label,
            "trade_inst_id": TRADE_INST_ID,
            "is_simulated": IS_SIMULATED,
            "time": datetime.now().isoformat(),
            "error": repr(e),
        }


def extract_execution_audit_summary(execution_result):
    """
    Extract compact execution audit fields from okx_trade.py enriched result.
    """
    summary = {}
    if not isinstance(execution_result, dict):
        return summary

    for symbol, res in execution_result.items():
        if not isinstance(res, dict):
            continue
        audit = res.get("execution_audit", {}) or {}
        pos = audit.get("actual_position", {}) or {}
        post = audit.get("post_snapshot", {}) or {}
        pre = audit.get("pre_snapshot", {}) or {}
        summary[symbol] = {
            "instId": res.get("instId"),
            "trade_mode": res.get("trade_mode"),
            "signal": res.get("signal"),
            "order_code": res.get("code"),
            "order_msg": res.get("msg"),
            "actual_pos": audit.get("actual_pos"),
            "actual_avg_px": audit.get("actual_avg_px"),
            "actual_mark_px": audit.get("actual_mark_px"),
            "actual_notional_usd": audit.get("actual_notional_usd"),
            "actual_upl": audit.get("actual_upl"),
            "actual_realized_pnl": audit.get("actual_realized_pnl"),
            "actual_fee": audit.get("actual_fee"),
            "position_raw": pos,
            "pre_total_eq": ((pre.get("balance") or {}).get("total_eq")),
            "post_total_eq": ((post.get("balance") or {}).get("total_eq")),
            "pre_available_usdt": ((pre.get("balance") or {}).get("avail_bal")),
            "post_available_usdt": ((post.get("balance") or {}).get("avail_bal")),
        }
    return summary


def execute_okx_order(okx_decision):
    print("=" * 80)
    print("[STEP 5] Execute OKX simulated order")

    if not isinstance(okx_decision, dict) or len(okx_decision) != 1:
        raise ValueError("FinMem cycle requires exactly one symbol decision")

    symbol, trade_info = next(iter(okx_decision.items()))
    if not isinstance(trade_info, dict):
        raise ValueError("FinMem trade decision must be a mapping")

    signal = str(trade_info.get("signal", "hold")).lower().strip()
    action = {
        "buy": OrderAction.BUY,
        "open_long": OrderAction.BUY,
        "long": OrderAction.BUY,
        "reverse_to_long": OrderAction.BUY,
        "sell": OrderAction.SELL,
        "open_short": OrderAction.SELL,
        "short": OrderAction.SELL,
        "reverse_to_short": OrderAction.SELL,
        "close": OrderAction.CLOSE,
        "close_long": OrderAction.CLOSE,
        "close_short": OrderAction.CLOSE,
    }.get(signal, OrderAction.HOLD)
    raw_inst_id = trade_info.get("instId") or trade_info.get("inst_id")
    trade_mode = str(trade_info.get("trade_mode", TRADE_MODE)).lower().strip()
    is_swap = trade_mode == "swap" or str(raw_inst_id or "").upper().endswith("-SWAP")
    instrument_id = str(
        raw_inst_id or (f"{symbol}-USDT-SWAP" if is_swap else f"{symbol}-USDT")
    ).upper()
    reference_price = safe_float(trade_info.get("price"), 0.0) or None
    notional_usdt = safe_float(trade_info.get("notional_usdt"), 0.0) or None

    bot = okxbot(IS_SIMULATED)
    backend = DecisionMappingBackend(
        bot,
        backend_id="finmem-okx",
        execution_mode=RUNTIME_MODE,
    )
    request = OrderRequest(
        strategy_id="finmem",
        instrument_id=instrument_id,
        market_type=MarketType.SWAP if is_swap else MarketType.SPOT,
        action=action,
        notional_usdt=notional_usdt if action in {OrderAction.BUY, OrderAction.SELL} else None,
        allow_backend_sizing=(
            action in {OrderAction.BUY, OrderAction.SELL} and notional_usdt is None
        ),
        reference_price=reference_price,
        margin_mode=trade_info.get("tdMode") or trade_info.get("td_mode"),
        position_side=trade_info.get("posSide") or trade_info.get("pos_side"),
        reduce_only=bool(trade_info.get("reduceOnly", trade_info.get("reduce_only", False))),
        leverage=int(safe_float(trade_info.get("leverage"), 0.0)) or None,
        reason=str(trade_info.get("risk_reason", "FinMem execution decision")),
        metadata={"legacy_decision": okx_decision},
    )
    cycle = TradingService(backend).execute(request)
    result = cycle.execution.legacy_payload
    if not result:
        result = {
            symbol: {
                "status": "error",
                "instId": instrument_id,
                "signal": signal,
                "error": cycle.execution.error_message or "common trading service failed",
            }
        }

    print("[EXECUTION RESULT]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("[COMMON EXECUTION REPORT]")
    print(json.dumps(cycle.model_dump(mode="json"), ensure_ascii=False, indent=2))

    return result


# ============================================================
# 8. 记录完整单轮流程
# ============================================================

def save_cycle_record(record):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RECORD_DIR / f"cycle_{SYMBOL.lower()}_{timestamp}.json"

    save_json(path, record)

    print("=" * 80)
    print("[CYCLE RECORD SAVED]")
    print(path)

    return path


# ============================================================
# 9. 主流程
# ============================================================

def main():
    if not ALLOW_NETWORK:
        raise PermissionError("the live FinMem cycle requires FINMEM_ALLOW_NETWORK=1")
    cycle_start = datetime.now().isoformat()

    record = {
        "cycle_start": cycle_start,
        "symbol": SYMBOL,
        "inst_id": INST_ID,
        "data_inst_id": DATA_INST_ID,
        "trade_inst_id": TRADE_INST_ID,
        "trade_mode": TRADE_MODE,
        "margin_mode": MARGIN_MODE,
        "leverage": LEVERAGE,
        "bar": BAR,
        "include_unconfirmed_kline": INCLUDE_UNCONFIRMED_KLINE,
        "execute_order": EXECUTE_ORDER,
        "runtime_mode": RUNTIME_MODE,
        "network_enabled": ALLOW_NETWORK,
        "account_query_enabled": QUERY_ACCOUNT,
        "virtual_paper_short_mode": VIRTUAL_PAPER_SHORT_MODE,
        "allow_short": ALLOW_SHORT,
        "is_simulated": IS_SIMULATED,
        "trade_amount_usdt": TRADE_AMOUNT_USDT,
        "strategy_initial_capital_usdt": STRATEGY_INITIAL_CAPITAL_USDT,
        "strategy_account_state_path": str(STRATEGY_ACCOUNT_STATE_PATH),
        "live_previous_days": LIVE_PREVIOUS_DAYS,
        "live_use_historical_news": LIVE_USE_HISTORICAL_NEWS,
        "historical_news_target_records": HISTORICAL_NEWS_TARGET_RECORDS,
        "historical_news_max_pages": HISTORICAL_NEWS_MAX_PAGES,
        "news_history_path": str(NEWS_HISTORY_PATH),
        "news_cache_min_news_per_day": NEWS_CACHE_MIN_NEWS_PER_DAY,
    }

    try:
        live_json_info = build_live_json()
        record["live_json_info"] = live_json_info

        latest_price = live_json_info["latest_item"]["prices"]

        record["real_okx_snapshot_before_cycle"] = get_real_okx_snapshot(label="before_cycle")

        strategy_before = mark_to_market_strategy_account(latest_price)
        record["strategy_account_before"] = strategy_before

        investor_action = run_investor_decision(live_json_info=live_json_info)
        record["investor_action"] = investor_action

        okx_decision = convert_action_to_okx_decision(investor_action)

        okx_decision, strategy_before_control = apply_strategy_account_control(
            okx_decision,
            latest_price=latest_price,
            investor_action=investor_action,
        )

        record["strategy_account_before_control"] = strategy_before_control
        record["okx_decision"] = okx_decision

        preflight_check(okx_decision)

        final_infos = list(okx_decision.values())
        final_signal = str(final_infos[0].get("signal", "hold")).lower().strip()
        risk_reason = final_infos[0].get("risk_reason", "")

        if final_signal == "hold":
            strategy_after = mark_to_market_strategy_account(latest_price)
            execution_result = {
                "status": "skipped_by_hold_or_risk",
                "reason": risk_reason or "final decision is hold",
                "okx_decision": okx_decision,
            }

        elif check_duplicate_execution(investor_action, okx_decision):
            strategy_after = mark_to_market_strategy_account(latest_price)
            execution_result = {
                "status": "skipped_duplicate",
                "reason": "same symbol/date/final_signal already executed",
                "okx_decision": okx_decision,
            }

        else:
            if EXECUTE_ORDER and not VIRTUAL_PAPER_SHORT_MODE:
                execution_result = execute_okx_order(okx_decision)

                strategy_after = update_strategy_account_after_execution(
                    okx_decision=okx_decision,
                    latest_price=latest_price,
                    execution_result=execution_result,
                )

                if order_was_successful(execution_result):
                    mark_executed(investor_action, okx_decision, execution_result)
                else:
                    print("[WARNING] Order was not successful; not marking executed.")

            else:
                execution_result = virtual_execute_strategy_order(okx_decision)
                strategy_after = update_strategy_account_after_execution(
                    okx_decision=okx_decision,
                    latest_price=latest_price,
                    execution_result=execution_result,
                )

                if order_was_successful(execution_result):
                    mark_executed(investor_action, okx_decision, execution_result)

        record["strategy_account_state"] = strategy_after
        record["execution_result"] = execution_result
        record["execution_audit_summary"] = extract_execution_audit_summary(execution_result)
        record["real_okx_snapshot_after_cycle"] = get_real_okx_snapshot(label="after_cycle")
        record["status"] = "success"

    except Exception as e:
        record["status"] = "failed"
        record["error"] = str(e)
        record["traceback"] = traceback.format_exc()

        print("=" * 80)
        print("[ERROR]")
        traceback.print_exc()

    finally:
        record["cycle_end"] = datetime.now().isoformat()
        save_cycle_record(record)

        try:
            from quant_bench.methods.finmem.live.dashboard_log import append_cycle_record
            append_cycle_record(record)
        except Exception as e:
            print("[WARNING] failed to append dashboard log:", e)

    print("=" * 80)
    print("[DONE]")
    print("cycle status:", record["status"])

if __name__ == "__main__":
    main()
