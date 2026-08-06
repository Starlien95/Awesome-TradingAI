# finmem_dashboard_logger_eth.py
"""Dual-ledger dashboard logger for a configured FinMem symbol.

It converts FinMem cycle records into dashboard-compatible logs and
maintains two ledgers:

1) Paper Benchmark Ledger
   - Uses the InvestorBench target position and reference price only.
   - Ignores OKX execution and risk control.

2) Strategy Account Ledger
   - Reads a symbol-scoped local strategy account state.
   - Tracks only the isolated strategy capital pool.
   - Keeps okx_* and btc_* column names for dashboard compatibility.
"""

from __future__ import annotations

import glob
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from quant_bench.runtime.core.atomic_io import atomic_write_csv, atomic_write_json

try:
    from quant_bench.methods.finmem.live.env import load_project_env
    load_project_env()
except Exception:
    pass


# ============================================================
# 1. Config
# ============================================================

PROJECT_ROOT = Path(os.getenv("FINMEM_WORKSPACE", "~/.local/share/quant-bench/methods/finmem")).expanduser().resolve()

TIMEFRAME = "1d"
SYMBOL = os.getenv("FINMEM_SYMBOL", "BTC").upper().strip()
MODEL_NAME = f"finmem_{SYMBOL.lower()}"

LOG_DIR = PROJECT_ROOT / "timeframes" / TIMEFRAME / "logs" / MODEL_NAME

METRICS_PATH = LOG_DIR / f"{MODEL_NAME}_metrics.csv"
TRADES_PATH = LOG_DIR / f"{MODEL_NAME}_trades.csv"
SIGNALS_PATH = LOG_DIR / f"{MODEL_NAME}_signals.csv"
VOLUME_PATH = LOG_DIR / f"{MODEL_NAME}_volume.csv"
METADATA_PATH = LOG_DIR / f"{MODEL_NAME}_metadata.json"

CYCLE_RECORD_DIR = PROJECT_ROOT / "live_cycle_records"

PAPER_STATE_PATH = PROJECT_ROOT / "live_decision" / f"paper_benchmark_state_{SYMBOL.lower()}.json"
OKX_STATE_PATH = PROJECT_ROOT / "live_decision" / f"okx_account_state_{SYMBOL.lower()}.json"
STRATEGY_ACCOUNT_STATE_PATH = PROJECT_ROOT / "live_decision" / f"strategy_account_state_{SYMBOL.lower()}.json"

PAPER_INITIAL_CAPITAL = 1.0
STRATEGY_INITIAL_CAPITAL_USDT = 10000.0
PERIODS_PER_YEAR = 365

PAPER_ALLOW_SHORT = True
IS_SIMULATED = os.getenv("FINMEM_MODE", "paper").lower().strip() != "live"


# ============================================================
# 2. Basic IO
# ============================================================

def load_json(path: Path) -> Dict[str, Any]:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    atomic_write_json(path, obj)


def load_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if path.exists():
        old_df = pd.read_csv(path)
        out = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out = new_df
    atomic_write_csv(out, path)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            v = value.strip()
            if not v or v.upper() == "AUTO":
                return default
            value = v
        x = float(value)
        if math.isnan(x):
            return default
        return x
    except Exception:
        return default


def infer_cycle_time(record: Dict[str, Any]) -> Tuple[str, int]:
    raw_time = record.get("cycle_end") or record.get("cycle_start") or datetime.now().isoformat()
    dt = pd.to_datetime(raw_time, errors="coerce")
    if pd.isna(dt):
        dt = pd.Timestamp.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S"), int(dt.timestamp())


def get_latest_cycle_record_path() -> Optional[Path]:
    candidates = sorted(glob.glob(str(CYCLE_RECORD_DIR / f"cycle_{SYMBOL.lower()}_*.json")))
    if not candidates:
        return None
    return Path(candidates[-1])


def already_logged(cycle_start: str) -> bool:
    trades = load_csv(TRADES_PATH)
    if trades.empty or "cycle_start" not in trades.columns:
        return False
    return str(cycle_start) in set(trades["cycle_start"].astype(str))


def json_dumps_compact(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return ""



def nested_get(obj: Dict[str, Any], path, default=None):
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def extract_snapshot_balance(snapshot: Dict[str, Any]) -> Dict[str, float]:
    bal = (snapshot or {}).get("balance", {}) or {}
    return {
        "total_eq": safe_float(bal.get("total_eq"), 0.0),
        "available_usdt": safe_float(bal.get("avail_bal"), 0.0),
        "frozen_usdt": safe_float(bal.get("frozen_bal"), 0.0),
        "cash_usdt": safe_float(bal.get("cash_bal"), 0.0),
    }


def compute_adverse_slippage_bps(final_signal: str, decision_price: float, actual_price: float) -> float:
    """
    Positive value = worse than decision/reference price.
    buy/open_long: worse if actual_price > decision_price
    sell/open_short: worse if actual_price < decision_price
    close: not classified here, returns 0 unless extended with close side.
    """
    decision_price = safe_float(decision_price, 0.0)
    actual_price = safe_float(actual_price, 0.0)
    if decision_price <= 0 or actual_price <= 0:
        return 0.0

    sig = str(final_signal).lower().strip()
    if sig in {"buy", "open_long", "reverse_to_long"}:
        return (actual_price / decision_price - 1.0) * 10000.0
    if sig in {"sell", "open_short", "reverse_to_short"}:
        return (decision_price / actual_price - 1.0) * 10000.0
    return 0.0


# ============================================================
# 3. Parse cycle record
# ============================================================

def position_to_position_signal(position: float) -> str:
    position = safe_float(position, 0.0)
    if position > 0:
        return "long"
    if position < 0:
        return "short"
    return "flat"


def normalize_trade_side(final_signal: str) -> str:
    signal = str(final_signal).lower().strip()
    if signal in {"buy", "open_long", "reverse_to_long"}:
        return "buy"
    if signal in {"sell", "open_short", "reverse_to_short"}:
        return "sell"
    if signal in {"close", "close_long", "close_short"}:
        return "close"
    return "hold"


def parse_cycle_record(record: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(record.get("symbol", SYMBOL)).upper()

    live_info = record.get("live_json_info", {}) or {}
    live_item = live_info.get("latest_item", {}) or {}
    price = safe_float(live_item.get("prices"), 0.0)

    investor_action = record.get("investor_action", {}) or {}
    raw_action = investor_action.get("raw_action", {}) or {}

    investor_signal = str(investor_action.get("signal", "hold")).lower().strip()
    investor_position = safe_float(
        investor_action.get(
            "target_position",
            raw_action.get("position", investor_action.get("position", 0.0)),
        ),
        0.0,
    )
    position_signal = investor_action.get("position_signal") or position_to_position_signal(investor_position)
    action_date = investor_action.get("target_date") or investor_action.get("date") or live_info.get("latest_date")

    okx_decision = record.get("okx_decision", {}) or {}
    symbol_decision = okx_decision.get(symbol, {}) if isinstance(okx_decision, dict) else {}
    final_signal = str(symbol_decision.get("signal", "hold")).lower().strip()
    quantity = symbol_decision.get("quantity", "0")
    notional_usdt = safe_float(symbol_decision.get("notional_usdt", quantity), 0.0)
    open_notional_usdt = safe_float(symbol_decision.get("open_notional_usdt", notional_usdt), 0.0)
    trade_mode = str(symbol_decision.get("trade_mode", "spot")).lower().strip()
    trade_inst_id = symbol_decision.get("instId", "")
    target_sign = safe_float(symbol_decision.get("target_sign", 0.0), 0.0)
    current_sign = safe_float(symbol_decision.get("current_sign", 0.0), 0.0)
    target_state = symbol_decision.get("target_state", "")
    current_state = symbol_decision.get("current_state", "")
    tgt_ccy = symbol_decision.get("tgtCcy", "")
    risk_reason = symbol_decision.get("risk_reason", "")
    original_signal = str(symbol_decision.get("original_signal", investor_signal)).lower().strip()
    decision_target_position = safe_float(symbol_decision.get("target_position", investor_position), investor_position)

    execution_result = record.get("execution_result", {}) or {}
    execution_status = execution_result.get("status", "") if isinstance(execution_result, dict) else ""

    order_id = ""
    order_code = ""
    order_success = False
    dry_run = False

    if isinstance(execution_result, dict) and symbol in execution_result:
        symbol_result = execution_result.get(symbol, {}) or {}
        order_code = str(symbol_result.get("code", ""))
        order_success = order_code == "0"
        dry_run = bool(symbol_result.get("dry_run", False))
        data = symbol_result.get("data", [])
        if isinstance(data, list) and data:
            order_id = str(data[0].get("ordId", ""))
    elif isinstance(execution_result, dict):
        execution_status = str(execution_result.get("status", execution_status))
        dry_run = bool(execution_result.get("dry_run", False))

    # Real OKX simulated-account execution audit.
    symbol_result = execution_result.get(symbol, {}) if isinstance(execution_result, dict) else {}
    execution_audit = symbol_result.get("execution_audit", {}) if isinstance(symbol_result, dict) else {}
    execution_summary = (record.get("execution_audit_summary", {}) or {}).get(symbol, {}) or {}

    before_snapshot = record.get("real_okx_snapshot_before_cycle", {}) or execution_audit.get("pre_snapshot", {}) or {}
    after_snapshot = record.get("real_okx_snapshot_after_cycle", {}) or execution_audit.get("post_snapshot", {}) or {}
    before_bal = extract_snapshot_balance(before_snapshot)
    after_bal = extract_snapshot_balance(after_snapshot)

    actual_position = execution_audit.get("actual_position", {}) or execution_summary.get("position_raw", {}) or {}
    actual_pos = safe_float(execution_audit.get("actual_pos", execution_summary.get("actual_pos", actual_position.get("pos"))), 0.0)
    actual_avg_px = safe_float(execution_audit.get("actual_avg_px", execution_summary.get("actual_avg_px", actual_position.get("avgPx"))), 0.0)
    actual_mark_px = safe_float(execution_audit.get("actual_mark_px", execution_summary.get("actual_mark_px", actual_position.get("markPx"))), 0.0)
    actual_notional_usd = safe_float(execution_audit.get("actual_notional_usd", execution_summary.get("actual_notional_usd", actual_position.get("notionalUsd"))), 0.0)
    actual_upl = safe_float(execution_audit.get("actual_upl", execution_summary.get("actual_upl", actual_position.get("upl"))), 0.0)
    actual_realized_pnl = safe_float(execution_audit.get("actual_realized_pnl", execution_summary.get("actual_realized_pnl", actual_position.get("realizedPnl"))), 0.0)
    actual_fee = safe_float(execution_audit.get("actual_fee", execution_summary.get("actual_fee", actual_position.get("fee"))), 0.0)
    actual_pos_side = str(actual_position.get("posSide", "")) if isinstance(actual_position, dict) else ""

    slippage_bps = compute_adverse_slippage_bps(final_signal, price, actual_avg_px)
    actual_notional_error_usdt = actual_notional_usd - notional_usdt if notional_usdt else 0.0

    trade_side = normalize_trade_side(final_signal)

    live_news_count_by_date = live_info.get("news_count_by_date", {}) or {}
    latest_news_count = len(live_item.get("news", []) or [])

    strategy_state = record.get("strategy_account_state") or {}
    if not strategy_state and STRATEGY_ACCOUNT_STATE_PATH.exists():
        try:
            strategy_state = load_json(STRATEGY_ACCOUNT_STATE_PATH)
        except Exception:
            strategy_state = {}

    return {
        "symbol": symbol,
        "price": price,
        "action_date": action_date,
        "investor_signal": investor_signal,
        "investor_position": investor_position,
        "target_position": investor_position,
        "position_signal": position_signal,
        "decision_target_position": decision_target_position,
        "original_signal": original_signal,
        "final_signal": final_signal,
        "trade_side": trade_side,
        "quantity": quantity,
        "notional_usdt": notional_usdt,
        "open_notional_usdt": open_notional_usdt,
        "trade_mode": trade_mode,
        "trade_inst_id": trade_inst_id,
        "target_sign": target_sign,
        "current_sign": current_sign,
        "target_state": target_state,
        "current_state": current_state,
        "tgt_ccy": tgt_ccy,
        "risk_reason": risk_reason,
        "execution_status": execution_status,
        "order_code": order_code,
        "order_id": order_id,
        "order_success": order_success,
        "dry_run": dry_run,
        "execute_order": bool(record.get("execute_order", False)),
        "is_simulated": bool(record.get("is_simulated", True)),

        # Strategy account state. Key btc_qty is kept for compatibility; in ETH logger it means ETH qty.
        "strategy_initial_capital_usdt": safe_float(
            strategy_state.get("initial_capital_usdt"),
            safe_float(record.get("strategy_initial_capital_usdt"), STRATEGY_INITIAL_CAPITAL_USDT),
        ),
        "strategy_cash_usdt": safe_float(strategy_state.get("cash_usdt"), STRATEGY_INITIAL_CAPITAL_USDT),
        "strategy_btc_qty": safe_float(strategy_state.get("btc_qty"), 0.0),
        "strategy_equity_usdt": safe_float(strategy_state.get("equity_usdt"), STRATEGY_INITIAL_CAPITAL_USDT),
        "strategy_position_value_usdt": safe_float(strategy_state.get("position_value_usdt"), 0.0),
        "strategy_position_state": strategy_state.get("position_state", "flat"),
        "strategy_position_sign": safe_float(strategy_state.get("position_sign"), 0.0),
        "stage_b_swap": bool(record.get("stage_b_swap", False) or strategy_state.get("stage_b_swap", False) or trade_mode == "swap"),
        "strategy_last_price": safe_float(strategy_state.get("last_price"), price),

        # Real OKX simulated-account fields. These are separate from local strategy ledger.
        "real_okx_total_eq_before": before_bal["total_eq"],
        "real_okx_total_eq_after": after_bal["total_eq"],
        "real_okx_available_usdt_before": before_bal["available_usdt"],
        "real_okx_available_usdt_after": after_bal["available_usdt"],
        "real_okx_cash_usdt_before": before_bal["cash_usdt"],
        "real_okx_cash_usdt_after": after_bal["cash_usdt"],
        "real_okx_pos": actual_pos,
        "real_okx_pos_side": actual_pos_side,
        "real_okx_avg_px": actual_avg_px,
        "real_okx_mark_px": actual_mark_px,
        "real_okx_notional_usd": actual_notional_usd,
        "real_okx_upl": actual_upl,
        "real_okx_realized_pnl": actual_realized_pnl,
        "real_okx_fee": actual_fee,
        "execution_slippage_bps": slippage_bps,
        "actual_notional_error_usdt": actual_notional_error_usdt,
        "real_okx_snapshot_before_json": json_dumps_compact(before_snapshot),
        "real_okx_snapshot_after_json": json_dumps_compact(after_snapshot),

        "live_start_date": live_info.get("live_start_date", ""),
        "live_end_date": live_info.get("live_end_date", ""),
        "latest_date": live_info.get("latest_date", ""),
        "live_previous_days": record.get("live_previous_days", live_info.get("previous_days", "")),
        "live_use_historical_news": bool(record.get("live_use_historical_news", False)),
        "historical_news_target_records": record.get("historical_news_target_records", ""),
        "historical_news_max_pages": record.get("historical_news_max_pages", ""),
        "news_mode": live_info.get("news_mode", ""),
        "latest_news_count": latest_news_count,
        "news_count_by_date_json": json_dumps_compact(live_news_count_by_date),
    }


# ============================================================
# 4. Performance metrics
# ============================================================

def compute_perf_from_series(equity_series: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> Dict[str, float]:
    equity = pd.to_numeric(equity_series, errors="coerce").dropna()

    if equity.empty:
        return {"cr": 0.0, "sr": 0.0, "av": 0.0, "mdd": 0.0}

    first = float(equity.iloc[0])
    last = float(equity.iloc[-1])
    cr = last / first - 1.0 if first != 0 else 0.0

    returns = equity.pct_change().dropna()

    if len(returns) >= 2:
        mean = float(returns.mean())
        std = float(returns.std(ddof=1))
        av = std * math.sqrt(periods_per_year)
        sr = mean / std * math.sqrt(periods_per_year) if std > 0 else 0.0
    else:
        av = 0.0
        sr = 0.0

    drawdown = equity / equity.cummax() - 1.0
    mdd = abs(float(drawdown.min())) if not drawdown.empty else 0.0

    return {"cr": cr, "sr": sr, "av": av, "mdd": mdd}


def position_signal_to_score(position: float) -> float:
    position = safe_float(position, 0.0)
    if position > 0:
        return 1.0
    if position < 0:
        return -1.0 if PAPER_ALLOW_SHORT else 0.0
    return 0.0


# ============================================================
# 5. Paper Benchmark Ledger
# ============================================================

def load_paper_state() -> Dict[str, Any]:
    if not PAPER_STATE_PATH.exists():
        return {
            "paper_equity": PAPER_INITIAL_CAPITAL,
            "paper_position": 0.0,
            "last_price": 0.0,
            "last_action_date": None,
        }
    return load_json(PAPER_STATE_PATH)


def investor_position_to_paper_position(position: float) -> float:
    position = safe_float(position, 0.0)
    if position > 0:
        return 1.0
    if position < 0:
        return -1.0 if PAPER_ALLOW_SHORT else 0.0
    return 0.0


def update_paper_ledger(parsed: Dict[str, Any]) -> Dict[str, Any]:
    state = load_paper_state()

    current_price = parsed["price"]
    last_price = safe_float(state.get("last_price"), 0.0)
    prev_position = safe_float(state.get("paper_position"), 0.0)
    prev_equity = safe_float(state.get("paper_equity"), PAPER_INITIAL_CAPITAL)

    if last_price > 0 and current_price > 0:
        price_return = current_price / last_price - 1.0
        paper_period_return = prev_position * price_return
        paper_equity = prev_equity * (1.0 + paper_period_return)
    else:
        price_return = 0.0
        paper_period_return = 0.0
        paper_equity = prev_equity

    new_position = investor_position_to_paper_position(parsed["target_position"])

    save_json(PAPER_STATE_PATH, {
        "paper_equity": paper_equity,
        "paper_position": new_position,
        "last_price": current_price,
        "last_action_date": parsed["action_date"],
        "updated_at": datetime.now().isoformat(),
    })

    return {
        "paper_equity": paper_equity,
        "paper_position": new_position,
        "paper_prev_position": prev_position,
        "paper_price_return": price_return,
        "paper_period_return": paper_period_return,
    }


# ============================================================
# 6. Strategy Account Ledger, exposed as okx_* fields
# ============================================================

def load_okx_state() -> Dict[str, Any]:
    if not OKX_STATE_PATH.exists():
        return {
            "okx_initial_equity": None,
            "last_okx_equity": None,
        }
    return load_json(OKX_STATE_PATH)


def update_okx_account_ledger(parsed: Dict[str, Any]) -> Dict[str, Any]:
    state = load_okx_state()

    strategy_initial = safe_float(parsed.get("strategy_initial_capital_usdt"), STRATEGY_INITIAL_CAPITAL_USDT)
    okx_equity = safe_float(parsed.get("strategy_equity_usdt"), strategy_initial)

    if state.get("okx_initial_equity") is None:
        initial_equity = strategy_initial
    else:
        initial_equity = safe_float(state.get("okx_initial_equity"), strategy_initial)

    last_equity = safe_float(state.get("last_okx_equity"), okx_equity)

    if last_equity > 0 and okx_equity > 0:
        okx_period_return = okx_equity / last_equity - 1.0
    else:
        okx_period_return = 0.0

    save_json(OKX_STATE_PATH, {
        "okx_initial_equity": initial_equity,
        "last_okx_equity": okx_equity,
        "ledger_source": "strategy_account_state",
        "updated_at": datetime.now().isoformat(),
    })

    cash = safe_float(parsed.get("strategy_cash_usdt"), strategy_initial)
    asset_qty = safe_float(parsed.get("strategy_btc_qty"), 0.0)
    price = safe_float(parsed.get("price"), 0.0)
    asset_value = safe_float(parsed.get("strategy_position_value_usdt"), asset_qty * price)

    return {
        "okx_read_success": True,
        "okx_error": "",
        "okx_cash_usdt": cash,
        "okx_available_usdt": cash,
        "okx_frozen_usdt": 0.0,
        "okx_total_eq_reported": okx_equity,
        "okx_btc_qty": asset_qty,
        "okx_btc_value_usdt": asset_value,
        "okx_relevant_equity": okx_equity,
        "okx_equity": okx_equity,
        "okx_initial_equity": initial_equity,
        "okx_period_return": okx_period_return,
        "ledger_source": "strategy_account_state",
    }


# ============================================================
# 7. Append logic
# ============================================================

def get_next_cycle_id() -> int:
    metrics = load_csv(METRICS_PATH)
    if metrics.empty or "cycle_id" not in metrics.columns:
        return 1
    return int(pd.to_numeric(metrics["cycle_id"], errors="coerce").max()) + 1


def compute_cycle_volume(parsed: Dict[str, Any]) -> float:
    """
    Approximate executed notional for dashboard volume.

    Stage B SWAP uses notional_usdt from run_one_live_cycle.py.
    For reverse signals, notional_usdt is close notional + open notional approximation.
    """
    if not parsed["order_success"]:
        return 0.0

    notional = safe_float(parsed.get("notional_usdt"), 0.0)
    if notional > 0:
        return notional

    final_signal = str(parsed["final_signal"]).lower().strip()
    qty = safe_float(parsed["quantity"], 0.0)
    price = safe_float(parsed["price"], 0.0)

    if final_signal in {"buy", "open_long"}:
        if str(parsed.get("tgt_ccy", "")).lower().strip() == "quote_ccy":
            return qty
        return qty * price

    if final_signal in {"sell", "open_short", "close", "reverse_to_long", "reverse_to_short"}:
        return qty * price

    return 0.0

def append_cycle_record(record: Optional[Dict[str, Any]] = None, record_path: Optional[Path] = None) -> Dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if record is None:
        if record_path is None:
            record_path = get_latest_cycle_record_path()
        if record_path is None:
            raise FileNotFoundError(f"No ETH cycle records found in {CYCLE_RECORD_DIR}")
        record = load_json(Path(record_path))

    cycle_start = str(record.get("cycle_start", ""))
    if cycle_start and already_logged(cycle_start):
        print(f"[ETH DASHBOARD LOGGER] Skip duplicate cycle_start: {cycle_start}")
        return {"status": "skipped_duplicate", "cycle_start": cycle_start}

    parsed = parse_cycle_record(record)

    cycle_id = get_next_cycle_id()
    datetime_str, timestamp = infer_cycle_time(record)

    paper = update_paper_ledger(parsed)
    okx = update_okx_account_ledger(parsed)

    cycle_volume_usdt = compute_cycle_volume(parsed)
    old_metrics = load_csv(METRICS_PATH)

    paper_series = pd.concat([
        pd.to_numeric(old_metrics.get("paper_equity", pd.Series(dtype=float)), errors="coerce"),
        pd.Series([paper["paper_equity"]]),
    ], ignore_index=True)

    okx_series = pd.concat([
        pd.to_numeric(old_metrics.get("okx_equity", pd.Series(dtype=float)), errors="coerce"),
        pd.Series([okx["okx_equity"]]),
    ], ignore_index=True)

    paper_perf = compute_perf_from_series(paper_series)
    okx_perf = compute_perf_from_series(okx_series)

    if not old_metrics.empty and "btc_price" in old_metrics.columns:
        first_price = safe_float(old_metrics["btc_price"].iloc[0], parsed["price"])
    else:
        first_price = parsed["price"]

    if first_price > 0:
        paper_bh_equity = PAPER_INITIAL_CAPITAL * (parsed["price"] / first_price)
    else:
        paper_bh_equity = PAPER_INITIAL_CAPITAL

    cumulative_volume = (
        safe_float(old_metrics["cycle_volume_usdt"].sum(), 0.0) + cycle_volume_usdt
        if not old_metrics.empty and "cycle_volume_usdt" in old_metrics.columns
        else cycle_volume_usdt
    )

    metrics_row = {
        "cycle_id": cycle_id,
        "timestamp": timestamp,
        "datetime": datetime_str,
        "bar_timestamp": timestamp,
        "bar_datetime": datetime_str,
        "method_family": "finmem",
        "strategy_id": MODEL_NAME,
        "frequency": TIMEFRAME,
        "symbol": parsed["symbol"],
        "action_date": parsed["action_date"],
        # Keep btc_price column for dashboard compatibility; for this logger it stores ETH price.
        "btc_price": parsed["price"],
        "asset_price": parsed["price"],

        "live_start_date": parsed["live_start_date"],
        "live_end_date": parsed["live_end_date"],
        "latest_date": parsed["latest_date"],
        "live_previous_days": parsed["live_previous_days"],
        "live_use_historical_news": parsed["live_use_historical_news"],
        "historical_news_target_records": parsed["historical_news_target_records"],
        "historical_news_max_pages": parsed["historical_news_max_pages"],
        "news_mode": parsed["news_mode"],
        "latest_news_count": parsed["latest_news_count"],
        "news_count_by_date_json": parsed["news_count_by_date_json"],

        "investor_signal": parsed["investor_signal"],
        "investor_position": parsed["investor_position"],
        "target_position": parsed["target_position"],
        "position_signal": parsed["position_signal"],
        "decision_target_position": parsed["decision_target_position"],
        "original_signal": parsed["original_signal"],
        "final_signal": parsed["final_signal"],
        "trade_side": parsed["trade_side"],
        "quantity": parsed["quantity"],
        "notional_usdt": parsed["notional_usdt"],
        "open_notional_usdt": parsed["open_notional_usdt"],
        "trade_mode": parsed["trade_mode"],
        "trade_inst_id": parsed["trade_inst_id"],
        "target_sign": parsed["target_sign"],
        "current_sign": parsed["current_sign"],
        "target_state": parsed["target_state"],
        "current_state": parsed["current_state"],
        "tgt_ccy": parsed["tgt_ccy"],
        "risk_reason": parsed["risk_reason"],
        "execution_status": parsed["execution_status"],
        "order_code": parsed["order_code"],
        "order_id": parsed["order_id"],
        "order_success": parsed["order_success"],
        "dry_run": parsed["dry_run"],
        "execute_order": parsed["execute_order"],
        "is_simulated": parsed["is_simulated"],

        "paper_equity": paper["paper_equity"],
        "paper_position": paper["paper_position"],
        "paper_prev_position": paper["paper_prev_position"],
        "paper_price_return": paper["paper_price_return"],
        "paper_period_return": paper["paper_period_return"],
        "paper_cr": paper_perf["cr"],
        "paper_sr": paper_perf["sr"],
        "paper_av": paper_perf["av"],
        "paper_mdd": paper_perf["mdd"],
        "paper_bh_equity": paper_bh_equity,
        "paper_bh_return_pct": (paper_bh_equity / PAPER_INITIAL_CAPITAL - 1.0) * 100.0,

        "okx_read_success": okx["okx_read_success"],
        "okx_error": okx["okx_error"],
        "okx_cash_usdt": okx["okx_cash_usdt"],
        "okx_available_usdt": okx["okx_available_usdt"],
        "okx_frozen_usdt": okx["okx_frozen_usdt"],
        "okx_total_eq_reported": okx["okx_total_eq_reported"],
        "okx_btc_qty": okx["okx_btc_qty"],
        "okx_btc_value_usdt": okx["okx_btc_value_usdt"],
        "okx_relevant_equity": okx["okx_relevant_equity"],
        "okx_equity": okx["okx_equity"],
        "okx_initial_equity": okx["okx_initial_equity"],
        "okx_period_return": okx["okx_period_return"],
        "okx_cr": okx_perf["cr"],
        "okx_sr": okx_perf["sr"],
        "okx_av": okx_perf["av"],
        "okx_mdd": okx_perf["mdd"],
        "okx_ledger_source": okx.get("ledger_source", "strategy_account_state"),

        "strategy_initial_capital_usdt": parsed["strategy_initial_capital_usdt"],
        "strategy_cash_usdt": parsed["strategy_cash_usdt"],
        "strategy_btc_qty": parsed["strategy_btc_qty"],
        "strategy_equity_usdt": parsed["strategy_equity_usdt"],
        "strategy_position_value_usdt": parsed["strategy_position_value_usdt"],
        "strategy_position_state": parsed["strategy_position_state"],
        "strategy_position_sign": parsed["strategy_position_sign"],
        "stage_b_swap": parsed["stage_b_swap"],

        "cycle_volume_usdt": cycle_volume_usdt,
        "cumulative_total_volume_usdt": cumulative_volume,
    }

    metrics_row["initial_capital_usdt"] = parsed["strategy_initial_capital_usdt"]
    metrics_row["cash_total_usdt"] = parsed["strategy_cash_usdt"]
    metrics_row["holdings_value_usdt"] = parsed["strategy_position_value_usdt"]
    metrics_row["total_equity_usdt"] = parsed["strategy_equity_usdt"]
    metrics_row["strategy_equity"] = parsed["strategy_equity_usdt"]
    metrics_row["strategy_pnl"] = (
        parsed["strategy_equity_usdt"] - parsed["strategy_initial_capital_usdt"]
    )
    metrics_row["strategy_returns_pct"] = (
        metrics_row["strategy_pnl"] / parsed["strategy_initial_capital_usdt"] * 100.0
        if parsed["strategy_initial_capital_usdt"] > 0
        else 0.0
    )
    metrics_row["baseline_equity"] = (
        metrics_row["paper_bh_equity"] * parsed["strategy_initial_capital_usdt"]
    )
    metrics_row["baseline_pnl"] = (
        metrics_row["baseline_equity"] - parsed["strategy_initial_capital_usdt"]
    )
    metrics_row["baseline_returns_pct"] = metrics_row["paper_bh_return_pct"]
    metrics_row["alpha_return_pct"] = metrics_row["strategy_returns_pct"] - metrics_row["baseline_returns_pct"]
    metrics_row["active_positions"] = int(parsed["strategy_position_sign"] != 0)
    metrics_row["gross_exposure"] = (
        parsed["strategy_position_value_usdt"] / parsed["strategy_equity_usdt"]
        if parsed["strategy_equity_usdt"] > 0
        else 0.0
    )
    metrics_row["cash_ratio"] = (
        parsed["strategy_cash_usdt"] / parsed["strategy_equity_usdt"]
        if parsed["strategy_equity_usdt"] > 0
        else 0.0
    )
    metrics_row["cash_usdt"] = metrics_row["okx_cash_usdt"]
    metrics_row["btc_qty"] = metrics_row["okx_btc_qty"]
    metrics_row["position_value_usdt"] = metrics_row["okx_btc_value_usdt"]

    append_csv(METRICS_PATH, metrics_row)

    trade_row = {
        "trade_id": f"{parsed['symbol']}-{cycle_id:08d}",
        "cycle_id": cycle_id,
        "timestamp": timestamp,
        "datetime": datetime_str,
        "trade_date": parsed["action_date"],
        "coin": parsed["symbol"],
        "side": parsed["trade_side"],
        "qty": safe_float(parsed["quantity"], 0.0),
        "fee_usdt": parsed["real_okx_fee"],
        "slippage_bps": parsed["execution_slippage_bps"],
        "cash_before": parsed["strategy_cash_usdt"],
        "cash_after": parsed["strategy_cash_usdt"],
        "position_qty_before": 0.0,
        "position_qty_after": parsed["strategy_btc_qty"],
        "reason": parsed["risk_reason"],
        "success": parsed["order_success"],
        "method_family": "finmem",
        "strategy_id": MODEL_NAME,
        "frequency": TIMEFRAME,
        "cycle_start": cycle_start,
        "symbol": parsed["symbol"],
        "action_date": parsed["action_date"],
        "price": parsed["price"],
        "live_start_date": parsed["live_start_date"],
        "live_end_date": parsed["live_end_date"],
        "news_mode": parsed["news_mode"],
        "latest_news_count": parsed["latest_news_count"],
        "investor_signal": parsed["investor_signal"],
        "investor_position": parsed["investor_position"],
        "target_position": parsed["target_position"],
        "position_signal": parsed["position_signal"],
        "original_signal": parsed["original_signal"],
        "final_signal": parsed["final_signal"],
        "trade_side": parsed["trade_side"],
        "quantity": parsed["quantity"],
        "notional_usdt": parsed["notional_usdt"],
        "open_notional_usdt": parsed["open_notional_usdt"],
        "trade_mode": parsed["trade_mode"],
        "trade_inst_id": parsed["trade_inst_id"],
        "target_sign": parsed["target_sign"],
        "current_sign": parsed["current_sign"],
        "target_state": parsed["target_state"],
        "current_state": parsed["current_state"],
        "tgt_ccy": parsed["tgt_ccy"],
        "cycle_volume_usdt": cycle_volume_usdt,
        "risk_reason": parsed["risk_reason"],
        "execution_status": parsed["execution_status"],
        "order_code": parsed["order_code"],
        "order_id": parsed["order_id"],
        "order_success": parsed["order_success"],
        "dry_run": parsed["dry_run"],
        "execute_order": parsed["execute_order"],
        "is_simulated": parsed["is_simulated"],
        "strategy_cash_usdt": parsed["strategy_cash_usdt"],
        "strategy_btc_qty": parsed["strategy_btc_qty"],
        "strategy_equity_usdt": parsed["strategy_equity_usdt"],
        "strategy_position_state": parsed["strategy_position_state"],
        "strategy_position_sign": parsed["strategy_position_sign"],
        "stage_b_swap": parsed["stage_b_swap"],
        "record_path": str(record_path or ""),
    }
    append_csv(TRADES_PATH, trade_row)

    signal_row = {
        "cycle_id": cycle_id,
        "timestamp": timestamp,
        "datetime": datetime_str,
        "bar_timestamp": timestamp,
        "bar_datetime": datetime_str,
        "coin": parsed["symbol"],
        "score": position_signal_to_score(parsed["target_position"]),
        "signal_label": parsed["final_signal"],
        "signal_score": position_signal_to_score(parsed["target_position"]),
        "method_family": "finmem",
        "strategy_id": MODEL_NAME,
        "frequency": TIMEFRAME,
        "price_at_pred": parsed["price"],
        "price_future": "",
        "true_return": "",
        "labeled": False,
        "investor_signal": parsed["investor_signal"],
        "investor_position": parsed["investor_position"],
        "target_position": parsed["target_position"],
        "position_signal": parsed["position_signal"],
        "final_signal": parsed["final_signal"],
        "trade_side": parsed["trade_side"],
    }
    append_csv(SIGNALS_PATH, signal_row)

    volume_old = load_csv(VOLUME_PATH)
    cumulative_volume_for_volume_file = (
        safe_float(volume_old["cycle_volume_usdt"].sum(), 0.0) + cycle_volume_usdt
        if not volume_old.empty and "cycle_volume_usdt" in volume_old.columns
        else cycle_volume_usdt
    )

    today = datetime_str[:10]
    if not volume_old.empty:
        tmp = volume_old.copy()
        tmp["datetime"] = pd.to_datetime(tmp["datetime"], errors="coerce")
        tmp["date"] = tmp["datetime"].dt.strftime("%Y-%m-%d")
        tmp["cycle_volume_usdt"] = pd.to_numeric(tmp["cycle_volume_usdt"], errors="coerce").fillna(0.0)
        today_volume = float(tmp.loc[tmp["date"] == today, "cycle_volume_usdt"].sum()) + cycle_volume_usdt
    else:
        today_volume = cycle_volume_usdt

    per_coin_daily = {parsed["symbol"]: today_volume} if today_volume > 0 else {}
    per_coin_cumulative = {parsed["symbol"]: cumulative_volume_for_volume_file} if cumulative_volume_for_volume_file > 0 else {}

    volume_row = {
        "cycle_id": cycle_id,
        "timestamp": timestamp,
        "datetime": datetime_str,
        "symbol": parsed["symbol"],
        "method_family": "finmem",
        "strategy_id": MODEL_NAME,
        "frequency": TIMEFRAME,
        "final_signal": parsed["final_signal"],
        "trade_side": parsed["trade_side"],
        "cycle_volume_usdt": cycle_volume_usdt,
        "daily_total_volume_usdt": today_volume,
        "cumulative_total_volume_usdt": cumulative_volume_for_volume_file,
        "per_coin_daily_json": json.dumps(per_coin_daily, ensure_ascii=False),
        "per_coin_cumulative_json": json.dumps(per_coin_cumulative, ensure_ascii=False),
    }
    append_csv(VOLUME_PATH, volume_row)

    metadata = {
        "model_name": MODEL_NAME,
        "timeframe": TIMEFRAME,
        "symbol": SYMBOL,
        "periods_per_year": PERIODS_PER_YEAR,
        "paper_initial_capital": PAPER_INITIAL_CAPITAL,
        "paper_allow_short": PAPER_ALLOW_SHORT,
        "stage_b_swap": True,
        "is_simulated": IS_SIMULATED,
        "account_ledger_source": "strategy_account_state",
        "strategy_initial_capital_usdt": STRATEGY_INITIAL_CAPITAL_USDT,
        "supports_final_signals": ["buy", "hold", "sell", "close", "reverse_to_long", "reverse_to_short"],
        "supports_rolling_window_fields": True,
        "metrics_path": str(METRICS_PATH),
        "trades_path": str(TRADES_PATH),
        "signals_path": str(SIGNALS_PATH),
        "volume_path": str(VOLUME_PATH),
        "paper_state_path": str(PAPER_STATE_PATH),
        "okx_state_path": str(OKX_STATE_PATH),
        "strategy_account_state_path": str(STRATEGY_ACCOUNT_STATE_PATH),
        "updated_at": datetime.now().isoformat(),
    }
    save_json(METADATA_PATH, metadata)

    from quant_bench.runtime.core.run_manifest import write_run_manifest

    write_run_manifest(
        LOG_DIR,
        strategy_id=MODEL_NAME,
        method_family="finmem",
        frequency=TIMEFRAME,
        mode=os.getenv("FINMEM_MODE", "paper"),
        paths={
            "metrics_path": str(METRICS_PATH),
            "trades_path": str(TRADES_PATH),
            "signals_path": str(SIGNALS_PATH),
            "volume_path": str(VOLUME_PATH),
        },
        extra={"symbol": SYMBOL},
    )

    print(f"[{SYMBOL} DASHBOARD LOGGER] appended cycle:", cycle_id)

    return {
        "status": "appended",
        "cycle_id": cycle_id,
        "metrics_path": str(METRICS_PATH),
        "trades_path": str(TRADES_PATH),
        "signals_path": str(SIGNALS_PATH),
        "volume_path": str(VOLUME_PATH),
        "paper_state_path": str(PAPER_STATE_PATH),
        "okx_state_path": str(OKX_STATE_PATH),
        "strategy_account_state_path": str(STRATEGY_ACCOUNT_STATE_PATH),
    }


def main() -> None:
    latest = get_latest_cycle_record_path()
    if latest is None:
        raise FileNotFoundError(f"No {SYMBOL} cycle records found in {CYCLE_RECORD_DIR}")

    print(f"[{SYMBOL} DASHBOARD LOGGER] latest record:", latest)
    result = append_cycle_record(record_path=latest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
