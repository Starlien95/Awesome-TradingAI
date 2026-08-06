from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from quant_bench.runtime.core.execution import TradeExecutor
from quant_bench.runtime.core.state_store import JsonStateStore

logger = logging.getLogger(__name__)


def _now_row_time(ts: float | None = None) -> tuple[float, str]:
    timestamp = float(ts if ts is not None else time.time())
    return timestamp, datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
        return default if math.isnan(out) or math.isinf(out) else out
    except (TypeError, ValueError):
        return default


class PerCoinOKXSpotBroker:
    """
    OKX spot broker for the FinGPT daily strategy.

    The strategy keeps independent per-symbol equity buckets in local state.
    A buy spends only that symbol's current strategy cash, so a later re-entry
    naturally uses less cash after losses and more cash after gains.

    Actual OKX is one account, so sells are limited to the quantity this broker
    recorded for this strategy instead of clearing the whole account position.
    """

    def __init__(
        self,
        state_path: str | Path,
        executor: TradeExecutor,
        coins: list[str],
        initial_capital_usdt: float,
        per_coin_capital_usdt: float,
        fee_bps: float,
        min_position_value_usdt: float = 2.0,
        order_settle_delay_sec: float = 1.5,
        require_full_cash: bool = True,
        order_cash_buffer_bps: float | None = None,
        reconcile_account_positions_enabled: bool = False,
    ):
        self.executor = executor
        self.coins = list(coins)
        self.initial_capital_usdt = float(initial_capital_usdt)
        self.per_coin_capital_usdt = float(per_coin_capital_usdt)
        self.fee_bps = float(fee_bps)
        self.min_position_value_usdt = float(min_position_value_usdt)
        self.order_settle_delay_sec = float(order_settle_delay_sec)
        self.require_full_cash = bool(require_full_cash)
        self.order_cash_buffer_bps = (
            float(order_cash_buffer_bps)
            if order_cash_buffer_bps is not None
            else max(0.0, self.fee_bps + 5.0)
        )
        self.reconcile_account_positions_enabled = bool(reconcile_account_positions_enabled)
        self.store = JsonStateStore(
            state_path,
            {
                "initial_capital_usdt": self.initial_capital_usdt,
                "cash_by_symbol": {coin: self.per_coin_capital_usdt for coin in self.coins},
                "position_qty_by_symbol": {coin: 0.0 for coin in self.coins},
                "last_signal_by_symbol": {coin: 0 for coin in self.coins},
                "external_cash_credit_by_symbol": {coin: 0.0 for coin in self.coins},
                "last_account_reconcile_by_symbol": {},
                "processed_trade_dates": [],
                "trade_counter": 0,
                "broker_mode": "okx_spot",
            },
        )
        self._normalize_state()
        self._audit_records: dict[str, list[dict[str, Any]]] = {
            "orders": [],
            "fills": [],
            "failed_orders": [],
            "account_snapshots": [],
            "order_events": [],
        }

    @property
    def state(self) -> dict[str, Any]:
        return self.store.state

    def _normalize_state(self) -> None:
        for coin in self.coins:
            self.state.setdefault("cash_by_symbol", {}).setdefault(coin, self.per_coin_capital_usdt)
            self.state.setdefault("position_qty_by_symbol", {}).setdefault(coin, 0.0)
            self.state.setdefault("last_signal_by_symbol", {}).setdefault(coin, 0)
            self.state.setdefault("external_cash_credit_by_symbol", {}).setdefault(coin, 0.0)
        self.state.setdefault("last_account_reconcile_by_symbol", {})
        self.state["broker_mode"] = "okx_spot"
        self.store.save()

    def has_processed_trade_date(self, trade_date: str) -> bool:
        return trade_date in set(self.state.get("processed_trade_dates", []))

    def mark_trade_date_done(self, trade_date: str) -> None:
        self.store.mark_done("processed_trade_dates", trade_date)

    def equity_snapshot(self, prices: dict[str, float]) -> dict[str, Any]:
        cash_total = 0.0
        holdings_value = 0.0
        active_positions = 0
        per_coin_value: dict[str, float] = {}
        for coin in self.coins:
            cash = float(self.state["cash_by_symbol"].get(coin, 0.0))
            qty = float(self.state["position_qty_by_symbol"].get(coin, 0.0))
            price = float(prices.get(coin, 0.0) or 0.0)
            value = qty * price if qty > 0 and price > 0 else 0.0
            cash_total += cash
            holdings_value += value
            per_coin_value[coin] = cash + value
            if value >= self.min_position_value_usdt:
                active_positions += 1
        total = cash_total + holdings_value
        return {
            "cash_total_usdt": cash_total,
            "holdings_value_usdt": holdings_value,
            "total_equity_usdt": total,
            "strategy_equity": total,
            "strategy_pnl": total - self.initial_capital_usdt,
            "strategy_returns_pct": (total / self.initial_capital_usdt - 1.0) * 100.0
            if self.initial_capital_usdt > 0
            else 0.0,
            "active_positions": active_positions,
            "gross_exposure": holdings_value / total if total > 0 else 0.0,
            "cash_ratio": cash_total / total if total > 0 else 0.0,
            "per_coin_equity": per_coin_value,
        }

    def drain_audit_records(self) -> dict[str, list[dict[str, Any]]]:
        out = self._audit_records
        self._audit_records = {
            "orders": [],
            "fills": [],
            "failed_orders": [],
            "account_snapshots": [],
            "order_events": [],
        }
        return out

    def reconcile_account_positions(
        self,
        prices: dict[str, float],
        timestamp: float | None = None,
        trade_date: str = "",
    ) -> None:
        if not self.reconcile_account_positions_enabled:
            return
        holdings = self._account_holdings()
        if holdings is None:
            return
        timestamp, dt = _now_row_time(timestamp)
        changed = False
        for coin in self.coins:
            price = float(prices.get(coin, 0.0) or 0.0)
            if price <= 0:
                continue
            changed = self._reconcile_coin_with_observed_qty(
                coin,
                price,
                float(holdings.get(coin, 0.0) or 0.0),
                timestamp,
                dt,
                trade_date,
            ) or changed
        if changed:
            self.store.save()

    def rebalance_coin(
        self,
        coin: str,
        target_position: int,
        price: float,
        trade_date: str,
        signal_row: dict[str, Any],
        timestamp: float | None = None,
    ) -> dict[str, Any] | None:
        if coin not in self.coins or price <= 0:
            return None
        timestamp, dt = _now_row_time(timestamp)
        self._reconcile_position_with_account(coin, price, timestamp, dt, trade_date)
        cash_before = float(self.state["cash_by_symbol"].get(coin, 0.0))
        qty_before = float(self.state["position_qty_by_symbol"].get(coin, 0.0))
        position_value_before = qty_before * price
        current_position = 1 if position_value_before >= self.min_position_value_usdt else 0
        if target_position == current_position:
            self.state["last_signal_by_symbol"][coin] = int(signal_row.get("model_score", 0))
            self.store.save()
            return None

        self.state["trade_counter"] = int(self.state.get("trade_counter", 0)) + 1
        trade_id = f"{trade_date}-{coin}-{self.state['trade_counter']:06d}"
        fee_rate = self.fee_bps / 10_000.0
        side = "buy" if target_position == 1 else "sell"
        notional = 0.0
        fee = 0.0
        qty_after = qty_before
        cash_after = cash_before

        if side == "buy":
            cash_buffer_rate = max(0.0, self.order_cash_buffer_bps) / 10_000.0
            target_notional = max(0.0, cash_before * (1.0 - cash_buffer_rate))
            if target_notional < self.min_position_value_usdt:
                self.store.save()
                return None
            account_cash = self.executor.get_usdt_balance()
            if self.require_full_cash and account_cash < target_notional:
                return self._failed_trade(
                    trade_id, timestamp, dt, trade_date, coin, side, price, qty_before,
                    target_notional, signal_row, cash_before, cash_after, qty_before, qty_after,
                    "insufficient_okx_usdt",
                )
            notional = target_notional if self.require_full_cash else min(target_notional, max(0.0, account_cash * 0.98))
            if notional < self.min_position_value_usdt:
                self.store.save()
                return None
            observed_before = self._account_qty(coin)
            result = self.executor.execute_trade(coin, "buy", notional, price)
            if not result:
                return self._failed_trade(
                    trade_id, timestamp, dt, trade_date, coin, side, price, qty_before,
                    notional, signal_row, cash_before, cash_after, qty_before, qty_after,
                    "okx_buy_failed", result,
                )
            observed_after = self._account_qty_after_settle(coin)
            actual_price = float(result.avg_fill_price or price)
            actual_notional = float(result.filled_notional_usdt or notional)
            fee = float(result.fee_usdt or (actual_notional * fee_rate))
            estimated_delta = max(0.0, actual_notional - fee) / actual_price if actual_price > 0 else 0.0
            qty_delta = self._observed_delta_or_estimate(observed_before, observed_after, estimated_delta)
            qty_after = qty_before + qty_delta
            cash_after = cash_before - actual_notional
            notional = actual_notional
        else:
            if position_value_before < self.min_position_value_usdt:
                self.state["position_qty_by_symbol"][coin] = 0.0
                self.store.save()
                return None
            observed_before = self._account_qty(coin)
            if observed_before is None:
                return self._failed_trade(
                    trade_id, timestamp, dt, trade_date, coin, side, price, qty_before,
                    position_value_before, signal_row, cash_before, cash_after, qty_before, qty_after,
                    "okx_holdings_unavailable",
                )
            if observed_before < qty_before * 0.98:
                self._reconcile_coin_with_observed_qty(
                    coin,
                    price,
                    observed_before,
                    timestamp,
                    dt,
                    trade_date,
                )
                cash_before = float(self.state["cash_by_symbol"].get(coin, 0.0))
                qty_before = float(self.state["position_qty_by_symbol"].get(coin, 0.0))
                position_value_before = qty_before * price
                current_position = 1 if position_value_before >= self.min_position_value_usdt else 0
                if target_position == current_position:
                    self.state["last_signal_by_symbol"][coin] = int(signal_row.get("model_score", 0))
                    self.store.save()
                    return None
                observed_before = self._account_qty(coin)
                if observed_before is None:
                    return self._failed_trade(
                        trade_id, timestamp, dt, trade_date, coin, side, price, qty_before,
                        position_value_before, signal_row, cash_before, cash_after, qty_before, qty_after,
                        "okx_holdings_unavailable",
                    )
            if observed_before < qty_before * 0.98:
                return self._failed_trade(
                    trade_id, timestamp, dt, trade_date, coin, side, price, qty_before,
                    position_value_before, signal_row, cash_before, cash_after, qty_before, qty_after,
                    "insufficient_strategy_qty_on_okx",
                )
            notional = position_value_before
            result = self.executor.execute_trade(coin, "sell", notional, price)
            if not result:
                return self._failed_trade(
                    trade_id, timestamp, dt, trade_date, coin, side, price, qty_before,
                    notional, signal_row, cash_before, cash_after, qty_before, qty_after,
                    "okx_sell_failed", result,
                )
            observed_after = self._account_qty_after_settle(coin)
            observed_sold = max(0.0, observed_before - observed_after) if observed_after is not None else 0.0
            filled_qty = float(result.filled_qty or 0.0)
            qty_sold = min(qty_before, observed_sold) if observed_sold > 0 else (min(qty_before, filled_qty) if filled_qty > 0 else qty_before)
            notional = float(result.filled_notional_usdt or (qty_sold * price))
            fee = float(result.fee_usdt or (notional * fee_rate))
            cash_after = cash_before + max(0.0, notional - fee)
            qty_after = max(0.0, qty_before - qty_sold)
            if qty_after * price < self.min_position_value_usdt:
                qty_after = 0.0

        self.state["cash_by_symbol"][coin] = cash_after
        self.state["position_qty_by_symbol"][coin] = qty_after
        self.state["last_signal_by_symbol"][coin] = int(signal_row.get("model_score", 0))
        self.store.save()
        return self._trade_row(
            trade_id, timestamp, dt, trade_date, coin, side, price,
            abs(qty_after - qty_before) if side == "buy" else qty_before,
            notional, fee, signal_row, cash_before, cash_after, qty_before, qty_after,
            "positive_to_okx_bucket" if side == "buy" else "negative_to_strategy_cash",
            True,
            result,
        )

    def _account_holdings(self) -> dict[str, float] | None:
        account_api = getattr(self.executor, "account_api", None)
        if not account_api:
            logger.warning("[%s] OKX account API unavailable for holdings reconciliation", self.executor.identifier)
            return None
        try:
            res = account_api.get_account_balance()
        except Exception as exc:
            logger.warning("[%s] OKX holdings reconciliation failed: %s", self.executor.identifier, exc)
            return None
        if not isinstance(res, dict) or res.get("code") != "0":
            logger.warning(
                "[%s] OKX holdings reconciliation returned code=%s msg=%s",
                self.executor.identifier,
                res.get("code") if isinstance(res, dict) else "",
                res.get("msg") if isinstance(res, dict) else res,
            )
            return None

        first = (res.get("data") or [{}])[0]
        details = first.get("details", []) if isinstance(first, dict) else []
        out: dict[str, float] = {}
        for detail in details:
            ccy = detail.get("ccy")
            qty = _to_float(detail.get("availBal"), 0.0)
            if qty > 0 and ccy and ccy != "USDT":
                out[f"{ccy}-USDT"] = qty
        return out

    def _account_qty(self, coin: str) -> float | None:
        holdings = self._account_holdings()
        if holdings is None:
            return None
        return float(holdings.get(coin, 0.0) or 0.0)

    def _account_qty_after_settle(self, coin: str) -> float | None:
        if self.order_settle_delay_sec > 0:
            time.sleep(self.order_settle_delay_sec)
        return self._account_qty(coin)

    def _reconcile_position_with_account(
        self,
        coin: str,
        price: float,
        timestamp: float,
        dt: str,
        trade_date: str,
    ) -> bool:
        if not self.reconcile_account_positions_enabled:
            return False
        holdings = self._account_holdings()
        if holdings is None:
            return False
        changed = self._reconcile_coin_with_observed_qty(
            coin,
            price,
            float(holdings.get(coin, 0.0) or 0.0),
            timestamp,
            dt,
            trade_date,
        )
        if changed:
            self.store.save()
        return changed

    def _reconcile_coin_with_observed_qty(
        self,
        coin: str,
        price: float,
        observed_qty: float,
        timestamp: float,
        dt: str,
        trade_date: str,
    ) -> bool:
        local_qty = float(self.state["position_qty_by_symbol"].get(coin, 0.0))
        if local_qty <= 0 or price <= 0:
            return False
        local_value = local_qty * price
        if local_value < self.min_position_value_usdt:
            return False
        if observed_qty >= local_qty * 0.98:
            return False

        observed_value = max(0.0, observed_qty) * price
        qty_after = observed_qty if observed_value >= self.min_position_value_usdt else 0.0
        missing_qty = max(0.0, local_qty - qty_after)
        fee_rate = self.fee_bps / 10_000.0
        cash_credit = max(0.0, missing_qty * price * (1.0 - fee_rate))
        cash_before = float(self.state["cash_by_symbol"].get(coin, 0.0))
        cash_after = cash_before + cash_credit

        self.state["cash_by_symbol"][coin] = cash_after
        self.state["position_qty_by_symbol"][coin] = qty_after
        credits = self.state.setdefault("external_cash_credit_by_symbol", {})
        credits[coin] = float(credits.get(coin, 0.0) or 0.0) + cash_credit
        reconcile_row = {
            "timestamp": timestamp,
            "datetime": dt,
            "trade_date": trade_date,
            "coin": coin,
            "local_qty_before": local_qty,
            "observed_qty": observed_qty,
            "local_qty_after": qty_after,
            "cash_before": cash_before,
            "cash_after": cash_after,
            "cash_credit_usdt": cash_credit,
            "reference_price": price,
            "reason": "external_position_reconciliation",
        }
        self.state.setdefault("last_account_reconcile_by_symbol", {})[coin] = reconcile_row
        self._audit_records["order_events"].append(
            {
                "event_type": "account_reconciliation",
                "exchange": "okx",
                "account_mode": "demo" if getattr(self.executor, "_okx_flag", "1") == "1" else "live",
                "trade_mode": "spot",
                **reconcile_row,
            }
        )
        logger.warning(
            "[%s] reconciled %s local_qty=%s observed_qty=%s cash_credit=%.6f",
            self.executor.identifier,
            coin,
            f"{local_qty:g}",
            f"{observed_qty:g}",
            cash_credit,
        )
        return True

    @staticmethod
    def _observed_delta_or_estimate(observed_before: float | None, observed_after: float | None, estimate: float) -> float:
        if observed_before is None or observed_after is None:
            return float(estimate)
        observed_delta = max(0.0, float(observed_after) - float(observed_before))
        if observed_delta > 0:
            return observed_delta
        return float(estimate)

    def _failed_trade(
        self,
        trade_id: str,
        timestamp: float,
        dt: str,
        trade_date: str,
        coin: str,
        side: str,
        price: float,
        qty: float,
        notional: float,
        signal_row: dict[str, Any],
        cash_before: float,
        cash_after: float,
        qty_before: float,
        qty_after: float,
        reason: str,
        execution_result: Any | None = None,
    ) -> dict[str, Any]:
        self.state["last_signal_by_symbol"][coin] = int(signal_row.get("model_score", 0))
        self.store.save()
        return self._trade_row(
            trade_id, timestamp, dt, trade_date, coin, side, price, qty, notional, 0.0,
            signal_row, cash_before, cash_after, qty_before, qty_after, reason, False, execution_result,
        )

    def _trade_row(
        self,
        trade_id: str,
        timestamp: float,
        dt: str,
        trade_date: str,
        coin: str,
        side: str,
        price: float,
        qty: float,
        notional: float,
        fee: float,
        signal_row: dict[str, Any],
        cash_before: float,
        cash_after: float,
        qty_before: float,
        qty_after: float,
        reason: str,
        success: bool,
        execution_result: Any | None = None,
    ) -> dict[str, Any]:
        actual_price = float(getattr(execution_result, "avg_fill_price", 0.0) or price)
        actual_qty = float(getattr(execution_result, "filled_qty", 0.0) or qty)
        actual_notional = float(getattr(execution_result, "filled_notional_usdt", 0.0) or notional)
        slippage = float(getattr(execution_result, "slippage_bps", 0.0) or 0.0)
        fee_usdt = float(getattr(execution_result, "fee_usdt", 0.0) or fee)
        row = {
            "trade_id": trade_id,
            "timestamp": timestamp,
            "datetime": dt,
            "trade_date": trade_date,
            "news_date": signal_row.get("news_date", ""),
            "coin": coin,
            "side": side,
            "price": actual_price,
            "reference_price": price,
            "actual_avg_price": actual_price if success else 0.0,
            "qty": actual_qty,
            "requested_qty": qty,
            "actual_qty": actual_qty if success else 0.0,
            "notional_usdt": actual_notional if success else notional,
            "requested_notional_usdt": notional,
            "actual_notional_usdt": actual_notional if success else 0.0,
            "fee_usdt": fee_usdt,
            "fee_ccy": getattr(execution_result, "fee_ccy", ""),
            "slippage_bps": slippage,
            "order_id": getattr(execution_result, "ord_id", ""),
            "client_order_id": getattr(execution_result, "cl_ord_id", ""),
            "order_status": getattr(execution_result, "status", ""),
            "fill_verified": bool(getattr(execution_result, "fill_verified", False)),
            "partial_fill": bool(getattr(execution_result, "partial_fill", False)),
            "exchange_error_code": getattr(execution_result, "okx_s_code", "") or getattr(execution_result, "okx_code", ""),
            "exchange_error_msg": getattr(execution_result, "okx_s_msg", "") or getattr(execution_result, "okx_msg", ""),
            "signal_label": signal_row.get("model_label", ""),
            "signal_score": signal_row.get("model_score", 0),
            "final_score": signal_row.get("final_score", 0.0),
            "threshold": signal_row.get("threshold", 0.0),
            "cash_before": cash_before,
            "cash_after": cash_after,
            "position_qty_before": qty_before,
            "position_qty_after": qty_after,
            "position_value_after": qty_after * price,
            "reason": reason,
            "success": success,
        }
        self._record_audit(row, execution_result)
        return row

    def _record_audit(self, row: dict[str, Any], execution_result: Any | None) -> None:
        context = {
            key: row.get(key, "")
            for key in [
                "trade_id",
                "timestamp",
                "datetime",
                "trade_date",
                "news_date",
                "coin",
            ]
        }
        if execution_result is not None:
            order_row = execution_result.to_order_row(context)
            fill_rows = execution_result.to_fill_rows(context)
            events = execution_result.to_event_records(context)
        else:
            order_row = dict(context)
            order_row.update(
                {
                    "exchange": "okx",
                    "account_mode": "demo" if getattr(self.executor, "_okx_flag", "1") == "1" else "live",
                    "trade_mode": "spot",
                    "inst_id": row.get("coin", ""),
                    "side": row.get("side", ""),
                    "ord_type": "market",
                    "td_mode": "cash",
                    "requested_notional_usdt": row.get("requested_notional_usdt", row.get("notional_usdt", 0.0)),
                    "reference_price": row.get("reference_price", 0.0),
                    "order_status": row.get("order_status", "failed_precheck"),
                    "filled_qty": 0.0,
                    "filled_notional_usdt": 0.0,
                    "fee_usdt": 0.0,
                    "slippage_bps": 0.0,
                    "fill_verified": False,
                    "partial_fill": False,
                    "success": False,
                    "error_message": row.get("reason", ""),
                }
            )
            fill_rows = []
            events = []

        snapshots = []
        ref_price = float(row.get("reference_price", 0.0) or 0.0)
        for phase, cash_key, qty_key in [
            ("before", "cash_before", "position_qty_before"),
            ("after", "cash_after", "position_qty_after"),
        ]:
            cash = float(row.get(cash_key, 0.0) or 0.0)
            qty = float(row.get(qty_key, 0.0) or 0.0)
            snap = dict(context)
            snap.update(
                {
                    "snapshot_phase": phase,
                    "account_mode": order_row.get("account_mode", ""),
                    "trade_mode": "spot",
                    "usdt_available": cash,
                    "holdings_value_usdt": qty * ref_price,
                    "estimated_total_equity_usdt": cash + qty * ref_price,
                    "holdings_json": {row.get("coin", ""): qty},
                }
            )
            snapshots.append(snap)

        self._audit_records["orders"].append(order_row)
        self._audit_records["fills"].extend(fill_rows)
        self._audit_records["account_snapshots"].extend(snapshots)
        self._audit_records["order_events"].extend(events)
        if not bool(row.get("success", False)):
            self._audit_records["failed_orders"].append(order_row)
