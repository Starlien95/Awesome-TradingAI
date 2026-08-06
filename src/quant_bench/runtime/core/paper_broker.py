from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_bench.runtime.core.atomic_io import atomic_write_csv
from quant_bench.runtime.core.state_store import JsonStateStore


def _now_row_time(ts: float | None = None) -> tuple[float, str]:
    timestamp = float(ts if ts is not None else time.time())
    return timestamp, datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


class PerCoinPaperBroker:
    """
    Paper broker for independent per-coin cash buckets.

    It matches the ten-coin backtest semantics: each symbol owns its own capital
    bucket and cannot borrow from another symbol's cash.
    """

    def __init__(
        self,
        state_path: str | Path,
        coins: list[str],
        initial_capital_usdt: float,
        per_coin_capital_usdt: float,
        fee_bps: float,
        slippage_bps: float,
        min_position_value_usdt: float = 2.0,
    ):
        self.coins = list(coins)
        self.initial_capital_usdt = float(initial_capital_usdt)
        self.per_coin_capital_usdt = float(per_coin_capital_usdt)
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.min_position_value_usdt = float(min_position_value_usdt)
        self.store = JsonStateStore(
            state_path,
            {
                "initial_capital_usdt": self.initial_capital_usdt,
                "cash_by_symbol": {coin: self.per_coin_capital_usdt for coin in self.coins},
                "position_qty_by_symbol": {coin: 0.0 for coin in self.coins},
                "last_signal_by_symbol": {coin: 0 for coin in self.coins},
                "processed_trade_dates": [],
                "trade_counter": 0,
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
        cash_before = float(self.state["cash_by_symbol"].get(coin, 0.0))
        qty_before = float(self.state["position_qty_by_symbol"].get(coin, 0.0))
        position_value_before = qty_before * price
        current_position = 1 if position_value_before >= self.min_position_value_usdt else 0
        if target_position == current_position:
            self.state["last_signal_by_symbol"][coin] = int(signal_row.get("model_score", 0))
            self.store.save()
            return None

        fee_rate = self.fee_bps / 10_000.0
        slippage_rate = self.slippage_bps / 10_000.0
        side = "buy" if target_position == 1 else "sell"
        self.state["trade_counter"] = int(self.state.get("trade_counter", 0)) + 1
        trade_id = f"{trade_date}-{coin}-{self.state['trade_counter']:06d}"

        if side == "buy":
            spendable = max(0.0, cash_before)
            if spendable < self.min_position_value_usdt:
                self.store.save()
                return None
            fill_price = price * (1.0 + slippage_rate)
            fee = spendable * fee_rate
            notional = max(0.0, spendable - fee)
            qty_delta = notional / fill_price if fill_price > 0 else 0.0
            cash_after = cash_before - spendable
            qty_after = qty_before + qty_delta
        else:
            if qty_before * price < self.min_position_value_usdt:
                self.store.save()
                return None
            fill_price = price * (1.0 - slippage_rate)
            notional = qty_before * fill_price
            fee = notional * fee_rate
            cash_after = cash_before + max(0.0, notional - fee)
            qty_after = 0.0

        slippage = ((fill_price - price) / price * 10_000.0) if side == "buy" else ((price - fill_price) / price * 10_000.0)
        self.state["cash_by_symbol"][coin] = cash_after
        self.state["position_qty_by_symbol"][coin] = qty_after
        self.state["last_signal_by_symbol"][coin] = int(signal_row.get("model_score", 0))
        self.store.save()

        row = {
            "trade_id": trade_id,
            "timestamp": timestamp,
            "datetime": dt,
            "trade_date": trade_date,
            "news_date": signal_row.get("news_date", ""),
            "coin": coin,
            "side": side,
            "price": fill_price,
            "reference_price": price,
            "actual_avg_price": fill_price,
            "qty": abs(qty_after - qty_before) if side == "buy" else qty_before,
            "requested_qty": 0.0 if side == "buy" else qty_before,
            "actual_qty": abs(qty_after - qty_before) if side == "buy" else qty_before,
            "notional_usdt": notional,
            "requested_notional_usdt": spendable if side == "buy" else position_value_before,
            "actual_notional_usdt": notional,
            "fee_usdt": fee,
            "fee_ccy": "USDT",
            "slippage_bps": slippage,
            "order_id": f"paper-{trade_id}",
            "client_order_id": f"paper-{trade_id}",
            "order_status": "filled",
            "fill_verified": True,
            "partial_fill": False,
            "signal_label": signal_row.get("model_label", ""),
            "signal_score": signal_row.get("model_score", 0),
            "final_score": signal_row.get("final_score", 0.0),
            "threshold": signal_row.get("threshold", 0.0),
            "cash_before": cash_before,
            "cash_after": cash_after,
            "position_qty_before": qty_before,
            "position_qty_after": qty_after,
            "position_value_after": qty_after * price,
            "reason": "positive_to_full_bucket" if side == "buy" else "negative_to_cash",
            "success": True,
        }
        self._record_audit(row, cash_before, cash_after, qty_before, qty_after)
        return row

    @staticmethod
    def write_records(records: list[dict[str, Any]], path: str | Path) -> None:
        if records:
            atomic_write_csv(pd.DataFrame(records), path)

    def _record_audit(
        self,
        row: dict[str, Any],
        cash_before: float,
        cash_after: float,
        qty_before: float,
        qty_after: float,
    ) -> None:
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
        order = dict(context)
        order.update(
            {
                "exchange": "paper",
                "account_mode": "paper",
                "trade_mode": "paper_spot",
                "inst_id": row.get("coin", ""),
                "side": row.get("side", ""),
                "ord_type": "market",
                "td_mode": "cash",
                "tgt_ccy": "quote_ccy" if row.get("side") == "buy" else "base_ccy",
                "requested_notional_usdt": row.get("requested_notional_usdt", 0.0),
                "requested_qty": row.get("requested_qty", 0.0),
                "submitted_sz": row.get("requested_notional_usdt", 0.0),
                "reference_price": row.get("reference_price", 0.0),
                "submit_ts": row.get("timestamp", 0.0),
                "ack_ts": row.get("timestamp", 0.0),
                "settle_ts": row.get("timestamp", 0.0),
                "order_id": row.get("order_id", ""),
                "client_order_id": row.get("client_order_id", ""),
                "order_status": "filled",
                "avg_fill_price": row.get("actual_avg_price", 0.0),
                "filled_qty": row.get("actual_qty", 0.0),
                "filled_notional_usdt": row.get("actual_notional_usdt", 0.0),
                "fee_usdt": row.get("fee_usdt", 0.0),
                "fee_ccy": row.get("fee_ccy", "USDT"),
                "slippage_bps": row.get("slippage_bps", 0.0),
                "fill_verified": True,
                "partial_fill": False,
                "success": True,
            }
        )
        fill = dict(context)
        fill.update(
            {
                "fill_index": 0,
                "exchange": "paper",
                "inst_id": row.get("coin", ""),
                "order_id": row.get("order_id", ""),
                "trade_id_exchange": row.get("order_id", ""),
                "fill_time": row.get("timestamp", 0.0),
                "side": row.get("side", ""),
                "fill_price": row.get("actual_avg_price", 0.0),
                "fill_qty": row.get("actual_qty", 0.0),
                "fill_notional_usdt": row.get("actual_notional_usdt", 0.0),
                "fee": row.get("fee_usdt", 0.0),
                "fee_ccy": "USDT",
            }
        )
        snapshots = []
        for phase, cash, qty in [("before", cash_before, qty_before), ("after", cash_after, qty_after)]:
            snap = dict(context)
            snap.update(
                {
                    "snapshot_phase": phase,
                    "account_mode": "paper",
                    "trade_mode": "paper_spot",
                    "usdt_available": cash,
                    "holdings_value_usdt": qty * float(row.get("reference_price", 0.0) or 0.0),
                    "estimated_total_equity_usdt": cash + qty * float(row.get("reference_price", 0.0) or 0.0),
                    "holdings_json": {row.get("coin", ""): qty},
                }
            )
            snapshots.append(snap)
        self._audit_records["orders"].append(order)
        self._audit_records["fills"].append(fill)
        self._audit_records["account_snapshots"].extend(snapshots)
