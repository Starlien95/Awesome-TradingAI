"""Deterministic long/flat/short paper ledger used by FinMem."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_bench.runtime.core.state_store import JsonStateStore


def target_sign(value: Any, *, allow_short: bool = True) -> int:
    number = float(value or 0.0)
    if number > 0:
        return 1
    if number < 0 and allow_short:
        return -1
    return 0


class PaperLedger:
    """A symbol-scoped paper ledger with atomic state persistence."""

    def __init__(
        self,
        path: str | Path,
        *,
        initial_capital_usdt: float = 10_000.0,
        fee_bps: float = 0.0,
        allow_short: bool = True,
    ) -> None:
        if initial_capital_usdt <= 0:
            raise ValueError("initial_capital_usdt must be positive")
        if fee_bps < 0:
            raise ValueError("fee_bps cannot be negative")
        self.initial_capital_usdt = float(initial_capital_usdt)
        self.fee_bps = float(fee_bps)
        self.allow_short = bool(allow_short)
        self.store = JsonStateStore(
            path,
            {
                "initial_capital_usdt": self.initial_capital_usdt,
                "equity_usdt": self.initial_capital_usdt,
                "cash_usdt": self.initial_capital_usdt,
                "position_sign": 0,
                "position_state": "flat",
                "last_price": 0.0,
                "entry_price": 0.0,
                "position_qty": 0.0,
                "trade_count": 0,
                "fee_paid_usdt": 0.0,
            },
        )
        self._normalize()

    @property
    def state(self) -> dict[str, Any]:
        return self.store.state

    @staticmethod
    def _state_name(sign: int) -> str:
        return "long" if sign > 0 else "short" if sign < 0 else "flat"

    def _normalize(self) -> None:
        sign = target_sign(self.state.get("position_sign", 0), allow_short=self.allow_short)
        self.state["position_sign"] = sign
        self.state["position_state"] = self._state_name(sign)
        self.store.save()

    def mark_to_market(self, price: float) -> dict[str, Any]:
        price = float(price)
        if price <= 0:
            raise ValueError("price must be positive")
        previous = float(self.state.get("last_price", 0.0) or 0.0)
        sign = int(self.state.get("position_sign", 0) or 0)
        equity = float(self.state.get("equity_usdt", self.initial_capital_usdt))
        price_return = price / previous - 1.0 if previous > 0 else 0.0
        equity = max(0.0, equity * (1.0 + sign * price_return))
        self.state.update(
            {
                "equity_usdt": equity,
                "cash_usdt": equity if sign == 0 else 0.0,
                "last_price": price,
                "last_price_return": price_return,
                "position_qty": equity / price if sign else 0.0,
                "position_state": self._state_name(sign),
            }
        )
        self.store.save()
        return dict(self.state)

    def rebalance(self, target_position: float, price: float) -> dict[str, Any]:
        before = self.mark_to_market(price)
        current = int(before["position_sign"])
        desired = target_sign(target_position, allow_short=self.allow_short)
        turnover = abs(desired - current)
        fee = before["equity_usdt"] * turnover * self.fee_bps / 10_000.0
        equity = max(0.0, float(before["equity_usdt"]) - fee)
        self.state.update(
            {
                "equity_usdt": equity,
                "cash_usdt": equity if desired == 0 else 0.0,
                "position_sign": desired,
                "position_state": self._state_name(desired),
                "position_qty": equity / float(price) if desired else 0.0,
                "entry_price": float(price) if desired != current and desired else 0.0,
                "trade_count": int(self.state.get("trade_count", 0)) + int(desired != current),
                "fee_paid_usdt": float(self.state.get("fee_paid_usdt", 0.0)) + fee,
            }
        )
        self.store.save()
        return {
            "changed": desired != current,
            "previous_sign": current,
            "target_sign": desired,
            "fee_usdt": fee,
            "state": dict(self.state),
        }
