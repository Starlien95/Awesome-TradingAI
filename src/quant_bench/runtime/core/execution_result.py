from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
        return default if math.isnan(out) or math.isinf(out) else out
    except (TypeError, ValueError):
        return default


def _json_dumps(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


@dataclass
class TradeExecutionResult:
    success: bool = False
    status: str = "failed"
    exchange: str = "okx"
    account_mode: str = ""
    trade_mode: str = "spot"
    inst_id: str = ""
    side: str = ""
    action: str = ""
    ord_type: str = "market"
    td_mode: str = ""
    tgt_ccy: str = ""
    requested_notional_usdt: float = 0.0
    requested_qty: float = 0.0
    submitted_sz: str = ""
    reference_price: float = 0.0
    submit_ts: float = 0.0
    ack_ts: float = 0.0
    settle_ts: float = 0.0
    ord_id: str = ""
    cl_ord_id: str = ""
    okx_code: str = ""
    okx_msg: str = ""
    okx_s_code: str = ""
    okx_s_msg: str = ""
    avg_fill_price: float = 0.0
    filled_qty: float = 0.0
    filled_notional_usdt: float = 0.0
    fee: float = 0.0
    fee_ccy: str = ""
    fee_usdt: float = 0.0
    slippage_bps: float = 0.0
    fill_verified: bool = False
    partial_fill: bool = False
    raw_response: dict[str, Any] = field(default_factory=dict)
    order_response: dict[str, Any] = field(default_factory=dict)
    fills_response: dict[str, Any] = field(default_factory=dict)
    fills: list[dict[str, Any]] = field(default_factory=list)
    error_type: str = ""
    error_message: str = ""

    def __bool__(self) -> bool:
        return bool(self.success)

    @classmethod
    def failure(
        cls,
        *,
        inst_id: str,
        side: str,
        action: str,
        requested_notional_usdt: float,
        reference_price: float,
        trade_mode: str = "spot",
        td_mode: str = "",
        tgt_ccy: str = "",
        submitted_sz: str = "",
        requested_qty: float = 0.0,
        error_type: str = "execution_error",
        error_message: str = "",
        raw_response: dict[str, Any] | None = None,
        submit_ts: float | None = None,
        ack_ts: float | None = None,
    ) -> TradeExecutionResult:
        return cls(
            success=False,
            status="failed",
            trade_mode=trade_mode,
            inst_id=inst_id,
            side=side,
            action=action,
            requested_notional_usdt=float(requested_notional_usdt or 0.0),
            requested_qty=float(requested_qty or 0.0),
            submitted_sz=str(submitted_sz or ""),
            reference_price=float(reference_price or 0.0),
            submit_ts=float(submit_ts or time.time()),
            ack_ts=float(ack_ts or 0.0),
            td_mode=td_mode,
            tgt_ccy=tgt_ccy,
            raw_response=raw_response or {},
            okx_code=str((raw_response or {}).get("code", "")),
            okx_msg=str((raw_response or {}).get("msg", "")),
            error_type=error_type,
            error_message=error_message,
        )

    def update_from_order_response(self, order_response: dict[str, Any]) -> None:
        self.order_response = order_response or {}
        data = (self.order_response.get("data") or [{}])[0] if isinstance(self.order_response, dict) else {}
        if not isinstance(data, dict):
            return
        state = str(data.get("state", "") or "")
        if state:
            self.status = state
        avg_px = _to_float(data.get("avgPx"), 0.0)
        acc_fill_sz = _to_float(data.get("accFillSz"), 0.0)
        fee = abs(_to_float(data.get("fee"), 0.0))
        fee_ccy = str(data.get("feeCcy", "") or "")
        if avg_px > 0:
            self.avg_fill_price = avg_px
        if acc_fill_sz > 0:
            self.filled_qty = acc_fill_sz
        if self.avg_fill_price > 0 and self.filled_qty > 0:
            self.filled_notional_usdt = self.avg_fill_price * self.filled_qty
            self.fill_verified = True
        if fee > 0:
            self.fee = fee
            self.fee_ccy = fee_ccy
            self.fee_usdt = fee if fee_ccy.upper() == "USDT" else fee * max(self.avg_fill_price, self.reference_price)

    def update_from_fills_response(self, fills_response: dict[str, Any]) -> None:
        self.fills_response = fills_response or {}
        rows = self.fills_response.get("data") if isinstance(self.fills_response, dict) else []
        if not isinstance(rows, list) or not rows:
            return
        parsed: list[dict[str, Any]] = []
        notional = 0.0
        qty = 0.0
        fee_usdt = 0.0
        fee_parts: list[str] = []
        fill_times: list[float] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            px = _to_float(row.get("fillPx"), 0.0)
            sz = _to_float(row.get("fillSz"), 0.0)
            fee = abs(_to_float(row.get("fee"), 0.0))
            fee_ccy = str(row.get("feeCcy", "") or "")
            fill_ts = _to_float(row.get("fillTime"), 0.0)
            if fill_ts > 10_000_000_000:
                fill_ts = fill_ts / 1000.0
            if fill_ts > 0:
                fill_times.append(fill_ts)
            if px > 0 and sz > 0:
                qty += sz
                notional += px * sz
            if fee > 0:
                fee_usdt += fee if fee_ccy.upper() == "USDT" else fee * max(px, self.reference_price)
                fee_parts.append(f"{fee:g} {fee_ccy}".strip())
            parsed.append(row)
        if qty > 0 and notional > 0:
            self.fills = parsed
            existing_qty = float(self.filled_qty or 0.0)
            fills_cover_order = existing_qty <= 0 or qty >= existing_qty * 0.999
            if fills_cover_order:
                self.filled_qty = qty
                self.filled_notional_usdt = notional
                self.avg_fill_price = notional / qty
            self.fill_verified = True
            self.status = "filled" if self.status in {"", "submitted", "accepted", "live"} else self.status
        if fee_usdt > 0 and (self.filled_qty <= 0 or qty >= float(self.filled_qty or 0.0) * 0.999):
            self.fee_usdt = fee_usdt
            self.fee = fee_usdt
            self.fee_ccy = "MIXED" if len(set(fee_parts)) > 1 else (fee_parts[0].split()[-1] if fee_parts else "")
        if fill_times:
            self.settle_ts = max(fill_times)
        self.recompute_derived()

    def recompute_derived(self) -> None:
        if self.reference_price > 0 and self.avg_fill_price > 0:
            if self.side == "sell":
                self.slippage_bps = (self.reference_price - self.avg_fill_price) / self.reference_price * 10_000.0
            else:
                self.slippage_bps = (self.avg_fill_price - self.reference_price) / self.reference_price * 10_000.0
        if self.requested_qty > 0 and self.filled_qty > 0:
            self.partial_fill = self.filled_qty < self.requested_qty * 0.999
        if self.success and not self.status:
            self.status = "filled" if self.fill_verified else "submitted_unverified"

    def to_order_row(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        row = dict(context or {})
        row.update(
            {
                "exchange": self.exchange,
                "account_mode": self.account_mode,
                "trade_mode": self.trade_mode,
                "inst_id": self.inst_id,
                "side": self.side,
                "ord_type": self.ord_type,
                "td_mode": self.td_mode,
                "tgt_ccy": self.tgt_ccy,
                "requested_notional_usdt": self.requested_notional_usdt,
                "requested_qty": self.requested_qty,
                "submitted_sz": self.submitted_sz,
                "reference_price": self.reference_price,
                "submit_ts": self.submit_ts,
                "ack_ts": self.ack_ts,
                "settle_ts": self.settle_ts,
                "order_id": self.ord_id,
                "client_order_id": self.cl_ord_id,
                "order_status": self.status,
                "okx_code": self.okx_code,
                "okx_msg": self.okx_msg,
                "okx_s_code": self.okx_s_code,
                "okx_s_msg": self.okx_s_msg,
                "avg_fill_price": self.avg_fill_price,
                "filled_qty": self.filled_qty,
                "filled_notional_usdt": self.filled_notional_usdt,
                "fee_usdt": self.fee_usdt,
                "fee_ccy": self.fee_ccy,
                "slippage_bps": self.slippage_bps,
                "fill_verified": self.fill_verified,
                "partial_fill": self.partial_fill,
                "success": self.success,
                "error_type": self.error_type,
                "error_message": self.error_message,
            }
        )
        return row

    def to_fill_rows(self, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        base = dict(context or {})
        if not self.fills and self.fill_verified and self.filled_qty > 0 and self.avg_fill_price > 0:
            row = dict(base)
            row.update(
                {
                    "fill_index": 0,
                    "exchange": self.exchange,
                    "inst_id": self.inst_id,
                    "order_id": self.ord_id,
                    "trade_id_exchange": "",
                    "fill_time": self.settle_ts,
                    "side": self.side,
                    "fill_price": self.avg_fill_price,
                    "fill_qty": self.filled_qty,
                    "fill_notional_usdt": self.filled_notional_usdt,
                    "fee": self.fee,
                    "fee_ccy": self.fee_ccy,
                    "raw_fill_json": _json_dumps(
                        {
                            "source": "aggregate_order_fill",
                            "fill_verified": self.fill_verified,
                            "fills_query_empty": True,
                        }
                    ),
                }
            )
            return [row]
        for index, fill in enumerate(self.fills):
            px = _to_float(fill.get("fillPx"), 0.0)
            sz = _to_float(fill.get("fillSz"), 0.0)
            fee = _to_float(fill.get("fee"), 0.0)
            row = dict(base)
            row.update(
                {
                    "fill_index": index,
                    "exchange": self.exchange,
                    "inst_id": self.inst_id,
                    "order_id": self.ord_id,
                    "trade_id_exchange": fill.get("tradeId", ""),
                    "fill_time": fill.get("fillTime", ""),
                    "side": fill.get("side", self.side),
                    "fill_price": px,
                    "fill_qty": sz,
                    "fill_notional_usdt": px * sz,
                    "fee": fee,
                    "fee_ccy": fill.get("feeCcy", ""),
                    "raw_fill_json": _json_dumps(fill),
                }
            )
            rows.append(row)
        return rows

    def to_event_records(self, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        base = dict(context or {})
        events = []
        for event_type, payload in [
            ("place_order_response", self.raw_response),
            ("order_query_response", self.order_response),
            ("fills_query_response", self.fills_response),
        ]:
            if payload:
                row = dict(base)
                row.update(
                    {
                        "event_type": event_type,
                        "event_ts": time.time(),
                        "order_id": self.ord_id,
                        "payload": payload,
                    }
                )
                events.append(row)
        return events
