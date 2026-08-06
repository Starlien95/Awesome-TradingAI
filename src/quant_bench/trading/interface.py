"""Public data structures for account and order execution boundaries."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Protocol, cast, runtime_checkable

from pydantic import Field, field_validator, model_validator

from quant_bench.contracts.models import StrictModel


class MarketType(str, Enum):
    """Supported market types."""

    SPOT = "spot"
    SWAP = "swap"


class OrderAction(str, Enum):
    """Account action requested by a strategy."""

    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    HOLD = "hold"


class OrderType(str, Enum):
    """Order type sent to a trading backend."""

    MARKET = "market"
    LIMIT = "limit"


class OrderRequest(StrictModel):
    """The only order request strategies may send to execution code."""

    schema_version: Literal["1"] = "1"
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    strategy_id: str
    decision_id: str | None = None
    instrument_id: str
    market_type: MarketType
    action: OrderAction
    order_type: OrderType = OrderType.MARKET
    quantity: float | None = Field(default=None, gt=0)
    notional_usdt: float | None = Field(default=None, gt=0)
    allow_backend_sizing: bool = False
    reference_price: float | None = Field(default=None, gt=0)
    limit_price: float | None = Field(default=None, gt=0)
    margin_mode: str | None = None
    position_side: str | None = None
    reduce_only: bool = False
    leverage: int | None = Field(default=None, ge=1)
    target_currency: str | None = None
    client_order_id: str | None = None
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("strategy_id", "instrument_id")
    @classmethod
    def require_non_empty_token(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("instrument_id")
    @classmethod
    def normalize_instrument_id(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def validate_order(self) -> OrderRequest:
        if self.market_type is MarketType.SWAP and not self.instrument_id.endswith("-SWAP"):
            raise ValueError("swap instrument_id must end with -SWAP")
        if self.market_type is MarketType.SPOT and self.instrument_id.endswith("-SWAP"):
            raise ValueError("spot instrument_id must not end with -SWAP")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit orders require limit_price")
        if self.action in {OrderAction.BUY, OrderAction.SELL}:
            sizing_values = [self.quantity is not None, self.notional_usdt is not None]
            if sum(sizing_values) > 1:
                raise ValueError("provide only one of quantity or notional_usdt")
            if not any(sizing_values) and not self.allow_backend_sizing:
                raise ValueError(
                    "buy and sell orders require quantity, notional_usdt, or allow_backend_sizing"
                )
        return self


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class AccountSnapshot(StrictModel):
    """Account state captured before or after an execution attempt."""

    schema_version: Literal["1"] = "1"
    backend_id: str
    execution_mode: str
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_equity: float = 0.0
    available_cash: float = 0.0
    frozen_cash: float = 0.0
    positions: dict[str, float] = Field(default_factory=dict)
    reference_prices: dict[str, float] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        backend_id: str,
        execution_mode: str,
        instrument_id: str | None = None,
        reference_price: float | None = None,
    ) -> AccountSnapshot:
        raw = dict(payload)
        total_equity = _safe_float(
            raw.get(
                "total_equity",
                raw.get(
                    "total_equity_usdt",
                    raw.get(
                        "estimated_total_equity_usdt",
                        raw.get("total_eq", raw.get("value", raw.get("strategy_equity", 0.0))),
                    ),
                ),
            ),
            0.0,
        )
        available_cash = _safe_float(
            raw.get(
                "available_cash",
                raw.get(
                    "usdt_available",
                    raw.get("cash", raw.get("cash_total_usdt", raw.get("avail_bal", 0.0))),
                ),
            ),
            0.0,
        )
        positions_payload = raw.get("positions", raw.get("holdings_json"))
        if isinstance(positions_payload, Mapping):
            positions = {
                str(symbol): float(_safe_float(quantity, 0.0) or 0.0)
                for symbol, quantity in positions_payload.items()
            }
        elif instrument_id and "position" in raw:
            positions = {instrument_id: float(_safe_float(raw.get("position"), 0.0) or 0.0)}
        else:
            positions = {}
        prices = {instrument_id: float(reference_price)} if instrument_id and reference_price else {}
        return cls(
            backend_id=backend_id,
            execution_mode=execution_mode,
            total_equity=float(total_equity or 0.0),
            available_cash=float(available_cash or 0.0),
            frozen_cash=float(
                _safe_float(raw.get("frozen_cash", raw.get("frozen_bal", 0.0)), 0.0) or 0.0
            ),
            positions=positions,
            reference_prices=prices,
            raw=raw,
        )


class ExecutionReport(StrictModel):
    """The common result returned by every trading backend."""

    schema_version: Literal["1"] = "1"
    request_id: str
    backend_id: str
    execution_mode: str
    instrument_id: str
    market_type: MarketType
    requested_action: OrderAction
    executed_action: OrderAction = OrderAction.HOLD
    submitted: bool = False
    status: Literal["skipped", "failed", "submitted", "partial", "filled"] = "skipped"
    order_id: str | None = None
    client_order_id: str | None = None
    exchange_response_code: str | None = None
    exchange_response_message: str | None = None
    expected_price: float | None = None
    estimated_price: float | None = None
    actual_average_fill_price: float | None = None
    filled_quantity: float = 0.0
    filled_notional: float = 0.0
    fee_amount: float | None = None
    fee_currency: str | None = None
    slippage_bps: float | None = None
    raw_order_response: Any = None
    raw_fill_response: Any = None
    error_type: str | None = None
    error_message: str | None = None
    legacy_payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def successful(self) -> bool:
        return self.submitted and self.status in {"submitted", "partial", "filled"}

    def __bool__(self) -> bool:
        return self.successful

    @property
    def avg_fill_price(self) -> float:
        return float(self.actual_average_fill_price or 0.0)

    @property
    def filled_notional_usdt(self) -> float:
        return self.filled_notional

    @property
    def fee_usdt(self) -> float:
        return float(self.fee_amount or 0.0)

    @property
    def fee_ccy(self) -> str:
        return self.fee_currency or ""

    @property
    def ord_id(self) -> str:
        return self.order_id or ""

    @property
    def cl_ord_id(self) -> str:
        return self.client_order_id or ""

    @property
    def fill_verified(self) -> bool:
        return self.status in {"partial", "filled"} and self.filled_quantity > 0

    @property
    def partial_fill(self) -> bool:
        return self.status == "partial"

    @property
    def okx_code(self) -> str:
        return self.exchange_response_code or ""

    @property
    def okx_msg(self) -> str:
        return self.exchange_response_message or ""

    @property
    def okx_s_code(self) -> str:
        return self.exchange_response_code or ""

    @property
    def okx_s_msg(self) -> str:
        return self.exchange_response_message or ""

    @classmethod
    def from_mapping(
        cls,
        request: OrderRequest,
        payload: Mapping[str, Any],
        *,
        backend_id: str,
        execution_mode: str,
    ) -> ExecutionReport:
        raw = dict(payload)
        status = str(
            raw.get("execution_status", raw.get("order_status", raw.get("status", "skipped")))
            or "skipped"
        ).lower()
        status = {
            "waiting": "skipped",
            "hold": "skipped",
            "success": "filled",
            "completed": "filled",
            "partially_filled": "partial",
            "accepted": "submitted",
            "live": "submitted",
            "error": "failed",
            "rejected": "failed",
        }.get(status, status)
        if status not in {"skipped", "failed", "submitted", "partial", "filled"}:
            status = "failed" if raw.get("error") else "submitted" if raw.get("submitted") else "skipped"
        normalized_status = cast(
            Literal["skipped", "failed", "submitted", "partial", "filled"], status
        )
        submitted = bool(raw.get("submitted", status in {"submitted", "partial", "filled"}))
        executed_raw = str(
            raw.get("executed_action", request.action.value if submitted else OrderAction.HOLD.value)
        ).lower()
        try:
            executed_action = OrderAction(executed_raw)
        except ValueError:
            executed_action = request.action if submitted else OrderAction.HOLD
        expected_price = _safe_float(raw.get("expected_price"), request.reference_price)
        actual_average_fill_price = _safe_float(
            raw.get(
                "actual_avg_fill_price",
                raw.get("actual_average_fill_price", raw.get("avg_fill_price")),
            ),
            None,
        )
        slippage_bps = _safe_float(raw.get("slippage_bps"), None)
        if (
            slippage_bps is None
            and expected_price is not None
            and expected_price > 0
            and actual_average_fill_price is not None
            and actual_average_fill_price > 0
        ):
            price_change_bps = (actual_average_fill_price / expected_price - 1.0) * 10_000.0
            if request.action is OrderAction.BUY:
                slippage_bps = price_change_bps
            elif request.action is OrderAction.SELL:
                slippage_bps = -price_change_bps
        return cls(
            request_id=request.request_id,
            backend_id=backend_id,
            execution_mode=execution_mode,
            instrument_id=request.instrument_id,
            market_type=request.market_type,
            requested_action=request.action,
            executed_action=executed_action,
            submitted=submitted,
            status=normalized_status,
            order_id=raw.get("order_id") or raw.get("ord_id"),
            client_order_id=raw.get("client_order_id") or raw.get("cl_ord_id"),
            exchange_response_code=(
                str(raw.get("exchange_response_code"))
                if raw.get("exchange_response_code") is not None
                else str(raw.get("code")) if raw.get("code") is not None else None
            ),
            exchange_response_message=raw.get("exchange_response_msg") or raw.get("msg"),
            expected_price=expected_price,
            estimated_price=_safe_float(
                raw.get("estimated_price", raw.get("reference_price")), request.reference_price
            ),
            actual_average_fill_price=actual_average_fill_price,
            filled_quantity=float(
                _safe_float(raw.get("filled_quantity", raw.get("filled_qty")), 0.0) or 0.0
            ),
            filled_notional=float(
                _safe_float(
                    raw.get("filled_notional", raw.get("filled_notional_usdt")), 0.0
                )
                or 0.0
            ),
            fee_amount=_safe_float(raw.get("fee_amount", raw.get("fee_usdt")), None),
            fee_currency=raw.get("fee_currency") or raw.get("fee_ccy"),
            slippage_bps=slippage_bps,
            raw_order_response=raw.get("raw_order_response", raw.get("response")),
            raw_fill_response=raw.get("raw_fill_response"),
            error_type=raw.get("error_type"),
            error_message=raw.get("error_message") or raw.get("error"),
            legacy_payload=raw,
        )

    @classmethod
    def failed(
        cls,
        request: OrderRequest,
        *,
        backend_id: str,
        execution_mode: str,
        error: Exception,
    ) -> ExecutionReport:
        return cls(
            request_id=request.request_id,
            backend_id=backend_id,
            execution_mode=execution_mode,
            instrument_id=request.instrument_id,
            market_type=request.market_type,
            requested_action=request.action,
            status="failed",
            expected_price=request.reference_price,
            estimated_price=request.reference_price,
            error_type=type(error).__name__,
            error_message=str(error),
        )

    def to_legacy_payload(self) -> dict[str, Any]:
        payload = dict(self.legacy_payload)
        payload.update(
            {
                "requested_action": self.requested_action.value.upper(),
                "executed_action": self.executed_action.value.upper(),
                "submitted": self.submitted,
                "status": self.status,
                "order_id": self.order_id,
                "client_order_id": self.client_order_id,
                "expected_price": self.expected_price,
                "estimated_price": self.estimated_price,
                "actual_avg_fill_price": self.actual_average_fill_price,
                "filled_quantity": self.filled_quantity,
                "filled_notional": self.filled_notional,
                "fee_amount": self.fee_amount,
                "fee_currency": self.fee_currency,
                "slippage_bps": self.slippage_bps,
                "raw_fill_response": self.raw_fill_response,
                "error_type": self.error_type,
                "error_message": self.error_message,
            }
        )
        return payload

    def to_order_row(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        row = dict(context or {})
        row.update(
            {
                "backend_id": self.backend_id,
                "execution_mode": self.execution_mode,
                "trade_mode": self.market_type.value,
                "inst_id": self.instrument_id,
                "side": self.executed_action.value,
                "requested_notional_usdt": self.legacy_payload.get("requested_notional_usdt", 0.0),
                "requested_qty": self.legacy_payload.get("requested_qty", 0.0),
                "reference_price": self.expected_price,
                "order_id": self.order_id,
                "client_order_id": self.client_order_id,
                "order_status": self.status,
                "exchange_response_code": self.exchange_response_code,
                "exchange_response_msg": self.exchange_response_message,
                "avg_fill_price": self.actual_average_fill_price,
                "filled_qty": self.filled_quantity,
                "filled_notional_usdt": self.filled_notional,
                "fee_usdt": self.fee_amount,
                "fee_ccy": self.fee_currency,
                "slippage_bps": self.slippage_bps,
                "fill_verified": self.fill_verified,
                "partial_fill": self.partial_fill,
                "success": self.successful,
                "error_type": self.error_type,
                "error_message": self.error_message,
            }
        )
        return row

    def to_fill_rows(self, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        base = dict(context or {})
        payload = self.raw_fill_response
        raw_rows = payload.get("data", []) if isinstance(payload, dict) else []
        rows: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_rows if isinstance(raw_rows, list) else []):
            if not isinstance(raw, Mapping):
                continue
            price = float(_safe_float(raw.get("fillPx"), 0.0) or 0.0)
            quantity = float(_safe_float(raw.get("fillSz"), 0.0) or 0.0)
            row = dict(base)
            row.update(
                {
                    "fill_index": index,
                    "backend_id": self.backend_id,
                    "inst_id": self.instrument_id,
                    "order_id": self.order_id,
                    "trade_id_exchange": raw.get("tradeId", ""),
                    "fill_time": raw.get("fillTime", ""),
                    "side": raw.get("side", self.executed_action.value),
                    "fill_price": price,
                    "fill_qty": quantity,
                    "fill_notional_usdt": price * quantity,
                    "fee": _safe_float(raw.get("fee"), 0.0),
                    "fee_ccy": raw.get("feeCcy", ""),
                    "raw_fill_json": dict(raw),
                }
            )
            rows.append(row)
        if not rows and self.fill_verified:
            row = dict(base)
            row.update(
                {
                    "fill_index": 0,
                    "backend_id": self.backend_id,
                    "inst_id": self.instrument_id,
                    "order_id": self.order_id,
                    "trade_id_exchange": "",
                    "fill_time": "",
                    "side": self.executed_action.value,
                    "fill_price": self.actual_average_fill_price,
                    "fill_qty": self.filled_quantity,
                    "fill_notional_usdt": self.filled_notional,
                    "fee": self.fee_amount,
                    "fee_ccy": self.fee_currency,
                    "raw_fill_json": {"source": "aggregate_execution_report"},
                }
            )
            rows.append(row)
        return rows

    def to_event_records(self, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        base = dict(context or {})
        records: list[dict[str, Any]] = []
        for event_type, payload in (
            ("place_order_response", self.raw_order_response),
            ("fills_query_response", self.raw_fill_response),
        ):
            if payload is None:
                continue
            row = dict(base)
            row.update(
                {
                    "event_type": event_type,
                    "event_ts": datetime.now(timezone.utc).timestamp(),
                    "order_id": self.order_id,
                    "payload": payload,
                }
            )
            records.append(row)
        return records


class TradingCycleResult(StrictModel):
    """One execution attempt and its before/after account state."""

    schema_version: Literal["1"] = "1"
    request: OrderRequest
    account_before: AccountSnapshot | None = None
    execution: ExecutionReport
    account_after: AccountSnapshot | None = None
    warnings: list[str] = Field(default_factory=list)


@runtime_checkable
class TradingBackend(Protocol):
    """Interface shared by paper, exchange-demo, and live accounts."""

    @property
    def backend_id(self) -> str: ...

    @property
    def execution_mode(self) -> str: ...

    def account_snapshot(self, request: OrderRequest) -> AccountSnapshot | Mapping[str, Any]: ...

    def submit_order(self, request: OrderRequest) -> ExecutionReport | Mapping[str, Any]: ...
