"""Adapters that route legacy brokers through the common trading interface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from quant_bench.trading.interface import (
    AccountSnapshot,
    ExecutionReport,
    OrderAction,
    OrderRequest,
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class RebalanceBrokerBackend:
    """Adapt a broker whose public operation is ``rebalance_coin``."""

    def __init__(self, broker: Any, *, backend_id: str, execution_mode: str) -> None:
        self.broker = broker
        self.backend_id = backend_id
        self.execution_mode = execution_mode

    def account_snapshot(self, request: OrderRequest) -> AccountSnapshot:
        raw = self.broker.equity_snapshot(
            {request.instrument_id: float(request.reference_price or 0.0)}
        )
        return AccountSnapshot.from_mapping(
            raw,
            backend_id=self.backend_id,
            execution_mode=self.execution_mode,
            instrument_id=request.instrument_id,
            reference_price=request.reference_price,
        )

    def submit_order(self, request: OrderRequest) -> ExecutionReport:
        metadata = request.metadata
        target_position = int(metadata.get("target_position", 0) or 0)
        payload = self.broker.rebalance_coin(
            request.instrument_id,
            target_position,
            float(request.reference_price or 0.0),
            str(metadata.get("trade_date", "")),
            dict(metadata.get("signal_row") or {}),
            timestamp=metadata.get("timestamp"),
        )
        if payload is None:
            payload = {
                "status": "skipped",
                "submitted": False,
                "executed_action": "hold",
            }
        return ExecutionReport.from_mapping(
            request,
            payload,
            backend_id=self.backend_id,
            execution_mode=self.execution_mode,
        )


class DecisionMappingBackend:
    """Adapt FinMem's symbol-keyed decision client without changing its logs."""

    def __init__(self, client: Any, *, backend_id: str, execution_mode: str) -> None:
        self.client = client
        self.backend_id = backend_id
        self.execution_mode = execution_mode

    def account_snapshot(self, request: OrderRequest) -> AccountSnapshot:
        raw = self.client.get_account_snapshot(instId=request.instrument_id, ccy="USDT")
        normalized = self._normalize_snapshot(raw, request)
        return AccountSnapshot.from_mapping(
            normalized,
            backend_id=self.backend_id,
            execution_mode=self.execution_mode,
            instrument_id=request.instrument_id,
            reference_price=request.reference_price,
        )

    @staticmethod
    def _normalize_snapshot(raw: Mapping[str, Any], request: OrderRequest) -> dict[str, Any]:
        payload = dict(raw)
        balance = payload.get("balance")
        if isinstance(balance, Mapping):
            payload["total_eq"] = balance.get("total_eq", 0.0)
            payload["avail_bal"] = balance.get(
                "avail_bal", balance.get("available", balance.get("cash_bal", 0.0))
            )
            payload["frozen_bal"] = balance.get("frozen_bal", balance.get("frozen", 0.0))

        position_wrapper = payload.get("position")
        position = (
            position_wrapper.get("position")
            if isinstance(position_wrapper, Mapping)
            else None
        )
        if isinstance(position, Mapping):
            payload["positions"] = {
                request.instrument_id: _as_float(
                    position.get("pos", position.get("position_qty")), 0.0
                )
            }
        payload["raw_snapshot"] = dict(raw)
        return payload

    @staticmethod
    def _symbol_payload(raw: Mapping[str, Any], request: OrderRequest) -> dict[str, Any]:
        symbol = request.instrument_id.split("-")[0]
        candidate = raw.get(symbol)
        if isinstance(candidate, Mapping):
            return dict(candidate)
        if len(raw) == 1:
            only_value = next(iter(raw.values()))
            if isinstance(only_value, Mapping):
                return dict(only_value)
        return dict(raw)

    @staticmethod
    def _first_order_row(payload: Mapping[str, Any]) -> dict[str, Any]:
        data = payload.get("data")
        if isinstance(data, list) and data and isinstance(data[0], Mapping):
            return dict(data[0])
        return {}

    @staticmethod
    def _order_detail_row(audit: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
        detail = audit.get("order_detail")
        if not isinstance(detail, Mapping):
            return {}, detail
        raw = detail.get("raw")
        if isinstance(raw, Mapping):
            data = raw.get("data")
            if isinstance(data, list) and data and isinstance(data[0], Mapping):
                return dict(data[0]), raw
        return {}, raw if raw is not None else dict(detail)

    def submit_order(self, request: OrderRequest) -> ExecutionReport:
        legacy_decision = request.metadata.get("legacy_decision")
        if not isinstance(legacy_decision, Mapping):
            raise ValueError("FinMem orders require metadata.legacy_decision")

        raw_result = self.client.execute_decision(dict(legacy_decision))
        if not isinstance(raw_result, Mapping):
            raise TypeError("FinMem execute_decision must return a mapping")

        result = self._symbol_payload(raw_result, request)
        audit = result.get("execution_audit")
        audit = dict(audit) if isinstance(audit, Mapping) else {}
        response_row = self._first_order_row(result)
        detail_row, raw_fill_response = self._order_detail_row(audit)

        response_code = result.get("code")
        row_code = response_row.get("sCode")
        response_ok = str(response_code) == "0" and (
            row_code in (None, "") or str(row_code) == "0"
        )
        state = str(detail_row.get("state", "")).lower()
        filled_quantity = _as_float(
            detail_row.get("accFillSz", detail_row.get("fillSz")), 0.0
        )
        average_price = _as_float(
            detail_row.get("avgPx", detail_row.get("fillPx")), 0.0
        )

        result_status = str(result.get("status", "")).lower()
        skipped = request.action is OrderAction.HOLD or result_status in {
            "waiting",
            "hold",
            "skip",
            "skipped",
        }
        failed = result_status in {"error", "failed", "invalid_signal", "invalid_decision"}
        submitted = response_ok and not skipped and not failed
        if failed or (response_code not in (None, "") and not response_ok):
            status = "failed"
        elif skipped:
            status = "skipped"
        elif state in {"filled", "fully_filled"} or (submitted and filled_quantity > 0):
            status = "filled"
        elif state in {"partially_filled", "partially-filled"}:
            status = "partial"
        elif submitted:
            status = "submitted"
        else:
            status = "skipped"

        flattened = {
            "status": status,
            "submitted": submitted,
            "executed_action": request.action.value if submitted else "hold",
            "order_id": response_row.get("ordId") or detail_row.get("ordId"),
            "client_order_id": response_row.get("clOrdId") or detail_row.get("clOrdId"),
            "exchange_response_code": response_code if response_code is not None else row_code,
            "exchange_response_msg": result.get("msg") or response_row.get("sMsg"),
            "actual_avg_fill_price": average_price or None,
            "filled_quantity": filled_quantity,
            "filled_notional": (
                filled_quantity * average_price if filled_quantity > 0 and average_price > 0 else 0.0
            ),
            "fee_amount": _as_float(detail_row.get("fee"), 0.0) or None,
            "fee_currency": detail_row.get("feeCcy"),
            "raw_order_response": result,
            "raw_fill_response": raw_fill_response,
            "error_message": result.get("error") or result.get("reason"),
        }
        report = ExecutionReport.from_mapping(
            request,
            flattened,
            backend_id=self.backend_id,
            execution_mode=self.execution_mode,
        )
        return report.model_copy(update={"legacy_payload": dict(raw_result)})


class PaperLedgerBackend:
    """Adapt FinMem's deterministic paper ledger to the common interface."""

    def __init__(self, ledger: Any, *, backend_id: str = "finmem-paper") -> None:
        self.ledger = ledger
        self.backend_id = backend_id
        self.execution_mode = "paper"

    def account_snapshot(self, request: OrderRequest) -> AccountSnapshot:
        state = dict(self.ledger.state)
        state["total_equity"] = state.get("equity_usdt", 0.0)
        state["available_cash"] = state.get("cash_usdt", 0.0)
        state["positions"] = {
            request.instrument_id: _as_float(state.get("position_qty"), 0.0)
            * _as_float(state.get("position_sign"), 0.0)
        }
        return AccountSnapshot.from_mapping(
            state,
            backend_id=self.backend_id,
            execution_mode=self.execution_mode,
            instrument_id=request.instrument_id,
            reference_price=request.reference_price,
        )

    def submit_order(self, request: OrderRequest) -> ExecutionReport:
        target_position = request.metadata.get("target_position", 0.0)
        price = float(request.reference_price or 0.0)
        payload = self.ledger.rebalance(target_position, price)
        changed = bool(payload.get("changed"))
        previous_sign = int(payload.get("previous_sign", 0) or 0)
        target_sign = int(payload.get("target_sign", 0) or 0)
        state = dict(payload.get("state") or {})
        turnover = abs(target_sign - previous_sign)
        equity = _as_float(state.get("equity_usdt"), 0.0)
        fee = _as_float(payload.get("fee_usdt"), 0.0)
        gross_equity = equity + fee
        canonical = {
            "status": "filled" if changed else "skipped",
            "submitted": changed,
            "executed_action": request.action.value if changed else "hold",
            "order_id": f"paper-{request.request_id}" if changed else None,
            "actual_avg_fill_price": price if changed else None,
            "filled_quantity": gross_equity * turnover / price if changed and price > 0 else 0.0,
            "filled_notional": gross_equity * turnover if changed else 0.0,
            "fee_amount": fee,
            "fee_currency": "USDT",
            "raw_order_response": payload,
        }
        report = ExecutionReport.from_mapping(
            request,
            canonical,
            backend_id=self.backend_id,
            execution_mode=self.execution_mode,
        )
        return report.model_copy(update={"legacy_payload": dict(payload)})
