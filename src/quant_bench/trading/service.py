"""Common trading execution service."""

from __future__ import annotations

from collections.abc import Mapping

from quant_bench.trading.interface import (
    AccountSnapshot,
    ExecutionReport,
    OrderRequest,
    TradingBackend,
    TradingCycleResult,
)


class TradingService:
    """Execute an order while recording common account and fill data."""

    def __init__(self, backend: TradingBackend) -> None:
        self.backend = backend

    def _snapshot(self, request: OrderRequest, warnings: list[str]) -> AccountSnapshot | None:
        try:
            snapshot = self.backend.account_snapshot(request)
            if isinstance(snapshot, AccountSnapshot):
                return snapshot
            if not isinstance(snapshot, Mapping):
                raise TypeError("account_snapshot must return AccountSnapshot or a mapping")
            return AccountSnapshot.from_mapping(
                snapshot,
                backend_id=self.backend.backend_id,
                execution_mode=self.backend.execution_mode,
                instrument_id=request.instrument_id,
                reference_price=request.reference_price,
            )
        except Exception as exc:
            warnings.append(f"account snapshot failed: {type(exc).__name__}: {exc}")
            return None

    def execute(self, request: OrderRequest) -> TradingCycleResult:
        warnings: list[str] = []
        before = self._snapshot(request, warnings)
        try:
            result = self.backend.submit_order(request)
            if isinstance(result, ExecutionReport):
                execution = result
            elif isinstance(result, Mapping):
                execution = ExecutionReport.from_mapping(
                    request,
                    result,
                    backend_id=self.backend.backend_id,
                    execution_mode=self.backend.execution_mode,
                )
            else:
                raise TypeError("submit_order must return ExecutionReport or a mapping")
        except Exception as exc:
            execution = ExecutionReport.failed(
                request,
                backend_id=self.backend.backend_id,
                execution_mode=self.backend.execution_mode,
                error=exc,
            )
        after = self._snapshot(request, warnings)
        return TradingCycleResult(
            request=request,
            account_before=before,
            execution=execution,
            account_after=after,
            warnings=warnings,
        )
