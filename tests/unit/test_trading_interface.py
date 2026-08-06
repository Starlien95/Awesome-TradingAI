from __future__ import annotations

from pathlib import Path

import pytest

from quant_bench.methods.finmem.paper import PaperLedger
from quant_bench.runtime.core.execution import TradeExecutor
from quant_bench.runtime.core.execution_result import TradeExecutionResult
from quant_bench.trading import (
    AccountSnapshot,
    DecisionMappingBackend,
    ExecutionReport,
    MarketType,
    OrderAction,
    OrderRequest,
    PaperLedgerBackend,
    RebalanceBrokerBackend,
    TradingService,
)


def test_order_request_requires_explicit_sizing_for_directional_orders() -> None:
    with pytest.raises(ValueError, match="quantity, notional_usdt, or allow_backend_sizing"):
        OrderRequest(
            strategy_id="example",
            instrument_id="BTC-USDT",
            market_type=MarketType.SPOT,
            action=OrderAction.BUY,
            reference_price=100_000.0,
        )

    request = OrderRequest(
        strategy_id="example",
        instrument_id="BTC-USDT",
        market_type=MarketType.SPOT,
        action=OrderAction.BUY,
        notional_usdt=250.0,
        reference_price=100_000.0,
    )

    assert request.instrument_id == "BTC-USDT"
    assert request.notional_usdt == 250.0


def test_order_request_rejects_market_and_instrument_mismatch() -> None:
    with pytest.raises(ValueError, match="must end with -SWAP"):
        OrderRequest(
            strategy_id="example",
            instrument_id="BTC-USDT",
            market_type=MarketType.SWAP,
            action=OrderAction.SELL,
            quantity=1.0,
            reference_price=100_000.0,
        )


def test_trading_service_normalizes_snapshots_and_legacy_execution_payloads() -> None:
    class ExampleBackend:
        backend_id = "example-paper"
        execution_mode = "paper"

        def __init__(self) -> None:
            self.snapshot_count = 0

        def account_snapshot(self, request: OrderRequest) -> dict[str, object]:
            self.snapshot_count += 1
            return {
                "total_eq": 1_000.0 + self.snapshot_count,
                "cash": 800.0,
                "position": 0.01 * self.snapshot_count,
            }

        def submit_order(self, request: OrderRequest) -> dict[str, object]:
            return {
                "status": "filled",
                "submitted": True,
                "executed_action": "BUY",
                "order_id": "paper-1",
                "actual_avg_fill_price": 100_100.0,
                "filled_quantity": 0.002,
                "filled_notional": 200.2,
                "fee_amount": 0.1,
                "fee_currency": "USDT",
            }

    request = OrderRequest(
        strategy_id="example",
        instrument_id="BTC-USDT",
        market_type=MarketType.SPOT,
        action=OrderAction.BUY,
        quantity=0.002,
        reference_price=100_000.0,
    )
    cycle = TradingService(ExampleBackend()).execute(request)

    assert isinstance(cycle.account_before, AccountSnapshot)
    assert cycle.account_before.total_equity == 1_001.0
    assert cycle.account_before.positions == {"BTC-USDT": 0.01}
    assert cycle.account_after.total_equity == 1_002.0
    assert isinstance(cycle.execution, ExecutionReport)
    assert cycle.execution.status == "filled"
    assert cycle.execution.order_id == "paper-1"
    assert cycle.execution.filled_notional == 200.2
    assert cycle.execution.successful is True


def test_trading_service_converts_backend_exceptions_into_failed_reports() -> None:
    class BrokenBackend:
        backend_id = "broken"
        execution_mode = "simulated"

        def account_snapshot(self, request: OrderRequest) -> dict[str, float]:
            return {"total_equity": 1_000.0, "available_cash": 1_000.0}

        def submit_order(self, request: OrderRequest) -> ExecutionReport:
            raise RuntimeError("exchange unavailable")

    request = OrderRequest(
        strategy_id="example",
        instrument_id="BTC-USDT-SWAP",
        market_type=MarketType.SWAP,
        action=OrderAction.SELL,
        quantity=1.0,
        reference_price=100_000.0,
    )
    cycle = TradingService(BrokenBackend()).execute(request)

    assert cycle.execution.status == "failed"
    assert cycle.execution.submitted is False
    assert cycle.execution.error_type == "RuntimeError"
    assert cycle.execution.error_message == "exchange unavailable"


def test_runtime_trade_executor_uses_the_common_order_interface() -> None:
    executor = TradeExecutor.__new__(TradeExecutor)
    executor.identifier = "runtime-model"
    executor._okx_flag = "1"
    executor.trade_mode = "spot"
    captured: dict[str, object] = {}

    def fake_snapshot(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "estimated_total_equity_usdt": 1_000.0,
            "usdt_available": 750.0,
            "holdings_json": {"BTC-USDT": 0.0025},
        }

    def fake_execute_trade(
        coin: str,
        action: str,
        amount_usdt: float,
        price: float,
        **kwargs: object,
    ) -> TradeExecutionResult:
        captured.update(
            coin=coin,
            action=action,
            amount_usdt=amount_usdt,
            price=price,
            kwargs=kwargs,
        )
        return TradeExecutionResult(
            success=True,
            status="filled",
            account_mode="demo",
            trade_mode="spot",
            inst_id=coin,
            side=action,
            action=action,
            requested_notional_usdt=amount_usdt,
            reference_price=price,
            ord_id="okx-1",
            avg_fill_price=100_050.0,
            filled_qty=0.002,
            filled_notional_usdt=200.1,
            fill_verified=True,
        )

    executor.get_account_snapshot = fake_snapshot
    executor.execute_trade = fake_execute_trade
    request = OrderRequest(
        strategy_id="runtime-model",
        instrument_id="BTC-USDT",
        market_type=MarketType.SPOT,
        action=OrderAction.BUY,
        notional_usdt=200.0,
        reference_price=100_000.0,
    )

    cycle = TradingService(executor).execute(request)

    assert captured["coin"] == "BTC-USDT"
    assert captured["amount_usdt"] == 200.0
    assert cycle.account_before.total_equity == 1_000.0
    assert cycle.execution.status == "filled"
    assert cycle.execution.order_id == "okx-1"
    assert cycle.execution.actual_average_fill_price == 100_050.0


def test_rebalance_broker_adapter_converts_strategy_metadata_to_common_execution() -> None:
    class ExampleRebalanceBroker:
        def equity_snapshot(self, prices: dict[str, float]) -> dict[str, object]:
            return {
                "total_equity_usdt": 1_000.0,
                "cash_total_usdt": 600.0,
                "per_coin_equity": {"BTC-USDT": 1_000.0},
            }

        def rebalance_coin(
            self,
            coin: str,
            target_position: int,
            price: float,
            trade_date: str,
            signal_row: dict[str, object],
            timestamp: float | None = None,
        ) -> dict[str, object]:
            assert coin == "BTC-USDT"
            assert target_position == 1
            assert trade_date == "2026-08-05"
            assert signal_row["model_score"] == 1
            assert timestamp == 123.0
            return {
                "status": "filled",
                "submitted": True,
                "executed_action": "BUY",
                "order_id": "paper-btc-1",
                "actual_avg_fill_price": price,
                "filled_quantity": 0.002,
                "filled_notional": 200.0,
                "success": True,
            }

    backend = RebalanceBrokerBackend(
        ExampleRebalanceBroker(),
        backend_id="fingpt-paper",
        execution_mode="paper",
    )
    request = OrderRequest(
        strategy_id="fingpt",
        instrument_id="BTC-USDT",
        market_type=MarketType.SPOT,
        action=OrderAction.BUY,
        allow_backend_sizing=True,
        reference_price=100_000.0,
        metadata={
            "target_position": 1,
            "trade_date": "2026-08-05",
            "signal_row": {"model_score": 1},
            "timestamp": 123.0,
        },
    )

    cycle = TradingService(backend).execute(request)

    assert cycle.execution.status == "filled"
    assert cycle.execution.order_id == "paper-btc-1"
    assert cycle.execution.to_legacy_payload()["success"] is True


def test_finmem_decision_adapter_preserves_legacy_result_and_extracts_fill() -> None:
    raw_result = {
        "BTC": {
            "code": "0",
            "msg": "",
            "data": [{"ordId": "okx-7", "sCode": "0", "sMsg": ""}],
            "instId": "BTC-USDT-SWAP",
            "signal": "buy",
            "execution_audit": {
                "order_detail": {
                    "ok": True,
                    "raw": {
                        "code": "0",
                        "data": [
                            {
                                "ordId": "okx-7",
                                "state": "filled",
                                "avgPx": "100100",
                                "accFillSz": "0.002",
                                "fee": "-0.1",
                                "feeCcy": "USDT",
                            }
                        ],
                    },
                },
                "actual_avg_px": 100_100.0,
            },
        }
    }

    class ExampleDecisionClient:
        def get_account_snapshot(self, instId: str, ccy: str) -> dict[str, object]:
            assert instId == "BTC-USDT-SWAP"
            assert ccy == "USDT"
            return {
                "balance": {
                    "total_eq": "1000",
                    "avail_bal": "700",
                    "frozen_bal": "10",
                },
                "position": {
                    "position": {"instId": instId, "pos": "0.002"},
                },
            }

        def execute_decision(self, decision: dict[str, object]) -> dict[str, object]:
            assert decision["BTC"]
            return raw_result

    request = OrderRequest(
        strategy_id="finmem",
        instrument_id="BTC-USDT-SWAP",
        market_type=MarketType.SWAP,
        action=OrderAction.BUY,
        notional_usdt=200.0,
        reference_price=100_000.0,
        metadata={"legacy_decision": {"BTC": {"signal": "buy"}}},
    )
    backend = DecisionMappingBackend(
        ExampleDecisionClient(),
        backend_id="finmem-okx",
        execution_mode="demo",
    )

    cycle = TradingService(backend).execute(request)

    assert cycle.account_before is not None
    assert cycle.account_before.total_equity == 1_000.0
    assert cycle.account_before.available_cash == 700.0
    assert cycle.account_before.positions == {"BTC-USDT-SWAP": 0.002}
    assert cycle.execution.status == "filled"
    assert cycle.execution.order_id == "okx-7"
    assert cycle.execution.actual_average_fill_price == 100_100.0
    assert cycle.execution.filled_quantity == 0.002
    assert cycle.execution.filled_notional == pytest.approx(200.2)
    assert cycle.execution.slippage_bps == pytest.approx(10.0)
    assert cycle.execution.legacy_payload == raw_result


def test_finmem_paper_ledger_adapter_preserves_existing_payload(tmp_path: Path) -> None:
    ledger = PaperLedger(tmp_path / "state.json", fee_bps=10.0)
    request = OrderRequest(
        strategy_id="finmem",
        instrument_id="BTC-USDT-SWAP",
        market_type=MarketType.SWAP,
        action=OrderAction.SELL,
        allow_backend_sizing=True,
        reference_price=50_000.0,
        metadata={"target_position": -1},
    )

    cycle = TradingService(PaperLedgerBackend(ledger)).execute(request)

    assert cycle.execution.status == "filled"
    assert cycle.execution.executed_action is OrderAction.SELL
    assert cycle.execution.fee_amount == 10.0
    assert cycle.execution.legacy_payload["target_sign"] == -1
    assert cycle.account_after is not None
    assert cycle.account_after.positions["BTC-USDT-SWAP"] < 0
