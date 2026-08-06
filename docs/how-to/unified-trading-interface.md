# Unified trading interface

`quant-bench` separates strategy decisions from account and order execution.
A strategy may decide what it wants to do, but it must not select credentials,
construct an exchange client, or define a private order-result format.

The common path is:

```text
strategy decision
    -> OrderRequest
    -> TradingService
    -> TradingBackend
    -> ExecutionReport + before/after AccountSnapshot
```

`TradingBackend` is the common trading-backend protocol. A backend owns one
paper, exchange-demo, or live account implementation. `TradingService` owns the
shared execution flow: account snapshot before the order, order submission,
error normalization, and account snapshot after the order.

## Public contracts

- `OrderRequest` contains the exact instrument, spot or perpetual-swap market,
  action, order type, requested size, reference price, and strategy metadata.
- `AccountSnapshot` provides total equity, available and frozen cash,
  positions, reference prices, and the untouched provider payload.
- `ExecutionReport` provides submission status, order identifiers, expected
  and actual prices, fill size, notional, fee, slippage, errors, and raw
  exchange responses.
- `TradingCycleResult` combines the request, before/after snapshots, execution
  report, and non-fatal snapshot warnings.

All four models are versioned and reject unknown fields. Directional orders
must specify quantity or USDT notional unless a reviewed legacy adapter is
explicitly responsible for sizing. Spot and perpetual-swap instrument names
cannot be mixed.

## Current method mapping

| Method family | Strategy-specific part | Common execution integration |
|---|---|---|
| Traditional model runtime and MacroHFT | signal and requested USDT amount | `TradeExecutor` implements `TradingBackend` |
| FinGPT news | sentiment, threshold, and target position | `RebalanceBrokerBackend` adapts paper and OKX spot brokers |
| FinMem paper ledger | target long, flat, or short position | `PaperLedgerBackend` |
| FinMem OKX cycle | InvestorBench decision and legacy symbol mapping | `DecisionMappingBackend` |
| Canonical and Qlib backtests | vectorized predictions and portfolio simulation | no exchange account; results remain in the experiment artifact contract |
| External `ai_trade` methods | complete child process with its own reviewed execution layer | process boundary controlled by `MethodRunner`; not an in-process backend |

The adapters preserve existing native artifacts and legacy payloads. This is
required for restart compatibility, dashboards, and historical audit logs.
New code should consume the common fields instead of adding another private
order schema.

## Adding a strategy that can trade

1. Keep model inference and signal generation independent from credentials and
   exchange software development kits.
2. Convert the final intent to one `OrderRequest` per instrument.
3. Select a configured backend outside strategy code.
4. Execute with `TradingService(backend).execute(request)`.
5. Store `TradingCycleResult` or its standard report fields in the run
   artifacts.

Only backend modules may call exchange order methods. A strategy module must
not call `place_order`, `close_positions`, `execute_decision`, or a private
paper-ledger mutation directly.

## External process boundary

The four `ai_trade` agents are integrated as separately deployed processes.
Their process adapter validates method capability, execution mode,
confirmation token, repository location, and durable run records. It does not
move account credentials or issue the child process's orders itself. Full
cross-process unification therefore means the external repository must keep
its own standardized execution layer; `quant-bench` must not duplicate or
silently replace it.
