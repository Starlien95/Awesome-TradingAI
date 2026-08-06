# Core CSV Schema

CSV contracts are versioned per artifact in
`quant_bench.contracts.csv`. The shared registry version is
`quant-bench.csv.v1`. Runtime manifests declare this value in
`csv_schema_version`; manifest artifact paths are UTF-8, relative to the run
root, and cannot escape it.

Canonical research files keep their benchmark contracts:

```text
predictions.csv:
timestamp,symbol,score,label

backtest/returns.csv:
timestamp,gross_return,net_return,turnover,cost,positions,equity
```

Canonical timestamps are timezone-aware ISO 8601 values. Runtime
`timestamp` and `bar_timestamp` are Unix seconds. `bar_timestamp` is the fact
time used for curve alignment; datetime text remains a human-readable alias.
FinGPT daily bars use explicit `Asia/Shanghai` local midnight.

Monetary columns ending in `_usdt` are absolute USDT amounts. Return columns
ending in `_pct` are percentage points. `gross_exposure` and `cash_ratio` are
unitless ratios. Extension columns are allowed after the required columns.

## Metrics

Required columns:

```text
timestamp,datetime,bar_timestamp,bar_datetime,
initial_capital_usdt,cash_total_usdt,holdings_value_usdt,total_equity_usdt,
strategy_equity,strategy_pnl,strategy_returns_pct,
baseline_equity,baseline_pnl,baseline_returns_pct,
btc_price,active_positions,gross_exposure,cash_ratio,
method_family,strategy_id,frequency
```

`strategy_equity` and `total_equity_usdt` both contain absolute strategy
equity in runtime metrics. Account-wide equity remains in explicitly named
account audit columns and must not silently replace strategy equity.

## Signals

Required columns:

```text
cycle_id,timestamp,datetime,bar_timestamp,bar_datetime,
coin,score,signal_label,signal_score,
price_at_pred,price_future,true_return,labeled,
method_family,strategy_id,frequency
```

## Trades

Required columns:

```text
trade_id,timestamp,datetime,trade_date,coin,side,price,qty,notional_usdt,
fee_usdt,slippage_bps,cash_before,cash_after,
position_qty_before,position_qty_after,reason,success
```

Enhanced execution columns written by new runs:

```text
reference_price,actual_avg_price,requested_qty,actual_qty,
requested_notional_usdt,actual_notional_usdt,fee_ccy,
order_id,client_order_id,order_status,fill_verified,partial_fill,
exchange_error_code,exchange_error_msg
```

## Orders

Orders and fills are backend-neutral audit contracts. Exchange-specific status
and error fields can be appended as extensions. JSON columns such as
`raw_fill_json` and `holdings_json` use deterministic key ordering.

One row per submitted or rejected order:

```text
trade_id,timestamp,datetime,trade_date,coin,exchange,account_mode,trade_mode,
inst_id,side,ord_type,td_mode,tgt_ccy,
requested_notional_usdt,requested_qty,submitted_sz,reference_price,
submit_ts,ack_ts,settle_ts,order_id,client_order_id,order_status,
okx_code,okx_msg,okx_s_code,okx_s_msg,
avg_fill_price,filled_qty,filled_notional_usdt,
fee_usdt,fee_ccy,slippage_bps,fill_verified,partial_fill,
success,error_type,error_message
```

## Fills

One row per exchange fill when the broker can query fills:

```text
trade_id,timestamp,datetime,trade_date,coin,fill_index,exchange,inst_id,
order_id,trade_id_exchange,fill_time,side,fill_price,fill_qty,
fill_notional_usdt,fee,fee_ccy,raw_fill_json
```

## Rebalance Plan

One row per strategy decision before order submission:

```text
decision_id,timestamp,datetime,bar_timestamp,bar_datetime,trade_date,
strategy_id,method_family,frequency,coin,side,reference_price,
current_qty,current_value_usdt,target_value_usdt,target_weight,
planned_delta_usdt,strategy_equity,cash_available_for_strategy,
reason,will_submit_order
```

## Account Snapshots

Before/after snapshots around each attempted order:

```text
trade_id,timestamp,datetime,trade_date,coin,snapshot_phase,
account_mode,trade_mode,usdt_available,holdings_value_usdt,
estimated_total_equity_usdt,holdings_json
```

## Volume

Required columns:

```text
timestamp,datetime,cycle_volume_usdt,cumulative_total_volume_usdt,
daily_total_volume_usdt,per_coin_cumulative_json,per_coin_daily_json
```

## FinMem compatibility columns

FinMem logs include all required canonical columns and retain several legacy
dashboard aliases such as `btc_price`, `okx_btc_qty`, and
`strategy_btc_qty`. For non-BTC runs those aliases contain the configured
asset's values. New consumers should use `symbol`, `asset_price`,
`total_equity_usdt`, `cash_total_usdt`, `holdings_value_usdt`,
`method_family`, `strategy_id`, and `frequency`.

Every symbol-scoped FinMem log directory also contains `run_manifest.json`.

## FinGPT research artifacts

`quant-bench fingpt backtest` writes the required metrics, signals, trades, and
volume contracts above. It also writes method-specific diagnostics:

```text
validation/parameter_trials.csv
validation/daily_scores.csv
validation/daily_signals.csv
global_tuned/daily_backtest.csv
global_tuned/per_symbol_summary.csv
global_tuned/portfolio_daily.csv
global_tuned/portfolio_summary.csv
global_tuned/trade_list.csv
```

Signal rows include `price_future` and `true_return` only for evaluation. The
position used by the backtest is delayed by at least one day. The final signal
row has `labeled: false` when no next price exists.

All signal rows for one research date share one `cycle_id`. Research metrics
and runtime metrics use the same absolute `strategy_equity` unit.

## Compatibility and migration

Readers may accept documented aliases for older files, but must merge aliases
row by row and surface quality issues. New producers write the required column
order from `quant_bench.contracts.csv`. A schema change requires synchronized
updates to the registry, producer, manifest, dashboard adapter, this document,
migration notes, and tests.
