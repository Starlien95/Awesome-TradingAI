# Trading flow

## Traditional ML and MacroHFT

```text
TimeframeRunner.start
  loop at update_interval_sec
    reload local configs
    fetch and cache market bars
    for each registered model
      read current strategy holdings
      adapter.predict -> ranked scores
      finalize prior labels and record current predictions
      adapter.generate_signals -> target changes
      write the rebalance plan
      TradeExecutor.execute_trade -> TradeExecutionResult
      write orders, fills, failures, snapshots, metrics, signals, trades, volume
    wait for the next closed bar
```

Each process owns its runner. Each model within that process owns its executor and accounting state.

## Target selection

The shared adapter calls `quant_bench.strategies.threshold_topk_dropout.select_target_weights`, which matches `ThresholdTopkDropoutStrategy` semantics:

1. Keep candidates with `score > threshold`.
2. Take the highest `top_k` candidates.
3. Put old holdings outside that target into the dropout set.
4. Remove at most `max_dropout`, smallest current weight first.
5. Merge retained holdings and new targets.
6. If the merge exceeds `top_k`, keep the highest absolute model scores.
7. Produce equal long target weights bounded by `risk_degree`.

`passive_topk_dropout` keeps retained holdings unchanged, sells removed targets, and uses available strategy cash to buy new targets. `target_weight` performs strict equal-weight rebalancing and produces higher turnover.

`min_position_value_usdt`, `rebalance_tolerance_usdt`, and `rebalance_tolerance_pct` suppress dust and immaterial orders. `max_position_count` and the old `max_sell_per_cycle` field remain compatibility inputs; `top_k` and `max_dropout` are authoritative.

## FinGPT news

FinGPT uses independent per-symbol capital buckets. Positive targets that symbol's invested bucket, negative targets cash, and neutral retains state. It does not rank symbols into a shared top-k portfolio. In OKX spot mode it can sell only the quantity recorded as strategy-owned.

## Safety consequences

A symbol can be held by more than one process because process state is independent. A retained symbol is not repeatedly bought under the passive policy. Account reconciliation, credentials, and mode selection are checked per exported config.
