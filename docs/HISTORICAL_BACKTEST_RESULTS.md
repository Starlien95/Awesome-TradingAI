# Historical backtest results

This page records completed, exploratory, interrupted, and missing historical evidence used by Awesome TradingAI. The studies use different assets, periods, features, return definitions, and execution assumptions. They are not ranked against one another and are separate from the continuously updated exchange-paper Leaderboard.

## Machine-learning study

The source workbook covers ADA, BTC, DOGE, ETH, HBAR, LINK, LTC, OKB, TRX, and XRP from 2020 through 2025. Each row below is the best reported configuration for one method. Return is annualized. `IR` is the workbook's information ratio. `52 features` identifies the extended feature set; `OHLCV` identifies the six-field market-data input.

| Method | Family | Best configuration | Annualized return | IR | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| GATs | Graph ML | 4h, 52 features | 71.70% | 1.27 | -33.97% |
| TCN | Time-series ML | 1h, OHLCV | 62.94% | 1.31 | -29.10% |
| XGBoost | Traditional ML | 4h, 52 features | 41.93% | 0.87 | -42.42% |
| DoubleEnsemble | Ensemble ML | 15m, 52 features | 37.48% | 0.84 | -48.39% |
| TabNet | Tabular ML | 15m, 52 features | 34.52% | 1.03 | -32.67% |
| LSTM | Time-series ML | 4h, 52 features | 32.90% | 0.66 | -44.07% |
| MLP | Traditional ML | 1h, 52 features | 30.33% | 0.87 | -35.24% |
| GRU | Time-series ML | 4h, 52 features | 29.66% | 0.83 | -38.14% |
| TRA | Time-series ML | 4h, 52 features | 25.08% | 0.44 | -52.94% |
| Localformer | Graph ML | 15m, OHLCV | 24.38% | 0.55 | -51.19% |
| Sandwich | Graph ML | 4h, 52 features | 23.16% | 0.50 | -35.89% |
| CatBoost | Traditional ML | 5m, 52 features | 21.46% | 0.46 | -61.93% |
| ALSTM | Time-series ML | 1h, 52 features | 21.17% | 0.56 | -51.26% |
| Linear | Traditional ML | 1h, 52 features | 21.08% | 0.51 | -44.59% |
| AdaRNN | Time-series ML | 4h, OHLCV | 19.41% | 0.37 | -45.59% |
| SFM | Time-series ML | 15m, OHLCV | 18.31% | 0.55 | -43.95% |
| GeneralPtNN | Time-series ML | 5m, 52 features | 16.44% | 0.44 | -34.04% |
| TCTS | Meta-framework | 15m, OHLCV | 10.92% | 0.33 | -50.06% |
| TFT | Time-series ML | 15m, 52 features | 10.53% | 0.31 | -50.30% |
| KRNN | Time-series ML | 1h, OHLCV | 9.78% | 0.26 | -35.75% |
| LightGBM | Traditional ML | 5m, 52 features | 8.57% | 0.25 | -52.05% |
| Transformer | Time-series ML | 1h, OHLCV | 5.32% | 0.16 | -41.11% |
| IGMTF | Graph ML | 15m, OHLCV | 1.72% | 0.05 | -58.33% |
| ADD | Meta-framework | 5m, OHLCV | -1.37% | -0.04 | -52.05% |

The workbook contains a duplicated LightGBM block and one malformed summary row. The public table keeps one valid best configuration per named method, producing 24 rows.

## Completed FinGPT, FinAgent, FinMem, and LLM studies

| Method | Period and assets | Total return | Native benchmark | Excess return | Sharpe / Sortino | Max drawdown | Sample and activity |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| FinGPT pretrained LoRA | 2022-06-01 to 2024-01-01, ETH-USDT | +79.60% | +29.37% | +50.23% | 0.98 / N/A | -38.00% | 175 trades; 6,939 classified news items; 74.66% exposure |
| FinGPT | 2025, ETH-USDT | -9.17% | -11.57% | +2.40% | 0.18 / N/A | -48.62% | 35 trades; 8,573 classified news items; 85.48% exposure |
| FinAgent Qwen | 2023-06-01 to 2023-12-31, BTCUSD | +48.93% | +57.67% | -8.74% | 2.08 / 3.35 | -14.79% | Completed full-stage run |
| FinAgent | 2023-06-01 to 2023-12-31, BTCUSD | +36.51% | +57.67% | -21.16% | 1.62 / 2.15 | -12.27% | Completed full-stage run |
| FinMem | 2025, BTCUSD | -20.76% | -7.35% | -13.41% | -0.78 / N/A | -42.46% | 210 active signals; 47.62% directional accuracy |
| LLM daily-chat baseline | 2025-01-01 to 2025-07-01, BTC-USDT perpetual | -74.20% | +12.82% | -87.02% | N/A | -91.07% | 31 closed trades; 16.13% win rate; failed baseline |
| TradingAgents | Historical archive | N/A | N/A | N/A | N/A | N/A | No completed historical artifact found |

The FinGPT pretrained LoRA study reported 44.56% annualized return and 50.69% annualized volatility. The 2025 FinGPT study reported -9.17% annualized return and 65.66% annualized volatility. Its decision threshold was selected on 2024 data and frozen before the 2025 test.

The FinMem BTC audit stores cumulative log reward. Its public total return uses `exp(cumulative_log_reward) - 1`, matching the supplied cumulative-profit curve. FinMem ETH is outside the current public scope.

## Exploratory and incomplete evidence

These records remain visible because incomplete and non-comparable outcomes are part of the evidence record. They do not enter the completed comparison.

| Method and variant | Progress | Observed return | Reference | Max drawdown | Last observation | Status |
| --- | ---: | ---: | --- | ---: | --- | --- |
| LLM daily-chat, two-month exploratory run | 100.00% | +154.62% | Buy and hold -11.88% | -54.84% | 2025-03-01 | Completed exploratory; 6 closed trades |
| LLM daily-chat, DeepSeek annual 4h | 26.53% | -26.37% | Annual target progress 26.53% | -94.17% | 2025-04-07 20:00 | Interrupted |
| LLM daily-chat, Qwen3-Max annual 4h | 16.35% | -31.25% | Annual target progress 16.35% | -58.25% | 2025-03-01 16:00 | Interrupted |
| LLM daily-chat, Qwen3.6 thinking annual 4h | 12.51% | +71.61% | Annual target progress 12.51% | -49.60% | 2025-02-15 16:00 | Interrupted |
| LLM daily-chat, Qwen3.6 without thinking annual 4h | 13.56% | -40.70% | Annual target progress 13.56% | -80.26% | 2025-02-19 12:00 | Interrupted |
| FinAgent Qwen, 2025 progress | 78.41% | +302.93% | Same-date excess -2.19 pp | N/A | 2025-06-11 | Progress snapshot |
| FinAgent DeepSeek, 2025 progress | 91.01% | +229.24% | Same-date excess -130.55 pp | N/A | 2025-10-08 | Progress snapshot |

The two FinAgent progress rows have different end dates. Their cumulative returns cannot be compared with each other or with the completed 2023 stage. The four annual LLM rows ended before their target period and do not have final results.

## Evidence rules and limitations

1. A row in the completed table requires a completed artifact and a defined end date.
2. Each study keeps its native benchmark, asset scope, return definition, and execution assumptions.
3. Progress snapshots and interrupted runs have their own table and are excluded from ranking.
4. Failed results and missing artifacts remain visible.
5. Historical backtesting, local paper trading, exchange paper trading, and live trading are separate evidence stages.
6. Raw news, prompts, model outputs, order records, account state, private paths, and source archives are not published.

The public website presents the same evidence in its Historical Backtests page while keeping it separate from the live-updating exchange-paper results.
