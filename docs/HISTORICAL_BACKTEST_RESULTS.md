# Historical backtest results

This page records a curated set of completed historical studies that are useful for understanding the benchmark. These results use different assets, periods, features, return definitions, and execution assumptions. They are not ranked against one another and are not part of the continuously updated exchange-paper Leaderboard.

## Selected machine-learning configurations

The source workbook covers ten cryptocurrencies from 2020 through 2025. The table selects five representative configurations that also map to methods used by the project. Return is annualized. The risk-adjusted measure is information ratio.

| Method | Configuration | Annualized return | Information ratio | Max drawdown |
| --- | --- | ---: | ---: | ---: |
| GATs | 4h, 52 features | 71.70% | 1.27 | -33.97% |
| TCN | 1h, OHLCV | 62.94% | 1.31 | -29.10% |
| XGBoost | 4h, 52 features | 41.93% | 0.87 | -42.42% |
| DoubleEnsemble | 15m, 52 features | 37.48% | 0.84 | -48.39% |
| TabNet | 15m, 52 features | 34.52% | 1.03 | -32.67% |

These are the workbook's reported best configurations. They do not establish expected future returns. Frequency and feature-set selection are part of the reported configuration.

## FinGPT, FinAgent, FinMem, and LLM studies

| Method | Period and assets | Total return | Native benchmark | Sharpe | Max drawdown | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| FinGPT pretrained LoRA | 2022-06-01 to 2024-01-01, ETH-USDT | +79.60% | +29.37% | 0.98 | -38.00% | Completed historical ETH long/flat study |
| FinGPT Sentiment SFT | 2025, ETH-USDT | -9.17% | -11.57% | 0.18 | -48.62% | Threshold selected on 2024 and frozen for the 2025 test |
| FinAgent Qwen | 2023-06-01 to 2023-12-31, BTCUSD | +48.93% | +57.67% | 2.08 | -14.79% | Completed, with lower raw return than buy-and-hold |
| FinAgent DeepSeek | 2023-06-01 to 2023-12-31, BTCUSD | +36.51% | +57.67% | 1.62 | -12.27% | Completed, with lower raw return than buy-and-hold |
| FinMem BTC | 2025, BTCUSD | -20.76% | -7.35% | -0.78 | -42.46% | Completed negative result |
| FinMem ETH | 2025, ETHUSD | -27.72% | -11.57% | -0.69 | -44.17% | Completed negative result |
| LLM daily-chat baseline | 2025 H1, BTC-USDT perpetual | -74.20% | +12.82% | N/A | -91.07% | Completed failed baseline with unacceptable risk |
| TradingAgents | No completed historical artifact found | N/A | N/A | N/A | N/A | Recorded as an evidence gap |

FinMem audit files store cumulative log reward. The total-return values above use `exp(cumulative_log_reward) - 1`, which is the conversion used by the supplied cumulative-profit curves.

## Exclusions and interpretation

- Interrupted 2025 annual 4-hour LLM runs are retained in the private archive for debugging and are not presented as completed results.
- FinAgent 2025 progress snapshots have different end dates, so they are excluded from the completed comparison.
- A two-month BTC perpetual exploratory run reported +154.62% with a -54.84% drawdown. It is not promoted because its short interval is not comparable with the completed six-month baseline.
- Raw news, prompts, model outputs, order records, account state, private paths, and source archives are not published here.
- Historical backtests, local paper trading, exchange paper trading, and live trading are separate evidence stages. A result in one stage does not validate another.

The public website presents these tables alongside the continuously updated exchange-paper results while keeping the protocols visibly separate.
