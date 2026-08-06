# FinGPT 新闻情绪研究与模拟盘

FinGPT 部分现在包含统一的离线回测、验证集调参、多币种独立资金桶、分类评估、dry-run 模拟盘和受控 OKX 执行入口。

## 离线示例

```bash
quant-bench fingpt init --workspace ./qb-workspace

CONFIG=./qb-workspace/methods/fingpt_news/research/config.yaml
quant-bench fingpt validate --config "$CONFIG"
quant-bench fingpt backtest \
  --config "$CONFIG" \
  --output ./qb-workspace/runs/fingpt-smoke
```

示例只使用 wheel 内置的合成价格和合成情绪数据，不访问网络。

输入价格需要 `date,symbol,close`。情绪文件需要 `date,symbol`，并提供 `article_score`、`prob_positive/prob_negative` 或 `model_score` 中的一种。支持 `asset` 与 `market` 两类新闻桶。

参数选择只读取 validation 时间段。冻结后的参数再用于 test 时间段，`chosen_config.json` 明确记录 `test_metrics_used_for_selection: false`。支持：

- `asset_only`、`market_only`、`hybrid_fixed`、`hybrid_adaptive`；
- 全局阈值与带覆盖率保护的逐币种阈值；
- 新闻 top-N、最小情绪强度、score power、固定或自适应 alpha；
- Sharpe、收益、Calmar、超额收益目标；
- 交易次数、有效信号天数和新闻数量门槛。

每个币种拥有独立资金桶。positive 目标为做多，negative 目标为空仓，neutral 保持已有目标。`signal_lag_days` 最小为 1，保证当日新闻不会获得同日已经发生的收益。手续费和滑点均计入换手成本。

离线模拟盘：

```bash
quant-bench fingpt paper-dry-run --workspace ./qb-workspace
```

联网 paper 模式必须显式提供 `--allow-network`，只访问配置的新闻、模型服务和公共价格，不创建账户 executor。OKX 下单还需要 `--execute-orders`、环境变量凭据以及 `DEMO_ORDERS` 或 `LIVE_ORDERS`。YAML 内联凭据会被拒绝。

默认参数文件为 `per_coin_params_default.csv`，使用统一通用初始值。`per_coin_params_sentiment_sft_2025.csv` 作为旧实验 recipe 保留，不再作为默认配置。

完整字段、产物、参数语义和公开边界见[英文文档](fingpt-news.md)。
