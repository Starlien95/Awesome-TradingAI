import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. 避免 FinMemAgent.load_checkpoint 因 OPENAI_API_KEY 报错
# ============================================================

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# 你的项目里 OpenAIEmbedding 会硬检查 OPENAI_API_KEY。
# 如果你实际用的是 DashScope，可以把 DASHSCOPE_API_KEY 映射给 OPENAI_API_KEY。
if "OPENAI_API_KEY" not in os.environ:
    if "DASHSCOPE_API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]
    else:
        # 仅用于加载 checkpoint，不实际请求 embedding 时通常够用
        os.environ["OPENAI_API_KEY"] = "dummy_key_for_loading_checkpoint_only"


from .engine.agent import FinMemAgent


# ============================================================
# 1. 基础工具函数
# ============================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_from_config(config: dict, ticker: str):
    env_config = config["env_config"]
    meta_config = config["meta_config"]

    start_date = env_config["test_start_time"]
    end_date = env_config["test_end_time"]
    data_path = env_config["env_data_path"][ticker]
    checkpoint_path = meta_config["test_checkpoint_save_path"]

    return data_path, checkpoint_path, start_date, end_date


def resolve_agent_checkpoint_path(checkpoint_path: str) -> str:
    """
    支持两种输入：
    1. results/.../test_checkpoint
    2. results/.../test_checkpoint/agent
    """
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.name == "agent":
        if checkpoint_path.exists():
            return str(checkpoint_path)
        raise FileNotFoundError(f"Agent checkpoint path not found: {checkpoint_path}")

    agent_path = checkpoint_path / "agent"

    if agent_path.exists():
        return str(agent_path)

    raise FileNotFoundError(
        f"Cannot find agent checkpoint. Tried:\n"
        f"  {checkpoint_path}\n"
        f"  {agent_path}"
    )


def infer_trading_days(ticker: str) -> int:
    """
    和官方 eval_pipeline.py 逻辑保持一致：
    股票/ETF 用 252，BTC/ETH 用 365。
    你如果跑其他币种，也建议用 365。
    """
    stock_like = {"MSFT", "JNJ", "UVV", "HON", "TSLA", "AAPL", "NIO", "ETF"}
    crypto_like = {
        "BTC", "ETH", "ADA", "DOGE", "HBAR",
        "LINK", "LTC", "OKB", "TRX", "XRP"
    }

    ticker = ticker.upper()

    if ticker in stock_like:
        return 252
    if ticker in crypto_like:
        return 365

    # 默认按 crypto 处理
    return 365


# ============================================================
# 2. 读取价格和 action
# ============================================================

def load_price_df(data_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for date_str, contents in data.items():
        if contents is None:
            continue

        # 兼容 "prices" 和 "price"
        price = contents.get("prices", contents.get("price", None))

        if price is None:
            continue

        if start_date <= date_str <= end_date:
            rows.append({
                "date": pd.to_datetime(date_str).date(),
                "price": float(price),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            f"No price data found in {data_path} "
            f"from {start_date} to {end_date}"
        )

    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_action_df(checkpoint_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    agent_path = resolve_agent_checkpoint_path(checkpoint_path)

    agent = FinMemAgent.load_checkpoint(path=agent_path)

    action_df = pd.DataFrame(agent.portfolio.get_action_record())

    if action_df.empty:
        raise ValueError("Action record is empty.")

    if "date" not in action_df.columns:
        raise ValueError("Action record has no 'date' column.")

    if "position" not in action_df.columns:
        raise ValueError(
            "Action record has no 'position' column. "
            "Please inspect agent.portfolio.get_action_record()."
        )

    action_df["date"] = pd.to_datetime(action_df["date"]).dt.date

    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()

    action_df = action_df[
        (action_df["date"] >= start) & (action_df["date"] <= end)
    ].copy()

    action_df = action_df.rename(columns={"position": "direction"})

    keep_cols = ["date", "symbol", "direction"]

    if "price" in action_df.columns:
        keep_cols.append("price")

    if "evidence" in action_df.columns:
        keep_cols.append("evidence")

    keep_cols = [c for c in keep_cols if c in action_df.columns]
    action_df = action_df[keep_cols].copy()

    action_df["direction"] = action_df["direction"].astype(float)

    return action_df.sort_values("date").reset_index(drop=True)


def infer_action_name(direction: float) -> str:
    if direction > 0:
        return "BUY / LONG"
    if direction < 0:
        return "SELL / SHORT"
    return "HOLD / NEUTRAL"


# ============================================================
# 3. 生成逐日审计表
# ============================================================

def build_audit_df(
    price_df: pd.DataFrame,
    action_df: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    # 去重：如果同一天有多条 action，保留最后一条
    if action_df["date"].duplicated().any():
        print("[WARNING] Duplicated action dates found. Keep the last record.")
        action_df = action_df.drop_duplicates(subset=["date"], keep="last")

    # 按官方 eval 逻辑：缺失日期补 direction=0
    full_dates = price_df["date"].tolist()
    existing_action_dates = set(action_df["date"].tolist())

    missing_dates = [d for d in full_dates if d not in existing_action_dates]

    if missing_dates:
        print("[WARNING] Missing action dates will be filled with direction=0:")
        print(missing_dates)

        missing_df = pd.DataFrame({
            "date": missing_dates,
            "symbol": [ticker] * len(missing_dates),
            "direction": [0.0] * len(missing_dates),
        })

        action_df = pd.concat([action_df, missing_df], axis=0)

    action_df = action_df.sort_values("date").reset_index(drop=True)

    # 构造 date_t -> date_t+1
    reward_df = price_df.copy()
    reward_df["next_date"] = reward_df["date"].shift(-1)
    reward_df["next_price"] = reward_df["price"].shift(-1)

    # 最后一日没有 next_price，不能计算收益
    reward_df = reward_df.dropna(subset=["next_price"]).copy()

    merged = reward_df.merge(
        action_df[["date", "direction"]],
        on="date",
        how="left",
    )

    merged["direction"] = merged["direction"].fillna(0.0)
    merged["action_name"] = merged["direction"].apply(infer_action_name)

    merged["log_return"] = np.log(merged["next_price"] / merged["price"])
    merged["simple_return"] = merged["next_price"] / merged["price"] - 1

    merged["agent_daily_log_reward"] = (
        merged["direction"] * merged["log_return"]
    )

    merged["agent_cum_log_reward"] = (
        merged["agent_daily_log_reward"].cumsum()
    )

    merged["buy_hold_cum_log_reward"] = (
        merged["log_return"].cumsum()
    )

    merged["is_direction_correct"] = np.where(
        merged["direction"] == 0,
        np.nan,
        np.sign(merged["direction"]) == np.sign(merged["log_return"]),
    )

    return merged


# ============================================================
# 4. 指标计算：官方 eval 风格 + 标准风格
# ============================================================

def standard_deviation(x):
    x = list(x)
    if len(x) <= 1:
        return 0.0

    mean = sum(x) / len(x)
    variance = sum((r - mean) ** 2 for r in x) / (len(x) - 1)
    return variance ** 0.5


def calculate_max_drawdown_eval_style(daily_returns):
    """
    复刻官方 eval_pipeline.py 的 MDD：
    把 daily_returns 当作 simple return 去累乘。
    注意：这里 daily_returns 实际是 log return。
    """
    cumulative_returns = [1.0]

    for r in daily_returns:
        cumulative_returns.append(cumulative_returns[-1] * (1 + r))

    peak = cumulative_returns[0]
    max_drawdown = 0.0

    for v in cumulative_returns:
        if v > peak:
            peak = v

        drawdown = (peak - v) / peak

        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown


def calculate_eval_style_metrics(daily_rewards, num_price_points, trading_days):
    """
    复刻官方 eval_pipeline.py：

    CR = sum(daily_log_rewards)
    AV = std(daily_log_rewards, ddof=1) * sqrt(trading_days)
    SR = [CR / (len(price_list) / trading_days)] / AV
    MDD = calculate_max_drawdown(daily_rewards)
    """
    daily_rewards = list(daily_rewards)

    cum_return = float(sum(daily_rewards))

    daily_std = standard_deviation(daily_rewards)
    ann_vol = daily_std * np.sqrt(trading_days)

    if ann_vol == 0:
        sharpe = np.nan
    else:
        sharpe = (cum_return / (num_price_points / trading_days)) / ann_vol

    mdd = calculate_max_drawdown_eval_style(daily_rewards)

    return {
        "Cumulative Return": cum_return,
        "Sharpe Ratio": sharpe,
        "Max DrawDown": mdd,
        "Annualized Volatility": ann_vol,
    }


def calculate_max_drawdown_standard_from_log_returns(daily_log_returns):
    """
    标准 log return 净值曲线：
    equity = exp(cumsum(log_returns))
    """
    daily_log_returns = np.array(daily_log_returns, dtype=float)

    equity = np.exp(
        np.concatenate([[0.0], np.cumsum(daily_log_returns)])
    )

    running_max = np.maximum.accumulate(equity)
    drawdowns = 1 - equity / running_max

    return float(np.max(drawdowns))


def calculate_standard_metrics(daily_log_rewards, trading_days):
    daily_log_rewards = np.array(daily_log_rewards, dtype=float)

    cum_return = float(daily_log_rewards.sum())

    if len(daily_log_rewards) <= 1:
        ann_vol = 0.0
        sharpe = np.nan
    else:
        ann_vol = float(
            np.std(daily_log_rewards, ddof=1) * np.sqrt(trading_days)
        )

        if ann_vol == 0:
            sharpe = np.nan
        else:
            sharpe = float(
                np.mean(daily_log_rewards) * trading_days / ann_vol
            )

    mdd = calculate_max_drawdown_standard_from_log_returns(daily_log_rewards)

    return {
        "Cumulative Log Return": cum_return,
        "Sharpe Ratio": sharpe,
        "Max DrawDown": mdd,
        "Annualized Volatility": ann_vol,
    }


def make_metrics_tables(audit_df: pd.DataFrame, num_price_points: int, trading_days: int):
    buy_hold_daily = audit_df["log_return"].tolist()
    agent_daily = audit_df["agent_daily_log_reward"].tolist()

    eval_buy_hold = calculate_eval_style_metrics(
        buy_hold_daily,
        num_price_points=num_price_points,
        trading_days=trading_days,
    )

    eval_agent = calculate_eval_style_metrics(
        agent_daily,
        num_price_points=num_price_points,
        trading_days=trading_days,
    )

    standard_buy_hold = calculate_standard_metrics(
        buy_hold_daily,
        trading_days=trading_days,
    )

    standard_agent = calculate_standard_metrics(
        agent_daily,
        trading_days=trading_days,
    )

    eval_df = pd.DataFrame({
        "Buy & Hold": eval_buy_hold,
        "Agent": eval_agent,
    })

    standard_df = pd.DataFrame({
        "Buy & Hold": standard_buy_hold,
        "Agent": standard_agent,
    })

    return eval_df, standard_df


# ============================================================
# 5. 绘图
# ============================================================

def build_curve_df(
    price_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    return_mode: str,
) -> pd.DataFrame:
    """
    使用 price_df 的日期作为曲线横轴：
    第一天收益为 0；
    后续每一天对应前一日收益累计后的结果。
    """
    curve_dates = price_df["date"].tolist()

    agent_cum_log = [0.0] + audit_df["agent_cum_log_reward"].tolist()
    buy_hold_cum_log = [0.0] + audit_df["buy_hold_cum_log_reward"].tolist()

    if len(agent_cum_log) != len(curve_dates):
        raise ValueError(
            f"Curve length mismatch: "
            f"dates={len(curve_dates)}, agent={len(agent_cum_log)}"
        )

    curve_df = pd.DataFrame({
        "date": pd.to_datetime(curve_dates),
        "agent_cum_log": agent_cum_log,
        "buy_hold_cum_log": buy_hold_cum_log,
    })

    if return_mode == "simple":
        curve_df["agent_curve"] = (
            np.exp(curve_df["agent_cum_log"]) - 1
        ) * 100

        curve_df["buy_hold_curve"] = (
            np.exp(curve_df["buy_hold_cum_log"]) - 1
        ) * 100

    elif return_mode == "log":
        curve_df["agent_curve"] = curve_df["agent_cum_log"] * 100
        curve_df["buy_hold_curve"] = curve_df["buy_hold_cum_log"] * 100

    else:
        raise ValueError("return_mode must be either 'simple' or 'log'.")

    return curve_df


def plot_curve(
    curve_df: pd.DataFrame,
    output_png: str,
    ticker: str,
    agent_label: str,
    title: str,
):
    agent_final = curve_df["agent_curve"].iloc[-1]
    buy_hold_final = curve_df["buy_hold_curve"].iloc[-1]

    plt.figure(figsize=(16, 8), dpi=150)

    # 曲线
    plt.plot(
        curve_df["date"],
        curve_df["agent_curve"],
        linewidth=2.8,
        label=agent_label,
        color="#2F6FDB",
    )

    plt.plot(
        curve_df["date"],
        curve_df["buy_hold_curve"],
        linewidth=2.5,
        linestyle="--",
        label="Buy & Hold Baseline",
        color="#2F8F3A",
    )

    # 0 线
    plt.axhline(0, color="gray", linewidth=1.0)

    # 标题、坐标轴
    plt.title(title, fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Profit (%)", fontsize=12)

    # 网格、图例
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left", fontsize=11)

    # 终点
    last_date = curve_df["date"].iloc[-1]

    plt.scatter(
        last_date,
        agent_final,
        s=45,
        color="#2F6FDB",
        zorder=5,
    )

    plt.scatter(
        last_date,
        buy_hold_final,
        s=45,
        color="#2F8F3A",
        zorder=5,
    )

    # 标注
    plt.annotate(
        f"{agent_label}: {agent_final:.2f}%",
        xy=(last_date, agent_final),
        xytext=(-145, -16),
        textcoords="offset points",
        bbox=dict(
            boxstyle="round,pad=0.25",
            fc="white",
            ec="#2F6FDB",
            lw=1.2,
        ),
        color="#2F6FDB",
        fontsize=11,
    )

    plt.annotate(
        f"Buy & Hold: {buy_hold_final:.2f}%",
        xy=(last_date, buy_hold_final),
        xytext=(-145, 16),
        textcoords="offset points",
        bbox=dict(
            boxstyle="round,pad=0.25",
            fc="white",
            ec="#2F8F3A",
            lw=1.2,
        ),
        color="#2F8F3A",
        fontsize=11,
    )

    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(output_png, bbox_inches="tight")
    print(f"[OK] Figure saved to: {output_png}")


# ============================================================
# 6. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot backtest cumulative profit curve from INVESTOR-BENCH checkpoint."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/main.json",
        help="Path to configs/main.json",
    )

    parser.add_argument(
        "--ticker",
        type=str,
        default="BTC",
        help="Ticker symbol, e.g., BTC",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override data path. If not set, read from config.",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Override test_checkpoint path. If not set, read from config.",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Override test start date. If not set, read from config.",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Override test end date. If not set, read from config.",
    )

    parser.add_argument(
        "--trading-days",
        type=int,
        default=None,
        help="Annual trading days. If not set, infer from ticker.",
    )

    parser.add_argument(
        "--return-mode",
        type=str,
        default="simple",
        choices=["simple", "log"],
        help=(
            "simple: plot exp(cumulative log return)-1; "
            "log: plot cumulative log return directly."
        ),
    )

    parser.add_argument(
        "--agent-label",
        type=str,
        default=None,
        help="Legend label for the agent curve.",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output file prefix.",
    )

    args = parser.parse_args()

    ticker = args.ticker.upper()

    config = load_config(args.config)

    config_data_path, config_checkpoint_path, config_start, config_end = (
        resolve_from_config(config, ticker)
    )

    data_path = args.data_path or config_data_path
    checkpoint_path = args.checkpoint_path or config_checkpoint_path
    start_date = args.start_date or config_start
    end_date = args.end_date or config_end
    trading_days = args.trading_days or infer_trading_days(ticker)

    agent_label = args.agent_label or f"{ticker} Agent"

    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = f"{ticker.lower()}_{start_date}_to_{end_date}"

    audit_csv = f"{output_prefix}_audit.csv"
    eval_metrics_csv = f"{output_prefix}_eval_style_metrics.csv"
    standard_metrics_csv = f"{output_prefix}_standard_metrics.csv"
    output_png = f"{output_prefix}_cumulative_profit_curve.png"

    title = (
        f"{ticker}USD Backtest Progress: "
        f"Current Cumulative Profit Curves with Buy & Hold Baseline"
    )

    print("=" * 80)
    print("[CONFIG]")
    print(f"ticker:          {ticker}")
    print(f"data_path:       {data_path}")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"start_date:      {start_date}")
    print(f"end_date:        {end_date}")
    print(f"trading_days:    {trading_days}")
    print(f"return_mode:     {args.return_mode}")
    print("=" * 80)

    # 读取数据
    price_df = load_price_df(data_path, start_date, end_date)
    action_df = load_action_df(checkpoint_path, start_date, end_date)

    print("\n[PRICE CHECK]")
    print(f"price rows:  {len(price_df)}")
    print(f"first date:  {price_df['date'].iloc[0]}")
    print(f"last date:   {price_df['date'].iloc[-1]}")
    print(f"first price: {price_df['price'].iloc[0]}")
    print(f"last price:  {price_df['price'].iloc[-1]}")

    print("\n[ACTION CHECK]")
    print(f"action rows before filling missing dates: {len(action_df)}")
    print(action_df.head(10).to_string(index=False))
    print("...")
    print(action_df.tail(10).to_string(index=False))

    # 构造审计表
    audit_df = build_audit_df(price_df, action_df, ticker)

    audit_df.to_csv(audit_csv, index=False)
    print(f"\n[OK] Audit CSV saved to: {audit_csv}")

    # 指标
    eval_df, standard_df = make_metrics_tables(
        audit_df=audit_df,
        num_price_points=len(price_df),
        trading_days=trading_days,
    )

    eval_df.to_csv(eval_metrics_csv)
    standard_df.to_csv(standard_metrics_csv)

    print("\n[EVAL-STYLE METRICS, should match eval_pipeline.py]")
    print(eval_df)

    print("\n[STANDARD METRICS, for comparison]")
    print(standard_df)

    print(f"\n[OK] Eval-style metrics saved to: {eval_metrics_csv}")
    print(f"[OK] Standard metrics saved to: {standard_metrics_csv}")

    # 决策统计
    print("\n[ACTION DISTRIBUTION]")
    print(audit_df["action_name"].value_counts())

    non_neutral = audit_df[audit_df["direction"] != 0].copy()

    if len(non_neutral) > 0:
        hit_rate = non_neutral["is_direction_correct"].mean()
    else:
        hit_rate = np.nan

    print("\n[DECISION QUALITY]")
    print(f"non-neutral days: {len(non_neutral)}")
    print(f"hit rate on non-neutral days: {hit_rate}")

    print("\n[WORST AGENT DAILY REWARDS]")
    print(
        audit_df.sort_values("agent_daily_log_reward", ascending=True)[
            [
                "date",
                "action_name",
                "price",
                "next_price",
                "log_return",
                "agent_daily_log_reward",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    # 绘图
    curve_df = build_curve_df(
        price_df=price_df,
        audit_df=audit_df,
        return_mode=args.return_mode,
    )

    plot_curve(
        curve_df=curve_df,
        output_png=output_png,
        ticker=ticker,
        agent_label=agent_label,
        title=title,
    )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
