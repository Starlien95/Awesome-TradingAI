import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 0. 防止 FinMemAgent.load_checkpoint 因 OPENAI_API_KEY 报错
# ============================================================

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

if "OPENAI_API_KEY" not in os.environ:
    if "DASHSCOPE_API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]
    else:
        os.environ["OPENAI_API_KEY"] = "dummy_key_for_loading_checkpoint_only"

from .engine.agent import FinMemAgent


# ============================================================
# 1. 配置区：你主要改这里
# ============================================================

RUNS = {
    "BTC": {
        "data_path": "data/btc.json",
        "checkpoint_path": "results/exp/qwen3-max/BTC/test_checkpoint",
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "agent_label": "BTC Agent",
    },
    "ETH": {
        "data_path": "data/eth.json",
        "checkpoint_path": "results/exp/qwen3-max/ETH/test_checkpoint",
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "agent_label": "ETH Agent",
    },
}

OUTPUT_DIR = "curve_outputs"

# simple: 普通累计收益率 = exp(cumulative log return) - 1，更适合画图展示
# log:    累计对数收益，和官方 eval 的 CR 更接近
RETURN_MODE = "simple"

TRADING_DAYS = 365


# ============================================================
# 2. 读取数据
# ============================================================

def resolve_agent_checkpoint_path(checkpoint_path: str) -> str:
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


def load_price_df(data_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for date_str, contents in data.items():
        if contents is None:
            continue

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
            f"No price data found in {data_path} from {start_date} to {end_date}"
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
        raise ValueError("Action record has no 'position' column.")

    action_df["date"] = pd.to_datetime(action_df["date"]).dt.date

    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()

    action_df = action_df[
        (action_df["date"] >= start) & (action_df["date"] <= end)
    ].copy()

    action_df = action_df.rename(columns={"position": "direction"})
    action_df["direction"] = action_df["direction"].astype(float)

    keep_cols = ["date", "symbol", "direction"]

    if "price" in action_df.columns:
        keep_cols.append("price")

    keep_cols = [c for c in keep_cols if c in action_df.columns]
    action_df = action_df[keep_cols].copy()

    return action_df.sort_values("date").reset_index(drop=True)


def infer_action_name(direction: float) -> str:
    if direction > 0:
        return "BUY / LONG"
    if direction < 0:
        return "SELL / SHORT"
    return "HOLD / NEUTRAL"


# ============================================================
# 3. 构造审计数据
# ============================================================

def build_audit_df(price_df: pd.DataFrame, action_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if action_df["date"].duplicated().any():
        print(f"[WARNING] {ticker}: duplicated action dates found, keep last.")
        action_df = action_df.drop_duplicates(subset=["date"], keep="last")

    full_dates = price_df["date"].tolist()
    existing_action_dates = set(action_df["date"].tolist())

    missing_dates = [d for d in full_dates if d not in existing_action_dates]

    if missing_dates:
        print(f"[WARNING] {ticker}: missing action dates filled with direction=0:")
        print(missing_dates)

        missing_df = pd.DataFrame({
            "date": missing_dates,
            "symbol": [ticker] * len(missing_dates),
            "direction": [0.0] * len(missing_dates),
        })

        action_df = pd.concat([action_df, missing_df], axis=0)

    action_df = action_df.sort_values("date").reset_index(drop=True)

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

    merged["agent_daily_log_reward"] = merged["direction"] * merged["log_return"]
    merged["agent_cum_log_reward"] = merged["agent_daily_log_reward"].cumsum()
    merged["buy_hold_cum_log_reward"] = merged["log_return"].cumsum()

    merged["is_direction_correct"] = np.where(
        merged["direction"] == 0,
        np.nan,
        np.sign(merged["direction"]) == np.sign(merged["log_return"]),
    )

    return merged


def build_curve_df(price_df: pd.DataFrame, audit_df: pd.DataFrame, return_mode: str) -> pd.DataFrame:
    curve_dates = pd.to_datetime(price_df["date"].tolist())

    agent_cum_log = [0.0] + audit_df["agent_cum_log_reward"].tolist()
    buy_hold_cum_log = [0.0] + audit_df["buy_hold_cum_log_reward"].tolist()

    if len(agent_cum_log) != len(curve_dates):
        raise ValueError(
            f"Curve length mismatch: dates={len(curve_dates)}, "
            f"agent={len(agent_cum_log)}"
        )

    curve_df = pd.DataFrame({
        "date": curve_dates,
        "agent_cum_log": agent_cum_log,
        "buy_hold_cum_log": buy_hold_cum_log,
    })

    if return_mode == "simple":
        curve_df["agent_curve"] = (np.exp(curve_df["agent_cum_log"]) - 1) * 100
        curve_df["buy_hold_curve"] = (np.exp(curve_df["buy_hold_cum_log"]) - 1) * 100
    elif return_mode == "log":
        curve_df["agent_curve"] = curve_df["agent_cum_log"] * 100
        curve_df["buy_hold_curve"] = curve_df["buy_hold_cum_log"] * 100
    else:
        raise ValueError("return_mode must be 'simple' or 'log'.")

    return curve_df


# ============================================================
# 4. 指标计算
# ============================================================

def standard_deviation(x):
    x = list(x)
    if len(x) <= 1:
        return 0.0
    mean = sum(x) / len(x)
    variance = sum((r - mean) ** 2 for r in x) / (len(x) - 1)
    return variance ** 0.5


def calculate_max_drawdown_eval_style(daily_returns):
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


def calculate_eval_style_metrics(daily_rewards, num_price_points, trading_days=365):
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


def calculate_metrics_for_run(audit_df: pd.DataFrame, num_price_points: int):
    buy_hold_daily = audit_df["log_return"].tolist()
    agent_daily = audit_df["agent_daily_log_reward"].tolist()

    buy_hold_metrics = calculate_eval_style_metrics(
        buy_hold_daily,
        num_price_points=num_price_points,
        trading_days=TRADING_DAYS,
    )

    agent_metrics = calculate_eval_style_metrics(
        agent_daily,
        num_price_points=num_price_points,
        trading_days=TRADING_DAYS,
    )

    return pd.DataFrame({
        "Buy & Hold": buy_hold_metrics,
        "Agent": agent_metrics,
    })


# ============================================================
# 5. 绘图
# ============================================================

def plot_single_curve(ticker: str, curve_df: pd.DataFrame, agent_label: str, output_path: str):
    agent_final = curve_df["agent_curve"].iloc[-1]
    buy_hold_final = curve_df["buy_hold_curve"].iloc[-1]

    plt.figure(figsize=(14, 7), dpi=150)

    plt.plot(
        curve_df["date"],
        curve_df["agent_curve"],
        linewidth=2.7,
        label=agent_label,
    )

    plt.plot(
        curve_df["date"],
        curve_df["buy_hold_curve"],
        linewidth=2.3,
        linestyle="--",
        label=f"{ticker} Buy & Hold",
    )

    plt.axhline(0, linewidth=1, color="gray")

    plt.title(
        f"{ticker}USD Backtest Progress: Cumulative Profit Curves with Buy & Hold Baseline",
        fontsize=15,
    )

    plt.xlabel("Date")
    plt.ylabel("Cumulative Profit (%)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left")

    last_date = curve_df["date"].iloc[-1]

    plt.scatter(last_date, agent_final, s=45, zorder=5)
    plt.scatter(last_date, buy_hold_final, s=45, zorder=5)

    plt.annotate(
        f"{agent_label}: {agent_final:.2f}%",
        xy=(last_date, agent_final),
        xytext=(-145, -16),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.25", fc="white"),
        fontsize=10,
    )

    plt.annotate(
        f"Buy & Hold: {buy_hold_final:.2f}%",
        xy=(last_date, buy_hold_final),
        xytext=(-145, 16),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.25", fc="white"),
        fontsize=10,
    )

    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"[OK] Single curve saved: {output_path}")


def plot_combined_curves(all_curves: dict, output_path: str):
    plt.figure(figsize=(16, 8), dpi=150)

    for ticker, info in all_curves.items():
        curve_df = info["curve_df"]
        agent_label = info["agent_label"]

        plt.plot(
            curve_df["date"],
            curve_df["agent_curve"],
            linewidth=2.5,
            label=agent_label,
        )

        plt.plot(
            curve_df["date"],
            curve_df["buy_hold_curve"],
            linewidth=2.0,
            linestyle="--",
            label=f"{ticker} Buy & Hold",
        )

    plt.axhline(0, linewidth=1, color="gray")

    plt.title(
        "Crypto Backtest Progress: Agent Cumulative Profit Curves with Buy & Hold Baselines",
        fontsize=15,
    )

    plt.xlabel("Date")
    plt.ylabel("Cumulative Profit (%)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left")

    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"[OK] Combined curve saved: {output_path}")


# ============================================================
# 6. 主流程
# ============================================================

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_curves = {}
    summary_rows = []

    for ticker, cfg in RUNS.items():
        print("=" * 80)
        print(f"[RUN] {ticker}")
        print(f"data_path:       {cfg['data_path']}")
        print(f"checkpoint_path: {cfg['checkpoint_path']}")
        print(f"start_date:      {cfg['start_date']}")
        print(f"end_date:        {cfg['end_date']}")

        price_df = load_price_df(
            data_path=cfg["data_path"],
            start_date=cfg["start_date"],
            end_date=cfg["end_date"],
        )

        action_df = load_action_df(
            checkpoint_path=cfg["checkpoint_path"],
            start_date=cfg["start_date"],
            end_date=cfg["end_date"],
        )

        audit_df = build_audit_df(
            price_df=price_df,
            action_df=action_df,
            ticker=ticker,
        )

        curve_df = build_curve_df(
            price_df=price_df,
            audit_df=audit_df,
            return_mode=RETURN_MODE,
        )

        metrics_df = calculate_metrics_for_run(
            audit_df=audit_df,
            num_price_points=len(price_df),
        )

        audit_path = output_dir / f"{ticker.lower()}_audit.csv"
        curve_csv_path = output_dir / f"{ticker.lower()}_curve.csv"
        metrics_path = output_dir / f"{ticker.lower()}_metrics.csv"
        single_png_path = output_dir / f"{ticker.lower()}_cumulative_profit_curve.png"

        audit_df.to_csv(audit_path, index=False)
        curve_df.to_csv(curve_csv_path, index=False)
        metrics_df.to_csv(metrics_path)

        print("\n[METRICS]")
        print(metrics_df)

        print("\n[ACTION DISTRIBUTION]")
        print(audit_df["action_name"].value_counts())

        non_neutral = audit_df[audit_df["direction"] != 0]

        if len(non_neutral) > 0:
            hit_rate = non_neutral["is_direction_correct"].mean()
        else:
            hit_rate = np.nan

        print("\n[DECISION QUALITY]")
        print(f"non-neutral days: {len(non_neutral)}")
        print(f"hit rate: {hit_rate}")

        plot_single_curve(
            ticker=ticker,
            curve_df=curve_df,
            agent_label=cfg["agent_label"],
            output_path=str(single_png_path),
        )

        all_curves[ticker] = {
            "curve_df": curve_df,
            "agent_label": cfg["agent_label"],
        }

        summary_rows.append({
            "ticker": ticker,
            "agent_final_profit_pct": curve_df["agent_curve"].iloc[-1],
            "buy_hold_final_profit_pct": curve_df["buy_hold_curve"].iloc[-1],
            "agent_cr_log": metrics_df.loc["Cumulative Return", "Agent"],
            "buy_hold_cr_log": metrics_df.loc["Cumulative Return", "Buy & Hold"],
            "agent_sr": metrics_df.loc["Sharpe Ratio", "Agent"],
            "buy_hold_sr": metrics_df.loc["Sharpe Ratio", "Buy & Hold"],
            "agent_mdd": metrics_df.loc["Max DrawDown", "Agent"],
            "buy_hold_mdd": metrics_df.loc["Max DrawDown", "Buy & Hold"],
            "agent_av": metrics_df.loc["Annualized Volatility", "Agent"],
            "buy_hold_av": metrics_df.loc["Annualized Volatility", "Buy & Hold"],
            "non_neutral_days": len(non_neutral),
            "hit_rate": hit_rate,
        })

    combined_png_path = output_dir / "combined_btc_eth_cumulative_profit_curve.png"

    plot_combined_curves(
        all_curves=all_curves,
        output_path=str(combined_png_path),
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    print("=" * 80)
    print("[SUMMARY]")
    print(summary_df)
    print(f"[OK] Summary saved: {summary_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
