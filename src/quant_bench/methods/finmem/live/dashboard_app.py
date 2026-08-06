# finmem_multi_asset_dashboard.py
"""Multi-asset Streamlit dashboard for FinMem runs.

It discovers every symbol-scoped logger directory under the configured
workspace. BTC and ETH are shown as examples when no run has been written.

    timeframes/1d/logs/finmem_btc/finmem_btc_metrics.csv
    timeframes/1d/logs/finmem_btc/finmem_btc_trades.csv
    timeframes/1d/logs/finmem_btc/finmem_btc_signals.csv
    timeframes/1d/logs/finmem_btc/finmem_btc_volume.csv

    timeframes/1d/logs/finmem_eth/finmem_eth_metrics.csv
    timeframes/1d/logs/finmem_eth/finmem_eth_trades.csv
    timeframes/1d/logs/finmem_eth/finmem_eth_signals.csv
    timeframes/1d/logs/finmem_eth/finmem_eth_volume.csv

Notes:
- For dashboard compatibility, ETH logs may still contain column names such as
  btc_price / btc_qty / okx_btc_qty. This dashboard normalizes them into:
    asset_price
    asset_qty
    asset_value_usdt
- okx_* fields represent the local 500-USDT strategy account ledger in your current logger.

Run:
    streamlit run finmem_multi_asset_dashboard.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. Basic config
# ============================================================

PROJECT_ROOT = Path(os.getenv("FINMEM_WORKSPACE", "~/.local/share/quant-bench/methods/finmem")).expanduser().resolve()
TIMEFRAMES_DIR = PROJECT_ROOT / "timeframes"

DEFAULT_ASSETS = {
    "BTC": {
        "model_name": "finmem_btc",
        "log_dir": TIMEFRAMES_DIR / "1d" / "logs" / "finmem_btc",
    },
    "ETH": {
        "model_name": "finmem_eth",
        "log_dir": TIMEFRAMES_DIR / "1d" / "logs" / "finmem_eth",
    },
}

COLORS = {
    "BTC": "#f59e0b",
    "ETH": "#38bdf8",
    "paper": "#22c55e",
    "account": "#a78bfa",
    "baseline": "#facc15",
    "drawdown": "#f43f5e",
    "positive": "#34d399",
    "negative": "#fb7185",
    "neutral": "#94a3b8",
    "panel": "rgba(15, 23, 42, 0.55)",
    "grid": "rgba(148, 163, 184, 0.16)",
}

METRIC_NUMERIC_COLS = [
    "cycle_id",
    "timestamp",
    "btc_price",
    "asset_price",
    "paper_equity",
    "paper_position",
    "paper_prev_position",
    "paper_price_return",
    "paper_period_return",
    "paper_cr",
    "paper_sr",
    "paper_av",
    "paper_mdd",
    "paper_bh_equity",
    "paper_bh_return_pct",
    "okx_cash_usdt",
    "okx_available_usdt",
    "okx_frozen_usdt",
    "okx_total_eq_reported",
    "okx_btc_qty",
    "okx_btc_value_usdt",
    "okx_relevant_equity",
    "okx_equity",
    "okx_initial_equity",
    "okx_period_return",
    "okx_cr",
    "okx_sr",
    "okx_av",
    "okx_mdd",
    "strategy_initial_capital_usdt",
    "strategy_cash_usdt",
    "strategy_btc_qty",
    "strategy_equity_usdt",
    "strategy_position_value_usdt",
    "strategy_equity",
    "strategy_pnl",
    "strategy_returns_pct",
    "baseline_equity",
    "baseline_pnl",
    "baseline_returns_pct",
    "alpha_return_pct",
    "cycle_volume_usdt",
    "cumulative_total_volume_usdt",
    "investor_position",
    "target_position",
    "decision_target_position",
]

REQUIRED_METRIC_COLS = [
    "datetime",
    "symbol",
    "asset_price",
    "paper_equity",
    "okx_equity",
    "paper_cr",
    "okx_cr",
]


st.set_page_config(
    page_title="FinMem Multi-Asset Dashboard",
    layout="wide",
    page_icon="📊",
)

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.52);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 14px;
        padding: 0.6rem 0.85rem;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. IO
# ============================================================

def load_json_file(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


@st.cache_data(ttl=10)
def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        unit = "ms" if ts.dropna().median() > 1e12 else "s"
        df["datetime"] = pd.to_datetime(ts, unit=unit, errors="coerce")
    else:
        df["datetime"] = pd.NaT

    for col in METRIC_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "cycle_id" not in df.columns:
        df["cycle_id"] = np.arange(1, len(df) + 1)

    df = df.sort_values("datetime", ascending=True).reset_index(drop=True)
    return df


def discover_runs() -> Dict[str, Dict[str, Path]]:
    """
    Discover all timeframes/*/logs/*/*_metrics.csv runs.
    Uses compound key: f"{asset}/{model_name}" to support multiple models
    per asset (e.g. BTC/finmem_btc and BTC/deepfund_btc).
    """
    runs: Dict[str, Dict[str, Path]] = {}

    # Add defaults first, even if files are currently absent.
    for asset, cfg in DEFAULT_ASSETS.items():
        model_name = cfg["model_name"]
        log_dir = cfg["log_dir"]
        key = f"{asset}/{model_name}"
        runs[key] = {
            "asset": asset,
            "model_name": model_name,
            "timeframe": "1d",
            "log_dir": log_dir,
            "metrics_path": log_dir / f"{model_name}_metrics.csv",
            "trades_path": log_dir / f"{model_name}_trades.csv",
            "signals_path": log_dir / f"{model_name}_signals.csv",
            "volume_path": log_dir / f"{model_name}_volume.csv",
            "metadata_path": log_dir / f"{model_name}_metadata.json",
        }

    # Add discovered runs.
    for metrics_path in sorted(TIMEFRAMES_DIR.glob("*/*/*/*_metrics.csv")):
        prefix = metrics_path.name.replace("_metrics.csv", "")
        model_name = metrics_path.parent.name
        metadata_path = metrics_path.parent / f"{prefix}_metadata.json"
        metadata = load_json_file(metadata_path)

        asset = str(metadata.get("symbol") or "").upper().strip()
        if not asset:
            # Infer from finmem_btc / finmem_eth / deepfund_btc.
            if model_name.lower().endswith("_btc"):
                asset = "BTC"
            elif model_name.lower().endswith("_eth"):
                asset = "ETH"
            else:
                asset = model_name.replace("finmem_", "").replace("deepfund_", "").upper()

        key = f"{asset}/{model_name}"

        if key in runs:
            runs[key].update({
                "model_name": model_name,
                "log_dir": metrics_path.parent,
                "metrics_path": metrics_path,
                "trades_path": metrics_path.parent / f"{prefix}_trades.csv",
                "signals_path": metrics_path.parent / f"{prefix}_signals.csv",
                "volume_path": metrics_path.parent / f"{prefix}_volume.csv",
                "metadata_path": metadata_path,
            })
        else:
            runs[key] = {
                "asset": asset,
                "model_name": model_name,
                "timeframe": "1d",
                "log_dir": metrics_path.parent,
                "metrics_path": metrics_path,
                "trades_path": metrics_path.parent / f"{prefix}_trades.csv",
                "signals_path": metrics_path.parent / f"{prefix}_signals.csv",
                "volume_path": metrics_path.parent / f"{prefix}_volume.csv",
                "metadata_path": metadata_path,
            }

    return runs


def normalize_metrics(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    out["asset"] = asset

    if "symbol" not in out.columns:
        out["symbol"] = asset
    out["symbol"] = out["symbol"].fillna(asset).astype(str).str.upper()

    # Normalize price.
    if "asset_price" not in out.columns:
        if "btc_price" in out.columns:
            out["asset_price"] = out["btc_price"]
        else:
            out["asset_price"] = np.nan

    # Normalize account qty/value.
    if "asset_qty" not in out.columns:
        out["asset_qty"] = out["okx_btc_qty"] if "okx_btc_qty" in out.columns else np.nan
    if "asset_value_usdt" not in out.columns:
        out["asset_value_usdt"] = out["okx_btc_value_usdt"] if "okx_btc_value_usdt" in out.columns else np.nan

    # Normalize strategy state.
    if "strategy_asset_qty" not in out.columns:
        out["strategy_asset_qty"] = out["strategy_btc_qty"] if "strategy_btc_qty" in out.columns else np.nan

    for col in REQUIRED_METRIC_COLS:
        if col not in out.columns:
            out[col] = np.nan

    return out


def normalize_trades(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["asset"] = asset
    if "symbol" not in out.columns:
        out["symbol"] = asset
    out["symbol"] = out["symbol"].fillna(asset).astype(str).str.upper()

    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")

    numeric_cols = [
        "cycle_id",
        "timestamp",
        "price",
        "investor_position",
        "target_position",
        "quantity",
        "cycle_volume_usdt",
        "strategy_cash_usdt",
        "strategy_btc_qty",
        "strategy_equity_usdt",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def load_asset_bundle(asset: str, info: Dict[str, Path]) -> Dict[str, object]:
    metrics = normalize_metrics(load_csv(str(info["metrics_path"])), asset)
    trades = normalize_trades(load_csv(str(info["trades_path"])), asset)
    signals = load_csv(str(info["signals_path"]))
    volume = load_csv(str(info["volume_path"]))
    metadata = load_json_file(info["metadata_path"])

    return {
        "asset": asset,
        "info": info,
        "metrics": metrics,
        "trades": trades,
        "signals": signals,
        "volume": volume,
        "metadata": metadata,
        "exists": not metrics.empty,
    }


def concat_nonempty(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ============================================================
# 3. Formatting and chart helpers
# ============================================================

def fmt_pct_decimal(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x) * 100:+.2f}%"


def fmt_pct_plain(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):+.2f}%"


def fmt_num(x, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):+.{digits}f}"


def fmt_money(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):,.4f}"


def fmt_price(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):,.2f}"


def style_figure(fig: go.Figure, height: int = 420, hovermode: str = "x unified") -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["panel"],
        margin=dict(l=18, r=18, t=58, b=18),
        hovermode=hovermode,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    return fig


def compute_drawdown(equity: pd.Series) -> pd.Series:
    equity = pd.to_numeric(equity, errors="coerce")
    peak = equity.cummax()
    return equity / peak - 1.0


def build_multi_equity_figure(metrics_all: pd.DataFrame, equity_col: str, title: str) -> go.Figure:
    fig = go.Figure()

    for asset in sorted(metrics_all["asset"].dropna().unique()):
        sub = metrics_all[metrics_all["asset"] == asset].copy()
        if sub.empty or equity_col not in sub.columns:
            continue

        color = COLORS.get(asset, COLORS["neutral"])
        fig.add_trace(
            go.Scatter(
                x=sub["datetime"],
                y=sub[equity_col],
                mode="lines+markers",
                name=f"{asset} {equity_col}",
                line=dict(color=color, width=2.7),
                marker=dict(size=7, color=color),
                customdata=np.stack([sub["cycle_id"], sub["final_signal"].astype(str)], axis=-1)
                if "final_signal" in sub.columns
                else None,
                hovertemplate=(
                    f"{asset}<br>C%{{customdata[0]}}<br>%{{x|%Y-%m-%d %H:%M}}"
                    "<br>Signal: %{customdata[1]}"
                    "<br>Equity: %{y:,.6f}<extra></extra>"
                )
                if "final_signal" in sub.columns
                else f"{asset}<br>%{{x|%Y-%m-%d %H:%M}}<br>Equity: %{{y:,.6f}}<extra></extra>",
            )
        )

    fig.update_layout(title=title)
    fig.update_yaxes(title_text="Equity")
    fig.update_xaxes(title_text="Time")
    return style_figure(fig, height=440)


def build_normalized_equity_figure(metrics_all: pd.DataFrame, equity_col: str, title: str) -> go.Figure:
    fig = go.Figure()

    for asset in sorted(metrics_all["asset"].dropna().unique()):
        sub = metrics_all[metrics_all["asset"] == asset].copy()
        if sub.empty or equity_col not in sub.columns:
            continue

        valid = sub[equity_col].dropna()
        if valid.empty or valid.iloc[0] == 0:
            continue

        sub["normalized_equity"] = sub[equity_col] / valid.iloc[0]
        color = COLORS.get(asset, COLORS["neutral"])

        fig.add_trace(
            go.Scatter(
                x=sub["datetime"],
                y=sub["normalized_equity"],
                mode="lines+markers",
                name=f"{asset}",
                line=dict(color=color, width=2.7),
                marker=dict(size=7, color=color),
                hovertemplate=f"{asset}<br>%{{x|%Y-%m-%d %H:%M}}<br>Norm Equity: %{{y:.6f}}<extra></extra>",
            )
        )

    fig.add_hline(y=1.0, line_dash="dot", line_color=COLORS["neutral"], opacity=0.5)
    fig.update_layout(title=title)
    fig.update_yaxes(title_text="Normalized Equity")
    fig.update_xaxes(title_text="Time")
    return style_figure(fig, height=440)


def build_strategy_vs_buyhold_figure(metrics: pd.DataFrame, asset: str, strategy_col: str = "okx_equity") -> go.Figure:
    """
    Single-asset comparison:
    - Strategy curve uses okx_equity for 500-USDT Strategy Account, or paper_equity for Paper Benchmark.
    - Buy & Hold is derived from paper_bh_equity and rescaled to the strategy curve's first value.
    """
    fig = go.Figure()

    if metrics.empty or strategy_col not in metrics.columns:
        return style_figure(fig, height=440)

    plot_df = metrics.copy().sort_values("datetime")
    strategy_valid = plot_df[strategy_col].dropna()
    if strategy_valid.empty:
        return style_figure(fig, height=440)

    strategy_start = float(strategy_valid.iloc[0])

    final_signal_series = (
        plot_df["final_signal"].astype(str)
        if "final_signal" in plot_df.columns
        else pd.Series([""] * len(plot_df), index=plot_df.index)
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["datetime"],
            y=plot_df[strategy_col],
            mode="lines+markers",
            name=f"{asset} Strategy",
            line=dict(color=COLORS.get(asset, COLORS["account"]), width=2.8),
            marker=dict(size=7),
            customdata=np.stack([plot_df["cycle_id"], final_signal_series], axis=-1),
            hovertemplate=(
                f"{asset} Strategy<br>C%{{customdata[0]}}<br>%{{x|%Y-%m-%d %H:%M}}"
                "<br>Signal: %{customdata[1]}"
                "<br>Equity: %{y:,.6f}<extra></extra>"
            ),
        )
    )

    if "paper_bh_equity" in plot_df.columns:
        bh = pd.to_numeric(plot_df["paper_bh_equity"], errors="coerce")
        bh_valid = bh.dropna()
        if not bh_valid.empty and float(bh_valid.iloc[0]) != 0:
            bh_scaled = bh / float(bh_valid.iloc[0]) * strategy_start
            fig.add_trace(
                go.Scatter(
                    x=plot_df["datetime"],
                    y=bh_scaled,
                    mode="lines",
                    name=f"{asset} Buy & Hold",
                    line=dict(color=COLORS["baseline"], width=2.3, dash="dash"),
                    hovertemplate=(
                        f"{asset} Buy & Hold<br>%{{x|%Y-%m-%d %H:%M}}"
                        "<br>Scaled Equity: %{y:,.6f}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(title=f"{asset} Strategy vs Buy & Hold")
    fig.update_yaxes(title_text="Equity")
    fig.update_xaxes(title_text="Time")
    return style_figure(fig, height=440)


def build_multi_strategy_vs_buyhold_normalized(metrics_all: pd.DataFrame, strategy_col: str = "okx_equity") -> go.Figure:
    """
    Multi-asset normalized comparison:
    solid line  = strategy
    dashed line = buy & hold
    """
    fig = go.Figure()

    for asset in sorted(metrics_all["asset"].dropna().unique()):
        sub = metrics_all[metrics_all["asset"] == asset].copy().sort_values("datetime")
        if sub.empty or strategy_col not in sub.columns:
            continue

        color = COLORS.get(asset, COLORS["neutral"])

        strategy = pd.to_numeric(sub[strategy_col], errors="coerce")
        strategy_valid = strategy.dropna()
        if not strategy_valid.empty and float(strategy_valid.iloc[0]) != 0:
            strategy_norm = strategy / float(strategy_valid.iloc[0])
            fig.add_trace(
                go.Scatter(
                    x=sub["datetime"],
                    y=strategy_norm,
                    mode="lines+markers",
                    name=f"{asset} Strategy",
                    line=dict(color=color, width=2.7),
                    marker=dict(size=7),
                    hovertemplate=f"{asset} Strategy<br>%{{x|%Y-%m-%d %H:%M}}<br>Norm: %{{y:.6f}}<extra></extra>",
                )
            )

        if "paper_bh_equity" in sub.columns:
            bh = pd.to_numeric(sub["paper_bh_equity"], errors="coerce")
            bh_valid = bh.dropna()
            if not bh_valid.empty and float(bh_valid.iloc[0]) != 0:
                bh_norm = bh / float(bh_valid.iloc[0])
                fig.add_trace(
                    go.Scatter(
                        x=sub["datetime"],
                        y=bh_norm,
                        mode="lines",
                        name=f"{asset} Buy & Hold",
                        line=dict(color=color, width=2.2, dash="dash"),
                        hovertemplate=f"{asset} Buy & Hold<br>%{{x|%Y-%m-%d %H:%M}}<br>Norm: %{{y:.6f}}<extra></extra>",
                    )
                )

    fig.add_hline(y=1.0, line_dash="dot", line_color=COLORS["neutral"], opacity=0.5)
    fig.update_layout(title="Strategy vs Buy & Hold Normalized")
    fig.update_yaxes(title_text="Normalized Equity")
    fig.update_xaxes(title_text="Time")
    return style_figure(fig, height=480)


def build_price_signal_figure(metrics: pd.DataFrame, asset: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=metrics["datetime"],
            y=metrics["asset_price"],
            mode="lines",
            name=f"{asset} Price",
            line=dict(color=COLORS.get(asset, COLORS["neutral"]), width=2.2),
            hovertemplate=f"%{{x|%Y-%m-%d %H:%M}}<br>{asset}: %{{y:,.2f}}<extra></extra>",
        )
    )

    if "investor_signal" in metrics.columns:
        sig_meta = [
            ("buy", "BUY", COLORS["positive"], "triangle-up"),
            ("hold", "HOLD", COLORS["neutral"], "circle"),
            ("sell", "SELL", COLORS["negative"], "triangle-down"),
        ]
        for sig, label, color, marker_symbol in sig_meta:
            subset = metrics[metrics["investor_signal"].astype(str).str.lower() == sig]
            if subset.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=subset["datetime"],
                    y=subset["asset_price"],
                    mode="markers",
                    name=f"Investor {label}",
                    marker=dict(size=11, color=color, symbol=marker_symbol, line=dict(width=1, color="white")),
                    customdata=np.stack([subset["cycle_id"], subset["final_signal"].astype(str)], axis=-1),
                    hovertemplate=(
                        "周期 C%{customdata[0]}<br>%{x|%Y-%m-%d %H:%M}"
                        f"<br>Investor {label}<br>Final: %{{customdata[1]}}"
                        f"<br>{asset}: %{{y:,.2f}}"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_yaxes(title_text=f"{asset} Price")
    fig.update_xaxes(title_text="Time")
    return style_figure(fig, height=420, hovermode="closest")


def build_account_position_figure(metrics: pd.DataFrame, asset: str) -> go.Figure:
    fig = go.Figure()

    if "asset_value_usdt" in metrics.columns:
        fig.add_trace(
            go.Scatter(
                x=metrics["datetime"],
                y=metrics["asset_value_usdt"],
                mode="lines+markers",
                name=f"{asset} Value USDT",
                line=dict(color=COLORS.get(asset, COLORS["account"]), width=2.3),
                marker=dict(size=6),
                hovertemplate=f"%{{x|%Y-%m-%d %H:%M}}<br>{asset} Value: %{{y:,.4f}}<extra></extra>",
            )
        )

    if "okx_equity" in metrics.columns:
        fig.add_trace(
            go.Scatter(
                x=metrics["datetime"],
                y=metrics["okx_equity"],
                mode="lines+markers",
                name="Strategy Equity",
                line=dict(color=COLORS["account"], width=2.5, dash="dash"),
                marker=dict(size=6),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Equity: %{y:,.4f}<extra></extra>",
            )
        )

    fig.update_yaxes(title_text="USDT")
    fig.update_xaxes(title_text="Time")
    return style_figure(fig, height=420)


def build_bar_metric_figure(latest_rows: pd.DataFrame, metric_col: str, title: str, pct: bool = True) -> go.Figure:
    fig = go.Figure()

    if latest_rows.empty or metric_col not in latest_rows.columns:
        return style_figure(fig, height=320)

    y = latest_rows[metric_col].astype(float)
    if pct:
        y = y * 100.0

    colors = [COLORS.get(a, COLORS["neutral"]) for a in latest_rows["asset"]]

    fig.add_trace(
        go.Bar(
            x=latest_rows["asset"],
            y=y,
            marker_color=colors,
            text=[f"{v:+.2f}%" if pct else f"{v:+.3f}" for v in y],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:+.4f}<extra></extra>",
        )
    )

    fig.update_layout(title=title)
    fig.update_yaxes(title_text="%" if pct else metric_col)
    return style_figure(fig, height=330, hovermode="closest")


# ============================================================
# 4. App
# ============================================================

st.title("📊 FinMem 多资产双账本面板")
st.caption(
    "整合 BTC 与 ETH 的 Paper Benchmark、500-USDT Strategy Account 账本，并展示 Strategy vs Buy & Hold 对比。"
)

runs = discover_runs()

available_assets = []
missing_assets = []

# Build display labels from compound keys (asset/model_name).
run_labels: Dict[str, str] = {}
bundles: Dict[str, Dict[str, object]] = {}
for key, info in runs.items():
    asset = info["asset"]
    run_labels[key] = f"{asset} ({info['model_name']})"
    bundle = load_asset_bundle(asset, info)
    bundles[key] = bundle
    if bundle["exists"]:
        available_assets.append(key)
    else:
        missing_assets.append(key)

st.sidebar.header("数据源")
sorted_keys = sorted(runs.keys())
selected_keys = st.sidebar.multiselect(
    "选择资产",
    sorted_keys,
    format_func=lambda k: run_labels.get(k, k),
    default=sorted(available_assets) if available_assets else sorted_keys,
)

ledger_view = st.sidebar.radio(
    "主账本视图",
    ["Strategy Account", "Paper Benchmark"],
    index=0,
)

if st.sidebar.button("刷新缓存"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
for key in sorted_keys:
    info = runs[key]
    exists = "✅" if key in available_assets else "⚠️"
    st.sidebar.caption(f"{exists} {run_labels[key]}: {info['metrics_path']}")

if not selected_keys:
    st.warning("请至少选择一个资产。")
    st.stop()

selected_bundles = [bundles[k] for k in selected_keys if bundles.get(k, {}).get("exists")]
if not selected_bundles:
    st.warning(
        "所选资产还没有 metrics.csv。请先运行对应 logger，例如 BTC/ETH 单轮模拟盘。"
    )
    st.stop()

metrics_all = concat_nonempty([b["metrics"] for b in selected_bundles])
trades_all = concat_nonempty([b["trades"] for b in selected_bundles])

if metrics_all.empty:
    st.warning("没有可用 metrics 数据。")
    st.stop()

# Latest rows per asset.
latest_rows = (
    metrics_all.sort_values(["asset", "datetime"])
    .groupby("asset", as_index=False)
    .tail(1)
    .sort_values("asset")
)

# ============================================================
# 4.1 Overview
# ============================================================

st.subheader("总览")

overview_cols = st.columns(4)
total_initial = latest_rows["okx_initial_equity"].fillna(500.0).sum() if "okx_initial_equity" in latest_rows else 500.0 * len(latest_rows)
total_equity = latest_rows["okx_equity"].fillna(0.0).sum() if "okx_equity" in latest_rows else np.nan
total_return = total_equity / total_initial - 1.0 if total_initial else np.nan
total_cash = latest_rows["okx_cash_usdt"].fillna(0.0).sum() if "okx_cash_usdt" in latest_rows else np.nan

overview_cols[0].metric("资产数", len(latest_rows))
overview_cols[1].metric("策略资金池总权益", fmt_money(total_equity), fmt_pct_decimal(total_return))
overview_cols[2].metric("策略现金合计", fmt_money(total_cash))
overview_cols[3].metric("总初始资金", fmt_money(total_initial))

st.markdown("#### 最新资产状态")

asset_cols = st.columns(len(latest_rows))
for col, (_, row) in zip(asset_cols, latest_rows.iterrows()):
    asset = row["asset"]
    with col:
        st.metric(f"{asset} Strategy Equity", fmt_money(row.get("okx_equity")), fmt_pct_decimal(row.get("okx_cr")))
        st.caption(
            f"price={fmt_price(row.get('asset_price'))} · "
            f"signal={row.get('final_signal', '—')} · "
            f"pos={row.get('position_signal', '—')}"
        )
        st.caption(
            f"cash={fmt_money(row.get('okx_cash_usdt'))} · "
            f"value={fmt_money(row.get('asset_value_usdt'))}"
        )

tab_overview, tab_asset, tab_trades, tab_raw = st.tabs(
    ["📈 多资产对比", "🔍 单资产详情", "🧾 合并交易记录", "📄 原始数据"]
)

with tab_overview:
    st.plotly_chart(
        build_multi_equity_figure(metrics_all, "okx_equity", "Strategy Account Equity by Asset"),
        use_container_width=True,
    )

    st.plotly_chart(
        build_multi_strategy_vs_buyhold_normalized(metrics_all, "okx_equity"),
        use_container_width=True,
    )

    st.plotly_chart(
        build_normalized_equity_figure(metrics_all, "paper_equity", "Paper Benchmark Normalized Equity"),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_bar_metric_figure(latest_rows, "okx_cr", "Latest Strategy CR", pct=True), use_container_width=True)
    with c2:
        st.plotly_chart(build_bar_metric_figure(latest_rows, "paper_cr", "Latest Paper CR", pct=True), use_container_width=True)

    compare_cols = [
        "asset",
        "cycle_id",
        "datetime",
        "action_date",
        "asset_price",
        "investor_signal",
        "investor_position",
        "position_signal",
        "final_signal",
        "okx_equity",
        "okx_cr",
        "okx_sr",
        "okx_av",
        "okx_mdd",
        "paper_equity",
        "paper_cr",
        "paper_sr",
        "paper_av",
        "paper_mdd",
        "okx_cash_usdt",
        "asset_qty",
        "asset_value_usdt",
        "risk_reason",
    ]
    compare_cols = [c for c in compare_cols if c in metrics_all.columns]

    with st.expander("查看最新对比表", expanded=True):
        st.dataframe(
            latest_rows[compare_cols],
            use_container_width=True,
            hide_index=True,
        )

with tab_asset:
    selected_asset_detail = st.selectbox(
        "选择单资产详情",
        sorted(metrics_all["asset"].dropna().unique()),
    )

    detail = metrics_all[metrics_all["asset"] == selected_asset_detail].copy()
    detail_latest = detail.iloc[-1]

    st.subheader(f"{selected_asset_detail} · {ledger_view}")

    metric_cols = st.columns(8)
    if ledger_view == "Strategy Account":
        metric_cols[0].metric("CR", fmt_pct_decimal(detail_latest.get("okx_cr")))
        metric_cols[1].metric("SR", fmt_num(detail_latest.get("okx_sr")))
        metric_cols[2].metric("AV", fmt_pct_decimal(detail_latest.get("okx_av")))
        metric_cols[3].metric("MDD", fmt_pct_decimal(detail_latest.get("okx_mdd")))
        metric_cols[4].metric("Equity", fmt_money(detail_latest.get("okx_equity")))
        metric_cols[5].metric("Price", fmt_price(detail_latest.get("asset_price")))
        metric_cols[6].metric("Investor", str(detail_latest.get("investor_signal", "—")))
        metric_cols[7].metric("Final", str(detail_latest.get("final_signal", "—")))

        st.plotly_chart(
            build_strategy_vs_buyhold_figure(detail, selected_asset_detail, "okx_equity"),
            use_container_width=True,
        )
        st.plotly_chart(
            build_account_position_figure(detail, selected_asset_detail),
            use_container_width=True,
        )

    else:
        metric_cols[0].metric("CR", fmt_pct_decimal(detail_latest.get("paper_cr")))
        metric_cols[1].metric("SR", fmt_num(detail_latest.get("paper_sr")))
        metric_cols[2].metric("AV", fmt_pct_decimal(detail_latest.get("paper_av")))
        metric_cols[3].metric("MDD", fmt_pct_decimal(detail_latest.get("paper_mdd")))
        metric_cols[4].metric("Equity", fmt_money(detail_latest.get("paper_equity")))
        metric_cols[5].metric("Price", fmt_price(detail_latest.get("asset_price")))
        metric_cols[6].metric("Investor", str(detail_latest.get("investor_signal", "—")))
        metric_cols[7].metric("Final", str(detail_latest.get("final_signal", "—")))

        st.plotly_chart(
            build_strategy_vs_buyhold_figure(detail, selected_asset_detail, "paper_equity"),
            use_container_width=True,
        )

    st.plotly_chart(
        build_price_signal_figure(detail, selected_asset_detail),
        use_container_width=True,
    )

    detail_cols = [
        "cycle_id",
        "datetime",
        "action_date",
        "asset_price",
        "investor_signal",
        "investor_position",
        "target_position",
        "position_signal",
        "final_signal",
        "trade_side",
        "quantity",
        "okx_equity",
        "okx_cr",
        "paper_equity",
        "paper_cr",
        "okx_cash_usdt",
        "asset_qty",
        "asset_value_usdt",
        "cycle_volume_usdt",
        "risk_reason",
    ]
    detail_cols = [c for c in detail_cols if c in detail.columns]

    with st.expander(f"{selected_asset_detail} metrics 明细"):
        st.dataframe(
            detail[detail_cols].sort_values("cycle_id", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

with tab_trades:
    if trades_all.empty:
        st.info("暂无 trades.csv 记录。")
    else:
        trade_cols = [
            "asset",
            "cycle_id",
            "datetime",
            "symbol",
            "action_date",
            "price",
            "investor_signal",
            "investor_position",
            "target_position",
            "position_signal",
            "final_signal",
            "trade_side",
            "quantity",
            "tgt_ccy",
            "cycle_volume_usdt",
            "risk_reason",
            "execution_status",
            "order_code",
            "order_id",
            "order_success",
            "dry_run",
            "execute_order",
            "is_simulated",
            "strategy_cash_usdt",
            "strategy_btc_qty",
            "strategy_equity_usdt",
            "strategy_position_state",
        ]
        trade_cols = [c for c in trade_cols if c in trades_all.columns]

        st.dataframe(
            trades_all[trade_cols].sort_values(["datetime", "asset"], ascending=[False, True]),
            use_container_width=True,
            hide_index=True,
        )

with tab_raw:
    st.subheader("All metrics")
    st.dataframe(
        metrics_all.sort_values(["datetime", "asset"], ascending=[False, True]),
        use_container_width=True,
        hide_index=True,
    )

    if not trades_all.empty:
        st.subheader("All trades")
        st.dataframe(
            trades_all.sort_values(["datetime", "asset"], ascending=[False, True]),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Loaded runs")
    run_rows = []
    for key, info in runs.items():
        run_rows.append({
            "asset": info["asset"],
            "model_name": info["model_name"],
            "metrics_path": str(info["metrics_path"]),
            "exists": Path(info["metrics_path"]).exists(),
        })
    st.dataframe(pd.DataFrame(run_rows), use_container_width=True, hide_index=True)
