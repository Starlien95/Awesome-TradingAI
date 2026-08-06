"""Plotly figures for the local analytics dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = [
    "#38BDF8",
    "#22C55E",
    "#F59E0B",
    "#A78BFA",
    "#F97316",
    "#14B8A6",
    "#F43F5E",
    "#84CC16",
    "#60A5FA",
    "#E879F9",
]


def _style(figure: go.Figure, *, height: int = 440, hovermode: str = "x unified") -> go.Figure:
    figure.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.62)",
        margin={"l": 28, "r": 24, "t": 54, "b": 38},
        hovermode=hovermode,
        legend={"orientation": "h", "y": 1.04, "x": 0},
    )
    figure.update_xaxes(gridcolor="rgba(148,163,184,0.14)", automargin=True)
    figure.update_yaxes(gridcolor="rgba(148,163,184,0.14)", automargin=True)
    return figure


def comparison_figure(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for index, (run_key, group) in enumerate(frame.groupby("run_key", sort=False)):
        name = str(group["display_name"].iloc[0])
        figure.add_trace(
            go.Scatter(
                x=group["datetime_utc"],
                y=group["view_return_pct"],
                mode="lines",
                name=name,
                line={"width": 2.1, "color": PALETTE[index % len(PALETTE)]},
                customdata=np.full(len(group), run_key),
                hovertemplate=(
                    f"{name}<br>%{{x|%Y-%m-%d %H:%M UTC}}"
                    "<br>区间收益 %{y:+.2f}%<extra></extra>"
                ),
            )
        )
    figure.add_hline(y=0, line_dash="dot", line_color="#64748B")
    figure.update_layout(title="归一化收益曲线")
    figure.update_yaxes(title_text="收益 (%)")
    return _style(figure, height=520)


def comparison_drawdown_figure(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for index, (_, group) in enumerate(frame.groupby("run_key", sort=False)):
        name = str(group["display_name"].iloc[0])
        figure.add_trace(
            go.Scatter(
                x=group["datetime_utc"],
                y=group["view_drawdown_pct"],
                mode="lines",
                name=name,
                line={"width": 1.8, "color": PALETTE[index % len(PALETTE)]},
                hovertemplate=(
                    f"{name}<br>%{{x|%Y-%m-%d %H:%M UTC}}"
                    "<br>回撤 %{y:.2f}%<extra></extra>"
                ),
            )
        )
    figure.update_layout(title="区间回撤")
    figure.update_yaxes(title_text="回撤 (%)")
    return _style(figure, height=380)


def run_performance_figure(frame: pd.DataFrame, display_name: str) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.52, 0.25, 0.23],
        vertical_spacing=0.08,
        subplot_titles=("累计收益", "回撤", "单周期收益"),
    )
    figure.add_trace(
        go.Scatter(
            x=frame["datetime_utc"],
            y=frame["return_pct"],
            name=display_name,
            line={"color": "#22C55E", "width": 2.6},
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>收益 %{y:+.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    if frame["baseline_return_pct"].notna().any():
        figure.add_trace(
            go.Scatter(
                x=frame["datetime_utc"],
                y=frame["baseline_return_pct"],
                name="Run 内置基准",
                line={"color": "#F59E0B", "width": 2.0, "dash": "dash"},
                hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>基准 %{y:+.2f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=frame["datetime_utc"],
            y=frame["drawdown_pct"],
            name="回撤",
            fill="tozeroy",
            line={"color": "#F43F5E", "width": 1.6},
            fillcolor="rgba(244,63,94,0.16)",
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>回撤 %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    bar_colors = np.where(frame["period_return"].fillna(0) >= 0, "#34D399", "#FB7185")
    figure.add_trace(
        go.Bar(
            x=frame["datetime_utc"],
            y=frame["period_return"] * 100.0,
            name="周期收益",
            marker_color=bar_colors,
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>周期收益 %{y:+.3f}%<extra></extra>",
        ),
        row=3,
        col=1,
    )
    figure.update_yaxes(title_text="收益 (%)", row=1, col=1)
    figure.update_yaxes(title_text="回撤 (%)", row=2, col=1)
    figure.update_yaxes(title_text="收益 (%)", row=3, col=1)
    return _style(figure, height=760)


def signal_score_figure(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if frame.empty:
        return _style(figure)
    for symbol, group in frame.groupby("symbol", sort=True):
        figure.add_trace(
            go.Scatter(
                x=group["datetime_utc"],
                y=group["score"],
                mode="lines+markers",
                name=str(symbol),
                marker={"size": 5},
                hovertemplate=(
                    f"{symbol}<br>%{{x|%Y-%m-%d %H:%M UTC}}"
                    "<br>score %{y:+.5f}<extra></extra>"
                ),
            )
        )
    figure.add_hline(y=0, line_dash="dot", line_color="#64748B")
    figure.update_layout(title="信号得分轨迹")
    figure.update_yaxes(title_text="score")
    return _style(figure, height=460)


def health_coverage_figure(summary: pd.DataFrame) -> go.Figure:
    current = summary.sort_values(["coverage_pct", "display_name"], na_position="first")
    colors = current["status"].map(
        {
            "healthy": "#22C55E",
            "completed": "#38BDF8",
            "degraded": "#F59E0B",
            "stale": "#F97316",
            "no_data": "#94A3B8",
            "error": "#F43F5E",
        }
    )
    figure = go.Figure(
        go.Bar(
            x=current["coverage_pct"],
            y=current["display_name"],
            orientation="h",
            marker_color=colors,
            customdata=np.stack(
                [current["gap_count"].fillna(0), current["rows"].fillna(0)], axis=-1
            ),
            hovertemplate=(
                "%{y}<br>覆盖率 %{x:.1f}%"
                "<br>缺口数 %{customdata[0]}<br>记录数 %{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(title="时间网格覆盖率")
    figure.update_xaxes(title_text="覆盖率 (%)", range=[0, 105])
    return _style(figure, height=max(360, 90 + len(current) * 32), hovermode="closest")
