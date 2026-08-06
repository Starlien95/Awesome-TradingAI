"""Read-only local analytics and runtime monitoring dashboard."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from platformdirs import user_data_path

from quant_bench.dashboard.analytics import (
    RunHealth,
    canonical_performance,
    canonical_signals,
    comparability_key,
    comparison_frame,
    evaluate_health,
    summarize_run,
)
from quant_bench.dashboard.charts import (
    comparison_drawdown_figure,
    comparison_figure,
    health_coverage_figure,
    run_performance_figure,
    signal_score_figure,
)
from quant_bench.dashboard.data import (
    RUN_DATA_FIELDS,
    DashboardIssue,
    RunData,
    RunSource,
    WorkspaceInventory,
    discover_workspace,
    load_run,
)

st.set_page_config(page_title="Awesome TradingAI", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.58);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 12px;
        padding: 0.55rem 0.75rem;
    }
    .qb-status {font-weight: 700; letter-spacing: .02em;}
    .qb-note {color: #94a3b8; font-size: .9rem;}
    div[data-testid="stMarkdownContainer"] code {
        overflow-wrap: anywhere;
        white-space: normal;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_WORKSPACE = Path(
    os.environ.get(
        "QUANT_BENCH_DASHBOARD_ROOT",
        str(user_data_path("quant-bench", appauthor=False)),
    )
).expanduser().resolve()

STATUS_LABELS = {
    "healthy": "HEALTHY",
    "completed": "COMPLETED",
    "degraded": "DEGRADED",
    "stale": "STALE",
    "no_data": "NO DATA",
    "error": "ERROR",
}


@st.cache_data(ttl=10, show_spinner=False)
def _cached_inventory(workspace: str) -> WorkspaceInventory:
    return discover_workspace(workspace)


def _file_fingerprint(
    source: RunSource,
    file_keys: tuple[str, ...],
) -> tuple[tuple[str, int, int], ...]:
    rows: list[tuple[str, int, int]] = []
    paths = [source.files[key] for key in file_keys if key in source.files]
    if source.manifest_path is not None:
        paths.append(source.manifest_path)
    for path in sorted(set(paths)):
        try:
            stat = path.stat()
        except OSError:
            rows.append((str(path), -1, -1))
        else:
            rows.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(rows)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_run(
    workspace: str,
    run_key: str,
    file_keys: tuple[str, ...],
    fingerprint: tuple[tuple[str, int, int], ...],
) -> RunData:
    del fingerprint
    inventory = discover_workspace(workspace)
    source = next(item for item in inventory.runs if item.key == run_key)
    return load_run(source, file_keys=file_keys)


def _load_run_cached(
    workspace: Path,
    source: RunSource,
    *,
    file_keys: tuple[str, ...] = RUN_DATA_FIELDS,
) -> RunData:
    return _cached_run(
        str(workspace),
        source.key,
        file_keys,
        _file_fingerprint(source, file_keys),
    )


def _format_number(value: Any, pattern: str, default: str = "—") -> str:
    if value is None or pd.isna(value):
        return default
    try:
        return pattern.format(value)
    except (TypeError, ValueError):
        return default


def _format_age(seconds: Any) -> str:
    if seconds is None or pd.isna(seconds):
        return "—"
    seconds = max(float(seconds), 0.0)
    if seconds >= 86400:
        return f"{seconds / 86400:.1f}d"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.0f}s"


def _issue_frame(issues: list[DashboardIssue]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "severity": issue.severity,
                "code": issue.code,
                "run_key": issue.run_key,
                "message": issue.message,
                "path": str(issue.path) if issue.path else "",
            }
            for issue in issues
        ]
    )


def _prepare_workspace(
    inventory: WorkspaceInventory,
) -> tuple[
    dict[str, RunData],
    dict[str, pd.DataFrame],
    dict[str, RunHealth],
    pd.DataFrame,
    list[DashboardIssue],
]:
    data_by_key: dict[str, RunData] = {}
    performance_by_key: dict[str, pd.DataFrame] = {}
    health_by_key: dict[str, RunHealth] = {}
    issues = list(inventory.issues)
    summaries: list[dict[str, Any]] = []
    for source in inventory.runs:
        data = _load_run_cached(inventory.root, source, file_keys=("metrics",))
        performance, run_issues = canonical_performance(data)
        health = evaluate_health(source, performance, run_issues)
        data_by_key[source.key] = data
        performance_by_key[source.key] = performance
        health_by_key[source.key] = health
        issues.extend(run_issues)
        summaries.append(summarize_run(source, performance, health))
    summary = pd.DataFrame(summaries)
    return data_by_key, performance_by_key, health_by_key, summary, issues


def _summary_display(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "display_name",
        "kind",
        "method_family",
        "frequency",
        "mode",
        "status",
        "rows",
        "coverage_pct",
        "gap_count",
        "total_return_pct",
        "max_drawdown_pct",
        "sharpe",
        "end",
        "age_seconds",
    ]
    display = summary[[column for column in columns if column in summary]].copy()
    display = display.rename(
        columns={
            "display_name": "run",
            "method_family": "method",
            "coverage_pct": "coverage (%)",
            "gap_count": "gaps",
            "total_return_pct": "return (%)",
            "max_drawdown_pct": "max drawdown (%)",
            "age_seconds": "age",
        }
    )
    if "age" in display:
        display["age"] = display["age"].map(_format_age)
    return display


def _render_overview(
    inventory: WorkspaceInventory,
    summary: pd.DataFrame,
    performance_by_key: dict[str, pd.DataFrame],
) -> None:
    st.title("Awesome TradingAI · Workspace Overview")
    st.caption("统一查看离线 benchmark、paper/demo runtime 与历史 timeframe 日志。所有数据来自本地 workspace。")
    if summary.empty:
        st.warning("当前 workspace 没有可展示的 run。")
        return

    filter_cols = st.columns(4)
    kinds = sorted(summary["kind"].dropna().unique().tolist())
    methods = sorted(summary["method_family"].dropna().unique().tolist())
    frequencies = sorted(summary["frequency"].dropna().unique().tolist())
    states = sorted(summary["status"].dropna().unique().tolist())
    selected_kinds = filter_cols[0].multiselect("Run type", kinds, default=kinds)
    selected_methods = filter_cols[1].multiselect("Method", methods, default=methods)
    selected_frequencies = filter_cols[2].multiselect("Frequency", frequencies, default=frequencies)
    selected_states = filter_cols[3].multiselect("Health", states, default=states)
    filtered = summary[
        summary["kind"].isin(selected_kinds)
        & summary["method_family"].isin(selected_methods)
        & summary["frequency"].isin(selected_frequencies)
        & summary["status"].isin(selected_states)
    ].copy()

    counts = Counter(summary["status"])
    metric_cols = st.columns(6)
    metric_cols[0].metric("Runs", len(summary))
    metric_cols[1].metric("Healthy", counts.get("healthy", 0))
    metric_cols[2].metric("Completed", counts.get("completed", 0))
    metric_cols[3].metric("Degraded", counts.get("degraded", 0))
    metric_cols[4].metric("Stale", counts.get("stale", 0))
    metric_cols[5].metric("Errors", counts.get("error", 0))

    available_keys = filtered["run_key"].tolist()
    selected_keys = st.multiselect(
        "曲线中的 runs",
        available_keys,
        default=available_keys[: min(len(available_keys), 12)],
        format_func=lambda key: str(summary.set_index("run_key").loc[key, "display_name"]),
    )
    common_window = st.toggle("使用共同有效区间", value=len(selected_keys) > 1)
    source_by_key = {source.key: source for source in inventory.runs}
    selected = [
        (source_by_key[key], performance_by_key[key])
        for key in selected_keys
        if key in source_by_key and key in performance_by_key
    ]
    comparison, window = comparison_frame(selected, common_window=common_window)
    comparable_groups = {comparability_key(source_by_key[key]) for key in selected_keys if key in source_by_key}
    if len(comparable_groups) > 1:
        st.warning("所选 run 跨越不同 protocol、dataset 或 runtime 语义。曲线用于探索，不生成统一排名。")
    if window.get("reason") == "EMPTY_COMMON_WINDOW":
        st.error("所选 runs 没有共同有效时间区间。切换到完整历史，或缩小选择范围。")
    elif not comparison.empty:
        if common_window:
            st.info(f"共同区间：{window['start']} 到 {window['end']}")
        st.plotly_chart(comparison_figure(comparison), use_container_width=True)
        with st.expander("查看区间回撤"):
            st.plotly_chart(comparison_drawdown_figure(comparison), use_container_width=True)
    else:
        st.info("当前筛选没有可画制的权益曲线。")

    st.subheader("Run Summary")
    st.dataframe(_summary_display(filtered), use_container_width=True, hide_index=True)


def _render_explorer(
    inventory: WorkspaceInventory,
    data_by_key: dict[str, RunData],
    performance_by_key: dict[str, pd.DataFrame],
    health_by_key: dict[str, RunHealth],
    issues: list[DashboardIssue],
) -> None:
    st.title("Run Explorer")
    source_by_key = {source.key: source for source in inventory.runs}
    selected_key = st.selectbox(
        "选择 run",
        list(source_by_key),
        format_func=lambda key: (
            f"{source_by_key[key].display_name} · {source_by_key[key].frequency}"
            f" · {source_by_key[key].kind}"
        ),
    )
    source = source_by_key[selected_key]
    data = _load_run_cached(inventory.root, source)
    performance = performance_by_key[selected_key]
    health = health_by_key[selected_key]
    st.subheader(source.display_name)
    st.caption(
        f"`{source.kind}` · `{source.method_family}` · `{source.frequency}` · "
        f"mode `{source.mode}` · status `{STATUS_LABELS[health.state]}`"
    )
    if source.mode == "live":
        st.warning("该 run 标记为 live。页面保持只读，不执行账户查询、持仓同步或订单操作。")

    latest_return = performance["return_pct"].dropna().iloc[-1] if not performance.empty and performance["return_pct"].notna().any() else np.nan
    max_drawdown = performance["drawdown_pct"].min() if not performance.empty else np.nan
    cards = st.columns(6)
    cards[0].metric("Health", STATUS_LABELS[health.state])
    cards[1].metric("Rows", health.rows)
    cards[2].metric("Return", _format_number(latest_return, "{:+.2f}%"))
    cards[3].metric("Max drawdown", _format_number(max_drawdown, "{:.2f}%"))
    cards[4].metric("Coverage", _format_number(health.coverage_ratio * 100 if health.coverage_ratio is not None else np.nan, "{:.1f}%"))
    cards[5].metric("Data age", _format_age(health.age_seconds))
    st.info(health.message)

    performance_tab, signals_tab, execution_tab, metadata_tab = st.tabs(
        ["Performance", "Signals", "Execution", "Files & Metadata"]
    )
    with performance_tab:
        if performance.empty:
            st.error("该 run 没有可解析的权益时间序列。")
        else:
            st.plotly_chart(
                run_performance_figure(performance, source.display_name),
                use_container_width=True,
            )
            with st.expander("最近数据行"):
                st.dataframe(performance.tail(200).sort_values("datetime_utc", ascending=False), use_container_width=True, hide_index=True)

    with signals_tab:
        signals = canonical_signals(data)
        if signals.empty:
            st.info("没有可用的 predictions 或 signals CSV。")
        else:
            st.plotly_chart(signal_score_figure(signals.tail(3000)), use_container_width=True)
            latest_time = signals["datetime_utc"].max()
            latest = signals[signals["datetime_utc"] == latest_time].sort_values("score", ascending=False)
            st.subheader("Latest signal snapshot")
            st.dataframe(latest, use_container_width=True, hide_index=True)

    with execution_tab:
        if all(
            frame.empty
            for frame in (
                data.trades,
                data.orders,
                data.fills,
                data.failed_orders,
                data.rebalance_plan,
                data.volume,
                data.account_snapshots,
            )
        ):
            st.info("该 run 没有 execution CSV。离线 benchmark 不要求这些 runtime 文件。")
        if not data.trades.empty:
            st.subheader("Trades")
            st.dataframe(data.trades.tail(300).iloc[::-1], use_container_width=True, hide_index=True)
        if not data.orders.empty:
            st.subheader("Orders")
            st.dataframe(data.orders.tail(300).iloc[::-1], use_container_width=True, hide_index=True)
        if not data.failed_orders.empty:
            st.subheader("Failed orders")
            st.dataframe(data.failed_orders.tail(300).iloc[::-1], use_container_width=True, hide_index=True)
        if not data.fills.empty:
            st.subheader("Fills")
            st.dataframe(data.fills.tail(300).iloc[::-1], use_container_width=True, hide_index=True)
        if not data.rebalance_plan.empty:
            st.subheader("Rebalance plan")
            st.dataframe(data.rebalance_plan.tail(300).iloc[::-1], use_container_width=True, hide_index=True)
        if not data.volume.empty:
            st.subheader("Volume")
            st.dataframe(data.volume.tail(300).iloc[::-1], use_container_width=True, hide_index=True)
        if not data.account_snapshots.empty:
            with st.expander("Account snapshots (local only)"):
                st.warning("该表包含本地账户状态，禁止直接复制到公共展示或公开 fixture。")
                st.dataframe(
                    data.account_snapshots.tail(300).iloc[::-1],
                    use_container_width=True,
                    hide_index=True,
                )

    with metadata_tab:
        file_rows = []
        for file_key, path in sorted(source.files.items()):
            try:
                stat = path.stat()
            except OSError:
                size = None
                modified = None
            else:
                size = stat.st_size
                modified = pd.Timestamp(stat.st_mtime, unit="s", tz="UTC")
            file_rows.append({"type": file_key, "path": str(path), "bytes": size, "modified_utc": modified})
        st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)
        with st.expander("Manifest"):
            st.json(source.manifest)
        run_issues = list(
            {
                (issue.code, str(issue.path), issue.message): issue
                for issue in [
                    *(item for item in issues if item.run_key == source.key),
                    *data.issues,
                ]
            }.values()
        )
        if run_issues:
            st.subheader("Run issues")
            st.dataframe(_issue_frame(run_issues), use_container_width=True, hide_index=True)


def _render_health(summary: pd.DataFrame, health_by_key: dict[str, RunHealth]) -> None:
    st.title("Runtime Health")
    st.caption("运行新鲜度按各自 frequency 判断。已完成的离线 benchmark 不因历史结束时间被标记为 stale。")
    if summary.empty:
        st.info("没有 run health 数据。")
        return
    st.plotly_chart(health_coverage_figure(summary), use_container_width=True)
    rows = []
    for run_key, health in health_by_key.items():
        rows.append(
            {
                "run_key": run_key,
                "state": health.state,
                "message": health.message,
                "rows": health.rows,
                "first_at": health.first_at,
                "last_at": health.last_at,
                "data_age": _format_age(health.age_seconds),
                "expected_interval": _format_age(health.expected_interval_seconds),
                "gaps": health.gap_count,
                "missing_intervals": health.missing_intervals,
                "duplicates": health.duplicate_count,
                "invalid_equity": health.invalid_equity_count,
                "coverage_pct": health.coverage_ratio * 100 if health.coverage_ratio is not None else np.nan,
                "issues": health.issue_count,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_quality(inventory: WorkspaceInventory, issues: list[DashboardIssue]) -> None:
    st.title("Data Quality")
    st.caption("读取错误会被隔离到对应 run。dashboard 不修改源 CSV，也不静默修补缺口。")
    issue_table = _issue_frame(issues)
    if issue_table.empty:
        st.success("没有发现结构化读取问题。")
    else:
        severity_filter = st.multiselect(
            "Severity",
            ["error", "warning", "info"],
            default=["error", "warning", "info"],
        )
        visible = issue_table[issue_table["severity"].isin(severity_filter)]
        st.dataframe(visible, use_container_width=True, hide_index=True)
        st.download_button(
            "下载问题清单 CSV",
            visible.to_csv(index=False).encode("utf-8"),
            file_name="quant_bench_dashboard_issues.csv",
            mime="text/csv",
        )

    st.subheader("Discovery contract")
    st.markdown(
        """
        - Canonical: `runs/<run_id>/run_manifest.json` with `backtest/returns.csv` and `predictions.csv`
        - Runtime: `runs/**/run_manifest.json` with manifest-declared metrics, signals, trades and volume paths
        - Legacy: `timeframes/<frequency>/logs/<model>/*_metrics.csv`
        """
    )
    st.caption(f"Workspace: `{inventory.root}`")


def _render_empty_workspace(inventory: WorkspaceInventory) -> None:
    st.title("Awesome TradingAI")
    st.warning("没有发现可视化 run。")
    st.code(
        "quant-bench quickstart --offline --workspace ./qb-workspace\n"
        "quant-bench dashboard --workspace ./qb-workspace",
        language="bash",
    )
    st.caption(f"已检查：`{inventory.root / 'runs'}` 和 `{inventory.root / 'timeframes'}`")
    if inventory.issues:
        st.dataframe(_issue_frame(list(inventory.issues)), use_container_width=True, hide_index=True)


def _render_dashboard(workspace: Path, page: str) -> None:
    inventory = _cached_inventory(str(workspace))
    if not inventory.runs:
        _render_empty_workspace(inventory)
        return
    with st.spinner("读取本地 run artifacts..."):
        data_by_key, performance_by_key, health_by_key, summary, issues = _prepare_workspace(inventory)
    if page == "Overview":
        _render_overview(inventory, summary, performance_by_key)
    elif page == "Run Explorer":
        _render_explorer(
            inventory,
            data_by_key,
            performance_by_key,
            health_by_key,
            issues,
        )
    elif page == "Health":
        _render_health(summary, health_by_key)
    else:
        _render_quality(inventory, issues)


st.sidebar.title("Awesome TradingAI")
st.sidebar.caption("Local analytics · read only · offline")
page = st.sidebar.radio("Page", ["Overview", "Run Explorer", "Health", "Data Quality"])
st.sidebar.text_input("Workspace", value=str(DEFAULT_WORKSPACE), disabled=True)
refresh_label = st.sidebar.selectbox("Auto refresh", ["Off", "10s", "30s", "60s"], index=0)
if st.sidebar.button("Refresh now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
st.sidebar.divider()
st.sidebar.caption("页面不训练、不推理、不访问交易所，也不执行订单。")

run_every = None if refresh_label == "Off" else refresh_label


@st.fragment(run_every=run_every)
def _workspace_fragment() -> None:
    _render_dashboard(DEFAULT_WORKSPACE, page)


_workspace_fragment()
