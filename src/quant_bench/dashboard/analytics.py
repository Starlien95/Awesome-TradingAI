"""Schema-tolerant analytics for canonical and runtime dashboard runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from typing import Any, Literal

import numpy as np
import pandas as pd

from quant_bench.dashboard.data import DashboardIssue, RunData, RunSource
from quant_bench.dashboard.schema import (
    PERFORMANCE_ALIASES,
    PERFORMANCE_COLUMNS,
    SIGNAL_ALIASES,
)

HealthState = Literal["healthy", "completed", "degraded", "stale", "no_data", "error"]


@dataclass(frozen=True)
class RunHealth:
    run_key: str
    state: HealthState
    rows: int
    first_at: pd.Timestamp | None
    last_at: pd.Timestamp | None
    age_seconds: float | None
    expected_interval_seconds: float | None
    gap_count: int
    missing_intervals: int
    duplicate_count: int
    invalid_equity_count: int
    coverage_ratio: float | None
    issue_count: int
    message: str


def timeframe_seconds(value: str) -> float | None:
    text = str(value).strip().lower()
    if len(text) < 2 or not text[:-1].isdigit():
        return None
    multiplier = {"m": 60, "h": 3600, "d": 86400, "w": 604800}.get(text[-1])
    return float(int(text[:-1]) * multiplier) if multiplier is not None else None


def _numeric_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in candidates:
        if column in frame.columns:
            parsed = pd.to_numeric(frame[column], errors="coerce")
            result = result.where(result.notna(), parsed)
    return result


def _datetime_column(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    for column in PERFORMANCE_ALIASES["datetime_utc"]:
        if column not in frame.columns:
            continue
        values = frame[column]
        if column in {"timestamp", "bar_timestamp"}:
            numeric = pd.to_numeric(values, errors="coerce")
            if numeric.notna().sum() >= max(int(len(values) * 0.8), 1):
                numeric = numeric.where(numeric > 0)
                finite = numeric.dropna()
                unit = "ms" if not finite.empty and float(finite.abs().median()) > 1e11 else "s"
                parsed = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
            else:
                parsed = pd.to_datetime(values, utc=True, errors="coerce")
        else:
            parsed = pd.to_datetime(values, utc=True, errors="coerce")
        result = result.where(result.notna(), parsed)
    return result


def _normalize_equity(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric[(numeric > 0) & np.isfinite(numeric)]
    if valid.empty:
        return pd.Series(np.nan, index=numeric.index, dtype=float)
    return numeric / float(valid.iloc[0]) * 100.0


def canonical_performance(data: RunData) -> tuple[pd.DataFrame, list[DashboardIssue]]:
    frame = data.metrics.reset_index(drop=True).copy()
    issues = list(data.issues)
    if frame.empty:
        return pd.DataFrame(columns=PERFORMANCE_COLUMNS), issues

    prepared = pd.DataFrame(index=frame.index)
    prepared["datetime_utc"] = _datetime_column(frame)
    prepared["equity"] = _numeric_column(
        frame,
        PERFORMANCE_ALIASES["equity"],
    )
    prepared["_explicit_return_pct"] = _numeric_column(
        frame,
        PERFORMANCE_ALIASES["return_pct"],
    )
    prepared["period_return"] = _numeric_column(
        frame,
        PERFORMANCE_ALIASES["period_return"],
    )
    prepared["baseline_equity"] = _numeric_column(
        frame,
        PERFORMANCE_ALIASES["baseline_equity"],
    )
    prepared["_explicit_baseline_return"] = _numeric_column(
        frame,
        PERFORMANCE_ALIASES["baseline_return_pct"],
    )
    valid_times = prepared["datetime_utc"].dropna()
    if not valid_times.is_monotonic_increasing:
        issues.append(
            DashboardIssue(
                code="OUT_OF_ORDER_TIMESTAMPS",
                message="metrics 时间戳不是升序，分析视图已稳定排序。",
                severity="warning",
                run_key=data.source.key,
                path=data.source.files.get("metrics"),
            )
        )
    prepared = prepared.sort_values("datetime_utc", kind="stable").reset_index(drop=True)
    explicit_return_pct = prepared.pop("_explicit_return_pct")
    explicit_baseline_return = prepared.pop("_explicit_baseline_return")

    if prepared["equity"].notna().sum() == 0 and explicit_return_pct.notna().any():
        prepared["equity"] = 100.0 * (1.0 + explicit_return_pct / 100.0)
    if explicit_return_pct.notna().any():
        prepared["equity_normalized"] = (100.0 + explicit_return_pct).where(
            explicit_return_pct.notna(),
            _normalize_equity(prepared["equity"]),
        )
    elif data.source.kind == "canonical" and prepared["period_return"].notna().any():
        prepared["equity_normalized"] = (
            1.0 + prepared["period_return"].fillna(0.0)
        ).cumprod() * 100.0
    else:
        prepared["equity_normalized"] = _normalize_equity(prepared["equity"])
    prepared["return_pct"] = prepared["equity_normalized"] - 100.0
    calculated_period_return = prepared["equity"].pct_change(fill_method=None)
    prepared["period_return"] = prepared["period_return"].where(
        prepared["period_return"].notna(), calculated_period_return
    )
    if prepared["baseline_equity"].notna().sum() == 0 and explicit_baseline_return.notna().any():
        prepared["baseline_equity"] = 100.0 * (1.0 + explicit_baseline_return / 100.0)
    if explicit_baseline_return.notna().any():
        prepared["baseline_normalized"] = (100.0 + explicit_baseline_return).where(
            explicit_baseline_return.notna(),
            _normalize_equity(prepared["baseline_equity"]),
        )
    else:
        prepared["baseline_normalized"] = _normalize_equity(prepared["baseline_equity"])
    prepared["baseline_return_pct"] = prepared["baseline_normalized"] - 100.0
    prepared["alpha_pct"] = prepared["return_pct"] - prepared["baseline_return_pct"]
    running_peak = prepared["equity_normalized"].cummax()
    prepared["drawdown_pct"] = (prepared["equity_normalized"] / running_peak - 1.0) * 100.0
    prepared = prepared.dropna(subset=["datetime_utc"])

    if prepared.empty:
        issues.append(
            DashboardIssue(
                code="MISSING_TIME_COLUMN",
                message="metrics 中没有可解析的 datetime、timestamp 或 bar time。",
                severity="error",
                run_key=data.source.key,
                path=data.source.files.get("metrics"),
            )
        )
        return pd.DataFrame(columns=list(PERFORMANCE_COLUMNS)), issues
    if prepared["equity_normalized"].notna().sum() == 0:
        issues.append(
            DashboardIssue(
                code="MISSING_EQUITY_COLUMN",
                message="metrics 中没有可用的 equity 或 cumulative return 列。",
                severity="error",
                run_key=data.source.key,
                path=data.source.files.get("metrics"),
            )
        )
    return prepared[list(PERFORMANCE_COLUMNS)].reset_index(drop=True), issues


def infer_interval(performance: pd.DataFrame, frequency: str) -> float | None:
    declared = timeframe_seconds(frequency)
    if len(performance) < 2:
        return declared
    diffs = performance["datetime_utc"].dropna().sort_values().diff().dt.total_seconds()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return declared
    observed = float(diffs.median())
    return declared or observed


def evaluate_health(
    source: RunSource,
    performance: pd.DataFrame,
    issues: list[DashboardIssue],
    *,
    now: datetime | pd.Timestamp | None = None,
) -> RunHealth:
    relevant_issues = [issue for issue in issues if not issue.run_key or issue.run_key == source.key]
    error_count = sum(issue.severity == "error" for issue in relevant_issues)
    if performance.empty:
        state: HealthState = "error" if error_count else "no_data"
        return RunHealth(
            run_key=source.key,
            state=state,
            rows=0,
            first_at=None,
            last_at=None,
            age_seconds=None,
            expected_interval_seconds=timeframe_seconds(source.frequency),
            gap_count=0,
            missing_intervals=0,
            duplicate_count=0,
            invalid_equity_count=0,
            coverage_ratio=None,
            issue_count=len(relevant_issues),
            message="没有可分析的权益时间序列。",
        )

    timestamps = performance["datetime_utc"].dropna().sort_values()
    first_at = timestamps.iloc[0]
    last_at = timestamps.iloc[-1]
    interval = infer_interval(performance, source.frequency)
    duplicate_count = int(timestamps.duplicated().sum())
    diffs = timestamps.drop_duplicates().diff().dt.total_seconds().dropna()
    gap_mask = diffs > interval * 1.5 if interval else pd.Series(False, index=diffs.index)
    gap_count = int(gap_mask.sum())
    missing_intervals = (
        int(sum(max(round(float(value) / interval) - 1, 0) for value in diffs[gap_mask]))
        if interval and gap_count
        else 0
    )
    expected_rows = (
        max(round((last_at - first_at).total_seconds() / interval) + 1, 1)
        if interval
        else len(timestamps)
    )
    coverage_ratio = min(len(timestamps.drop_duplicates()) / expected_rows, 1.0)
    invalid_equity_count = int(
        ((performance["equity_normalized"] <= 0) | ~np.isfinite(performance["equity_normalized"])).sum()
    )
    reference_now = pd.Timestamp(now or datetime.now(timezone.utc))
    if reference_now.tzinfo is None:
        reference_now = reference_now.tz_localize("UTC")
    else:
        reference_now = reference_now.tz_convert("UTC")
    age_seconds = max((reference_now - last_at).total_seconds(), 0.0)

    if source.status == "failed" or error_count or invalid_equity_count:
        state = "error"
        message = "run 存在失败状态、读取错误或无效权益值。"
    elif duplicate_count or gap_count or any(issue.severity == "warning" for issue in relevant_issues):
        state = "degraded"
        message = "run 可读取，存在重复时间、周期缺口或兼容性告警。"
    elif source.status in {"completed", "historical"} or source.mode in {
        "research",
        "research_backtest",
        "backtest",
    }:
        state = "completed"
        message = "离线 benchmark 已完成，历史结束时间不参与 runtime 新鲜度告警。"
    elif interval and age_seconds > max(interval * 3.0, 3600.0):
        state = "stale"
        message = "最后有效记录已超过三个预期周期。"
    else:
        state = "healthy"
        message = "时间序列连续，未发现结构化读取错误。"

    return RunHealth(
        run_key=source.key,
        state=state,
        rows=len(performance),
        first_at=first_at,
        last_at=last_at,
        age_seconds=age_seconds,
        expected_interval_seconds=interval,
        gap_count=gap_count,
        missing_intervals=missing_intervals,
        duplicate_count=duplicate_count,
        invalid_equity_count=invalid_equity_count,
        coverage_ratio=coverage_ratio,
        issue_count=len(relevant_issues),
        message=message,
    )


def summarize_run(source: RunSource, performance: pd.DataFrame, health: RunHealth) -> dict[str, Any]:
    valid = performance.dropna(subset=["equity_normalized"]).copy()
    total_return = float(valid["return_pct"].iloc[-1]) if not valid.empty else np.nan
    max_drawdown = float(valid["drawdown_pct"].min()) if not valid.empty else np.nan
    period_returns = valid["period_return"].replace([np.inf, -np.inf], np.nan).dropna()
    interval = health.expected_interval_seconds
    periods_per_year = 365.0 * 86400.0 / interval if interval else np.nan
    volatility = (
        float(period_returns.std(ddof=1) * sqrt(periods_per_year) * 100.0)
        if len(period_returns) > 1 and np.isfinite(periods_per_year)
        else np.nan
    )
    sharpe = (
        float(period_returns.mean() / period_returns.std(ddof=1) * sqrt(periods_per_year))
        if len(period_returns) > 1
        and period_returns.std(ddof=1) > 0
        and np.isfinite(periods_per_year)
        else np.nan
    )
    return {
        "run_key": source.key,
        "run_id": source.run_id,
        "display_name": f"{source.display_name} · {source.frequency} · {source.mode}",
        "kind": source.kind,
        "method_family": source.method_family,
        "frequency": source.frequency,
        "mode": source.mode,
        "status": health.state,
        "rows": health.rows,
        "start": health.first_at,
        "end": health.last_at,
        "age_seconds": health.age_seconds,
        "coverage_pct": health.coverage_ratio * 100.0 if health.coverage_ratio is not None else np.nan,
        "gap_count": health.gap_count,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_drawdown,
        "annualized_volatility_pct": volatility,
        "sharpe": sharpe,
        "protocol_id": source.protocol_id,
        "dataset_sha256": source.dataset_sha256,
    }


def comparability_key(source: RunSource) -> str:
    if source.protocol_id and source.dataset_sha256:
        return f"verified:{source.protocol_id}:{source.dataset_sha256}"
    return f"unverified:{source.key}"


def comparison_frame(
    selected: list[tuple[RunSource, pd.DataFrame]],
    *,
    common_window: bool,
    limit_per_run: int = 600,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared: list[tuple[RunSource, pd.DataFrame]] = []
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for source, frame in selected:
        current = frame.dropna(subset=["datetime_utc", "equity_normalized"]).copy()
        current = current.sort_values("datetime_utc")
        if current.empty:
            continue
        prepared.append((source, current))
        starts.append(current["datetime_utc"].iloc[0])
        ends.append(current["datetime_utc"].iloc[-1])
    common_start = max(starts) if starts else None
    common_end = min(ends) if ends else None
    if common_window and common_start is not None and common_end is not None and common_start > common_end:
        return pd.DataFrame(), {
            "mode": "common_window",
            "start": common_start,
            "end": common_end,
            "reason": "EMPTY_COMMON_WINDOW",
        }

    rows: list[pd.DataFrame] = []
    for source, frame in prepared:
        current = frame
        if common_window and common_start is not None and common_end is not None:
            current = current[
                (current["datetime_utc"] >= common_start)
                & (current["datetime_utc"] <= common_end)
            ].copy()
        if current.empty:
            continue
        base = float(current["equity_normalized"].iloc[0])
        current["view_return_pct"] = (current["equity_normalized"] / base - 1.0) * 100.0
        current["view_drawdown_pct"] = (
            current["equity_normalized"] / current["equity_normalized"].cummax() - 1.0
        ) * 100.0
        current["run_key"] = source.key
        current["display_name"] = (
            f"{source.display_name} · {source.frequency} · {source.mode}"
        )
        if limit_per_run > 0 and len(current) > limit_per_run:
            required = {0, len(current) - 1}
            required.add(int(current["view_return_pct"].idxmin() - current.index[0]))
            required.add(int(current["view_return_pct"].idxmax() - current.index[0]))
            required.add(int(current["view_drawdown_pct"].idxmin() - current.index[0]))
            sample_count = max(limit_per_run - len(required), 0)
            sampled = np.linspace(0, len(current) - 1, sample_count, dtype=int)
            positions = sorted(required | {int(value) for value in sampled})[:limit_per_run]
            current = current.iloc[positions]
        rows.append(
            current[
                [
                    "datetime_utc",
                    "run_key",
                    "display_name",
                    "view_return_pct",
                    "view_drawdown_pct",
                ]
            ]
        )
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return combined, {
        "mode": "common_window" if common_window else "full_history",
        "start": common_start if common_window else None,
        "end": common_end if common_window else None,
        "reason": None,
    }


def canonical_signals(data: RunData) -> pd.DataFrame:
    frame = data.signals.copy()
    if frame.empty:
        return frame
    result = pd.DataFrame(index=frame.index)
    result["datetime_utc"] = _datetime_column(frame)
    symbol_column = next((name for name in SIGNAL_ALIASES["symbol"] if name in frame), None)
    result["symbol"] = frame[symbol_column].astype(str) if symbol_column else "unknown"
    result["score"] = _numeric_column(frame, SIGNAL_ALIASES["score"])
    result["label"] = _numeric_column(frame, SIGNAL_ALIASES["label"])
    label_column = next(
        (name for name in SIGNAL_ALIASES["signal_label"] if name in frame), None
    )
    if label_column:
        result["signal_label"] = frame[label_column].astype(str)
    else:
        result["signal_label"] = np.where(result["score"] > 0, "positive", "non_positive")
    return result.dropna(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)
