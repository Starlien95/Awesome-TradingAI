from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from quant_bench.dashboard.analytics import (
    canonical_performance,
    comparison_frame,
    evaluate_health,
)
from quant_bench.dashboard.data import RunData, RunSource


def _source(
    tmp_path: Path,
    *,
    key: str = "runtime:r1",
    kind: str = "runtime",
    status: str = "active",
    frequency: str = "1h",
) -> RunSource:
    return RunSource(
        key=key,
        run_id=key,
        display_name=key,
        kind=kind,  # type: ignore[arg-type]
        method_family="test",
        frequency=frequency,
        mode="research" if kind == "canonical" else "demo",
        status=status,
        run_dir=tmp_path,
        manifest_path=None,
        manifest={},
        files={},
    )


def test_runtime_metrics_are_normalized_to_dashboard_contract(tmp_path: Path) -> None:
    source = _source(tmp_path)
    data = RunData(
        source=source,
        metrics=pd.DataFrame(
            {
                "bar_timestamp": [1_700_000_000, 1_700_003_600, 1_700_007_200],
                "total_equity_usdt": [1_000.0, 1_010.0, 989.8],
                "baseline_equity": [1_000.0, 1_005.0, 1_010.0],
            }
        ),
    )

    performance, issues = canonical_performance(data)

    assert not issues
    assert performance["equity_normalized"].tolist() == pytest.approx([100.0, 101.0, 98.98])
    assert performance["return_pct"].tolist() == pytest.approx([0.0, 1.0, -1.02])
    assert performance["drawdown_pct"].iloc[-1] == pytest.approx(-2.0)


def test_canonical_iso_timestamps_preserve_first_period_return(tmp_path: Path) -> None:
    source = _source(tmp_path, key="canonical:r1", kind="canonical", status="completed")
    data = RunData(
        source=source,
        metrics=pd.DataFrame(
            {
                "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
                "net_return": [0.10, 0.10],
                "equity": [110.0, 121.0],
            }
        ),
    )

    performance, issues = canonical_performance(data)

    assert not issues
    assert performance["datetime_utc"].notna().all()
    assert performance["return_pct"].tolist() == pytest.approx([10.0, 21.0])


def test_health_distinguishes_completed_research_from_stale_runtime(tmp_path: Path) -> None:
    metrics = pd.DataFrame(
        {
            "timestamp": [1_700_000_000, 1_700_003_600],
            "equity": [100.0, 101.0],
        }
    )
    canonical = _source(tmp_path, key="canonical:r1", kind="canonical", status="completed")
    runtime = _source(tmp_path, key="runtime:r1")
    canonical_perf, canonical_issues = canonical_performance(RunData(source=canonical, metrics=metrics))
    runtime_perf, runtime_issues = canonical_performance(RunData(source=runtime, metrics=metrics))
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)

    canonical_health = evaluate_health(canonical, canonical_perf, canonical_issues, now=now)
    runtime_health = evaluate_health(runtime, runtime_perf, runtime_issues, now=now)

    assert canonical_health.state == "completed"
    assert runtime_health.state == "stale"


def test_runtime_style_research_backtest_is_completed(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        key="runtime:fingpt",
        status="completed",
        frequency="1d",
    )
    source = RunSource(**{**source.__dict__, "mode": "research_backtest"})
    metrics = pd.DataFrame(
        {"timestamp": [1_700_000_000, 1_700_086_400], "strategy_returns_pct": [0.0, 1.0]}
    )
    performance, issues = canonical_performance(RunData(source=source, metrics=metrics))

    health = evaluate_health(
        source,
        performance,
        issues,
        now=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    assert health.state == "completed"


def test_performance_coalesces_numeric_aliases_and_sorts_time(tmp_path: Path) -> None:
    source = _source(tmp_path, key="runtime:mixed", frequency="1d")
    metrics = pd.DataFrame(
        {
            "bar_timestamp": [1_700_086_400, 1_700_000_000],
            "strategy_equity": [None, None],
            "total_equity_usdt": [110.0, 100.0],
            "baseline_returns_pct": [10.0, None],
            "baseline_equity": [None, 100.0],
        }
    )

    performance, issues = canonical_performance(RunData(source=source, metrics=metrics))

    assert performance["equity_normalized"].tolist() == pytest.approx([100.0, 110.0])
    assert performance["baseline_normalized"].tolist() == pytest.approx([100.0, 110.0])
    assert performance["datetime_utc"].is_monotonic_increasing
    assert any(issue.code == "OUT_OF_ORDER_TIMESTAMPS" for issue in issues)


def test_common_window_rebases_each_selected_run(tmp_path: Path) -> None:
    left_source = _source(tmp_path, key="runtime:left")
    right_source = _source(tmp_path, key="runtime:right")
    left = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"]
            ),
            "equity_normalized": [100.0, 101.0, 102.0],
        }
    )
    right = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                ["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z", "2024-01-01T03:00:00Z"]
            ),
            "equity_normalized": [100.0, 98.0, 99.0],
        }
    )

    result, window = comparison_frame(
        [(left_source, left), (right_source, right)], common_window=True
    )

    assert window["start"] == pd.Timestamp("2024-01-01T01:00:00Z")
    assert window["end"] == pd.Timestamp("2024-01-01T02:00:00Z")
    first_returns = result.groupby("run_key")["view_return_pct"].first()
    assert first_returns.to_dict() == pytest.approx({"runtime:left": 0.0, "runtime:right": 0.0})
