"""Qlib record adapters for frequency-safe cryptocurrency portfolio analysis."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
from qlib.data import D
from qlib.workflow.record_temp import PortAnaRecord


class CryptoPortAnaRecord(PortAnaRecord):  # type: ignore[misc]
    """Resolve a benchmark at the workflow frequency before Qlib creates its account.

    Qlib 0.9.7 creates ``Account`` with a daily default before resetting it to
    the executor frequency. A string benchmark therefore triggers a daily or
    1-minute lookup even when the provider contains only hourly bars. Supplying
    a precomputed Series keeps Qlib's public backtest implementation intact and
    removes that hidden cross-frequency data requirement.
    """

    def _benchmark_returns(self, benchmark: str | list[str], start: Any, end: Any) -> pd.Series:
        executor_kwargs = self.executor_config.get("kwargs", {})
        frequency = str(executor_kwargs.get("time_per_step", "day"))
        instruments = [benchmark] if isinstance(benchmark, str) else list(benchmark)
        frame = D.features(
            instruments,
            ["$close/Ref($close,1)-1"],
            start_time=start,
            end_time=end,
            freq=frequency,
        )
        if frame.empty:
            raise ValueError(
                f"benchmark has no {frequency} data in the backtest window: {instruments}"
            )
        values = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
        if isinstance(values.index, pd.MultiIndex) and "datetime" in values.index.names:
            values = values.groupby(level="datetime").mean()
        values = values.dropna().sort_index()
        if values.empty:
            raise ValueError(f"benchmark returns are empty after validation: {instruments}")
        values.name = "benchmark_return"
        return cast(pd.Series, values)

    def _generate(self, **kwargs: Any) -> dict[str, Any]:
        benchmark = self.backtest_config.get("benchmark")
        if isinstance(benchmark, (str, list)):
            prediction = self.load("pred.pkl")
            dates = prediction.index.get_level_values("datetime")
            start = self.backtest_config.get("start_time") or dates.min()
            end = self.backtest_config.get("end_time") or dates.max()
            self.backtest_config["benchmark"] = self._benchmark_returns(benchmark, start, end)
        return cast(dict[str, Any], super()._generate(**kwargs))
