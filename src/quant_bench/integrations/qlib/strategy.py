"""Qlib strategy adapter for the shared selection function."""

from __future__ import annotations

from typing import Any

import pandas as pd
from qlib.backtest.position import Position
from qlib.contrib.strategy import WeightStrategyBase

from quant_bench.strategies import select_target_weights


class ThresholdTopkDropoutStrategy(WeightStrategyBase):
    def __init__(
        self,
        *args: Any,
        topk: int = 5,
        min_score: float = 0.0,
        max_dropout: int = 1,
        enable_short: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.topk = topk
        self.min_score = min_score
        self.max_dropout = max_dropout
        self.enable_short = enable_short

    def generate_target_weight_position(
        self, score: pd.Series | pd.DataFrame, current: Any, **kwargs: Any
    ) -> dict[str, float]:
        del kwargs
        if isinstance(score, pd.DataFrame):
            if score.shape[1] != 1:
                raise ValueError("strategy score DataFrame must contain exactly one column")
            score = score.iloc[:, 0]
        if isinstance(current, Position):
            try:
                current_weights = current.get_stock_weight_dict(only_stock=False)
            except (AttributeError, KeyError, TypeError, ValueError):
                current_weights = {}
        elif isinstance(current, dict):
            current_weights = current
        else:
            current_weights = {}
        return select_target_weights(
            score,
            current_weights,
            top_k=self.topk,
            threshold=self.min_score,
            max_dropout=self.max_dropout,
            enable_short=self.enable_short,
        )
