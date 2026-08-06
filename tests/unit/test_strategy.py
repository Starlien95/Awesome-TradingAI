from __future__ import annotations

import pandas as pd
import pytest

from quant_bench.strategies import select_target_weights


def test_topk_threshold_and_equal_weights() -> None:
    scores = pd.Series({"A": 0.3, "B": 0.2, "C": -0.1})
    assert select_target_weights(scores, top_k=2, threshold=0.0) == {"A": 0.5, "B": 0.5}


def test_dropout_retains_only_the_allowed_old_holding() -> None:
    scores = pd.Series({"A": 0.5, "B": 0.4})
    current = {"X": 0.1, "Y": 0.4}
    result = select_target_weights(scores, current, top_k=3, max_dropout=1)
    assert "X" not in result
    assert "Y" in result
    assert len(result) == 3
    assert sum(result.values()) == pytest.approx(1.0)


def test_long_short_budget() -> None:
    scores = pd.Series({"A": 0.5, "B": -0.4})
    result = select_target_weights(scores, top_k=2, enable_short=True)
    assert result == {"A": 0.5, "B": -0.5}
