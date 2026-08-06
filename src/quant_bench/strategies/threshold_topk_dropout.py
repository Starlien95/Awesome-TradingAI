"""Pure target-selection semantics shared by research and runtime adapters."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def select_target_weights(
    scores: pd.Series,
    current_weights: Mapping[str, float] | None = None,
    *,
    top_k: int = 5,
    threshold: float = 0.0,
    max_dropout: int = 1,
    enable_short: bool = False,
) -> dict[str, float]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if max_dropout < 0:
        raise ValueError("max_dropout must be non-negative")
    score = scores.dropna().astype(float)
    if score.index.has_duplicates:
        raise ValueError("scores must have unique instrument names")
    current = dict(current_weights or {})

    long_targets = score[score > threshold].nlargest(top_k)
    short_targets = score[score < -threshold].nsmallest(top_k) if enable_short else pd.Series(dtype=float)
    target_holdings = set(long_targets.index) | set(short_targets.index)
    current_holdings = set(current)
    dropout_candidates = current_holdings - target_holdings
    ordered_dropout = sorted(dropout_candidates, key=lambda instrument: abs(current.get(instrument, 0.0)))
    dropped = set(ordered_dropout[:max_dropout])
    final_holdings = target_holdings | (current_holdings - dropped)

    if len(final_holdings) > top_k:
        ranking = pd.Series(
            {instrument: abs(float(score.get(instrument, 0.0))) for instrument in final_holdings}
        )
        final_holdings = set(ranking.nlargest(top_k).index)
    if not final_holdings:
        return {}

    long_final = [
        instrument
        for instrument in final_holdings
        if instrument in long_targets.index
        or (current.get(instrument, 0.0) > 0 and float(score.get(instrument, 0.0)) >= 0)
    ]
    short_final = [
        instrument
        for instrument in final_holdings
        if enable_short
        and (
            instrument in short_targets.index
            or (current.get(instrument, 0.0) < 0 and float(score.get(instrument, 0.0)) < 0)
        )
    ]
    weights: dict[str, float] = {}
    if enable_short:
        if long_final:
            weights.update({instrument: 0.5 / len(long_final) for instrument in long_final})
        if short_final:
            weights.update({instrument: -0.5 / len(short_final) for instrument in short_final})
    elif long_final:
        weights.update({instrument: 1.0 / len(long_final) for instrument in long_final})
    return weights
