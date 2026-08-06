from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _json_float(value: Any):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def object_to_current_frame(obj: Any, value_name: str) -> pd.DataFrame:
    """Convert Qlib prediction/label objects to a 1-column DataFrame.

    TSDatasetH returns labels as TSDataSampler. For signal analysis we need the
    current row of each sample, not the whole lookback window.
    """
    if isinstance(obj, pd.Series):
        return obj.rename(value_name).to_frame()
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.rename(columns={obj.columns[0]: value_name})
        return obj.iloc[:, [0]].rename(columns={obj.columns[0]: value_name})

    if hasattr(obj, "get_index") and hasattr(obj, "idx_map") and hasattr(obj, "idx_arr") and hasattr(obj, "data_arr"):
        rows = np.asarray(obj.idx_map)[:, 0].astype(int)
        cols = np.asarray(obj.idx_map)[:, 1].astype(int)
        data_rows = np.asarray(obj.idx_arr)[rows, cols]
        valid = np.isfinite(data_rows)
        values = np.full(len(rows), np.nan, dtype=float)
        if valid.any():
            values[valid] = np.asarray(obj.data_arr)[data_rows[valid].astype(int), 0].astype(float)
        return pd.DataFrame({value_name: values}, index=obj.get_index())

    for method_name in ("to_pandas", "to_frame"):
        method = getattr(obj, method_name, None)
        if callable(method):
            return object_to_current_frame(method(), value_name)

    raise TypeError(f"Unsupported prediction/label object: {type(obj)!r}")


def prediction_distribution_stats(scores: pd.Series) -> dict:
    scores = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if scores.empty:
        return {}
    return {
        "count": int(scores.shape[0]),
        "mean": _json_float(scores.mean()),
        "std": _json_float(scores.std()),
        "min": _json_float(scores.min()),
        "p01": _json_float(scores.quantile(0.01)),
        "p05": _json_float(scores.quantile(0.05)),
        "p50": _json_float(scores.quantile(0.50)),
        "p95": _json_float(scores.quantile(0.95)),
        "p99": _json_float(scores.quantile(0.99)),
        "max": _json_float(scores.max()),
        "positive_ratio": _json_float((scores > 0).mean()),
        "negative_ratio": _json_float((scores < 0).mean()),
    }
