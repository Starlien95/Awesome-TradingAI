from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_robust_zscore(
    feature_df: pd.DataFrame,
    window: int = 288,
    clip: float | None = 3.0,
) -> pd.DataFrame:
    """
    RobustZScoreNorm approximation for live inference.

    Qlib's training processor fits median/MAD on the historical train window.
    Those fitted processor parameters are not available for these external
    TabNet/TCN artifacts, so live inference uses only recent past rows from the
    current symbol. This avoids future leakage and keeps feature magnitudes close
    to the training processor's robust z-score scale.
    """
    if feature_df.empty:
        return feature_df
    work = feature_df.copy()
    lookback = work.tail(max(1, int(window)))
    values = lookback.to_numpy(dtype=float)
    center = np.nanmedian(values, axis=0)
    scale = np.nanmedian(np.abs(values - center), axis=0) * 1.4826
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    normalized = (work.to_numpy(dtype=float) - center) / scale
    if clip is not None:
        normalized = np.clip(normalized, -float(clip), float(clip))
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=float(clip or 0.0), neginf=-float(clip or 0.0))
    return pd.DataFrame(normalized, index=work.index, columns=work.columns)
