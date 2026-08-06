import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CustomFeatureCalculator:
    """
    Pandas-based implementation of CustomFeature_RL (52 features).
    Translates Qlib expressions into pandas operations to avoid Qlib initialization overhead.
    Maximum lookback window is 288 klines.
    """
    def __init__(self):
        self.eps = 1e-12
        self.W = 96

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the 52 features. df should have: open, high, low, close, volume.
        Expects index to be time or monotonic.
        """
        df = df.copy()

        # rename columns strictly lowercase
        df.columns = [c.lower() for c in df.columns]

        close = df['close']
        open_ = df['open']
        high = df['high']
        low = df['low']
        volume = df['volume']

        eps = self.eps
        W = self.W

        # helper operators to match Qlib exactly
        def Ref(series, d): return series.shift(d)
        def Mean(series, d): return series.rolling(d, min_periods=1).mean()
        def Std(series, d): return series.rolling(d, min_periods=1).std().fillna(0)
        def Max(series, d): return series.rolling(d, min_periods=1).max()
        def Min(series, d): return series.rolling(d, min_periods=1).min()
        def Sum(series, d): return series.rolling(d, min_periods=1).sum()
        # approximate Qlib EMA: Qlib EMA uses center of gravity EMA, pandas ewm(span) is standard. We use ewm with adjust=False.
        def EMA(series, d): return series.ewm(span=d, adjust=False).mean()
        def Corr(s1, s2, d): return s1.rolling(d, min_periods=1).corr(s2).fillna(0)
        def If(cond, s1, s2):
            val1 = s1 if isinstance(s1, pd.Series) else pd.Series(s1, index=cond.index)
            val2 = s2 if isinstance(s2, pd.Series) else pd.Series(s2, index=cond.index)
            return pd.Series(np.where(cond, val1, val2), index=cond.index)
        def Abs(series): return series.abs()
        def Log(series): return np.log(series)

        features = {}

        # 1. momentum_features
        features['feat_00'] = (close / Ref(close, 6) - 1) * 10
        features['feat_01'] = (close / Ref(close, 12) - 1) * 10
        features['feat_02'] = (close / Ref(close, 24) - 1) * 5
        features['feat_03'] = (close / Ref(close, 48) - 1) * 5
        features['feat_04'] = (close / Ref(close, 96) - 1) * 3
        features['feat_05'] = (close / Ref(close, 144) - 1) * 3
        features['feat_06'] = (close / Ref(close, 288) - 1) * 2
        features['feat_07'] = ((close / Ref(close, 6) - 1) - (Ref(close, 6) / Ref(close, 12) - 1)) * 20

        # 2. ma_features
        features['feat_08'] = (Mean(close, 12) / Ref(Mean(close, 12), 6) - 1) * 20
        features['feat_09'] = (Mean(close, 48) / Ref(Mean(close, 48), 12) - 1) * 10
        features['feat_10'] = (Mean(close, 144) / Ref(Mean(close, 144), 24) - 1) * 5
        features['feat_11'] = (close / (Mean(close, 12) + eps) - 1) * 10
        features['feat_12'] = (close / (Mean(close, 48) + eps) - 1) * 5
        features['feat_13'] = (close / (Mean(close, 144) + eps) - 1) * 3
        features['feat_14'] = (close / (Mean(close, 288) + eps) - 1) * 2
        features['feat_15'] = (Mean(close, 12) - Ref(Mean(close, 12), 6)) / (Mean(close, 12) + eps) * 20
        features['feat_16'] = (Mean(close, 48) - Ref(Mean(close, 48), 12)) / (Mean(close, 48) + eps) * 10
        features['feat_17'] = (Mean(close, 12) - Mean(close, 48)) / (Mean(close, 48) + eps) * 10

        # 3. macd_features
        macd_line = EMA(close, 24) - EMA(close, 52)
        features['feat_18'] = macd_line / (close + eps) * 100
        features['feat_19'] = EMA(macd_line, 18) / (close + eps) * 100
        features['feat_20'] = (macd_line - EMA(macd_line, 18)) / (close + eps) * 100

        # 4. volatility_features
        features['feat_21'] = Std(close, 12) / (Mean(Std(close, 12), W) + eps) - 1
        features['feat_22'] = Std(close, 48) / (Mean(Std(close, 48), W) + eps) - 1
        features['feat_23'] = Std(close, 144) / (Mean(Std(close, 144), W) + eps) - 1
        features['feat_24'] = Std(close, 12) / (Std(close, 48) + eps) - 1
        features['feat_25'] = Std(close, 48) / (Std(close, 144) + eps) - 1
        features['feat_26'] = Std(close, 24) / (Mean(Std(close, 24), W) + eps) - 1

        # 5. bollinger_features
        boll_top = 2 * Std(close, 48) / (Mean(close, 48) + eps)
        features['feat_27'] = boll_top / (Mean(boll_top, W) + eps) - 1
        features['feat_28'] = (close - (Mean(close, 48) - 2 * Std(close, 48))) / (4 * Std(close, 48) + eps) - 0.5
        features['feat_29'] = Std(close, 24) / (Std(close, 96) + eps) - 1

        # 6. volume_features
        features['feat_30'] = Log(volume / (Mean(volume, 24) + eps) + eps)
        features['feat_31'] = Log(volume / (Mean(volume, 96) + eps) + eps)
        features['feat_32'] = Corr(close, volume, 48)
        features['feat_33'] = Corr(close, volume, 144)
        vol_ratio = Std(volume, 24) / (Mean(volume, 24) + eps)
        features['feat_34'] = vol_ratio / (Mean(vol_ratio, W) + eps) - 1
        features['feat_35'] = Mean(volume, 12) / (Mean(volume, 48) + eps) - 1

        # 7. candlestick_features
        features['feat_36'] = (close - open_) / (high - low + eps)
        hl_ratio = (high - low) / (close + eps)
        features['feat_37'] = hl_ratio / (Mean(hl_ratio, W) + eps) - 1
        features['feat_38'] = (high - If(open_ > close, open_, close)) / (high - low + eps)
        features['feat_39'] = (If(open_ < close, open_, close) - low) / (high - low + eps)
        features['feat_40'] = (2*close - high - low) / (high - low + eps)

        # 8. rsi_features
        close_up = If(close > Ref(close, 1), 1, 0)
        features['feat_41'] = Sum(close_up, 24) / 24 - 0.5
        features['feat_42'] = Sum(close_up, 96) / 96 - 0.5
        close_gain = If(close > Ref(close, 1), close - Ref(close, 1), 0)
        close_diff_abs = Abs(close - Ref(close, 1))
        features['feat_43'] = Sum(close_gain, 24) / (Sum(close_diff_abs, 24) + eps) - 0.5
        features['feat_44'] = Sum(If(close > Ref(close, 1), 1, -1), 6) / 6

        # 9. quantile_features
        features['feat_45'] = (close - Min(low, 48)) / (Max(high, 48) - Min(low, 48) + eps) - 0.5
        features['feat_46'] = (close - Min(low, 144)) / (Max(high, 144) - Min(low, 144) + eps) - 0.5
        features['feat_47'] = (close - Min(low, 288)) / (Max(high, 288) - Min(low, 288) + eps) - 0.5
        features['feat_48'] = (close - Max(high, 48)) / (Max(high, 48) + eps) * 10

        # 10. atr_features
        tr = high - low
        atr_12_c = Mean(tr, 12) / (close + eps)
        features['feat_49'] = atr_12_c / (Mean(atr_12_c, W) + eps) - 1
        atr_24_c = Mean(tr, 24) / (close + eps)
        features['feat_50'] = atr_24_c / (Mean(atr_24_c, W) + eps) - 1
        features['feat_51'] = Mean(tr, 12) / (Mean(tr, 48) + eps) - 1

        feature_df = pd.DataFrame(features, index=df.index)

        # Ensure we return valid finite numbers, handle nans properly
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feature_df.ffill(inplace=True)
        feature_df.fillna(0, inplace=True)

        return feature_df
