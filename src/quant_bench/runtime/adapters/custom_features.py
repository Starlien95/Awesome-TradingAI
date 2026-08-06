"""
MLP/RL 用特征配置：仅提供特征表达式与名称，不依赖 qlib.init()。
特征计算可在 predict 中基于 pandas 自行实现或接入其他库。
"""


class CustomFeature_RL:
    """
    为 MLP/GRU/LSTM/RL 模型优化的特征集配置（无 qlib 依赖）。
    最大回看窗口 288 根 K 线；共 52 个特征。
    """

    def __init__(self, **kwargs):
        pass

    def get_feature_config(self):
        eps = 1e-12
        W = 96

        momentum_features = [
            "($close / Ref($close, 6) - 1) * 10",
            "($close / Ref($close, 12) - 1) * 10",
            "($close / Ref($close, 24) - 1) * 5",
            "($close / Ref($close, 48) - 1) * 5",
            "($close / Ref($close, 96) - 1) * 3",
            "($close / Ref($close, 144) - 1) * 3",
            "($close / Ref($close, 288) - 1) * 2",
            "(($close / Ref($close, 6) - 1) - (Ref($close, 6) / Ref($close, 12) - 1)) * 20",
        ]

        ma_features = [
            "(Mean($close, 12) / Ref(Mean($close, 12), 6) - 1) * 20",
            "(Mean($close, 48) / Ref(Mean($close, 48), 12) - 1) * 10",
            "(Mean($close, 144) / Ref(Mean($close, 144), 24) - 1) * 5",
            f"($close / (Mean($close, 12) + {eps}) - 1) * 10",
            f"($close / (Mean($close, 48) + {eps}) - 1) * 5",
            f"($close / (Mean($close, 144) + {eps}) - 1) * 3",
            f"($close / (Mean($close, 288) + {eps}) - 1) * 2",
            f"(Mean($close, 12) - Ref(Mean($close, 12), 6)) / (Mean($close, 12) + {eps}) * 20",
            f"(Mean($close, 48) - Ref(Mean($close, 48), 12)) / (Mean($close, 48) + {eps}) * 10",
            f"(Mean($close, 12) - Mean($close, 48)) / (Mean($close, 48) + {eps}) * 10",
        ]

        macd_features = [
            f"(EMA($close, 24) - EMA($close, 52)) / ($close + {eps}) * 100",
            f"EMA(EMA($close, 24) - EMA($close, 52), 18) / ($close + {eps}) * 100",
            f"((EMA($close, 24) - EMA($close, 52)) - EMA(EMA($close, 24) - EMA($close, 52), 18)) / ($close + {eps}) * 100",
        ]

        volatility_features = [
            f"Std($close, 12) / (Mean(Std($close, 12), {W}) + {eps}) - 1",
            f"Std($close, 48) / (Mean(Std($close, 48), {W}) + {eps}) - 1",
            f"Std($close, 144) / (Mean(Std($close, 144), {W}) + {eps}) - 1",
            f"Std($close, 12) / (Std($close, 48) + {eps}) - 1",
            f"Std($close, 48) / (Std($close, 144) + {eps}) - 1",
            f"Std($close, 24) / (Mean(Std($close, 24), {W}) + {eps}) - 1",
        ]

        bollinger_features = [
            f"(2 * Std($close, 48) / (Mean($close, 48) + {eps})) / (Mean(2 * Std($close, 48) / (Mean($close, 48) + {eps}), {W}) + {eps}) - 1",
            f"($close - (Mean($close, 48) - 2 * Std($close, 48))) / (4 * Std($close, 48) + {eps}) - 0.5",
            f"Std($close, 24) / (Std($close, 96) + {eps}) - 1",
        ]

        volume_features = [
            f"Log($volume / (Mean($volume, 24) + {eps}) + {eps})",
            f"Log($volume / (Mean($volume, 96) + {eps}) + {eps})",
            "Corr($close, $volume, 48)",
            "Corr($close, $volume, 144)",
            f"(Std($volume, 24) / (Mean($volume, 24) + {eps})) / (Mean(Std($volume, 24) / (Mean($volume, 24) + {eps}), {W}) + {eps}) - 1",
            f"Mean($volume, 12) / (Mean($volume, 48) + {eps}) - 1",
        ]

        candlestick_features = [
            f"($close - $open) / ($high - $low + {eps})",
            f"(($high - $low) / ($close + {eps})) / (Mean(($high - $low) / ($close + {eps}), {W}) + {eps}) - 1",
            f"($high - If($open > $close, $open, $close)) / ($high - $low + {eps})",
            f"(If($open < $close, $open, $close) - $low) / ($high - $low + {eps})",
            f"(2*$close - $high - $low) / ($high - $low + {eps})",
        ]

        rsi_features = [
            "Sum(If($close > Ref($close, 1), 1, 0), 24) / 24 - 0.5",
            "Sum(If($close > Ref($close, 1), 1, 0), 96) / 96 - 0.5",
            f"Sum(If($close > Ref($close, 1), $close - Ref($close, 1), 0), 24) / (Sum(Abs($close - Ref($close, 1)), 24) + {eps}) - 0.5",
            "Sum(If($close > Ref($close, 1), 1, -1), 6) / 6",
        ]

        quantile_features = [
            f"($close - Min($low, 48)) / (Max($high, 48) - Min($low, 48) + {eps}) - 0.5",
            f"($close - Min($low, 144)) / (Max($high, 144) - Min($low, 144) + {eps}) - 0.5",
            f"($close - Min($low, 288)) / (Max($high, 288) - Min($low, 288) + {eps}) - 0.5",
            f"($close - Max($high, 48)) / (Max($high, 48) + {eps}) * 10",
        ]

        atr_features = [
            f"(Mean($high - $low, 12) / ($close + {eps})) / (Mean(Mean($high - $low, 12) / ($close + {eps}), {W}) + {eps}) - 1",
            f"(Mean($high - $low, 24) / ($close + {eps})) / (Mean(Mean($high - $low, 24) / ($close + {eps}), {W}) + {eps}) - 1",
            f"Mean($high - $low, 12) / (Mean($high - $low, 48) + {eps}) - 1",
        ]

        all_features = (
            momentum_features
            + ma_features
            + macd_features
            + volatility_features
            + bollinger_features
            + volume_features
            + candlestick_features
            + rsi_features
            + quantile_features
            + atr_features
        )
        names = [f"feat_{i:02d}" for i in range(len(all_features))]
        return all_features, names
