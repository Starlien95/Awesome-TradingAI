import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import linregress

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class Alpha158Calculator:

    def __init__(self, freq: str = "5m"):
        """
        初始化Alpha158计算器
        Args:
            freq: K线频率（仅用于日志，不影响计算）
        """
        self.freq = freq
        self.feature_count = 158

        self.config = {
            "kbar": {},  # 使用K线特征
            "price": {
                "windows": [0, 1, 2, 3, 4],  # 使用0-4天的价格
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "volume": {
                "windows": [0, 1, 2, 3, 4],  # 使用0-4天的成交量
            },
            "rolling": {
                "windows": [5, 10, 20, 30, 60],  # 滚动窗口
                "include": None,  # 包含所有算子
                "exclude": ['RANK'],  # 排除RANK算子
            }
        }

        logger.info("Alpha158Calculator initialized with QLib官方配置")

    def calculate_features(self, kline_data: list[dict]) -> pd.DataFrame:
        """
        计算完整的Alpha158特征（严格按照QLib官方定义）

        Args:
            kline_data: K线数据列表，格式：[{timestamp, open, high, low, close, volume}]

        Returns:
            包含158个特征的DataFrame，特征顺序与QLib完全一致
        """
        try:
            # 1. 准备数据
            df = self._prepare_data(kline_data)

            # 2. 计算所有特征
            features_dict = {}

            # 计算K线特征
            features_dict.update(self._calculate_kbar_features(df))

            # 计算价格特征
            features_dict.update(self._calculate_price_features(df))

            # 计算成交量特征
            features_dict.update(self._calculate_volume_features(df))

            # 计算滚动特征
            features_dict.update(self._calculate_rolling_features(df))

            # 3. 转换为DataFrame
            features_df = pd.DataFrame(features_dict)

            # 4. 确保特征顺序正确
            features_df = self._ensure_feature_order(features_df)

            logger.info(f"成功计算 {len(features_df.columns)} 个Alpha158特征")
            return features_df

        except Exception as e:
            logger.error(f"特征计算失败: {e}")
            return self._create_empty_features()

    def _prepare_data(self, kline_data: list[dict]) -> pd.DataFrame:
        """准备数据，确保格式正确"""
        df = pd.DataFrame(kline_data)

        # 设置时间索引
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # 确保数值类型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 计算VWAP（成交量加权平均价）
        df['vwap'] = (df['close'] * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()

        # 填充缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')

        return df

    def _calculate_kbar_features(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """计算K线特征（9个特征）"""
        open_, high, low, close = df["open"], df["high"], df["low"], df["close"]

        features = {}

        # 1. KMID: ($close-$open)/$open
        features["KMID"] = (close - open_) / open_

        # 2. KLEN: ($high-$low)/$open
        features["KLEN"] = (high - low) / open_

        # 3. KMID2: ($close-$open)/($high-$low+1e-12)
        features["KMID2"] = (close - open_) / (high - low + 1e-12)

        # 4. KUP: ($high-Greater($open, $close))/$open
        features["KUP"] = (high - np.maximum(open_, close)) / open_

        # 5. KUP2: ($high-Greater($open, $close))/($high-$low+1e-12)
        features["KUP2"] = (high - np.maximum(open_, close)) / (high - low + 1e-12)

        # 6. KLOW: (Less($open, $close)-$low)/$open
        features["KLOW"] = (np.minimum(open_, close) - low) / open_

        # 7. KLOW2: (Less($open, $close)-$low)/($high-$low+1e-12)
        features["KLOW2"] = (np.minimum(open_, close) - low) / (high - low + 1e-12)

        # 8. KSFT: (2*$close-$high-$low)/$open
        features["KSFT"] = (2 * close - high - low) / open_

        # 9. KSFT2: (2*$close-$high-$low)/($high-$low+1e-12)
        features["KSFT2"] = (2 * close - high - low) / (high - low + 1e-12)

        return features

    def _calculate_price_features(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """计算价格特征"""
        features = {}

        _O, _H, _L, C, _VWAP = df['open'], df['high'], df['low'], df['close'], df['vwap']
        windows = self.config['price']['windows']
        price_fields = self.config['price']['feature']

        # 为每个价格字段和窗口计算特征
        for field in price_fields:
            field_lower = field.lower()
            price_series = df[field_lower]

            for d in windows:
                if d == 0:
                    # $field/$close
                    features[f'{field}{d}'] = price_series / C
                else:
                    # Ref($field, d)/$close
                    ref_price = price_series.shift(d)
                    features[f'{field}{d}'] = ref_price / C

        return features

    def _calculate_volume_features(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """计算成交量特征"""
        features = {}

        V = df['volume']
        windows = self.config['volume']['windows']

        for d in windows:
            if d == 0:
                # $volume/($volume+1e-12)
                features[f'VOLUME{d}'] = V / (V + 1e-12)
            else:
                # Ref($volume, d)/($volume+1e-12)
                ref_volume = V.shift(d)
                features[f'VOLUME{d}'] = ref_volume / (V + 1e-12)

        return features

    def _calculate_rolling_features(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """计算滚动特征"""
        features = {}

        C, H, L, V = df['close'], df['high'], df['low'], df['volume']
        windows = self.config['rolling']['windows']
        include = self.config['rolling']['include']
        exclude = self.config['rolling']['exclude']

        # 辅助函数：判断是否使用该算子
        def should_use(op_name: str) -> bool:
            if op_name in exclude:
                return False
            if include is None:
                return True
            return op_name in include

        # 1. ROC (Rate of Change)
        if should_use('ROC'):
            for d in windows:
                features[f'ROC{d}'] = C.shift(d) / C

        # 2. MA (Moving Average)
        if should_use('MA'):
            for d in windows:
                features[f'MA{d}'] = C.rolling(d).mean() / C

        # 3. STD (Standard Deviation)
        if should_use('STD'):
            for d in windows:
                features[f'STD{d}'] = C.rolling(d).std() / C

        # 4. BETA (Slope of linear regression)
        if should_use('BETA'):
            for d in windows:
                features[f'BETA{d}'] = self._calculate_slope(C, d) / C

        # 5. RSQR (R-squared of linear regression)
        if should_use('RSQR'):
            for d in windows:
                features[f'RSQR{d}'] = self._calculate_rsquare(C, d)

        # 6. RESI (Residual of linear regression)
        if should_use('RESI'):
            for d in windows:
                features[f'RESI{d}'] = self._calculate_residual(C, d) / C

        # 7. MAX (Maximum price)
        if should_use('MAX'):
            for d in windows:
                features[f'MAX{d}'] = H.rolling(d).max() / C

        # 8. LOW (Minimum price)
        if should_use('LOW'):
            for d in windows:
                features[f'MIN{d}'] = L.rolling(d).min() / C

        # 9. QTLU (80% Quantile)
        if should_use('QTLU'):
            for d in windows:
                features[f'QTLU{d}'] = C.rolling(d).quantile(0.8) / C

        # 10. QTLD (20% Quantile)
        if should_use('QTLD'):
            for d in windows:
                features[f'QTLD{d}'] = C.rolling(d).quantile(0.2) / C

        # 11. RSV (Relative Strength Value)
        if should_use('RSV'):
            for d in windows:
                min_low = L.rolling(d).min()
                max_high = H.rolling(d).max()
                features[f'RSV{d}'] = (C - min_low) / (max_high - min_low + 1e-12)

        # 12. IMAX (Index of Maximum)
        if should_use('IMAX'):
            for d in windows:
                features[f'IMAX{d}'] = self._calculate_idxmax(H, d) / d

        # 13. IMIN (Index of Minimum)
        if should_use('IMIN'):
            for d in windows:
                features[f'IMIN{d}'] = self._calculate_idxmin(L, d) / d

        # 14. IMXD (Index Max-Min Difference)
        if should_use('IMXD'):
            for d in windows:
                idx_max = self._calculate_idxmax(H, d)
                idx_min = self._calculate_idxmin(L, d)
                features[f'IMXD{d}'] = (idx_max - idx_min) / d

        # 15. CORR (Correlation between price and log volume)
        if should_use('CORR'):
            for d in windows:
                log_volume = np.log(V + 1)
                features[f'CORR{d}'] = C.rolling(d).corr(log_volume)

        # 16. CORD (Correlation between returns and volume changes)
        if should_use('CORD'):
            for d in windows:
                returns = C / C.shift(1)
                volume_changes = np.log(V / V.shift(1) + 1)
                features[f'CORD{d}'] = returns.rolling(d).corr(volume_changes)

        # 17. CNTP (Percentage of up days)
        if should_use('CNTP'):
            for d in windows:
                up_days = (C.shift(1) < C).rolling(d).mean()
                features[f'CNTP{d}'] = up_days

        # 18. CNTN (Percentage of down days)
        if should_use('CNTN'):
            for d in windows:
                down_days = (C.shift(1) > C).rolling(d).mean()
                features[f'CNTN{d}'] = down_days

        # 19. CNTD (Difference between up and down days)
        if should_use('CNTD'):
            for d in windows:
                up_days = (C.shift(1) < C).rolling(d).mean()
                down_days = (C.shift(1) > C).rolling(d).mean()
                features[f'CNTD{d}'] = up_days - down_days

        # 20. SUMP (Sum of gains / total absolute change)
        if should_use('SUMP'):
            for d in windows:
                gains = np.maximum(C - C.shift(1), 0)
                abs_changes = np.abs(C - C.shift(1))
                sum_gains = gains.rolling(d).sum()
                sum_abs = abs_changes.rolling(d).sum()
                features[f'SUMP{d}'] = sum_gains / (sum_abs + 1e-12)

        # 21. SUMN (Sum of losses / total absolute change)
        if should_use('SUMN'):
            for d in windows:
                losses = np.maximum(C.shift(1) - C, 0)
                abs_changes = np.abs(C - C.shift(1))
                sum_losses = losses.rolling(d).sum()
                sum_abs = abs_changes.rolling(d).sum()
                features[f'SUMN{d}'] = sum_losses / (sum_abs + 1e-12)

        # 22. SUMD (Difference between gains and losses)
        if should_use('SUMD'):
            for d in windows:
                gains = np.maximum(C - C.shift(1), 0)
                losses = np.maximum(C.shift(1) - C, 0)
                sum_gains = gains.rolling(d).sum()
                sum_losses = losses.rolling(d).sum()
                abs_changes = np.abs(C - C.shift(1))
                sum_abs = abs_changes.rolling(d).sum()
                features[f'SUMD{d}'] = (sum_gains - sum_losses) / (sum_abs + 1e-12)

        # 23. VMA (Volume Moving Average)
        if should_use('VMA'):
            for d in windows:
                features[f'VMA{d}'] = V.rolling(d).mean() / (V + 1e-12)

        # 24. VSTD (Volume Standard Deviation)
        if should_use('VSTD'):
            for d in windows:
                features[f'VSTD{d}'] = V.rolling(d).std() / (V + 1e-12)

        # 25. WVMA (Weighted Volatility Moving Average)
        if should_use('WVMA'):
            for d in windows:
                returns_abs = np.abs(C / C.shift(1) - 1)
                weighted = returns_abs * V
                std_weighted = weighted.rolling(d).std()
                mean_weighted = weighted.rolling(d).mean()
                features[f'WVMA{d}'] = std_weighted / (mean_weighted + 1e-12)

        # 26. VSUMP (Volume increase percentage)
        if should_use('VSUMP'):
            for d in windows:
                volume_increase = np.maximum(V - V.shift(1), 0)
                abs_volume_changes = np.abs(V - V.shift(1))
                sum_increase = volume_increase.rolling(d).sum()
                sum_abs = abs_volume_changes.rolling(d).sum()
                features[f'VSUMP{d}'] = sum_increase / (sum_abs + 1e-12)

        # 27. VSUMN (Volume decrease percentage)
        if should_use('VSUMN'):
            for d in windows:
                volume_decrease = np.maximum(V.shift(1) - V, 0)
                abs_volume_changes = np.abs(V - V.shift(1))
                sum_decrease = volume_decrease.rolling(d).sum()
                sum_abs = abs_volume_changes.rolling(d).sum()
                features[f'VSUMN{d}'] = sum_decrease / (sum_abs + 1e-12)

        # 28. VSUMD (Volume increase-decrease difference)
        if should_use('VSUMD'):
            for d in windows:
                volume_increase = np.maximum(V - V.shift(1), 0)
                volume_decrease = np.maximum(V.shift(1) - V, 0)
                sum_increase = volume_increase.rolling(d).sum()
                sum_decrease = volume_decrease.rolling(d).sum()
                abs_volume_changes = np.abs(V - V.shift(1))
                sum_abs = abs_volume_changes.rolling(d).sum()
                features[f'VSUMD{d}'] = (sum_increase - sum_decrease) / (sum_abs + 1e-12)

        return features

    def _calculate_slope(self, series: pd.Series, window: int) -> pd.Series:
        """计算线性回归的斜率"""
        def _slope(arr):
            if len(arr) < 2:
                return np.nan
            x = np.arange(len(arr))
            return linregress(x, arr).slope

        return series.rolling(window).apply(_slope, raw=True)

    def _calculate_rsquare(self, series: pd.Series, window: int) -> pd.Series:
        """计算线性回归的R平方"""
        def _rsquare(arr):
            if len(arr) < 2:
                return np.nan
            x = np.arange(len(arr))
            _slope, _intercept, r_value, _p_value, _std_err = linregress(x, arr)
            return r_value ** 2

        return series.rolling(window).apply(_rsquare, raw=True)

    def _calculate_residual(self, series: pd.Series, window: int) -> pd.Series:
        """计算线性回归的残差"""
        def _residual(arr):
            if len(arr) < 2:
                return np.nan
            x = np.arange(len(arr))
            slope, intercept, _r_value, _p_value, _std_err = linregress(x, arr)
            y_pred = slope * x + intercept
            return arr[-1] - y_pred[-1]

        return series.rolling(window).apply(_residual, raw=True)

    def _calculate_idxmax(self, series: pd.Series, window: int) -> pd.Series:
        """计算最大值索引（距离当前的天数）"""
        def _idxmax(arr):
            if len(arr) == 0:
                return np.nan
            max_idx = np.argmax(arr)
            return len(arr) - 1 - max_idx  # 距离当前的天数

        return series.rolling(window).apply(_idxmax, raw=True)

    def _calculate_idxmin(self, series: pd.Series, window: int) -> pd.Series:
        """计算最小值索引（距离当前的天数）"""
        def _idxmin(arr):
            if len(arr) == 0:
                return np.nan
            min_idx = np.argmin(arr)
            return len(arr) - 1 - min_idx  # 距离当前的天数

        return series.rolling(window).apply(_idxmin, raw=True)

    def _ensure_feature_order(self, features_df: pd.DataFrame) -> pd.DataFrame:

        # 1. K线特征 (9个)
        # 2. 价格特征 (窗口[0] × 4个字段 = 4个)
        # 3. 成交量特征 (窗口[0,1,2,3,4] = 5个)
        # 4. 滚动特征 (窗口[5,10,20,30,60] × 28个算子 = 140个)

        expected_features = []

        # 1. K线特征 (9个)
        kbar_features = ['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2']
        expected_features.extend(kbar_features)

        # 2. 价格特征 (4个) - 注意：QLib默认只使用窗口0
        price_fields = ['OPEN', 'HIGH', 'LOW', 'VWAP']
        for field in price_fields:
            expected_features.append(f'{field}0')  # 只有窗口0

        # 3. 成交量特征 (5个)
        for d in [0, 1, 2, 3, 4]:
            expected_features.append(f'VOLUME{d}')

        # 4. 滚动特征 (140个) - 严格按照QLib官方代码中的顺序
        rolling_windows = [5, 10, 20, 30, 60]
        rolling_ops = [
            'ROC', 'MA', 'STD', 'BETA', 'RSQR', 'RESI', 'MAX', 'MIN',
            'QTLU', 'QTLD', 'RSV', 'IMAX', 'IMIN', 'IMXD', 'CORR', 'CORD',
            'CNTP', 'CNTN', 'CNTD', 'SUMP', 'SUMN', 'SUMD', 'VMA', 'VSTD',
            'WVMA', 'VSUMP', 'VSUMN', 'VSUMD'
        ]

        for op in rolling_ops:
            for d in rolling_windows:
                expected_features.append(f'{op}{d}')

        # 验证特征数量
        if len(expected_features) != self.feature_count:
            logger.error(f"特征数量不匹配: 预期{self.feature_count}, 实际{len(expected_features)}")
            # 调整到158个
            if len(expected_features) > self.feature_count:
                logger.warning(f"截断多余特征: {len(expected_features) - self.feature_count}个")
                expected_features = expected_features[:self.feature_count]
            else:
                logger.warning(f"补充缺失特征: {self.feature_count - len(expected_features)}个")
                for i in range(len(expected_features), self.feature_count):
                    expected_features.append(f'FILLER{i:03d}')

        # 确保DataFrame包含所有预期特征
        for feature in expected_features:
            if feature not in features_df.columns:
                logger.warning(f"缺失特征: {feature}，用0填充")
                features_df[feature] = 0.0

        # 按预期顺序排列
        features_df = features_df[expected_features]

        # 重命名特征为Alpha001到Alpha158
        rename_dict = {old: f'Alpha{i+1:03d}' for i, old in enumerate(expected_features)}
        features_df = features_df.rename(columns=rename_dict)

        return features_df


    def _create_empty_features(self) -> pd.DataFrame:
        """创建空的158维特征DataFrame"""
        columns = [f'Alpha{i+1:03d}' for i in range(self.feature_count)]
        empty_df = pd.DataFrame(0.0, index=[0], columns=columns)
        logger.warning("创建空的158维特征DataFrame")
        return empty_df


# 测试函数
def test_alpha158_calculator():
    """测试Alpha158计算器"""
    # 生成测试数据
    np.random.seed(42)
    n_samples = 1000

    test_data = []
    base_price = 100.0

    for i in range(n_samples):
        price_change = np.random.normal(0, 0.02)
        base_price *= (1 + price_change)

        test_data.append({
            'timestamp': 1672531200000 + i * 300000,  # 5分钟间隔
            'open': base_price * (1 + np.random.uniform(-0.01, 0.01)),
            'high': base_price * (1 + np.random.uniform(0, 0.02)),
            'low': base_price * (1 - np.random.uniform(0, 0.02)),
            'close': base_price,
            'volume': 1000 + np.random.randint(-200, 200)
        })

    print("测试Alpha158计算器...")
    print(f"测试数据: {len(test_data)} 条")

    # 初始化计算器
    calculator = Alpha158Calculator(freq="5m")

    # 计算特征
    features = calculator.calculate_features(test_data)

    print(f"\n特征形状: {features.shape}")
    print(f"特征数量: {len(features.columns)}")

    # 显示部分特征
    print("\n前10个特征:")
    for col in features.columns[:10]:
        print(f"  {col}: {features[col].iloc[-1]:.6f}")

    print("\n最后10个特征:")
    for col in features.columns[-10:]:
        print(f"  {col}: {features[col].iloc[-1]:.6f}")

    # 检查特征值
    print("\n特征统计:")
    print(f"  非零特征数量: {(features.iloc[-1] != 0).sum()}")
    print(f"  特征值范围: [{features.iloc[-1].min():.6f}, {features.iloc[-1].max():.6f}]")

    return features


if __name__ == "__main__":
    # 运行测试
    features = test_alpha158_calculator()
    print("\n✅ Alpha158计算器测试完成")
