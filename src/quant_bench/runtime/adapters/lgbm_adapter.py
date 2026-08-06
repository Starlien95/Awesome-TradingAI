import logging
import os
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd

from quant_bench.runtime.adapters.alpha158 import Alpha158Calculator
from quant_bench.runtime.adapters.base import BaseModelAdapter
from quant_bench.runtime.core.config import ConfigManager

logger = logging.getLogger(__name__)

class LGBMAdapter(BaseModelAdapter):
    def __init__(self):
        self.model = None
        # 根据 alpha158 特征类实例化参数
        self.feature_calculator = Alpha158Calculator(freq="5m")

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LGBM 找不到模型文件: {model_path}")

        try:
            if model_path.endswith('.pkl'):
                self.model = joblib.load(model_path)
            else:
                self.model = lgb.Booster(model_file=model_path)
        except Exception:
            try:
                self.model = joblib.load(model_path)
            except Exception:
                self.model = lgb.Booster(model_file=model_path)

        logger.info(f"✅ LGBM 适配器加载模型成功: {model_path}")

    def predict(self, shared_data: dict[str, pd.DataFrame], config: ConfigManager) -> list[dict[str, Any]]:
        scores = []
        for coin, df in shared_data.items():
            if len(df) < 65: # Alpha158 至少需要 60 根 K线
                continue

            try:
                feature_df = self.feature_calculator.calculate_features(df)
                if feature_df.empty:
                    continue

                current_features = feature_df.iloc[[-1]]
                pred_score = self.model.predict(current_features)[0]

                scores.append({
                    'coin': coin,
                    'score': float(pred_score)
                })
            except Exception as e:
                logger.error(f"[LGBM] 预测 {coin} 出错: {e}")

        # 按分数从高到低排序
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores

    def generate_signals(
        self,
        predictions: list[dict[str, Any]],
        current_positions: dict[str, float],
        current_prices: dict[str, float],
        config: ConfigManager,
        budget_snapshot: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return super().generate_signals(
            predictions,
            current_positions,
            current_prices,
            config,
            budget_snapshot=budget_snapshot,
        )
