import logging
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd

from quant_bench.runtime.adapters.base import BaseModelAdapter
from quant_bench.runtime.core.config import ConfigManager

logger = logging.getLogger(__name__)

class XGBoostAdapter(BaseModelAdapter):
    """
    Qlib XGBoost 模型适配器 (4h 频段)
    """
    def __init__(self):
        self.model = None

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[XGBoost] 模型文件不存在: {model_path}")

        try:
            self.model = joblib.load(model_path)
            model_to_use = getattr(self.model, 'model', self.model)
            if not hasattr(model_to_use, 'predict'):
                raise ValueError(f"[XGBoost] 加载对象不具备 predict 能力: {type(self.model)}")
            logger.info("✅ XGBoost 加载成功")
        except Exception as e:
            self.model = None
            logger.error(f"❌ XGBoost 加载失败: {e}")
            raise

    def predict(self, shared_data: dict[str, pd.DataFrame], config: ConfigManager) -> list[dict[str, Any]]:
        if self.model is None:
            return []
        scores = []
        for coin, df in shared_data.items():
            if len(df) < 50:
                continue

            try:
                import xgboost as xgb

                from quant_bench.runtime.adapters.custom_feature_calculator import CustomFeatureCalculator
                calc = CustomFeatureCalculator()
                feature_df = calc.calculate_features(df)
                if feature_df.empty:
                    continue

                last_features = feature_df.iloc[[-1]]

                # Qlib XGBoost wrapper stores the native booster in .model
                model_to_use = getattr(self.model, 'model', self.model)

                if hasattr(model_to_use, 'predict'):
                    try:
                        # Raw booster requires DMatrix
                        if isinstance(model_to_use, xgb.core.Booster):
                            dtest = xgb.DMatrix(last_features)
                            pred = model_to_use.predict(dtest)
                        else:
                            pred = model_to_use.predict(last_features)
                    except Exception:
                        # Fallback attempt with DMatrix if normal predict fails
                        dtest = xgb.DMatrix(last_features)
                        pred = model_to_use.predict(dtest)
                else:
                    raise RuntimeError(f"[XGBoost] 已加载模型不具备 predict 能力: {type(model_to_use)}")

                score = float(pred[0]) if isinstance(pred, (list, tuple, np.ndarray)) else float(pred)

                scores.append({'coin': coin, 'score': score})
            except Exception as e:
                logger.error(f"[XGBoost] 预测异常：{e}")

        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores
