import logging
import os
from typing import Any

import joblib
import pandas as pd

from quant_bench.runtime.adapters.base import BaseModelAdapter
from quant_bench.runtime.core.config import ConfigManager

logger = logging.getLogger(__name__)

class DoubleEnsembleAdapter(BaseModelAdapter):
    """
    Qlib DoubleEnsemble 模型适配器 (15m 频段)
    """
    def __init__(self):
        self.model = None

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[DoubleEnsemble] 模型文件不存在: {model_path}")

        try:
            self.model = joblib.load(model_path)
            ensemble = getattr(self.model, 'ensemble', None)
            has_ensemble = ensemble is not None and len(ensemble) > 0
            if not has_ensemble and not hasattr(self.model, 'predict'):
                raise ValueError(f"[DoubleEnsemble] 加载对象不具备 predict/ensemble 能力: {type(self.model)}")
            logger.info("✅ DoubleEnsemble 加载成功")
        except Exception as e:
            self.model = None
            logger.error(f"❌ DoubleEnsemble 加载失败: {e}")
            raise

    def predict(self, shared_data: dict[str, pd.DataFrame], config: ConfigManager) -> list[dict[str, Any]]:
        if self.model is None:
            return []
        scores = []
        for coin, df in shared_data.items():
            # For 15m DE, max sequence lookback depends on the handler (up to 288 for CustomFeature_RL)
            if len(df) < 50:
                continue

            try:
                import numpy as np

                from quant_bench.runtime.adapters.custom_feature_calculator import CustomFeatureCalculator
                calc = CustomFeatureCalculator()
                feature_df = calc.calculate_features(df)
                if feature_df.empty:
                    continue

                last_features = feature_df.iloc[[-1]]

                # DoubleEnsemble wrapper extraction from Qlib
                ensemble = getattr(self.model, 'ensemble', None)
                sub_features = getattr(self.model, 'sub_features', None)
                sub_weights = getattr(self.model, 'sub_weights', None)

                if ensemble and len(ensemble) > 0:
                    sub_preds = []
                    for i, sub in enumerate(ensemble):
                        if hasattr(sub, 'predict'):
                            # Filter features if sub_features is available
                            if sub_features is not None and len(sub_features) > i:
                                current_sub_features = sub_features[i]
                                # Ensure we only pick columns that exist in the dataframe
                                input_data = last_features[current_sub_features]
                            else:
                                input_data = last_features

                            # LightGBM booster can take DataFrame directly
                            sub_preds.append(sub.predict(input_data)[0])
                        else:
                            raise RuntimeError(f"[DoubleEnsemble] 子模型不具备 predict 能力: {type(sub)}")

                    if sub_weights is not None and len(sub_weights) == len(ensemble):
                        score = np.sum(np.array(sub_preds) * np.array(sub_weights))
                    else:
                        score = np.mean(sub_preds)
                elif hasattr(self.model, 'predict'):
                    pred = self.model.predict(last_features)
                    score = float(pred[0]) if isinstance(pred, (list, tuple, np.ndarray)) else float(pred)
                else:
                    raise RuntimeError(f"[DoubleEnsemble] 已加载模型不具备 predict/ensemble 能力: {type(self.model)}")

                scores.append({'coin': coin, 'score': float(score)})
            except Exception as e:
                logger.error(f"[DoubleEnsemble] 预测异常：{e}")

        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores
