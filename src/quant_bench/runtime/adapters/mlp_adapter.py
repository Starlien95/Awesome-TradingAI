import logging
import os
from typing import Any

import pandas as pd
import torch
import torch.nn as nn

from quant_bench.runtime.adapters.base import BaseModelAdapter
from quant_bench.runtime.adapters.custom_features import CustomFeature_RL
from quant_bench.runtime.adapters.torch_serialization import torch_load_compat
from quant_bench.runtime.core.config import ConfigManager

logger = logging.getLogger(__name__)

class MLPStrategy(nn.Module):
    def __init__(self, input_dim=52, hidden_dim=64, output_dim=1, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class MLPAdapter(BaseModelAdapter):
    def __init__(self):
        self.feature_calculator = CustomFeature_RL()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _build_model_from_state(self, state: dict):
        first_weight = state.get("net.0.weight")
        if first_weight is None:
            raise ValueError("MLP state_dict missing net.0.weight")
        hidden_dim = int(first_weight.shape[0])
        input_dim = int(first_weight.shape[1])
        output_weight = state.get("net.8.weight")
        output_dim = int(output_weight.shape[0]) if output_weight is not None else 1
        self.model = MLPStrategy(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=0.0,
        ).to(self.device)

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[MLP] 模型文件不存在: {model_path}")

        try:
            checkpoint = torch_load_compat(model_path, self.device)
            state = None
            if isinstance(checkpoint, dict):
                state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
            elif hasattr(checkpoint, "state_dict"):
                state = checkpoint.state_dict()
            if state is None:
                raise ValueError("[MLP] 无法从 checkpoint 解析 state_dict")
            self._build_model_from_state(state)
            self.model.load_state_dict(state, strict=True)
            self.model.eval()
            logger.info("✅ MLP 适配器加载模型成功。")
        except Exception as e:
            self.model = None
            logger.error(f"❌ MLP 加载失败: {e}")
            raise e

    def predict(self, shared_data: dict[str, pd.DataFrame], config: ConfigManager) -> list[dict[str, Any]]:
        if self.model is None:
            return []
        self.model.eval()
        scores = []
        for coin, df in shared_data.items():
            if len(df) < 288:  # CustomFeature_RL 最大回看窗口288
                continue

            try:
                from quant_bench.runtime.adapters.custom_feature_calculator import CustomFeatureCalculator
                calc = CustomFeatureCalculator()
                feature_df = calc.calculate_features(df)
                if feature_df.empty:
                    continue

                # get last row
                last_features = feature_df.iloc[-1].values
                # convert to tensor [1, 52]
                tensor_input = torch.tensor(last_features, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    pred_tensor = self.model(tensor_input)

                score = pred_tensor.item()

                scores.append({
                    'coin': coin,
                    'score': float(score)
                })
            except Exception as e:
                logger.error(f"[MLP] 预测异常：{e}")

        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores
