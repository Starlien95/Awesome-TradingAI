import logging
import os
from typing import Any

import pandas as pd

from quant_bench.runtime.adapters.base import BaseModelAdapter
from quant_bench.runtime.adapters.torch_serialization import torch_load_compat
from quant_bench.runtime.core.config import ConfigManager

logger = logging.getLogger(__name__)

class GATSAdapter(BaseModelAdapter):
    """
    Qlib GATs (Graph Attention) 模型适配器 (4h 频段)
    """
    def __init__(self):
        self.model = None

    def load_model(self, model_path: str):
        # Prefer state_dict if available in the same directory
        dir_path = os.path.dirname(model_path)
        state_dict_path = os.path.join(dir_path, "gats_4h_state_dict.pth")

        target_path = state_dict_path if os.path.exists(state_dict_path) else model_path

        if not os.path.exists(target_path):
            raise FileNotFoundError(f"[GATS] 模型权重不存在: {target_path}")

        try:
            import torch
            from qlib.utils import init_instance_by_config

            # GATS config for 4h freq
            model_config = {
                'class': 'GATs',
                'module_path': 'qlib.contrib.model.pytorch_gats_ts',
                'kwargs': {
                    'd_feat': 52,
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.0,
                    'base_model': 'GRU',
                }
            }
            self.model = init_instance_by_config(model_config)

            # Load weights
            checkpoint = torch_load_compat(target_path, torch.device('cpu'))
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            else:
                state_dict = checkpoint

            # Robustly find the PyTorch module within the Qlib wrapper
            inner_model = None
            for attr in dir(self.model):
                if attr.startswith("__"):
                    continue
                try:
                    val = getattr(self.model, attr)
                    if isinstance(val, torch.nn.Module):
                        inner_model = val
                        break
                except Exception:
                    continue

            if inner_model is not None:
                inner_model.load_state_dict(state_dict, strict=True)
                self.model = inner_model
                self.model.eval()
                logger.info(f"✅ GATS 模型加载成功 (Source: {os.path.basename(target_path)})")
            else:
                raise ValueError(f"Could not find nn.Module in {type(self.model)}")
        except Exception as e:
            self.model = None
            logger.error(f"❌ GATS 加载失败: {e}")
            raise

    def predict(self, shared_data: dict[str, pd.DataFrame], config: ConfigManager) -> list[dict[str, Any]]:
        if self.model is None:
            return []
        scores = []
        for coin, df in shared_data.items():
            # GATs expects a sequence (step_len=24 in config)
            if len(df) < 50:
                continue

            try:
                import torch

                from quant_bench.runtime.adapters.custom_feature_calculator import CustomFeatureCalculator
                calc = CustomFeatureCalculator()
                feature_df = calc.calculate_features(df)
                if len(feature_df) < 24:
                    continue

                # Take last 24 rows for time-series input
                last_24_features = feature_df.iloc[-24:].values
                # format for PyTorch: [1, 24, 52]
                tensor_input = torch.tensor(last_24_features, dtype=torch.float32).unsqueeze(0)

                # Move input to the same device as model
                device = next(self.model.parameters()).device
                tensor_input = tensor_input.to(device)

                with torch.no_grad():
                    pred_tensor = self.model(tensor_input)
                score = pred_tensor.item()

                scores.append({'coin': coin, 'score': float(score)})
            except Exception as e:
                logger.error(f"[GATS] 预测异常：{e}")

        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores
