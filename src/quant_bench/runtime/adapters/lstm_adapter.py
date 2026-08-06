import logging
import os
from typing import Any

import pandas as pd
import torch
import torch.nn as nn

from quant_bench.runtime.adapters.base import BaseModelAdapter
from quant_bench.runtime.adapters.torch_serialization import torch_load_compat
from quant_bench.runtime.core.config import ConfigManager

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    def __init__(self, d_feat: int, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze(-1)


class LSTMAdapter(BaseModelAdapter):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _build_from_state(self, state: dict):
        input_dim = int(state["rnn.weight_ih_l0"].shape[1])
        hidden_size = int(state["rnn.weight_hh_l0"].shape[1])
        layer_ids = {
            int(k.split("_l")[-1].split(".")[0])
            for k in state
            if k.startswith("rnn.weight_ih_l")
        }
        num_layers = max(layer_ids) + 1 if layer_ids else 1
        self.model = LSTMModel(input_dim, hidden_size, num_layers, dropout=0.0).to(self.device)

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[LSTM] model file not found: {model_path}")
        try:
            checkpoint = torch_load_compat(model_path, self.device)
            if isinstance(checkpoint, dict):
                state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
            elif hasattr(checkpoint, "state_dict"):
                state = checkpoint.state_dict()
            else:
                state = None
            if state is None:
                raise ValueError(f"cannot parse state_dict from {model_path}")
            self._build_from_state(state)
            self.model.load_state_dict(state, strict=True)
            self.model.eval()
            logger.info("[LSTM] model loaded from %s", os.path.basename(model_path))
        except Exception as exc:
            self.model = None
            logger.error("[LSTM] load failed: %s", exc)
            raise

    def predict(self, shared_data: dict[str, pd.DataFrame], config: ConfigManager) -> list[dict[str, Any]]:
        if self.model is None:
            return []
        from quant_bench.runtime.adapters.custom_feature_calculator import CustomFeatureCalculator

        step_len = int(config.get("model.step_len", config.get("strategy.step_len", 48)))
        calc = CustomFeatureCalculator()
        scores: list[dict[str, Any]] = []
        self.model.eval()

        for coin, df in shared_data.items():
            try:
                feature_df = calc.calculate_features(df)
                if len(feature_df) < step_len:
                    continue
                values = feature_df.iloc[-step_len:].values
                tensor_input = torch.tensor(values, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    score = float(self.model(tensor_input).item())
                scores.append({"coin": coin, "score": score})
            except Exception as exc:
                logger.error("[LSTM] predict failed for %s: %s", coin, exc)

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores
