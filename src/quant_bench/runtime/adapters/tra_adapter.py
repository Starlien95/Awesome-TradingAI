import logging
import os
from typing import Any

import pandas as pd
import torch

from quant_bench.runtime.adapters.base import BaseModelAdapter
from quant_bench.runtime.adapters.torch_serialization import pickle_load_torch_device
from quant_bench.runtime.core.config import ConfigManager

logger = logging.getLogger(__name__)


class TRAAdapter(BaseModelAdapter):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wrapper = None
        self.model = None
        self.tra = None
        self.num_states = 1

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[TRA] model file not found: {model_path}")
        try:
            with open(model_path, "rb") as f:
                self.wrapper = pickle_load_torch_device(f, self.device)
            self.model = getattr(self.wrapper, "model", None)
            self.tra = getattr(self.wrapper, "tra", None)
            if self.model is None or self.tra is None:
                raise ValueError("qlib TRAModel wrapper does not contain model and tra modules")
            self.model.to(self.device).eval()
            self.tra.to(self.device).eval()
            self.num_states = int(getattr(self.tra, "num_states", 1))
            logger.info("[TRA] qlib wrapper loaded from %s", os.path.basename(model_path))
        except Exception as exc:
            logger.error("[TRA] load failed: %s", exc)
            self.wrapper = None
            self.model = None
            self.tra = None
            raise

    def predict(self, shared_data: dict[str, pd.DataFrame], config: ConfigManager) -> list[dict[str, Any]]:
        if self.model is None or self.tra is None:
            return []
        from quant_bench.runtime.adapters.custom_feature_calculator import CustomFeatureCalculator

        seq_len = int(config.get("model.seq_len", config.get("strategy.seq_len", 36)))
        calc = CustomFeatureCalculator()
        scores: list[dict[str, Any]] = []

        for coin, df in shared_data.items():
            try:
                feature_df = calc.calculate_features(df)
                if len(feature_df) < seq_len:
                    continue
                values = feature_df.iloc[-seq_len:].values
                data = torch.tensor(values, dtype=torch.float32).unsqueeze(0).to(self.device)
                state = torch.zeros((1, seq_len, self.num_states), dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    hidden = self.model(data)
                    all_preds, _choice, prob = self.tra(hidden, state)
                    score = (all_preds * prob).sum(dim=1) if prob is not None else all_preds.mean(dim=1)
                scores.append({"coin": coin, "score": float(score.item())})
            except Exception as exc:
                logger.error("[TRA] predict failed for %s: %s", coin, exc)

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores
