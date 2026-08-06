import logging
import os
from typing import Any

import pandas as pd
import torch

from quant_bench.runtime.adapters.base import BaseModelAdapter
from quant_bench.runtime.adapters.custom_feature_calculator import CustomFeatureCalculator
from quant_bench.runtime.adapters.feature_normalizer import rolling_robust_zscore
from quant_bench.runtime.adapters.torch_serialization import pickle_load_torch_device
from quant_bench.runtime.core.config import ConfigManager

logger = logging.getLogger(__name__)


class TabNetAdapter(BaseModelAdapter):
    """Adapter for qlib.contrib.model.pytorch_tabnet.TabnetModel."""

    def __init__(self):
        self.wrapper = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_calculator = CustomFeatureCalculator()

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[TabNet] model file not found: {model_path}")
        try:
            with open(model_path, "rb") as handle:
                self.wrapper = pickle_load_torch_device(handle, self.device)
            self.model = getattr(self.wrapper, "tabnet_model", None)
            if self.model is None:
                raise ValueError("loaded object does not contain tabnet_model")
            self.model.to(self.device).eval()
            logger.info("[TabNet] qlib TabNet loaded from %s", os.path.basename(model_path))
        except Exception as exc:
            logger.error("[TabNet] load failed: %s", exc)
            self.wrapper = None
            self.model = None
            raise

    def predict(self, shared_data: dict[str, pd.DataFrame], config: ConfigManager) -> list[dict[str, Any]]:
        if self.model is None:
            return []

        scores: list[dict[str, Any]] = []
        min_rows = int(config.get("model.min_feature_rows", 288))
        d_feat = int(getattr(self.wrapper, "d_feat", config.get("model.d_feat", 52)))

        for coin, df in shared_data.items():
            if len(df) < min_rows:
                continue
            try:
                feature_df = self.feature_calculator.calculate_features(df)
                if feature_df.empty:
                    continue
                if bool(config.get("model.live_robust_zscore", True)):
                    feature_df = rolling_robust_zscore(
                        feature_df,
                        window=int(config.get("model.robust_zscore_window", 288)),
                        clip=float(config.get("model.robust_zscore_clip", 3.0)),
                    )
                values = feature_df.iloc[-1].values.astype("float32")
                tensor_input = torch.tensor(values, dtype=torch.float32, device=self.device).reshape(1, -1)
                priors = torch.ones((1, d_feat), dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    pred = self.model(tensor_input, priors)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    score = float(pred.reshape(-1)[0].detach().cpu().item())
                scores.append({"coin": coin, "score": score})
            except Exception as exc:
                logger.error("[TabNet] predict failed for %s: %s", coin, exc)

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores
