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


def _install_tcn_pickle_compat() -> None:
    """
    The saved TCN pickle references a training-time forward patch named
    `global_tcn_forward_patch`. Install the same callable before unpickling.
    """
    import qlib.contrib.model.pytorch_tcn_ts as tcn_module

    def global_tcn_forward_patch(self, x):
        output = self.tcn(x)
        output = self.linear(output[:, :, -1])
        return output.squeeze()

    tcn_module.global_tcn_forward_patch = global_tcn_forward_patch
    tcn_module.TCNModel.global_tcn_forward_patch = global_tcn_forward_patch
    tcn_module.TCNModel.forward = global_tcn_forward_patch


class TCNAdapter(BaseModelAdapter):
    """Adapter for qlib.contrib.model.pytorch_tcn_ts TCN/TCNModel."""

    def __init__(self):
        self.wrapper = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_calculator = CustomFeatureCalculator()

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[TCN] model file not found: {model_path}")
        try:
            _install_tcn_pickle_compat()
            with open(model_path, "rb") as handle:
                loaded = pickle_load_torch_device(handle, self.device)
            self.wrapper = loaded
            self.model = getattr(loaded, "TCN_model", loaded if isinstance(loaded, torch.nn.Module) else None)
            if self.model is None:
                raise ValueError("loaded object does not contain TCN_model")
            self.model.to(self.device).eval()
            logger.info("[TCN] qlib TCN loaded from %s", os.path.basename(model_path))
        except Exception as exc:
            logger.error("[TCN] load failed: %s", exc)
            self.wrapper = None
            self.model = None
            raise

    def predict(self, shared_data: dict[str, pd.DataFrame], config: ConfigManager) -> list[dict[str, Any]]:
        if self.model is None:
            return []

        step_len = int(config.get("model.step_len", config.get("strategy.step_len", 20)))
        min_rows = int(config.get("model.min_feature_rows", max(288, step_len)))
        scores: list[dict[str, Any]] = []

        for coin, df in shared_data.items():
            if len(df) < min_rows:
                continue
            try:
                feature_df = self.feature_calculator.calculate_features(df)
                if len(feature_df) < step_len:
                    continue
                if bool(config.get("model.live_robust_zscore", True)):
                    feature_df = rolling_robust_zscore(
                        feature_df,
                        window=int(config.get("model.robust_zscore_window", 288)),
                        clip=float(config.get("model.robust_zscore_clip", 3.0)),
                    )
                values = feature_df.iloc[-step_len:].values.astype("float32")
                # qlib TCNModel is Conv1d-based and expects [batch, d_feat, step_len].
                tensor_input = torch.tensor(values.T, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    pred = self.model(tensor_input)
                    score = float(pred.reshape(-1)[0].detach().cpu().item())
                scores.append({"coin": coin, "score": score})
            except Exception as exc:
                logger.error("[TCN] predict failed for %s: %s", coin, exc)

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores
