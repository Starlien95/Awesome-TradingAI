"""MacroHFT runtime integration for locally exported TorchScript models.

The official MacroHFT repository did not contain a software license at the
audited commit.  This module therefore contains no source copied from that
repository.  Users who have permission to use the upstream implementation can
export their local checkpoints with ``tools/convert_macrohft_external.py`` and
load the resulting TorchScript bundle here.

The trend and volatility context follows the algorithm described in the
MacroHFT paper: a first-order low-pass filter followed by a linear slope, plus
rolling return volatility.  The model architecture remains inside the user's
locally exported artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch

try:
    from scipy.signal import butter, filtfilt

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from quant_bench.runtime.adapters.base import BaseModelAdapter
from quant_bench.runtime.adapters.custom_feature_calculator import CustomFeatureCalculator
from quant_bench.runtime.core.config import ConfigManager

logger = logging.getLogger(__name__)

SINGLE_FEATURES = [f"feat_{index:02d}" for index in range(52)]
TREND_FEATURES = [f"feat_{index:02d}_t60" for index in range(52)]


def _low_pass_slope(window: pd.Series) -> float:
    """Return the slope of a first-order Butterworth-smoothed window."""

    if len(window) < 5 or not HAS_SCIPY:
        return 0.0
    try:
        coefficients_b, coefficients_a = butter(1, 0.05, btype="low")
        values = window.to_numpy(dtype=float)
        smoothed = filtfilt(coefficients_b, coefficients_a, values)
        positions = np.arange(len(smoothed), dtype=float)
        return float(np.polyfit(positions, smoothed, deg=1)[0])
    except (TypeError, ValueError):
        return 0.0


def _add_market_context(
    features: pd.DataFrame,
    close: pd.Series,
    trend_window: int = 60,
    context_window: int = 360,
) -> pd.DataFrame:
    result = features.copy()
    for column in [name for name in result.columns if name.startswith("feat_")]:
        result[f"{column}_t60"] = result[column].rolling(trend_window, min_periods=1).mean()
    result["slope_360"] = close.rolling(context_window).apply(_low_pass_slope, raw=False)
    result["vol_360"] = close.pct_change().fillna(0).rolling(context_window).std()
    return result.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)


class MacroHFTV1Adapter(BaseModelAdapter):
    """Run a six-subagent MacroHFT policy exported as TorchScript.

    ``model.path`` points to the TorchScript hyperagent.  Six TorchScript
    subagents named ``slope_1.pt`` through ``vol_3.pt`` live in
    ``model.subagents_dir``.  If the directory is omitted, it is derived from a
    hyperagent filename ending in ``_hyperagent.pt``.
    """

    SUBAGENT_NAMES: ClassVar[tuple[str, ...]] = (
        "slope_1",
        "slope_2",
        "slope_3",
        "vol_1",
        "vol_2",
        "vol_3",
    )

    def __init__(self) -> None:
        self.feature_calculator = CustomFeatureCalculator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.subagents: list[Any] = []
        self.hyperagent: Any | None = None
        self._previous_action: dict[str, int] = {}

    def load_model(self, model_path: str) -> None:
        artifact = Path(model_path).expanduser().resolve()
        if artifact.suffix.lower() in {".pkl", ".pickle", ".pth"}:
            raise ValueError(
                "MacroHFT legacy pickle/state-dict loading is not distributed because the upstream "
                "source has no license. Export a trusted local checkpoint with "
                "tools/convert_macrohft_external.py and configure the resulting *_hyperagent.pt file."
            )
        if artifact.suffix.lower() != ".pt":
            raise ValueError(f"MacroHFT expects a TorchScript .pt hyperagent: {artifact}")
        if not artifact.is_file():
            raise FileNotFoundError(f"MacroHFT hyperagent not found: {artifact}")

        self.hyperagent = torch.jit.load(str(artifact), map_location=self.device)
        self.hyperagent.eval()
        derived = artifact.with_name(artifact.name.removesuffix("_hyperagent.pt") + "_subagents")
        self.set_subagents_dir(str(derived))
        logger.info("Loaded MacroHFT TorchScript hyperagent: %s", artifact)

    def set_subagents_dir(self, subagents_dir: str) -> None:
        directory = Path(subagents_dir).expanduser().resolve()
        loaded: list[Any] = []
        for name in self.SUBAGENT_NAMES:
            artifact = directory / f"{name}.pt"
            if not artifact.is_file():
                raise FileNotFoundError(f"MacroHFT TorchScript subagent not found: {artifact}")
            module = torch.jit.load(str(artifact), map_location=self.device)
            module.eval()
            loaded.append(module)
        self.subagents = loaded
        logger.info("Loaded %d MacroHFT TorchScript subagents from %s", len(loaded), directory)

    @torch.no_grad()
    def _score_one_coin(self, frame: pd.DataFrame, previous_action: int) -> tuple[float, int]:
        features = self.feature_calculator.calculate_features(frame)
        if features.empty:
            return 0.0, previous_action

        normalized = frame.copy()
        normalized.columns = [str(column).lower() for column in normalized.columns]
        features = _add_market_context(features, normalized["close"])
        latest = features.iloc[-1]

        single_state = torch.from_numpy(
            latest[SINGLE_FEATURES].to_numpy(dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        trend_state = torch.from_numpy(
            latest[TREND_FEATURES].to_numpy(dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        context_state = torch.from_numpy(
            latest[["slope_360", "vol_360"]].to_numpy(dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        action_state = torch.tensor([previous_action], dtype=torch.long, device=self.device)

        subagent_q = torch.stack(
            [module(single_state, trend_state, action_state) for module in self.subagents],
            dim=1,
        )
        weights = self.hyperagent(single_state, trend_state, context_state, action_state)
        combined_q = (weights.unsqueeze(-1) * subagent_q).sum(dim=1)
        hold_value = float(combined_q[0, 0].item())
        buy_value = float(combined_q[0, 1].item())
        return buy_value - hold_value, int(buy_value > hold_value)

    def predict(
        self,
        shared_data: dict[str, pd.DataFrame],
        config: ConfigManager,
    ) -> list[dict[str, Any]]:
        del config
        if self.hyperagent is None or len(self.subagents) != len(self.SUBAGENT_NAMES):
            logger.error("MacroHFT TorchScript bundle has not been loaded")
            return []

        predictions: list[dict[str, Any]] = []
        for coin, frame in shared_data.items():
            if len(frame) < 360:
                logger.debug("MacroHFT skipped %s because fewer than 360 bars are available", coin)
                continue
            try:
                previous = self._previous_action.get(coin, 0)
                score, action = self._score_one_coin(frame, previous)
                self._previous_action[coin] = action
                predictions.append({"coin": coin, "score": float(score)})
            except Exception as exc:
                logger.error("MacroHFT inference failed for %s: %s", coin, exc, exc_info=True)

        predictions.sort(key=lambda item: item["score"], reverse=True)
        return predictions

    def generate_signals(
        self,
        predictions: list[dict[str, Any]],
        current_positions: dict[str, float],
        current_prices: dict[str, float],
        config: ConfigManager,
        budget_snapshot: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        allowed_coins = {coin for coin in (config.coins or []) if coin}
        if not allowed_coins:
            return super().generate_signals(
                predictions, current_positions, current_prices, config, budget_snapshot
            )

        filtered = [prediction for prediction in predictions if prediction.get("coin") in allowed_coins]
        signals = super().generate_signals(
            filtered,
            current_positions,
            current_prices,
            config,
            budget_snapshot,
        )

        minimum_value = float(config.min_position_value_usdt)
        forced_sells = {signal["coin"] for signal in signals if signal.get("side") == "sell"}
        for coin, quantity in current_positions.items():
            if coin in allowed_coins or coin in forced_sells:
                continue
            price = float(current_prices.get(coin, 0.0))
            value = float(quantity) * price
            if quantity > 0 and price > 0 and value >= minimum_value:
                signals.append(
                    {
                        "coin": coin,
                        "side": "sell",
                        "amount_usdt": value,
                        "reason": "macrohft_single_coin_whitelist",
                        "target_value_usdt": 0.0,
                    }
                )

        return [
            signal
            for signal in signals
            if signal.get("side") != "buy" or signal.get("coin") in allowed_coins
        ]
