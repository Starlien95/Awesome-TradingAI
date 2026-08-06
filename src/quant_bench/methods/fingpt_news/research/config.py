"""Typed configuration for FinGPT sentiment research workflows."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

StrategyVariant = Literal["asset_only", "market_only", "hybrid_fixed", "hybrid_adaptive"]
Objective = Literal["sharpe", "total_return", "calmar", "excess_total_return"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DateWindow(StrictModel):
    start: str
    end: str

    @model_validator(mode="after")
    def validate_order(self) -> DateWindow:
        import pandas as pd

        start = pd.Timestamp(self.start)
        end = pd.Timestamp(self.end)
        if start >= end:
            raise ValueError(f"date window requires start < end: {self.start} >= {self.end}")
        return self


class InputSpec(StrictModel):
    prices: Path
    validation_sentiment: Path
    test_sentiment: Path
    price_date_column: str = "date"
    price_symbol_column: str = "symbol"
    price_close_column: str = "close"


class AggregationGrid(StrictModel):
    asset_top_n: list[int] = Field(default_factory=lambda: [12])
    market_top_n: list[int] = Field(default_factory=lambda: [24])
    min_abs_score: list[float] = Field(default_factory=lambda: [0.0])
    score_power: list[float] = Field(default_factory=lambda: [0.0])

    @model_validator(mode="after")
    def validate_grid(self) -> AggregationGrid:
        if not all((self.asset_top_n, self.market_top_n, self.min_abs_score, self.score_power)):
            raise ValueError("aggregation grids cannot be empty")
        if any(value < 0 for value in self.asset_top_n + self.market_top_n):
            raise ValueError("top_n values must be non-negative")
        if any(value < 0 for value in self.min_abs_score + self.score_power):
            raise ValueError("min_abs_score and score_power must be non-negative")
        return self

    def combinations(self) -> list[dict[str, float | int]]:
        return [
            {
                "asset_top_n": int(asset_top_n),
                "market_top_n": int(market_top_n),
                "min_abs_score": float(min_abs_score),
                "score_power": float(score_power),
            }
            for asset_top_n, market_top_n, min_abs_score, score_power in product(
                self.asset_top_n,
                self.market_top_n,
                self.min_abs_score,
                self.score_power,
            )
        ]


class SignalGrid(StrictModel):
    variants: list[StrategyVariant] = Field(default_factory=lambda: ["asset_only", "hybrid_adaptive"])
    thresholds: list[float] = Field(default_factory=lambda: [0.03, 0.05, 0.08, 0.10])
    fixed_asset_alpha: list[float] = Field(default_factory=lambda: [0.70])
    adaptive_k: list[float] = Field(default_factory=lambda: [6.0])
    adaptive_min_alpha: list[float] = Field(default_factory=lambda: [0.50])
    adaptive_max_alpha: list[float] = Field(default_factory=lambda: [0.90])
    market_beta: dict[str, float] = Field(default_factory=dict)
    conflict_policy: Literal["none", "neutral"] = "none"
    conflict_threshold: float = 0.08

    @model_validator(mode="after")
    def validate_grid(self) -> SignalGrid:
        if not self.variants or not self.thresholds:
            raise ValueError("signal variants and thresholds cannot be empty")
        if any(value < 0 for value in self.thresholds):
            raise ValueError("thresholds must be non-negative")
        if any(not 0 <= value <= 1 for value in self.fixed_asset_alpha):
            raise ValueError("fixed_asset_alpha must be within [0, 1]")
        for minimum, maximum in product(self.adaptive_min_alpha, self.adaptive_max_alpha):
            if not 0 <= minimum <= maximum <= 1:
                raise ValueError("adaptive alpha bounds must satisfy 0 <= min <= max <= 1")
        if any(value <= 0 for value in self.adaptive_k):
            raise ValueError("adaptive_k must be positive")
        return self

    def variant_combinations(self) -> list[dict[str, float | str | None]]:
        rows: list[dict[str, float | str | None]] = []
        for variant in self.variants:
            if variant == "asset_only":
                rows.append({"variant": variant, "fixed_alpha": 1.0, "adaptive_k": None,
                             "min_alpha": None, "max_alpha": None})
            elif variant == "market_only":
                rows.append({"variant": variant, "fixed_alpha": 0.0, "adaptive_k": None,
                             "min_alpha": None, "max_alpha": None})
            elif variant == "hybrid_fixed":
                rows.extend(
                    {"variant": variant, "fixed_alpha": float(alpha), "adaptive_k": None,
                     "min_alpha": None, "max_alpha": None}
                    for alpha in self.fixed_asset_alpha
                )
            else:
                rows.extend(
                    {"variant": variant, "fixed_alpha": None, "adaptive_k": float(k),
                     "min_alpha": float(minimum), "max_alpha": float(maximum)}
                    for k, minimum, maximum in product(
                        self.adaptive_k, self.adaptive_min_alpha, self.adaptive_max_alpha
                    )
                )
        return rows


class BacktestSpec(StrictModel):
    initial_capital_usdt: float = 10_000.0
    per_symbol_capital_usdt: float | None = None
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    signal_lag_days: int = 1
    annualization: int = 365
    calendar_policy: Literal["intersection"] = "intersection"

    @model_validator(mode="after")
    def validate_values(self) -> BacktestSpec:
        if self.initial_capital_usdt <= 0:
            raise ValueError("initial_capital_usdt must be positive")
        if self.per_symbol_capital_usdt is not None and self.per_symbol_capital_usdt <= 0:
            raise ValueError("per_symbol_capital_usdt must be positive")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("fee_bps and slippage_bps must be non-negative")
        if self.signal_lag_days < 1:
            raise ValueError("signal_lag_days must be at least 1 to prevent look-ahead")
        if self.annualization < 1:
            raise ValueError("annualization must be positive")
        return self


class SelectionSpec(StrictModel):
    objective: Objective = "sharpe"
    max_trials: int = 500
    min_trades: int = 1
    min_signal_days: int = 1
    min_classified_news: int = 1
    enable_per_symbol_thresholds: bool = True

    @model_validator(mode="after")
    def validate_values(self) -> SelectionSpec:
        if self.max_trials < 1:
            raise ValueError("max_trials must be positive")
        if min(self.min_trades, self.min_signal_days, self.min_classified_news) < 0:
            raise ValueError("selection coverage guards must be non-negative")
        return self


class FinGPTResearchConfig(StrictModel):
    schema_version: int = 1
    strategy_id: str = "FinGPTSentimentResearchV1"
    symbols: list[str]
    inputs: InputSpec
    validation: DateWindow
    test: DateWindow
    aggregation: AggregationGrid = Field(default_factory=AggregationGrid)
    signals: SignalGrid = Field(default_factory=SignalGrid)
    backtest: BacktestSpec = Field(default_factory=BacktestSpec)
    selection: SelectionSpec = Field(default_factory=SelectionSpec)
    model_id: str = "external-sentiment-model"
    adapter_id: str = "external-adapter"
    dataset_id: str = "user-provided"

    @model_validator(mode="after")
    def validate_protocol(self) -> FinGPTResearchConfig:
        import pandas as pd

        normalized = [symbol.strip().upper() for symbol in self.symbols if symbol.strip()]
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("symbols must be non-empty and unique")
        self.symbols = normalized
        if pd.Timestamp(self.validation.end) > pd.Timestamp(self.test.start):
            raise ValueError("validation window must end on or before the test window starts")
        per_symbol = self.backtest.per_symbol_capital_usdt
        if per_symbol is None:
            self.backtest.per_symbol_capital_usdt = self.backtest.initial_capital_usdt / len(normalized)
        expected = float(self.backtest.per_symbol_capital_usdt) * len(normalized)
        if abs(expected - self.backtest.initial_capital_usdt) > 1e-6:
            raise ValueError(
                "initial_capital_usdt must equal per_symbol_capital_usdt multiplied by symbol count"
            )
        trial_count = (
            len(self.aggregation.combinations())
            * len(self.signals.variant_combinations())
            * len(self.signals.thresholds)
        )
        if trial_count > self.selection.max_trials:
            raise ValueError(
                f"configured grid has {trial_count} trials, above max_trials={self.selection.max_trials}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> FinGPTResearchConfig:
        source = Path(path).expanduser().resolve()
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        config = cls.model_validate(payload)
        for field in ("prices", "validation_sentiment", "test_sentiment"):
            value = getattr(config.inputs, field)
            if not value.is_absolute():
                setattr(config.inputs, field, (source.parent / value).resolve())
        return config

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), allow_unicode=True, sort_keys=False)
