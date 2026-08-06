"""Authoritative typed configuration models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentSpec(StrictConfig):
    name: str = Field(min_length=1)
    seed: int = 42
    workspace: Path | None = None


class DataSpec(StrictConfig):
    dataset_id: str
    source: str
    path: str
    license: Literal["allowed", "metadata_only", "forbidden", "synthetic"]
    checksum: str | None = None


class UniverseSpec(StrictConfig):
    universe_id: str
    symbols: list[str] = Field(min_length=1)

    @field_validator("symbols")
    @classmethod
    def unique_symbols(cls, value: list[str]) -> list[str]:
        normalized = [symbol.strip().upper() for symbol in value]
        if any(not symbol for symbol in normalized):
            raise ValueError("symbols cannot contain empty values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("symbols must be unique")
        return normalized


class FrequencySpec(StrictConfig):
    id: str
    pandas_rule: str
    annualization: int = Field(gt=0)


class FeatureSpec(StrictConfig):
    feature_set_id: str
    plugin: str
    version: str = "1"
    parameters: dict[str, Any] = Field(default_factory=dict)


class LabelSpec(StrictConfig):
    label_id: str
    task_type: Literal["regression", "classification", "ranking", "rl"]
    horizon_bars: int = Field(gt=0)
    formula: str
    scale: float = 1.0

    @field_validator("formula")
    @classmethod
    def validate_parentheses(cls, value: str) -> str:
        depth = 0
        for char in value:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    raise ValueError("label formula has an unmatched closing parenthesis")
        if depth:
            raise ValueError("label formula has unmatched opening parentheses")
        return value


class ModelSpec(StrictConfig):
    model_id: str
    plugin: str
    backend: Literal["builtin", "qlib", "custom_qlib", "runtime"]
    module_path: str | None = None
    class_name: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_qlib_target(self) -> ModelSpec:
        if self.backend in {"qlib", "custom_qlib"} and not (self.module_path and self.class_name):
            raise ValueError("Qlib models require module_path and class_name")
        return self


class TimeRange(StrictConfig):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def require_order(self) -> TimeRange:
        if self.start >= self.end:
            raise ValueError("time range start must be earlier than end")
        return self


class SplitSpec(StrictConfig):
    train: TimeRange
    valid: TimeRange
    test: TimeRange
    purge_bars: int = Field(default=0, ge=0)
    embargo_bars: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def require_non_overlapping_ranges(self) -> SplitSpec:
        if self.train.end >= self.valid.start:
            raise ValueError("train and valid ranges must not overlap")
        if self.valid.end >= self.test.start:
            raise ValueError("valid and test ranges must not overlap")
        return self


class ExecutionCostSpec(StrictConfig):
    open_cost: float = Field(default=0.001, ge=0, lt=1)
    close_cost: float = Field(default=0.001, ge=0, lt=1)
    slippage_bps: float = Field(default=0.0, ge=0)
    min_cost: float = Field(default=0.0, ge=0)


class StrategySpec(StrictConfig):
    strategy_id: str = "threshold_topk_dropout_v1"
    top_k: int = Field(default=5, gt=0)
    threshold: float = 0.0
    max_dropout: int = Field(default=1, ge=0)
    risk_degree: float = Field(default=0.95, gt=0, le=1)
    rebalance_policy: Literal["passive_topk_dropout", "target_weight"] = "passive_topk_dropout"


class BenchmarkProtocolSpec(StrictConfig):
    protocol_id: str
    tier: Literal["smoke", "standard", "publication"]
    split: SplitSpec
    strategy: StrategySpec = Field(default_factory=StrategySpec)
    costs: ExecutionCostSpec = Field(default_factory=ExecutionCostSpec)
    initial_cash: float = Field(default=100_000.0, gt=0)
    seeds: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_seed_count(self) -> BenchmarkProtocolSpec:
        minimum = {"smoke": 1, "standard": 5, "publication": 20}[self.tier]
        if len(self.seeds) < minimum:
            raise ValueError(f"{self.tier} protocol requires at least {minimum} seed(s)")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("protocol seeds must be unique")
        return self


class TrackerSpec(StrictConfig):
    backend: Literal["local", "qlib", "mlflow", "none"] = "local"
    uri: str | None = None


class RuntimeSpec(StrictConfig):
    mode: Literal["research", "paper", "demo", "live"] = "research"
    allow_network: bool = False

    @model_validator(mode="after")
    def protect_live_mode(self) -> RuntimeSpec:
        if self.mode == "live":
            raise ValueError("live mode is not accepted by the research ExperimentConfig")
        return self


class ExperimentConfig(StrictConfig):
    schema_version: Literal["1"] = "1"
    recipe_id: str
    experiment: ExperimentSpec
    data: DataSpec
    universe: UniverseSpec
    frequency: FrequencySpec
    features: FeatureSpec
    label: LabelSpec
    model: ModelSpec
    protocol: BenchmarkProtocolSpec
    tracker: TrackerSpec = Field(default_factory=TrackerSpec)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)

    @model_validator(mode="after")
    def validate_semantics(self) -> ExperimentConfig:
        if self.label.horizon_bars > self.protocol.split.purge_bars and self.protocol.tier != "smoke":
            raise ValueError(
                "purge_bars must cover label.horizon_bars for standard and publication protocols"
            )
        if self.runtime.mode == "research" and self.runtime.allow_network:
            raise ValueError("research mode must remain offline; use a data command for network access")
        return self
