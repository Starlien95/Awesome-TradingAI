"""Versioned data models shared by the CLI, plugins, and artifact stores."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Base model that rejects unknown fields at public boundaries."""

    model_config = ConfigDict(extra="forbid")


class DataSourceInfo(StrictModel):
    plugin_id: str
    source_version: str = "unknown"
    exchange: str | None = None
    market_type: str | None = None


class DataLicenseInfo(StrictModel):
    redistribution: Literal["allowed", "metadata_only", "forbidden", "synthetic"]
    terms_url: str | None = None


class DataTimeInfo(StrictModel):
    timezone: str = "UTC"
    frequency: str
    start: datetime
    end: datetime

    @field_validator("start", "end")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must include an explicit timezone")
        return value


class DatasetManifest(StrictModel):
    schema_version: Literal["1"] = "1"
    dataset_id: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: DataSourceInfo
    license: DataLicenseInfo
    time: DataTimeInfo
    universe_id: str
    symbols: list[str]
    rows: int = Field(ge=0)
    validation_report: str | None = None
    parent_content_sha256: str | None = None
    transformations: list[str] = Field(default_factory=list)


class ArtifactRef(StrictModel):
    path: str
    media_type: str
    schema_id: str | None = None
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    visibility: Literal["public", "private", "trusted_local_only"] = "public"


class CodeVersion(StrictModel):
    git_commit: str | None = None
    dirty: bool = False


class EnvironmentInfo(StrictModel):
    python: str
    quant_bench: str
    pyqlib: str | None = None
    packages: dict[str, str] = Field(default_factory=dict)


class RunManifest(StrictModel):
    schema_version: Literal["1"] = "1"
    run_id: str
    status: Literal["validating", "running", "completed", "failed"]
    recipe_id: str
    protocol_id: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str | None = None
    feature_schema_sha256: str | None = None
    code: CodeVersion
    environment: EnvironmentInfo
    seeds: list[int]
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    failed_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class ModelCapabilities(StrictModel):
    task_types: list[str]
    feature_shapes: list[str]
    dataset_kinds: list[str]
    requires_gpu: bool = False
    deterministic: bool = False
    native_artifact_formats: list[str] = Field(default_factory=list)
    optional_extra: str | None = None
    required_metadata: list[str] = Field(default_factory=list)
    supports_incremental_fit: bool = False
    supports_online_inference: bool = False


class ModelCatalogEntry(StrictModel):
    model_id: str
    display_name: str
    aliases: list[str] = Field(default_factory=list)
    backend: Literal["builtin", "qlib", "custom_qlib", "runtime"]
    module_path: str
    class_name: str
    family: str
    verified_against: str
    default_kwargs: dict[str, Any] = Field(default_factory=dict)
    supported_feature_sets: list[str] = Field(
        default_factory=lambda: ["crypto_ohlcv6_v1", "crypto_alpha52_v1"]
    )
    capabilities: ModelCapabilities
    status: Literal["stable", "experimental", "requires_metadata"] = "stable"
    notes: str | None = None


class PredictionFrame:
    """A small runtime wrapper around a canonical prediction DataFrame."""

    REQUIRED_COLUMNS = ("timestamp", "symbol", "score")

    def __init__(self, frame: pd.DataFrame):
        missing = [column for column in self.REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"prediction frame is missing columns: {', '.join(missing)}")
        self.frame = frame.copy()


class ExperimentResult(StrictModel):
    run_id: str
    run_dir: Path
    status: Literal["completed", "failed"]
    metrics: dict[str, float | int | None] = Field(default_factory=dict)
    manifest_path: Path
