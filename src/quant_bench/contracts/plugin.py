"""Protocols implemented by built-in and third-party plugins."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from quant_bench.contracts.models import ArtifactRef, ModelCapabilities, PredictionFrame


class DatasetView:
    """Backend-neutral train, validation, and test tables."""

    def __init__(
        self,
        train_features: pd.DataFrame,
        train_labels: pd.Series,
        valid_features: pd.DataFrame,
        valid_labels: pd.Series,
        test_features: pd.DataFrame,
        test_labels: pd.Series,
    ) -> None:
        self.train_features = train_features
        self.train_labels = train_labels
        self.valid_features = valid_features
        self.valid_labels = valid_labels
        self.test_features = test_features
        self.test_labels = test_labels


class ArtifactWriter(Protocol):
    root: Path

    def write_bytes(
        self,
        relative_path: str,
        data: bytes,
        media_type: str,
        *,
        schema_id: str | None = None,
        visibility: str = "public",
    ) -> ArtifactRef: ...


class ArtifactReader(Protocol):
    root: Path

    def read_bytes(self, relative_path: str) -> bytes: ...


@runtime_checkable
class ModelPlugin(Protocol):
    plugin_id: str
    contract_version: str

    def capabilities(self) -> ModelCapabilities: ...

    def fit(self, dataset: DatasetView, context: Any) -> Any: ...

    def predict(self, model: Any, dataset: DatasetView) -> PredictionFrame: ...

    def save(self, model: Any, target: ArtifactWriter) -> ArtifactRef: ...

    def load(self, source: ArtifactReader) -> Any: ...
