"""A deterministic ridge baseline used by the offline quickstart."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quant_bench.contracts import ArtifactReader, ArtifactRef, ArtifactWriter, DatasetView, ModelCapabilities
from quant_bench.contracts.models import PredictionFrame


@dataclass
class NumpyLinearModel:
    columns: tuple[str, ...]
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    intercept: float


class NumpyLinearPlugin:
    plugin_id = "numpy_linear_v1"
    contract_version = "1"

    def __init__(self, ridge: float = 1.0e-6) -> None:
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        self.ridge = float(ridge)

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            task_types=["regression"],
            feature_shapes=["tabular"],
            dataset_kinds=["canonical"],
            deterministic=True,
            native_artifact_formats=["npz"],
        )

    def fit(self, dataset: DatasetView, context: Any = None) -> NumpyLinearModel:
        del context
        features = dataset.train_features.astype(float)
        labels = dataset.train_labels.astype(float)
        valid = features.notna().all(axis=1) & labels.notna()
        x = features.loc[valid].to_numpy(dtype=float)
        y = labels.loc[valid].to_numpy(dtype=float)
        if len(x) == 0:
            raise ValueError("no finite training samples are available")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale == 0] = 1.0
        standardized = (x - mean) / scale
        design = np.column_stack([np.ones(len(standardized)), standardized])
        penalty = np.eye(design.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        weights = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
        return NumpyLinearModel(
            columns=tuple(features.columns),
            mean=mean,
            scale=scale,
            coefficients=weights[1:],
            intercept=float(weights[0]),
        )

    def predict(self, model: NumpyLinearModel, dataset: DatasetView) -> PredictionFrame:
        features = dataset.test_features.loc[:, list(model.columns)].astype(float)
        finite = features.notna().all(axis=1)
        scores = pd.Series(np.nan, index=features.index, dtype=float, name="score")
        if finite.any():
            x = features.loc[finite].to_numpy(dtype=float)
            scores.loc[finite] = ((x - model.mean) / model.scale) @ model.coefficients + model.intercept
        return PredictionFrame(scores.rename("score").reset_index())

    def save(self, model: NumpyLinearModel, target: ArtifactWriter) -> ArtifactRef:
        buffer = io.BytesIO()
        np.savez(
            buffer,
            columns=np.asarray(model.columns),
            mean=model.mean,
            scale=model.scale,
            coefficients=model.coefficients,
            intercept=np.asarray([model.intercept]),
        )
        return target.write_bytes("model/model.npz", buffer.getvalue(), "application/x-npz")

    def load(self, source: ArtifactReader) -> NumpyLinearModel:
        with np.load(io.BytesIO(source.read_bytes("model/model.npz")), allow_pickle=False) as payload:
            return NumpyLinearModel(
                columns=tuple(str(value) for value in payload["columns"].tolist()),
                mean=payload["mean"],
                scale=payload["scale"],
                coefficients=payload["coefficients"],
                intercept=float(payload["intercept"][0]),
            )
