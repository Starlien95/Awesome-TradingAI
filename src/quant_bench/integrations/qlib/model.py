"""Generic adapter for every catalogued Qlib model class."""

from __future__ import annotations

import importlib
from typing import Any

import pandas as pd

from quant_bench.contracts import ArtifactReader, ArtifactWriter, ModelCapabilities, PredictionFrame
from quant_bench.registry import ModelRegistry


class QlibModelPlugin:
    plugin_id = "qlib_model_v1"
    contract_version = "1"

    def __init__(self, model_id: str, parameters: dict[str, Any] | None = None) -> None:
        self.registry = ModelRegistry()
        self.entry = self.registry.get(model_id)
        if self.entry.backend not in {"qlib", "custom_qlib"}:
            raise ValueError(f"{model_id} is not a Qlib-backed model")
        self.parameters = parameters or {}

    def capabilities(self) -> ModelCapabilities:
        return self.entry.capabilities

    def create_model(self) -> Any:
        parameters = {**self.entry.default_kwargs, **self.parameters}
        try:
            module = importlib.import_module(self.entry.module_path)
        except ModuleNotFoundError as exc:
            extra = self.entry.capabilities.optional_extra or "qlib"
            raise RuntimeError(f"{self.entry.model_id} requires optional extra(s): {extra}") from exc
        model_class = getattr(module, self.entry.class_name)
        return model_class(**parameters)

    def fit(self, dataset: Any, context: Any = None) -> Any:
        del context
        model = self.create_model()
        model.fit(dataset)
        return model

    def predict(self, model: Any, dataset: Any) -> PredictionFrame:
        prediction = model.predict(dataset)
        if isinstance(prediction, pd.DataFrame):
            if prediction.shape[1] != 1:
                raise ValueError("Qlib prediction DataFrame must contain one score column")
            prediction = prediction.iloc[:, 0]
        if not isinstance(prediction, pd.Series):
            raise TypeError(f"unsupported Qlib prediction type: {type(prediction).__name__}")
        frame = prediction.rename("score").reset_index()
        rename = {"datetime": "timestamp", "instrument": "symbol"}
        frame = frame.rename(columns=rename)
        return PredictionFrame(frame)

    def save(self, model: Any, target: ArtifactWriter) -> Any:
        del model, target
        raise RuntimeError("Qlib models must be persisted by Qlib Recorder or a native model exporter")

    def load(self, source: ArtifactReader) -> Any:
        del source
        raise RuntimeError(
            "public Qlib model loading rejects arbitrary pickle; use a native exporter or trusted migration"
        )
