"""Built-in model catalog and external entry-point discovery."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterable
from importlib import metadata, resources
from typing import Any

import yaml

from quant_bench.config.models import ModelSpec
from quant_bench.contracts import ModelCatalogEntry


class ModelRegistry:
    """Resolve every supported model through one catalog."""

    ENTRY_POINT_GROUP = "quant_bench.models"

    def __init__(self, entries: Iterable[ModelCatalogEntry] | None = None) -> None:
        loaded = list(entries) if entries is not None else self._load_builtin_entries()
        self._entries: dict[str, ModelCatalogEntry] = {}
        self._aliases: dict[str, str] = {}
        for entry in loaded:
            key = entry.model_id.lower()
            if key in self._entries:
                raise ValueError(f"duplicate model_id: {entry.model_id}")
            self._entries[key] = entry
            for alias in [entry.model_id, *entry.aliases]:
                normalized = alias.lower()
                previous = self._aliases.get(normalized)
                if previous and previous != key:
                    raise ValueError(f"duplicate model alias: {alias}")
                self._aliases[normalized] = key

    @staticmethod
    def _load_builtin_entries() -> list[ModelCatalogEntry]:
        path = resources.files("quant_bench").joinpath("resources").joinpath("model_catalog.yaml")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
            raise ValueError("model_catalog.yaml must contain a models list")
        return [ModelCatalogEntry.model_validate(item) for item in raw["models"]]

    def list(self) -> list[ModelCatalogEntry]:
        return sorted(self._entries.values(), key=lambda item: item.model_id)

    def get(self, model_id_or_alias: str) -> ModelCatalogEntry:
        normalized = model_id_or_alias.lower()
        key = self._aliases.get(normalized, normalized)
        try:
            return self._entries[key]
        except KeyError as exc:
            available = ", ".join(entry.model_id for entry in self.list())
            raise KeyError(f"unknown model {model_id_or_alias!r}; available models: {available}") from exc

    def to_model_spec(
        self,
        model_id_or_alias: str,
        *,
        parameters: dict[str, Any] | None = None,
        feature_dim: int | None = None,
    ) -> ModelSpec:
        entry = self.get(model_id_or_alias)
        kwargs = _merge_dicts(entry.default_kwargs, parameters or {})
        if feature_dim is not None:
            kwargs = apply_feature_dimension(entry.model_id, kwargs, feature_dim)
        plugin = "numpy_linear_v1" if entry.backend == "builtin" else "qlib_model_v1"
        return ModelSpec(
            model_id=entry.model_id,
            plugin=plugin,
            backend=entry.backend,
            module_path=entry.module_path if entry.backend != "builtin" else None,
            class_name=entry.class_name if entry.backend != "builtin" else None,
            parameters=kwargs,
        )

    def availability(self, model_id_or_alias: str) -> tuple[bool, str]:
        entry = self.get(model_id_or_alias)
        if entry.backend == "builtin":
            return True, "built in"
        try:
            spec = importlib.util.find_spec(entry.module_path)
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if spec is None:
            return False, f"module is not installed: {entry.module_path}"
        return True, "module available"

    @classmethod
    def external_entry_points(cls) -> tuple[metadata.EntryPoint, ...]:
        points = metadata.entry_points()
        return tuple(points.select(group=cls.ENTRY_POINT_GROUP))

    def instantiate(self, model_id_or_alias: str, parameters: dict[str, Any] | None = None) -> Any:
        entry = self.get(model_id_or_alias)
        kwargs = _merge_dicts(entry.default_kwargs, parameters or {})
        module = importlib.import_module(entry.module_path)
        model_class = getattr(module, entry.class_name)
        return model_class(**kwargs)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in base.items():
        result[key] = _merge_dicts(value, {}) if isinstance(value, dict) else value
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def apply_feature_dimension(model_id: str, parameters: dict[str, Any], feature_dim: int) -> dict[str, Any]:
    """Apply structural dimensions without replacing public default hyperparameters."""

    result = _merge_dicts(parameters, {})
    if model_id == "qlib_mlp":
        pt_kwargs = dict(result.get("pt_model_kwargs", {}))
        pt_kwargs["input_dim"] = feature_dim
        result["pt_model_kwargs"] = pt_kwargs
    elif model_id in {"qlib_tabnet", "qlib_tft"}:
        result["d_feat"] = feature_dim
    elif model_id == "qlib_tcts":
        result["input_dim"] = feature_dim
        result["d_feat"] = feature_dim
    elif model_id == "qlib_tra":
        model_config = dict(result.get("model_config", {}))
        model_config["input_size"] = feature_dim
        result["model_config"] = model_config
    elif model_id in {
        "qlib_adarnn",
        "qlib_add",
        "qlib_alstm",
        "qlib_alstm_ts",
        "qlib_gats",
        "qlib_gats_ts",
        "qlib_gru",
        "qlib_gru_ts",
        "qlib_hist",
        "qlib_igmtf",
        "qlib_lstm",
        "qlib_lstm_ts",
        "qlib_sfm",
        "qlib_tcn",
        "qlib_tcn_ts",
        "qlib_transformer",
        "qlib_transformer_ts",
    }:
        result["d_feat"] = feature_dim
    elif model_id in {"qlib_krnn", "qlib_sandwich"}:
        result["fea_dim"] = feature_dim
    return result
