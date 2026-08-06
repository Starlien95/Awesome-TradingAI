from __future__ import annotations

from pathlib import Path

import yaml

from quant_bench.registry import ModelRegistry

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_catalog_contains_all_verified_models() -> None:
    registry = ModelRegistry()
    entries = registry.list()
    assert len(entries) == 34
    assert len({entry.model_id for entry in entries}) == 34
    assert len([entry for entry in entries if entry.backend in {"qlib", "custom_qlib"}]) == 33
    assert {entry.backend for entry in entries} == {"builtin", "qlib", "custom_qlib"}
    assert registry.get("lgb").model_id == "qlib_lightgbm"
    assert registry.get("TFTModel").model_id == "qlib_tft"
    assert registry.get("CustomADARNN").model_id == "qlib_adarnn"


def test_every_legacy_workflow_model_resolves() -> None:
    registry = ModelRegistry()
    failures: list[str] = []
    workflow_root = REPOSITORY_ROOT / "src" / "quant_bench" / "resources" / "qlib_workflows"
    for path in sorted(workflow_root.glob("**/*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = config["task"]["model"]
        try:
            registry.get(config.get("model_id") or model["class"])
        except KeyError as exc:
            failures.append(f"{path}: {exc}")
    assert failures == []


def test_default_config_can_be_generated_for_every_model() -> None:
    registry = ModelRegistry()
    for entry in registry.list():
        spec = registry.to_model_spec(entry.model_id, feature_dim=6)
        assert spec.model_id == entry.model_id
        if entry.backend != "builtin":
            assert spec.module_path == entry.module_path
            assert spec.class_name == entry.class_name
