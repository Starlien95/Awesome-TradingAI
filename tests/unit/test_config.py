from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from quant_bench.config import config_sha256, load_config
from quant_bench.config.models import LabelSpec, TimeRange


def test_builtin_config_is_valid_and_hash_is_stable() -> None:
    first = load_config("crypto_smoke_v1")
    second = load_config("crypto_smoke_v1")
    assert first.recipe_id == "crypto_smoke_v1"
    assert config_sha256(first) == config_sha256(second)
    assert len(config_sha256(first)) == 64


def test_config_override_is_typed() -> None:
    config = load_config("crypto_smoke_v1", ["protocol.strategy.top_k=2", "experiment.seed=7"])
    assert config.protocol.strategy.top_k == 2
    assert config.experiment.seed == 7


def test_config_override_accepts_json_scientific_notation() -> None:
    config = load_config("crypto_smoke_v1", ["model.parameters.ridge=1e-06"])
    assert config.model.parameters["ridge"] == 1e-6
    assert isinstance(config.model.parameters["ridge"], float)


def test_unknown_override_is_rejected() -> None:
    with pytest.raises(KeyError, match="does not exist"):
        load_config("crypto_smoke_v1", ["model.unknown=1"])


@pytest.mark.parametrize(
    "formula",
    ["Ref($close, -1) / $close - 1)", "(Ref($close, -1) / $close - 1"],
)
def test_invalid_label_parentheses_are_rejected(formula: str) -> None:
    with pytest.raises(ValidationError):
        LabelSpec(
            label_id="invalid",
            task_type="regression",
            horizon_bars=1,
            formula=formula,
        )


def test_time_range_requires_order() -> None:
    value = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValidationError):
        TimeRange(start=value, end=value)
