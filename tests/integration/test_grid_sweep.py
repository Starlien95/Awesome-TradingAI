from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_bench.tuning import parse_grid_parameter, run_grid_sweep


def test_parse_grid_parameter_accepts_json_scalars() -> None:
    path, values = parse_grid_parameter("model.parameters.ridge=0.0,1e-6,true")
    assert path == "model.parameters.ridge"
    assert values == [0.0, 1e-6, True]


def test_offline_grid_sweep_is_resumable(tmp_path: Path) -> None:
    results = run_grid_sweep(
        "crypto_smoke_v1",
        tmp_path,
        ["model.parameters.ridge=0.0,0.001"],
        study_name="ridge-smoke",
    )
    frame = pd.read_csv(results)
    assert len(frame) == 2
    assert set(frame["status"]) == {"completed"}
    assert frame["run_id"].nunique() == 2

    resumed = run_grid_sweep(
        "crypto_smoke_v1",
        tmp_path,
        ["model.parameters.ridge=0.0,0.001"],
        study_name="ridge-smoke",
        resume=True,
    )
    assert len(pd.read_csv(resumed)) == 2


def test_grid_sweep_preserves_small_float_parameter_types(tmp_path: Path) -> None:
    results = run_grid_sweep(
        "crypto_smoke_v1",
        tmp_path,
        ["model.parameters.ridge=0.000001,0.0001"],
        study_name="small-ridge-values",
    )
    frame = pd.read_csv(results)
    assert len(frame) == 2
    assert set(frame["status"]) == {"completed"}


def test_grid_sweep_refuses_accidental_explosion(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="above --max-trials"):
        run_grid_sweep(
            "crypto_smoke_v1",
            tmp_path,
            ["model.parameters.ridge=0,1,2"],
            study_name="too-large",
            max_trials=2,
        )
