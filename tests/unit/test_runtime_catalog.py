from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from quant_bench.runtime.processes import (
    PROCESSES,
    export_process_configs,
    inspect_process_configs,
    list_processes,
    start_process,
)


def test_runtime_catalog_covers_migrated_processes() -> None:
    rows = list_processes()
    assert len(rows) == 7
    strategy_ids = {strategy for row in rows for strategy in row["strategies"]}
    assert {
        "MLP_1h",
        "TCN_1h",
        "XGBoost_4h",
        "GATS_4h",
        "LSTM_4h",
        "TRA_4h",
        "DoubleEnsemble_15m",
        "TabNet_15m",
        "MacroHFTv1_5m",
        "MacroHFTv1_15m",
        "MacroHFTv1_1h",
        "MacroHFTv1_4h",
    } == strategy_ids


def test_exported_runtime_configs_are_safe_and_editable(tmp_path: Path) -> None:
    written = export_process_configs("traditional_1h", tmp_path)
    assert {path.name for path in written} == {"config_mlp.yaml", "config_tcn.yaml"}

    for path in written:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["api"]["is_simulated"] is True
        assert not config["api"]["api_key"]
        assert not config["api"]["secret_key"]
        assert not config["api"]["passphrase"]
        assert config["strategy"]["threshold"] == 0.0
        assert config["strategy"]["risk_degree"] == 0.95
        assert not Path(config["model"]["path"]).is_absolute()

    rows = inspect_process_configs("traditional_1h", tmp_path)
    assert all(row["valid"] for row in rows)
    assert all(not row["credentials_present"] for row in rows)
    assert all(not row["model_exists"] for row in rows)
    assert all(not row["ready"] for row in rows)


def test_runtime_start_requires_exact_confirmation_before_loading_backends(tmp_path: Path) -> None:
    process_id = next(iter(PROCESSES))
    with pytest.raises(PermissionError, match="DEMO_ORDERS"):
        start_process(process_id, tmp_path, tmp_path, mode="demo", confirm="")
    with pytest.raises(PermissionError, match="LIVE_ORDERS"):
        start_process(process_id, tmp_path, tmp_path, mode="live", confirm="DEMO_ORDERS")
