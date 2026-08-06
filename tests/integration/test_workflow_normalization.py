from __future__ import annotations

from pathlib import Path

import yaml

from quant_bench.config.legacy_workflows import audit_workflows, dump_workflow, normalize_workflow

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = REPOSITORY_ROOT / "src" / "quant_bench" / "resources" / "qlib_workflows"


def test_all_workflows_are_normalized_and_idempotent() -> None:
    files = sorted(WORKFLOW_ROOT.glob("**/*.yaml"))
    assert len(files) == 202
    for path in files:
        text = path.read_text(encoding="utf-8")
        config = yaml.safe_load(text)
        assert dump_workflow(normalize_workflow(config, path)) == text
        handler_frequency = config["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["freq"]
        portfolio_record = config["task"]["record"][2]
        assert portfolio_record["class"] == "CryptoPortAnaRecord"
        assert portfolio_record["module_path"] == "quant_bench.integrations.qlib.records"
        portfolio_config = portfolio_record["kwargs"]["config"]
        executor = portfolio_config["executor"]
        assert executor["class"] == "SimulatorExecutor"
        assert executor["kwargs"]["time_per_step"] == handler_frequency
        assert executor["kwargs"]["generate_portfolio_metrics"] is True
        assert portfolio_config["backtest"]["benchmark"] == "BTC-USDT"


def test_workflow_audit_has_no_release_blocking_findings() -> None:
    audit = audit_workflows(WORKFLOW_ROOT, require_normalized=True)
    assert audit.files == 202
    assert audit.as_dict() == {
        "files": 202,
        "passed": True,
        "yaml_errors": [],
        "unsupported_models": [],
        "absolute_paths": [],
        "stock_defaults": [],
        "invalid_labels": [],
        "non_normalized": [],
    }
