from __future__ import annotations

from pathlib import Path

from quant_bench.config import load_config
from quant_bench.experiments import Experiment
from quant_bench.reporting import write_workspace_summary


def test_report_summarizes_completed_offline_run(tmp_path: Path) -> None:
    config = load_config("crypto_smoke_v1")
    config.experiment.workspace = tmp_path
    result = Experiment(config).run()

    report = write_workspace_summary(tmp_path)
    text = report.read_text(encoding="utf-8")
    assert result.run_id in text
    assert "Directly comparable: yes" in text
    assert "not evidence of paper, demo, or live trading performance" in text
