from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
import yaml

from quant_bench.methods.fingpt_news import run_fingpt_news


def test_fingpt_fixture_dry_run_writes_dashboard_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def blocked_socket(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("FinGPT dry-run attempted network access")

    monkeypatch.setattr(socket, "socket", blocked_socket)
    run_fingpt_news.WORKSPACE_ROOT = tmp_path
    config = run_fingpt_news.CURRENT_DIR / "configs" / "sentiment_sft_live.yaml"
    runner = run_fingpt_news.FinGPTNewsPaperRunner(config, dry_run=True)
    runner.run_once(trade=True, fetch=True, force_trade=True)

    run_dir = tmp_path / "runs" / "fingpt_news" / "FinGPTSentimentSFT_1d"
    assert (run_dir / "run_manifest.json").is_file()
    assert (run_dir / "metrics" / "FinGPTSentimentSFT_1d_metrics.csv").is_file()
    assert (run_dir / "signals" / "FinGPTSentimentSFT_1d_signals.csv").is_file()
    assert (run_dir / "trades" / "FinGPTSentimentSFT_1d_trades.csv").is_file()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["csv_schema_version"] == "quant-bench.csv.v1"
    assert manifest["metrics_path"] == "metrics/FinGPTSentimentSFT_1d_metrics.csv"


def test_fingpt_capability_gates_fail_before_network_or_executor(tmp_path: Path) -> None:
    run_fingpt_news.WORKSPACE_ROOT = tmp_path
    source = run_fingpt_news.CURRENT_DIR / "configs" / "sentiment_sft_live.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["runtime"]["dry_run"] = False
    payload["runtime"]["run_dir"] = "paper-gate"
    payload["trade"]["mode"] = "paper_spot"
    config = tmp_path / "paper.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(PermissionError, match="allow-network"):
        run_fingpt_news.FinGPTNewsPaperRunner(config)

    runner = run_fingpt_news.FinGPTNewsPaperRunner(config, allow_network=True)
    assert runner.executor is None
    assert isinstance(runner.price_provider, run_fingpt_news.OKXPublicPriceProvider)

    payload["trade"]["mode"] = "okx_spot"
    payload["api"]["okx_api_key"] = "inline-is-forbidden"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="inline OKX credentials"):
        run_fingpt_news.FinGPTNewsPaperRunner(
            config,
            allow_network=True,
            execute_orders=True,
            confirmation="DEMO_ORDERS",
        )


def test_fingpt_dry_run_rejects_order_capability(tmp_path: Path) -> None:
    run_fingpt_news.WORKSPACE_ROOT = tmp_path
    config = run_fingpt_news.CURRENT_DIR / "configs" / "sentiment_sft_live.yaml"
    with pytest.raises(PermissionError, match="dry-run"):
        run_fingpt_news.FinGPTNewsPaperRunner(config, dry_run=True, execute_orders=True)
