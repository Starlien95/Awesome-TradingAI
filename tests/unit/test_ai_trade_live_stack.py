from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from quant_bench.integrations.ai_trade import live_stack


def _fake_repo(tmp_path: Path) -> Path:
    files = {
        "deploy/common.sh": "",
        "deploy/start_live_stack.sh": "--mode simulated\n--mode simulated\n",
        "deploy/status_live_stack.sh": "",
        "deploy/stop_live_stack.sh": "",
        "requirements.server.txt": "",
        "scripts/ensure_finagent_crypto_data.py": "",
        "scripts/run_benchmark_4h_live.py": "",
        "src/runtime_profiles.py": '"is_simulated": True\n"is_simulated": True\n',
        "subprojects/finagent_dvampire/tools/live_mi_decision.py": "",
        "tradingagents/scripts/run_btc_4h_paper_live.py": "",
        ".env": "\n".join(
            (
                "DEEPSEEK_API_KEY=placeholder",
                "DASHSCOPE_API_KEY=placeholder",
                "ZI1A=placeholder",
                "ZI1S=placeholder",
                "ZI2A=placeholder",
                "ZI2S=placeholder",
                "ZI3A=placeholder",
                "ZI3S=placeholder",
                "OKX_PASSPHRASE=placeholder",
                "TRADINGAGENTS_OKX_API_KEY=placeholder",
                "TRADINGAGENTS_OKX_SECRET_KEY=placeholder",
                "TRADINGAGENTS_OKX_PASSPHRASE=placeholder",
            )
        ),
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def test_catalog_exposes_four_simulated_4h_children() -> None:
    rows = live_stack.list_live_stack_processes()
    assert [row["process_id"] for row in rows] == [
        "benchmark_deepseek",
        "benchmark_qwen",
        "finagent_live",
        "tradingagents_live",
    ]
    assert {row["timeframe"] for row in rows} == {"4h"}
    assert {row["environment"] for row in rows} == {"okx_simulated"}
    assert [row["credential_profile"] for row in rows] == [
        "zi1",
        "zi2",
        "zi3",
        "tradingagents_explicit",
    ]
    assert all("credential_groups" not in row for row in rows)


def test_check_is_offline_and_reports_only_credential_presence(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path)
    result = live_stack.inspect_live_stack(root)
    assert result["valid"] is True
    assert result["ready"] is True
    assert all(row["credentials_present"] for row in result["processes"])
    assert all(row["missing_credential_groups"] == [] for row in result["processes"])
    assert "placeholder" not in json.dumps(result)


def test_check_rejects_a_changed_live_mode_contract(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path)
    (root / "deploy/start_live_stack.sh").write_text("--mode live\n", encoding="utf-8")
    result = live_stack.inspect_live_stack(root)
    assert result["ready"] is False
    assert result["simulated_contract"]["agent_modes_simulated"] is False


def test_status_filters_one_child_and_summarizes_latest_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fake_repo(tmp_path)
    process = live_stack.AI_TRADE_PROCESSES[0]
    events = root / process.normalized_events
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(
        json.dumps(
            {
                "bar_open_time": "2026-07-22 20:00:00",
                "status": "ok",
                "signal": {"action": "hold"},
                "execution_result": {"execution_status": "skipped", "submitted": False},
                "portfolio_after": {"total_equity": 100_123.0},
                "model": {"provider": "deepseek", "model_name": "deepseek-v4-pro"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = "\n".join(
        (
            "benchmark_deepseek RUNNING pid=42 log=/srv/ai_trade/logs/live_stack/benchmark_deepseek.log",
            "benchmark_qwen STOPPED log=/srv/ai_trade/logs/live_stack/benchmark_qwen.log",
        )
    )
    monkeypatch.setattr(
        live_stack,
        "_run_control_script",
        lambda _root, _action: subprocess.CompletedProcess([], 0, output, ""),
    )

    result = live_stack.status_live_stack(root, "benchmark_deepseek")
    assert result["healthy"] is True
    assert result["processes"][0]["pid"] == 42
    assert result["processes"][0]["latest_event"]["total_equity"] == 100_123.0


def test_control_requires_exact_tokens_before_running_scripts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fake_repo(tmp_path)
    invoked = False

    def fake_run(_root: Path, _action: str) -> subprocess.CompletedProcess[str]:
        nonlocal invoked
        invoked = True
        return subprocess.CompletedProcess([], 0, "ok", "")

    monkeypatch.setattr(live_stack, "_run_control_script", fake_run)
    with pytest.raises(PermissionError, match="SIMULATED_ORDERS"):
        live_stack.start_live_stack(root, confirm="")
    with pytest.raises(PermissionError, match="STOP_LIVE_STACK"):
        live_stack.stop_live_stack(root, confirm="")
    assert invoked is False

    started = live_stack.start_live_stack(root, confirm="SIMULATED_ORDERS")
    assert started["returncode"] == 0
    assert invoked is True


def test_external_output_is_terminal_safe() -> None:
    decoded = live_stack._decode_external_output("warning \ufffd 中文\x00")
    assert "\ufffd" not in decoded
    assert "\x00" not in decoded
    decoded.encode("ascii")


def test_cli_lists_ai_trade_children(capsys: pytest.CaptureFixture[str]) -> None:
    from quant_bench.cli.app import main

    assert main(["runtime", "ai-trade", "list", "--process", "benchmark_qwen"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["process_id"] for row in payload] == ["benchmark_qwen"]
