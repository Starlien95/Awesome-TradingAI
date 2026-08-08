from __future__ import annotations

import json
from pathlib import Path

from quant_bench.cli.app import main
from quant_bench.methods import ExecutionMode, MethodRunSpec, get_method_registry, get_method_runner


def _fake_ai_trade_repo(tmp_path: Path) -> Path:
    files = {
        "main.py": "",
        "scripts/run_benchmark_4h_live.py": "",
        "tradingagents/scripts/run_btc_4h_backtest.py": "",
        "tradingagents/scripts/run_btc_4h_paper_live.py": "",
        "subprojects/finagent_dvampire/tools/main_mi_w_decision.py": "",
        "subprojects/finagent_dvampire/tools/live_mi_decision.py": "",
        "subprojects/finagent_dvampire/configs/exp/trading_mi_w_decision/BTCUSD_deepseek.py": "",
        "subprojects/finagent_dvampire/configs/deepseek_config.json": "{}",
        "subprojects/finagent_dvampire/res/prompts/asset_infos/exp_cryptos.json": "{}",
        "subprojects/finagent_dvampire/datasets/custom_cryptos/price/BTCUSD.parquet": "",
        "subprojects/finagent_dvampire/datasets/custom_cryptos/news/BTCUSD.parquet": "",
        "data/crypto_data.db": "",
        ".env": "\n".join(
            (
                "DEEPSEEK_API_KEY=secret-deepseek",
                "DASHSCOPE_API_KEY_YU=secret-qwen",
                "ZI1A=secret-zi1-key",
                "ZI1S=secret-zi1-secret",
                "ZI2A=secret-zi2-key",
                "ZI2S=secret-zi2-secret",
                "ZI3A=secret-zi3-key",
                "ZI3S=secret-zi3-secret",
                "OKX_PASSPHRASE=secret-passphrase",
                "TRADINGAGENTS_OKX_API_KEY=secret-ta-key",
                "TRADINGAGENTS_OKX_SECRET_KEY=secret-ta-secret",
                "TRADINGAGENTS_OKX_PASSPHRASE=secret-ta-passphrase",
            )
        ),
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def test_catalog_unifies_all_existing_method_families() -> None:
    registry = get_method_registry()
    descriptors = registry.list()
    ids = {descriptor.method_id for descriptor in descriptors}
    assert {
        "canonical",
        "qlib",
        "fingpt_news",
        "finmem",
        "runtime:traditional_15m",
        "runtime:traditional_1h",
        "runtime:traditional_4h",
        "runtime:macrohft_5m",
        "runtime:macrohft_15m",
        "runtime:macrohft_1h",
        "runtime:macrohft_4h",
        "ai_trade:benchmark_deepseek",
        "ai_trade:benchmark_qwen",
        "ai_trade:finagent_live",
        "ai_trade:tradingagents_live",
    } == ids
    assert {descriptor.method_id for descriptor in registry.list(ExecutionMode.LIVE)} == {
        "finmem",
        "runtime:traditional_15m",
        "runtime:traditional_1h",
        "runtime:traditional_4h",
        "runtime:macrohft_5m",
        "runtime:macrohft_15m",
        "runtime:macrohft_1h",
        "runtime:macrohft_4h",
        "ai_trade:benchmark_deepseek",
        "ai_trade:benchmark_qwen",
        "ai_trade:finagent_live",
        "ai_trade:tradingagents_live",
    }
    finagent = registry.get("finagent").descriptor
    assert finagent.frequencies_by_mode["backtest"] == ["1d"]
    assert finagent.frequencies_by_mode["exchange-paper"] == ["4h"]


def test_ai_trade_backtest_accepts_existing_qwen_yu_key_without_leaking_value(
    tmp_path: Path,
) -> None:
    repo = _fake_ai_trade_repo(tmp_path / "repo")
    spec = MethodRunSpec(
        method_id="benchmark_qwen",
        mode=ExecutionMode.BACKTEST,
        workspace=tmp_path / "workspace",
        repo=repo,
        frequency="4h",
        start="2025-01-01",
        end="2025-07-01",
        allow_network=True,
    )
    result = get_method_runner().check(spec)
    payload = result.model_dump_json()
    assert result.ready is True
    assert "--runtime_profile" in result.command
    assert "secret-qwen" not in payload


def test_ai_trade_live_requires_dedicated_per_agent_credentials(tmp_path: Path) -> None:
    repo = _fake_ai_trade_repo(tmp_path / "repo")
    spec = MethodRunSpec(
        method_id="tradingagents",
        mode=ExecutionMode.LIVE,
        workspace=tmp_path / "workspace",
        repo=repo,
        frequency="4h",
        allow_network=True,
        execute_orders=True,
        confirm="LIVE_ORDERS",
        once=True,
    )
    missing = get_method_runner().check(spec)
    assert missing.ready is False
    messages = " ".join(issue.message for issue in missing.issues)
    assert "AI_TRADE_TRADINGAGENTS_LIVE_LIVE_API_KEY" not in messages
    assert "AI_TRADE_TRADINGAGENTS_LIVE_API_KEY" in messages

    with (repo / ".env").open("a", encoding="utf-8") as stream:
        stream.write(
            "\nAI_TRADE_TRADINGAGENTS_LIVE_API_KEY=dedicated-key"
            "\nAI_TRADE_TRADINGAGENTS_LIVE_SECRET_KEY=dedicated-secret"
            "\nAI_TRADE_TRADINGAGENTS_LIVE_PASSPHRASE=dedicated-pass\n"
        )
    ready = get_method_runner().check(spec)
    serialized = ready.model_dump_json()
    assert ready.ready is True
    assert "dedicated-key" not in serialized


def test_finagent_local_paper_needs_no_okx_key_and_exchange_paper_uses_zi3(
    tmp_path: Path,
) -> None:
    repo = _fake_ai_trade_repo(tmp_path / "repo")
    env_path = repo / ".env"
    env_path.write_text("DEEPSEEK_API_KEY=deepseek-only\n", encoding="utf-8")
    paper = MethodRunSpec(
        method_id="finagent",
        mode=ExecutionMode.LOCAL_PAPER,
        workspace=tmp_path / "workspace",
        repo=repo,
        allow_network=True,
        once=True,
    )
    paper_result = get_method_runner().check(paper)
    assert paper_result.ready is True
    assert paper_result.spec.frequency == "4h"
    paper_mode_index = paper_result.command.index("--mode")
    assert paper_result.command[paper_mode_index : paper_mode_index + 2] == [
        "--mode",
        "paper",
    ]

    env_path.write_text(
        "\n".join(
            (
                "DEEPSEEK_API_KEY=secret",
                "ZI3A=zi3-key",
                "ZI3S=zi3-secret",
                "OKX_PASSPHRASE=passphrase",
            )
        ),
        encoding="utf-8",
    )
    exchange_paper = paper.model_copy(
        update={
            "mode": ExecutionMode.EXCHANGE_PAPER,
            "execute_orders": True,
            "confirm": "EXCHANGE_PAPER_ORDERS",
        }
    )
    exchange_paper_result = get_method_runner().check(exchange_paper)
    assert exchange_paper_result.ready is True
    exchange_mode_index = exchange_paper_result.command.index("--mode")
    assert exchange_paper_result.command[exchange_mode_index : exchange_mode_index + 2] == [
        "--mode",
        "simulated",
    ]
    key_index = exchange_paper_result.command.index("--okx-api-key")
    assert exchange_paper_result.command[key_index : key_index + 2] == [
        "--okx-api-key",
        "ZI3A",
    ]


def test_finagent_backtest_defaults_to_its_real_daily_frequency(tmp_path: Path) -> None:
    repo = _fake_ai_trade_repo(tmp_path / "repo")
    spec = MethodRunSpec(
        method_id="finagent",
        mode=ExecutionMode.BACKTEST,
        workspace=tmp_path / "workspace",
        repo=repo,
        allow_network=True,
    )
    result = get_method_runner().check(spec)
    assert result.ready is True
    assert result.spec.frequency == "1d"
    rejected = get_method_runner().check(spec.model_copy(update={"frequency": "4h"}))
    assert rejected.ready is False
    assert "unsupported_frequency" in {issue.code for issue in rejected.issues}


def test_finagent_resume_reuses_previous_native_tag(tmp_path: Path) -> None:
    repo = _fake_ai_trade_repo(tmp_path / "repo")
    native = tmp_path / "previous-run-id" / "native"
    native.mkdir(parents=True)
    spec = MethodRunSpec(
        method_id="finagent",
        mode=ExecutionMode.BACKTEST,
        workspace=tmp_path / "workspace",
        repo=repo,
        allow_network=True,
        resume=True,
        resume_from=native,
    )
    result = get_method_runner().check(spec)
    assert result.ready is True
    assert "tag=previous-run-id" in result.command
    assert f"workdir={native.as_posix()}" in result.command


def test_cli_exposes_one_methods_command_surface(
    capsys,
    tmp_path: Path,
) -> None:
    assert main(["methods", "show", "finagent"]) == 0
    descriptor = json.loads(capsys.readouterr().out)
    assert descriptor["method_id"] == "ai_trade:finagent_live"
    assert set(descriptor["modes"]) == {
        "backtest",
        "local-paper",
        "exchange-paper",
        "live",
    }

    assert main(["methods", "list", "--mode", "simulated"]) == 0
    legacy_rows = json.loads(capsys.readouterr().out)
    assert legacy_rows
    assert all("exchange-paper" in row["modes"] for row in legacy_rows)

    assert (
        main(
            [
                "methods",
                "check",
                "canonical",
                "--mode",
                "live",
                "--workspace",
                str(tmp_path),
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert [issue["code"] for issue in result["issues"]] == ["unsupported_mode"]
