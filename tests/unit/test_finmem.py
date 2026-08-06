from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from quant_bench.methods.finmem.config import FinMemMode, FinMemRuntimeConfig
from quant_bench.methods.finmem.data import inspect_dataset
from quant_bench.methods.finmem.operations import map_action
from quant_bench.methods.finmem.paper import PaperLedger
from quant_bench.methods.finmem.workspace import initialize_workspace


def _write_dataset(path: Path, start: int = 1, end: int = 10) -> None:
    payload = {
        f"2024-01-{day:02d}": {"prices": float(100 + day), "news": []}
        for day in range(start, end + 1)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_finmem_public_import_and_safety_contract(tmp_path: Path) -> None:
    paper = FinMemRuntimeConfig(tmp_path, mode=FinMemMode.PAPER, allow_network=True)
    assert paper.symbol == "BTC"
    assert paper.data_inst_id == "BTC-USDT"

    with pytest.raises(ValueError, match="account reads"):
        FinMemRuntimeConfig(
            tmp_path,
            mode=FinMemMode.PAPER,
            allow_network=True,
            query_account=True,
        )
    with pytest.raises(PermissionError, match="DEMO_ORDERS"):
        FinMemRuntimeConfig(
            tmp_path,
            mode=FinMemMode.DEMO,
            allow_network=True,
            execute_orders=True,
        )
    live = FinMemRuntimeConfig(
        tmp_path,
        symbol="eth",
        mode=FinMemMode.LIVE,
        allow_network=True,
        query_account=True,
        execute_orders=True,
        confirmation="LIVE_ORDERS",
    )
    assert live.symbol == "ETH"


def test_inspect_dataset_checks_daily_contract(tmp_path: Path) -> None:
    path = tmp_path / "btc.json"
    _write_dataset(path)
    result = inspect_dataset(path)
    assert result["symbol"] == "BTC"
    assert result["days"] == 10
    assert result["start_date"] == "2024-01-01"
    assert len(result["sha256"]) == 64

    bad = tmp_path / "bad.json"
    bad.write_text('{"2024-01-01":{"prices":"unknown","news":[]}}', encoding="utf-8")
    with pytest.raises(ValueError, match="prices must be numeric"):
        inspect_dataset(bad)


def test_initialize_workspace_supports_multiple_symbols(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_dataset(data_dir / "btc.json", 1, 10)
    _write_dataset(data_dir / "eth.json", 3, 12)
    workspace = tmp_path / "workspace"

    result = initialize_workspace(workspace, data_dir, symbols=["BTC", "ETH"])
    config_path = workspace / "configs" / "investorbench.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert str(config_path) in result["written"]
    assert config["env_config"]["trading_symbols"] == ["BTC", "ETH"]
    assert config["portfolio_config"]["type"] == "multi-assets"
    assert config["env_config"]["warmup_start_time"] == "2024-01-03"
    assert config["env_config"]["test_end_time"] == "2024-01-10"
    assert "api_key" not in config["chat_config"]

    with pytest.raises(FileExistsError):
        initialize_workspace(workspace, data_dir, symbols=["BTC", "ETH"])


def test_paper_ledger_marks_previous_position_before_rebalance(tmp_path: Path) -> None:
    ledger = PaperLedger(tmp_path / "paper.json", initial_capital_usdt=1_000, fee_bps=5)
    opened = ledger.rebalance(1, 100)
    assert opened["changed"] is True
    assert opened["state"]["position_state"] == "long"

    held = ledger.rebalance(1, 110)
    assert held["changed"] is False
    assert held["state"]["equity_usdt"] == pytest.approx(1_099.45)

    reversed_position = ledger.rebalance(-1, 110)
    assert reversed_position["target_sign"] == -1
    assert reversed_position["fee_usdt"] == pytest.approx(1.09945)
    assert json.loads((tmp_path / "paper.json").read_text(encoding="utf-8"))["position_sign"] == -1


def test_map_action_supports_any_symbol_and_short() -> None:
    result = map_action(
        {"symbol": "LINK", "date": "2025-01-01", "position": -1},
        trade_mode="swap",
        notional_usdt=250,
        allow_short=True,
    )
    assert result["LINK"]["instId"] == "LINK-USDT-SWAP"
    assert result["LINK"]["signal"] == "sell"
    assert result["LINK"]["target_sign"] == -1


def test_finmem_logger_writes_canonical_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINMEM_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("FINMEM_SYMBOL", "LINK")
    from quant_bench.methods.finmem.live import dashboard_log

    logger = importlib.reload(dashboard_log)
    result = logger.append_cycle_record(
        {
            "cycle_start": "2025-01-01T00:05:00",
            "cycle_end": "2025-01-01T00:06:00",
            "symbol": "LINK",
            "status": "success",
            "live_json_info": {
                "live_start_date": "2024-12-22",
                "live_end_date": "2025-01-01",
                "latest_date": "2025-01-01",
                "latest_item": {"prices": 20.0, "news": []},
                "news_mode": "none",
            },
            "investor_action": {"symbol": "LINK", "date": "2025-01-01", "position": 1},
            "okx_decision": {
                "LINK": {
                    "signal": "buy",
                    "quantity": "1000",
                    "notional_usdt": "1000",
                    "target_sign": 1,
                }
            },
            "strategy_account_state": {
                "initial_capital_usdt": 1000.0,
                "cash_usdt": 0.0,
                "btc_qty": 50.0,
                "equity_usdt": 1000.0,
                "position_value_usdt": 1000.0,
                "position_sign": 1,
                "position_state": "long",
            },
            "execution_result": {"LINK": {"code": "0", "data": [{"sCode": "0"}]}},
        }
    )
    assert result["status"] == "appended"
    metrics = __import__("pandas").read_csv(result["metrics_path"])
    signals = __import__("pandas").read_csv(result["signals_path"])
    trades = __import__("pandas").read_csv(result["trades_path"])
    assert {
        "initial_capital_usdt",
        "cash_total_usdt",
        "total_equity_usdt",
        "method_family",
        "strategy_id",
        "frequency",
    }.issubset(metrics.columns)
    assert {"signal_label", "signal_score", "method_family"}.issubset(signals.columns)
    assert {"trade_id", "trade_date", "coin", "side", "success"}.issubset(trades.columns)
    assert (Path(result["metrics_path"]).parent / "run_manifest.json").is_file()
