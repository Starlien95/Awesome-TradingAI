"""Command helpers for the optional FinMem runtime."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from quant_bench.methods.finmem.config import FinMemMode, FinMemRuntimeConfig
from quant_bench.methods.finmem.paper import PaperLedger, target_sign
from quant_bench.trading import (
    MarketType,
    OrderAction,
    OrderRequest,
    PaperLedgerBackend,
    TradingService,
)

INVESTORBENCH_COMMANDS = {"warmup", "warmup-checkpoint", "test", "test-checkpoint", "eval"}
NETWORK_COMMANDS = INVESTORBENCH_COMMANDS - {"eval"}


def doctor(workspace: str | Path | None = None, data_dir: str | Path | None = None) -> dict[str, Any]:
    """Inspect optional dependencies and local paths without network access."""

    module_names = {
        "cvxpy": "cvxpy",
        "feedparser": "feedparser",
        "httpx": "httpx",
        "json_repair": "json_repair",
        "loguru": "loguru",
        "orjson": "orjson",
        "qdrant_client": "qdrant_client",
        "python_okx": "okx",
        "streamlit": "streamlit",
        "textblob": "textblob",
        "typer": "typer",
    }
    dependencies = {name: importlib.util.find_spec(module) is not None for name, module in module_names.items()}
    root = Path(workspace).expanduser().resolve() if workspace else None
    source = Path(data_dir).expanduser().resolve() if data_dir else None
    return {
        "network_checked": False,
        "account_checked": False,
        "workspace": str(root) if root else None,
        "workspace_exists": bool(root and root.is_dir()),
        "data_dir": str(source) if source else None,
        "data_dir_exists": bool(source and source.is_dir()),
        "dependencies": dependencies,
        "credentials_present": {
            "llm_or_embedding": bool(os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")),
            "okx_demo": all(
                os.getenv(name)
                for name in ("OKX_API_KEY_SIMU", "OKX_SECRET_KEY_SIMU", "OKX_PASSPHRASE")
            ),
            "okx_live": all(
                os.getenv(name)
                for name in ("OKX_API_KEY", "OKX_SECRET_KEY", "OKX_PASSPHRASE")
            ),
        },
    }


def run_investorbench(command: str, config: str | Path, *, allow_network: bool = False) -> int:
    """Run a vendored InvestorBench command with an explicit network boundary."""

    if command not in INVESTORBENCH_COMMANDS:
        raise ValueError(f"unsupported InvestorBench command: {command}")
    if command in NETWORK_COMMANDS and not allow_network:
        raise PermissionError(f"{command} uses LLM/embedding/Qdrant services; pass --allow-network")
    path = Path(config).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "quant_bench.methods.finmem.investorbench.run",
            command,
            "--config-path",
            str(path),
        ],
        cwd=path.parent,
        check=False,
    )
    return int(completed.returncode)


def map_action(
    action: dict[str, Any],
    *,
    trade_mode: str = "swap",
    notional_usdt: float = 10_000.0,
    allow_short: bool = True,
    leverage: int = 1,
) -> dict[str, Any]:
    from quant_bench.methods.finmem.live.decision import InvestorActionAdapter

    adapter = InvestorActionAdapter(
        trade_amount_usdt=notional_usdt,
        allow_short=allow_short,
        trade_mode=trade_mode,
        td_mode="cross" if trade_mode == "swap" else "cash",
        leverage=leverage,
    )
    return adapter.convert_single_action(action)


def paper_step(
    state_path: str | Path,
    *,
    price: float,
    target_position: float,
    initial_capital_usdt: float = 10_000.0,
    fee_bps: float = 0.0,
) -> dict[str, Any]:
    ledger = PaperLedger(
        state_path,
        initial_capital_usdt=initial_capital_usdt,
        fee_bps=fee_bps,
        allow_short=True,
    )
    current = int(ledger.state.get("position_sign", 0) or 0)
    desired = target_sign(target_position, allow_short=True)
    if desired == current:
        action = OrderAction.HOLD
    elif desired == 0:
        action = OrderAction.CLOSE
    elif desired > current:
        action = OrderAction.BUY
    else:
        action = OrderAction.SELL
    request = OrderRequest(
        strategy_id="finmem",
        instrument_id="BTC-USDT-SWAP",
        market_type=MarketType.SWAP,
        action=action,
        allow_backend_sizing=action in {OrderAction.BUY, OrderAction.SELL},
        reference_price=price,
        reason="FinMem target-position paper step",
        metadata={"target_position": target_position},
    )
    cycle = TradingService(PaperLedgerBackend(ledger)).execute(request)
    return cycle.execution.legacy_payload


def read_action(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("action JSON root must be an object")
    return payload


def run_live_process(
    process: str,
    *,
    workspace: str | Path,
    symbol: str,
    mode: str,
    investor_config: str | Path,
    allow_network: bool,
    query_account: bool = False,
    execute_orders: bool = False,
    confirmation: str = "",
    env_file: str | Path | None = None,
) -> int:
    """Launch one cycle or scheduler with the same validated safety contract."""

    if process not in {"cycle", "schedule"}:
        raise ValueError(f"unsupported FinMem live process: {process}")
    config = FinMemRuntimeConfig(
        Path(workspace),
        symbol=symbol,
        mode=FinMemMode(mode),
        allow_network=allow_network,
        query_account=query_account,
        execute_orders=execute_orders,
        confirmation=confirmation,
    )
    investor_path = Path(investor_config).expanduser().resolve()
    if not investor_path.is_file():
        raise FileNotFoundError(investor_path)
    env = os.environ.copy()
    env.update(
        {
            "FINMEM_WORKSPACE": str(config.method_root),
            "FINMEM_SYMBOL": config.symbol,
            "FINMEM_MODE": config.mode.value,
            "FINMEM_ALLOW_NETWORK": "1" if config.allow_network else "0",
            "FINMEM_QUERY_ACCOUNT": "1" if config.query_account else "0",
            "FINMEM_EXECUTE_ORDER": "1" if config.execute_orders else "0",
            "FINMEM_ORDER_CONFIRM": config.confirmation,
            "FINMEM_INVESTOR_CONFIG": str(investor_path),
        }
    )
    if env_file:
        path = Path(env_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        env["FINMEM_ENV_FILE"] = str(path)
    module = (
        "quant_bench.methods.finmem.live.cycle"
        if process == "cycle"
        else "quant_bench.methods.finmem.live.scheduler"
    )
    completed = subprocess.run([sys.executable, "-m", module], env=env, check=False)
    return int(completed.returncode)


def run_dashboard(workspace: str | Path, streamlit_args: list[str] | None = None) -> int:
    if importlib.util.find_spec("streamlit") is None:
        raise RuntimeError('FinMem dashboard requires: pip install "quant-bench[dashboard]"')
    app_path = Path(__file__).with_name("live") / "dashboard_app.py"
    env = os.environ.copy()
    env["FINMEM_WORKSPACE"] = str(Path(workspace).expanduser().resolve() / "methods" / "finmem")
    completed = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), *(streamlit_args or [])],
        env=env,
        check=False,
    )
    return int(completed.returncode)
