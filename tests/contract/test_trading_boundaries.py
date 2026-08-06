from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src" / "quant_bench"
BACKEND_FILES = {
    Path("methods/fingpt_news/execution/okx_spot_broker.py"),
    Path("methods/finmem/live/exchange.py"),
    Path("runtime/core/execution.py"),
    Path("runtime/core/paper_broker.py"),
    Path("trading/adapters.py"),
}
PRIVATE_EXECUTION_CALLS = {
    "close_positions",
    "execute_decision",
    "place_order",
    "rebalance",
    "rebalance_coin",
}


def test_private_order_calls_are_confined_to_backends() -> None:
    violations: list[str] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        relative = path.relative_to(SOURCE_ROOT)
        if relative in BACKEND_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in PRIVATE_EXECUTION_CALLS
            ):
                violations.append(f"{relative}:{node.lineno}:{node.func.attr}")

    assert violations == []


def test_order_capable_entrypoints_use_the_common_trading_service() -> None:
    entrypoints = (
        Path("runtime/core/runner.py"),
        Path("methods/fingpt_news/run_fingpt_news.py"),
        Path("methods/finmem/operations.py"),
        Path("methods/finmem/live/cycle.py"),
    )
    for relative in entrypoints:
        source = (SOURCE_ROOT / relative).read_text(encoding="utf-8")
        assert "TradingService" in source, relative
