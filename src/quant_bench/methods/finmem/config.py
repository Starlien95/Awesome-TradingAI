"""Typed runtime boundaries for the FinMem method family."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class FinMemMode(str, Enum):
    """Execution modes exposed by the FinMem command surface."""

    OFFLINE = "offline"
    PAPER = "paper"
    DEMO = "demo"
    LIVE = "live"


ORDER_CONFIRMATIONS = {
    FinMemMode.DEMO: "DEMO_ORDERS",
    FinMemMode.LIVE: "LIVE_ORDERS",
}


@dataclass(frozen=True)
class FinMemRuntimeConfig:
    """Validated paths and safety switches for one FinMem process.

    ``paper`` may use public market/news and LLM endpoints when
    ``allow_network`` is set.  Account reads and order writes are separate
    capabilities.  Merely constructing this object never accesses a network.
    """

    workspace: Path
    symbol: str = "BTC"
    mode: FinMemMode = FinMemMode.OFFLINE
    allow_network: bool = False
    query_account: bool = False
    execute_orders: bool = False
    confirmation: str = ""

    def __post_init__(self) -> None:
        workspace = Path(self.workspace).expanduser().resolve()
        symbol = self.symbol.upper().strip()
        if not symbol or not symbol.replace("_", "").isalnum():
            raise ValueError(f"invalid FinMem symbol: {self.symbol!r}")
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "symbol", symbol)

        if self.mode == FinMemMode.OFFLINE and self.allow_network:
            raise ValueError("offline mode cannot enable network access")
        if (self.query_account or self.execute_orders) and not self.allow_network:
            raise ValueError("account access and orders require --allow-network")
        if self.query_account and self.mode not in {FinMemMode.DEMO, FinMemMode.LIVE}:
            raise ValueError("account reads are available only in demo or live mode")
        if self.execute_orders:
            expected = ORDER_CONFIRMATIONS.get(self.mode)
            if expected is None:
                raise ValueError("orders are available only in demo or live mode")
            if self.confirmation != expected:
                raise PermissionError(f"order execution requires confirmation token {expected}")

    @property
    def method_root(self) -> Path:
        return self.workspace / "methods" / "finmem"

    @property
    def run_root(self) -> Path:
        return self.workspace / "runs" / "finmem" / self.symbol.lower()

    @property
    def state_root(self) -> Path:
        return self.method_root / "state" / self.symbol.lower()

    @property
    def data_inst_id(self) -> str:
        return f"{self.symbol}-USDT"

    @property
    def trade_inst_id(self) -> str:
        return f"{self.symbol}-USDT-SWAP"
