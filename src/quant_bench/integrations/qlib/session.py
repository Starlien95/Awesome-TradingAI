"""Explicit Qlib initialization boundary."""

from __future__ import annotations

from types import TracebackType
from typing import Any


class QlibSession:
    """Initialize Qlib only after the caller explicitly enters the context."""

    def __init__(self, init_config: dict[str, Any]) -> None:
        self.init_config = dict(init_config)
        self.qlib: Any = None

    def __enter__(self) -> QlibSession:
        try:
            import qlib
        except ModuleNotFoundError as exc:
            raise RuntimeError('Qlib support requires: pip install "quant-bench[qlib]"') from exc
        qlib.init(**self.init_config)
        self.qlib = qlib
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.qlib = None
