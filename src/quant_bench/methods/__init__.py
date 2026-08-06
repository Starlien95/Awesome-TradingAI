"""Built-in benchmark Method families and unified execution interface."""

from quant_bench.methods.adapters import builtin_adapters
from quant_bench.methods.interface import (
    ExecutionMode,
    MethodCheckResult,
    MethodDescriptor,
    MethodRegistry,
    MethodRunner,
    MethodRunResult,
    MethodRunSpec,
)


def get_method_registry() -> MethodRegistry:
    """Return the complete built-in Method registry."""

    return MethodRegistry(builtin_adapters())


def get_method_runner() -> MethodRunner:
    """Return the unified Method runner used by the CLI and Python callers."""

    return MethodRunner(get_method_registry())


__all__ = [
    "ExecutionMode",
    "MethodCheckResult",
    "MethodDescriptor",
    "MethodRegistry",
    "MethodRunResult",
    "MethodRunSpec",
    "MethodRunner",
    "get_method_registry",
    "get_method_runner",
]
