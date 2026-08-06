from __future__ import annotations

import importlib.util
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _release_checker():
    path = REPOSITORY_ROOT / "tools" / "check_release.py"
    spec = importlib.util.spec_from_file_location("quant_bench_release_check", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_policy_has_no_open_errors() -> None:
    checker = _release_checker()
    assert checker.audit_repository(REPOSITORY_ROOT) == []
