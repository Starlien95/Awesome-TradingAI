from __future__ import annotations

import sys
import types
from pathlib import Path

import yaml

from quant_bench.integrations.qlib.catalog import resolve_workflow
from quant_bench.integrations.qlib.workflow import run_workflow


def test_qlib_workflow_freezes_absolute_provider_uri(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_workflow(path: str, *, experiment_name: str, uri_folder: str) -> None:
        calls.append((path, experiment_name, uri_folder))

    qlib_module = types.ModuleType("qlib")
    qlib_module.__path__ = []  # type: ignore[attr-defined]
    cli_module = types.ModuleType("qlib.cli")
    cli_module.__path__ = []  # type: ignore[attr-defined]
    run_module = types.ModuleType("qlib.cli.run")
    run_module.workflow = fake_workflow  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qlib", qlib_module)
    monkeypatch.setitem(sys.modules, "qlib.cli", cli_module)
    monkeypatch.setitem(sys.modules, "qlib.cli.run", run_module)

    provider = tmp_path / "provider"
    provider.mkdir()
    workflow = resolve_workflow("158", "lgb", "1h")
    run_dir = run_workflow(workflow, tmp_path / "workspace", provider_uri=provider)

    resolved_path = run_dir / "resolved_workflow.yaml"
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    assert resolved["qlib_init"]["provider_uri"] == str(provider.resolve())
    assert calls == [(str(resolved_path), "quant-bench", str((tmp_path / "workspace" / "mlruns").resolve()))]
