from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_bench.methods.interface import (
    AdapterCheck,
    AdapterOutcome,
    ExecutionMode,
    MethodDescriptor,
    MethodRegistry,
    MethodRunner,
    MethodRunSpec,
)


class FakeAdapter:
    descriptor = MethodDescriptor(
        method_id="fake",
        display_name="Fake",
        family="test",
        adapter="test",
        description="Interface test adapter.",
        modes=[ExecutionMode.BACKTEST, ExecutionMode.EXCHANGE_PAPER, ExecutionMode.LIVE],
        network_required_modes=[ExecutionMode.EXCHANGE_PAPER, ExecutionMode.LIVE],
        account_modes=[ExecutionMode.EXCHANGE_PAPER, ExecutionMode.LIVE],
        order_required_modes=[ExecutionMode.EXCHANGE_PAPER, ExecutionMode.LIVE],
        resumable_modes=[ExecutionMode.BACKTEST],
        aliases=["fake_alias"],
    )

    def __init__(self) -> None:
        self.check_calls = 0
        self.run_calls = 0

    def check(self, spec: MethodRunSpec) -> AdapterCheck:
        self.check_calls += 1
        return AdapterCheck(command=["fake", spec.mode.value])

    def run(self, spec: MethodRunSpec, run_dir: Path) -> AdapterOutcome:
        self.run_calls += 1
        native = run_dir / "native"
        native.mkdir()
        (native / "result.txt").write_text(spec.mode.value, encoding="utf-8")
        return AdapterOutcome(artifacts={"native_run_dir": str(native)})

    def status(self, spec: MethodRunSpec) -> dict[str, str]:
        return {"mode": spec.mode.value}


def _spec(tmp_path: Path, mode: ExecutionMode | str, **updates: object) -> MethodRunSpec:
    payload = {
        "method_id": "fake_alias",
        "mode": mode,
        "workspace": tmp_path,
        **updates,
    }
    return MethodRunSpec(**payload)


def test_registry_resolves_aliases_and_filters_modes() -> None:
    adapter = FakeAdapter()
    registry = MethodRegistry([adapter])
    assert registry.get("FAKE_ALIAS") is adapter
    assert [row.method_id for row in registry.list(ExecutionMode.BACKTEST)] == ["fake"]


def test_execution_mode_uses_canonical_values_and_accepts_legacy_aliases(
    tmp_path: Path,
) -> None:
    assert [mode.value for mode in ExecutionMode] == [
        "backtest",
        "local-paper",
        "exchange-paper",
        "live",
    ]
    assert ExecutionMode("paper") is ExecutionMode.LOCAL_PAPER
    assert ExecutionMode("simulated") is ExecutionMode.EXCHANGE_PAPER
    assert ExecutionMode.PAPER is ExecutionMode.LOCAL_PAPER
    assert ExecutionMode.SIMULATED is ExecutionMode.EXCHANGE_PAPER
    assert _spec(tmp_path, "paper").mode is ExecutionMode.LOCAL_PAPER
    assert _spec(tmp_path, "simulated").mode is ExecutionMode.EXCHANGE_PAPER


def test_run_spec_rejects_credentials_that_would_be_persisted(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="credentials must come from environment"):
        _spec(
            tmp_path,
            ExecutionMode.BACKTEST,
            parameters={"api_key": "must-not-be-recorded"},
        )


def test_exchange_paper_and_live_orders_use_distinct_exact_safety_tokens(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter()
    runner = MethodRunner(MethodRegistry([adapter]))

    missing = runner.check(_spec(tmp_path, ExecutionMode.EXCHANGE_PAPER))
    assert missing.valid is True
    assert missing.ready is False
    assert {issue.code for issue in missing.issues} == {
        "network_permission_required",
        "order_execution_required",
    }

    wrong = runner.check(
        _spec(
            tmp_path,
            ExecutionMode.EXCHANGE_PAPER,
            allow_network=True,
            execute_orders=True,
            confirm="LIVE_ORDERS",
        )
    )
    assert wrong.ready is False
    assert [issue.code for issue in wrong.issues] == ["confirmation_required"]

    ready = runner.check(
        _spec(
            tmp_path,
            ExecutionMode.EXCHANGE_PAPER,
            allow_network=True,
            execute_orders=True,
            confirm="EXCHANGE_PAPER_ORDERS",
        )
    )
    assert ready.ready is True

    legacy_ready = runner.check(
        _spec(
            tmp_path,
            ExecutionMode("simulated"),
            allow_network=True,
            execute_orders=True,
            confirm="SIMULATED_ORDERS",
        )
    )
    assert legacy_ready.ready is True


def test_run_writes_durable_spec_result_and_latest_record(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    runner = MethodRunner(MethodRegistry([adapter]))
    result = runner.run(_spec(tmp_path, ExecutionMode.BACKTEST))

    assert result.status == "completed"
    assert adapter.run_calls == 1
    assert (result.run_dir / "run_spec.json").is_file()
    assert (result.run_dir / "method_run.json").is_file()
    assert (result.run_dir / "checksums.sha256").is_file()
    latest = json.loads((tmp_path / "method_runs" / "fake" / "latest.json").read_text())
    assert latest["run_id"] == result.run_id
    assert latest["artifacts"]["native_run_dir"].endswith("native")


def test_dry_run_records_plan_without_calling_adapter(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    runner = MethodRunner(MethodRegistry([adapter]))
    result = runner.run(_spec(tmp_path, ExecutionMode.BACKTEST, dry_run=True))
    assert result.status == "planned"
    assert adapter.run_calls == 0


def test_unsupported_mode_stops_before_adapter_check(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    adapter.descriptor = adapter.descriptor.model_copy(
        update={"modes": [ExecutionMode.BACKTEST]}
    )
    runner = MethodRunner(MethodRegistry([adapter]))
    result = runner.check(_spec(tmp_path, ExecutionMode.LOCAL_PAPER))
    assert result.ready is False
    assert [issue.code for issue in result.issues] == ["unsupported_mode"]
    assert adapter.check_calls == 0


def test_resume_is_rejected_when_method_does_not_declare_it(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    adapter.descriptor = adapter.descriptor.model_copy(update={"resumable_modes": []})
    runner = MethodRunner(MethodRegistry([adapter]))
    result = runner.check(_spec(tmp_path, ExecutionMode.BACKTEST, resume=True))
    assert result.ready is False
    assert [issue.code for issue in result.issues] == ["resume_unsupported"]


def test_resume_uses_previous_native_run_or_requires_explicit_source(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    runner = MethodRunner(MethodRegistry([adapter]))
    missing = runner.check(_spec(tmp_path, ExecutionMode.BACKTEST, resume=True))
    assert [issue.code for issue in missing.issues] == ["resume_source_required"]

    first = runner.run(_spec(tmp_path, ExecutionMode.BACKTEST))
    resumed = runner.check(_spec(tmp_path, ExecutionMode.BACKTEST, resume=True))
    assert resumed.ready is True
    assert resumed.spec.resume_from == Path(first.artifacts["native_run_dir"])


def test_failed_adapter_is_persisted_before_exception_is_raised(tmp_path: Path) -> None:
    class BrokenAdapter(FakeAdapter):
        def run(self, spec: MethodRunSpec, run_dir: Path) -> AdapterOutcome:
            raise ValueError("broken")

    runner = MethodRunner(MethodRegistry([BrokenAdapter()]))
    with pytest.raises(ValueError, match="broken"):
        runner.run(_spec(tmp_path, ExecutionMode.BACKTEST))
    latest = json.loads((tmp_path / "method_runs" / "fake" / "latest.json").read_text())
    assert latest["status"] == "failed"
    assert latest["error_type"] == "ValueError"
