"""Unified Method interface for research and trading execution modes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, field_validator

from quant_bench.contracts.models import StrictModel

_EXECUTION_MODE_ALIASES = {
    "paper": "local-paper",
    "simulated": "exchange-paper",
}

_EXCHANGE_PAPER_CONFIRMATIONS = {"EXCHANGE_PAPER_ORDERS", "SIMULATED_ORDERS"}


class ExecutionMode(str, Enum):
    """User-visible execution modes shared by every Method adapter."""

    BACKTEST = "backtest"
    LOCAL_PAPER = "local-paper"
    EXCHANGE_PAPER = "exchange-paper"
    LIVE = "live"

    # Keep the original Python names for callers that already import them.
    PAPER = LOCAL_PAPER
    SIMULATED = EXCHANGE_PAPER

    @classmethod
    def _missing_(cls, value: object) -> ExecutionMode | None:
        if isinstance(value, str):
            canonical = _EXECUTION_MODE_ALIASES.get(value.strip().lower())
            if canonical is not None:
                return cls(canonical)
        return None

    @classmethod
    def cli_values(cls) -> tuple[str, ...]:
        """Return canonical CLI values followed by legacy input aliases."""

        return tuple(mode.value for mode in cls) + tuple(_EXECUTION_MODE_ALIASES)


class MethodDescriptor(StrictModel):
    """Static capability declaration for one selectable Method."""

    schema_version: Literal["1"] = "1"
    method_id: str
    display_name: str
    family: str
    adapter: str
    description: str
    modes: list[ExecutionMode]
    frequencies: list[str] = Field(default_factory=list)
    default_frequency: str | None = None
    frequencies_by_mode: dict[str, list[str]] = Field(default_factory=dict)
    default_frequency_by_mode: dict[str, str] = Field(default_factory=dict)
    network_required_modes: list[ExecutionMode] = Field(default_factory=list)
    network_optional_modes: list[ExecutionMode] = Field(default_factory=list)
    account_modes: list[ExecutionMode] = Field(default_factory=list)
    order_required_modes: list[ExecutionMode] = Field(default_factory=list)
    order_optional_modes: list[ExecutionMode] = Field(default_factory=list)
    resumable_modes: list[ExecutionMode] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    external: bool = False

    @field_validator("method_id")
    @classmethod
    def validate_method_id(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized or any(character.isspace() for character in normalized):
            raise ValueError("method_id must be a non-empty token without whitespace")
        return normalized


class MethodRunSpec(StrictModel):
    """Complete request for one Method execution."""

    schema_version: Literal["1"] = "1"
    method_id: str
    mode: ExecutionMode
    workspace: Path
    config: Path | None = None
    repo: Path | None = None
    frequency: str | None = None
    start: str | None = None
    end: str | None = None
    initial_capital: float = Field(default=100_000.0, gt=0)
    fee_rate: float = Field(default=1.5e-4, ge=0)
    resume: bool = False
    resume_from: Path | None = None
    allow_network: bool = False
    query_account: bool = False
    execute_orders: bool = False
    confirm: str = ""
    once: bool = False
    dry_run: bool = False
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("workspace", "config", "repo", "resume_from")
    @classmethod
    def expand_paths(cls, value: Path | None) -> Path | None:
        return value.expanduser().resolve() if value is not None else None

    @field_validator("parameters")
    @classmethod
    def reject_secret_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        sensitive_fragments = ("api_key", "secret", "passphrase", "password", "credential")
        rejected = [
            key
            for key in value
            if any(fragment in key.lower() for fragment in sensitive_fragments)
        ]
        if rejected:
            raise ValueError(
                "credentials must come from environment variables, not parameters: "
                + ", ".join(sorted(rejected))
            )
        return value


class CheckIssue(StrictModel):
    code: str
    message: str
    blocking: bool = True


class AdapterCheck(StrictModel):
    """Adapter-local readiness result before generic safety checks are added."""

    valid: bool = True
    ready: bool = True
    issues: list[CheckIssue] = Field(default_factory=list)
    command: list[str] | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class MethodCheckResult(StrictModel):
    method: MethodDescriptor
    spec: MethodRunSpec
    valid: bool
    ready: bool
    issues: list[CheckIssue]
    command: list[str] | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class AdapterOutcome(StrictModel):
    """Normalized return value from a concrete Method adapter."""

    status: str = "completed"
    returncode: int = 0
    command: list[str] | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class MethodRunResult(StrictModel):
    schema_version: Literal["1"] = "1"
    run_id: str
    method_id: str
    mode: ExecutionMode
    status: str
    run_dir: Path
    started_at: datetime
    completed_at: datetime | None = None
    returncode: int | None = None
    command: list[str] | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


class MethodAdapter(Protocol):
    """Seam implemented by every built-in or external Method."""

    descriptor: MethodDescriptor

    def check(self, spec: MethodRunSpec) -> AdapterCheck: ...

    def run(self, spec: MethodRunSpec, run_dir: Path) -> AdapterOutcome: ...

    def status(self, spec: MethodRunSpec) -> dict[str, Any]: ...


class MethodRegistry:
    """Registry and alias resolver for Method adapters."""

    def __init__(self, adapters: list[MethodAdapter] | None = None) -> None:
        self._adapters: dict[str, MethodAdapter] = {}
        self._aliases: dict[str, str] = {}
        for adapter in adapters or []:
            self.register(adapter)

    def register(self, adapter: MethodAdapter) -> None:
        descriptor = adapter.descriptor
        if descriptor.method_id in self._adapters:
            raise ValueError(f"duplicate method_id: {descriptor.method_id}")
        self._adapters[descriptor.method_id] = adapter
        for alias in descriptor.aliases:
            normalized = alias.strip().lower()
            if normalized in self._aliases or normalized in self._adapters:
                raise ValueError(f"duplicate Method alias: {alias}")
            self._aliases[normalized] = descriptor.method_id

    def get(self, method_id: str) -> MethodAdapter:
        normalized = method_id.strip().lower()
        canonical = self._aliases.get(normalized, normalized)
        try:
            return self._adapters[canonical]
        except KeyError as exc:
            choices = ", ".join(sorted(self._adapters))
            raise KeyError(f"unknown Method {method_id!r}; choose one of: {choices}") from exc

    def list(self, mode: ExecutionMode | None = None) -> list[MethodDescriptor]:
        descriptors = [adapter.descriptor for adapter in self._adapters.values()]
        if mode is not None:
            descriptors = [descriptor for descriptor in descriptors if mode in descriptor.modes]
        return sorted(descriptors, key=lambda item: item.method_id)


class MethodRunner:
    """Deep control-plane module for checking, running, and inspecting Methods."""

    def __init__(self, registry: MethodRegistry) -> None:
        self.registry = registry

    @staticmethod
    def _confirmation_token(mode: ExecutionMode) -> str | None:
        return {
            ExecutionMode.EXCHANGE_PAPER: "EXCHANGE_PAPER_ORDERS",
            ExecutionMode.LIVE: "LIVE_ORDERS",
        }.get(mode)

    def _generic_issues(
        self,
        descriptor: MethodDescriptor,
        spec: MethodRunSpec,
    ) -> list[CheckIssue]:
        issues: list[CheckIssue] = []
        if spec.mode not in descriptor.modes:
            supported = ", ".join(mode.value for mode in descriptor.modes)
            issues.append(
                CheckIssue(
                    code="unsupported_mode",
                    message=f"{descriptor.method_id} supports: {supported}",
                )
            )
            return issues
        allowed_frequencies = descriptor.frequencies_by_mode.get(
            spec.mode.value,
            descriptor.frequencies,
        )
        if spec.frequency and allowed_frequencies and spec.frequency not in allowed_frequencies:
            issues.append(
                CheckIssue(
                    code="unsupported_frequency",
                    message=(
                        f"{descriptor.method_id} frequency must be one of: "
                        f"{', '.join(allowed_frequencies)}"
                    ),
                )
            )
        if spec.mode in descriptor.network_required_modes and not spec.allow_network:
            issues.append(
                CheckIssue(
                    code="network_permission_required",
                    message=f"{spec.mode.value} requires --allow-network",
                )
            )
        if spec.query_account and spec.mode not in descriptor.account_modes:
            issues.append(
                CheckIssue(
                    code="account_access_unsupported",
                    message=f"{descriptor.method_id} does not expose account reads in {spec.mode.value}",
                )
            )
        if spec.query_account and not spec.allow_network:
            issues.append(
                CheckIssue(
                    code="account_network_required",
                    message="account reads require --allow-network",
                )
            )
        order_modes = descriptor.order_required_modes + descriptor.order_optional_modes
        if spec.execute_orders and spec.mode not in order_modes:
            issues.append(
                CheckIssue(
                    code="order_execution_unsupported",
                    message=f"{descriptor.method_id} cannot place orders in {spec.mode.value}",
                )
            )
        if spec.mode in descriptor.order_required_modes and not spec.execute_orders:
            issues.append(
                CheckIssue(
                    code="order_execution_required",
                    message=f"{descriptor.method_id} {spec.mode.value} requires --execute-orders",
                )
            )
        if spec.execute_orders:
            if not spec.allow_network:
                issues.append(
                    CheckIssue(
                        code="order_network_required",
                        message="order execution requires --allow-network",
                    )
                )
            expected = self._confirmation_token(spec.mode)
            confirmation_valid = (
                spec.confirm in _EXCHANGE_PAPER_CONFIRMATIONS
                if spec.mode is ExecutionMode.EXCHANGE_PAPER
                else spec.confirm == expected
            )
            if expected and not confirmation_valid:
                issues.append(
                    CheckIssue(
                        code="confirmation_required",
                        message=f"order execution requires --confirm {expected}",
                    )
                )
        if spec.resume and spec.mode not in descriptor.resumable_modes:
            issues.append(
                CheckIssue(
                    code="resume_unsupported",
                    message=f"{descriptor.method_id} does not support resume in {spec.mode.value}",
                )
            )
        if (
            spec.resume
            and spec.mode in descriptor.resumable_modes
            and spec.resume_from is None
        ):
            issues.append(
                CheckIssue(
                    code="resume_source_required",
                    message="resume requires --resume-from or a previous unified native run",
                )
            )
        return issues

    def _resolve_resume_source(
        self,
        descriptor: MethodDescriptor,
        spec: MethodRunSpec,
    ) -> MethodRunSpec:
        if not spec.resume or spec.resume_from is not None:
            return spec
        latest_path = self._method_root(spec, descriptor.method_id) / "latest.json"
        if not latest_path.is_file():
            return spec
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return spec
        native = (latest.get("artifacts") or {}).get("native_run_dir")
        if not native:
            return spec
        candidate = Path(native).expanduser().resolve()
        return spec.model_copy(update={"resume_from": candidate}) if candidate.exists() else spec

    def check(self, spec: MethodRunSpec) -> MethodCheckResult:
        adapter = self.registry.get(spec.method_id)
        canonical_spec = spec.model_copy(update={"method_id": adapter.descriptor.method_id})
        canonical_spec = self._resolve_resume_source(adapter.descriptor, canonical_spec)
        if canonical_spec.frequency is None:
            default_frequency = adapter.descriptor.default_frequency_by_mode.get(
                canonical_spec.mode.value,
                adapter.descriptor.default_frequency,
            )
            canonical_spec = canonical_spec.model_copy(update={"frequency": default_frequency})
        generic = self._generic_issues(adapter.descriptor, canonical_spec)
        if canonical_spec.mode not in adapter.descriptor.modes:
            local = AdapterCheck(valid=False, ready=False)
        else:
            local = adapter.check(canonical_spec)
        issues = [*generic, *local.issues]
        invalid_codes = {
            "account_access_unsupported",
            "order_execution_unsupported",
            "resume_unsupported",
            "unsupported_frequency",
            "unsupported_mode",
        }
        return MethodCheckResult(
            method=adapter.descriptor,
            spec=canonical_spec,
            valid=local.valid and not any(issue.code in invalid_codes for issue in generic),
            ready=local.ready and not any(issue.blocking for issue in issues),
            issues=issues,
            command=local.command,
            details=local.details,
        )

    @staticmethod
    def _method_root(spec: MethodRunSpec, method_id: str) -> Path:
        safe_id = method_id.replace(":", "__").replace("/", "_")
        return spec.workspace / "method_runs" / safe_id

    @staticmethod
    def _atomic_json(path: Path, payload: StrictModel | dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = payload.model_dump(mode="json") if isinstance(payload, StrictModel) else payload
        encoded = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _write_checksums(run_dir: Path) -> None:
        lines = []
        for name in ("method_run.json", "run_spec.json"):
            path = run_dir / name
            if path.is_file():
                lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}")
        target = run_dir / "checksums.sha256"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    def run(self, spec: MethodRunSpec) -> MethodRunResult:
        check = self.check(spec)
        if not check.ready:
            messages = "; ".join(issue.message for issue in check.issues if issue.blocking)
            raise RuntimeError(f"Method is not ready: {messages}")
        canonical_spec = check.spec
        started_at = datetime.now(timezone.utc)
        run_id = f"{started_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        run_dir = self._method_root(canonical_spec, check.method.method_id) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        self._atomic_json(run_dir / "run_spec.json", canonical_spec)
        result = MethodRunResult(
            run_id=run_id,
            method_id=check.method.method_id,
            mode=canonical_spec.mode,
            status="planned" if canonical_spec.dry_run else "running",
            run_dir=run_dir,
            started_at=started_at,
            command=check.command,
            details={"check": check.model_dump(mode="json")},
        )
        self._atomic_json(run_dir / "method_run.json", result)
        self._atomic_json(self._method_root(canonical_spec, check.method.method_id) / "latest.json", result)
        if canonical_spec.dry_run:
            self._write_checksums(run_dir)
            return result

        adapter = self.registry.get(check.method.method_id)
        try:
            outcome = adapter.run(canonical_spec, run_dir)
            result.status = outcome.status if outcome.returncode == 0 else "failed"
            result.returncode = outcome.returncode
            result.command = outcome.command or result.command
            result.artifacts = outcome.artifacts
            result.details = {**result.details, **outcome.details}
            result.completed_at = datetime.now(timezone.utc)
        except Exception as exc:
            result.status = "failed"
            result.completed_at = datetime.now(timezone.utc)
            result.error_type = type(exc).__name__
            result.error_message = str(exc)
            self._atomic_json(run_dir / "method_run.json", result)
            self._atomic_json(self._method_root(canonical_spec, check.method.method_id) / "latest.json", result)
            self._write_checksums(run_dir)
            raise
        self._atomic_json(run_dir / "method_run.json", result)
        self._atomic_json(self._method_root(canonical_spec, check.method.method_id) / "latest.json", result)
        self._write_checksums(run_dir)
        return result

    def status(self, spec: MethodRunSpec) -> dict[str, Any]:
        adapter = self.registry.get(spec.method_id)
        canonical_spec = spec.model_copy(update={"method_id": adapter.descriptor.method_id})
        latest_path = self._method_root(canonical_spec, adapter.descriptor.method_id) / "latest.json"
        latest = (
            json.loads(latest_path.read_text(encoding="utf-8"))
            if latest_path.is_file()
            else None
        )
        return {
            "method": adapter.descriptor.model_dump(mode="json"),
            "latest_run": latest,
            "adapter": adapter.status(canonical_spec),
        }
