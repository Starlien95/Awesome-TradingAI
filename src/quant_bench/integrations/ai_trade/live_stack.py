"""Safe control-plane integration for the external ai_trade 4H agent stack.

The agent implementations, credentials, checkpoints, and trading records remain
in a separate ai_trade checkout.  This module only discovers that checkout,
checks its local contract, and delegates to its reviewed deploy scripts.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AiTradeProcess:
    process_id: str
    project: str
    credential_profile: str
    model_provider: str
    model_name: str
    timeframe: str
    environment: str
    entrypoint: str
    normalized_events: str
    credential_groups: tuple[tuple[str, ...], ...]


AI_TRADE_PROCESSES: tuple[AiTradeProcess, ...] = (
    AiTradeProcess(
        process_id="benchmark_deepseek",
        project="benchmark",
        credential_profile="zi1",
        model_provider="deepseek",
        model_name="deepseek-v4-pro",
        timeframe="4h",
        environment="okx_simulated",
        entrypoint="scripts/run_benchmark_4h_live.py",
        normalized_events=(
            "trading_record/live_stack/benchmark-deepseek-btc-4h/"
            "normalized_events/events.jsonl"
        ),
        credential_groups=(
            ("DEEPSEEK_API_KEY", "DEEPSEEK"),
            ("ZI1A",),
            ("ZI1S",),
            ("OKX_PASSPHRASE",),
        ),
    ),
    AiTradeProcess(
        process_id="benchmark_qwen",
        project="benchmark",
        credential_profile="zi2",
        model_provider="qwen",
        model_name="qwen3.6-max-preview",
        timeframe="4h",
        environment="okx_simulated",
        entrypoint="scripts/run_benchmark_4h_live.py",
        normalized_events=(
            "trading_record/live_stack/benchmark-qwen-btc-4h-zi2/"
            "normalized_events/events.jsonl"
        ),
        credential_groups=(
            ("DASHSCOPE_API_KEY", "DASHSCOPE_APIKEY"),
            ("ZI2A",),
            ("ZI2S",),
            ("OKX_PASSPHRASE",),
        ),
    ),
    AiTradeProcess(
        process_id="finagent_live",
        project="finagent",
        credential_profile="zi3",
        model_provider="deepseek",
        model_name="deepseek-v4-pro",
        timeframe="4h",
        environment="okx_simulated",
        entrypoint="subprojects/finagent_dvampire/tools/live_mi_decision.py",
        normalized_events=(
            "subprojects/finagent_dvampire/workdir/"
            "live_trading_mi_w_decision_deepseek_4h/"
            "BTCUSD_deepseek_okx_simu_4h/normalized_events/events.jsonl"
        ),
        credential_groups=(
            ("DEEPSEEK_API_KEY", "DEEPSEEK"),
            ("ZI3A",),
            ("ZI3S",),
            ("OKX_PASSPHRASE",),
        ),
    ),
    AiTradeProcess(
        process_id="tradingagents_live",
        project="tradingagents",
        credential_profile="tradingagents_explicit",
        model_provider="deepseek",
        model_name="deepseek-v4-pro",
        timeframe="4h",
        environment="okx_simulated",
        entrypoint="tradingagents/scripts/run_btc_4h_paper_live.py",
        normalized_events=(
            "logs/tradingagents_live_stack_main_simu/BTCUSDT/"
            "normalized_events/events.jsonl"
        ),
        credential_groups=(
            ("DEEPSEEK_API_KEY", "DEEPSEEK"),
            ("TRADINGAGENTS_OKX_API_KEY",),
            ("TRADINGAGENTS_OKX_SECRET_KEY",),
            ("TRADINGAGENTS_OKX_PASSPHRASE",),
        ),
    ),
)

_PROCESS_BY_ID = {process.process_id: process for process in AI_TRADE_PROCESSES}
_CONTROL_SCRIPTS = {
    "start": "deploy/start_live_stack.sh",
    "status": "deploy/status_live_stack.sh",
    "stop": "deploy/stop_live_stack.sh",
}
_REQUIRED_FILES = (
    "deploy/common.sh",
    "requirements.server.txt",
    "scripts/ensure_finagent_crypto_data.py",
    *_CONTROL_SCRIPTS.values(),
)
_STATUS_PATTERN = re.compile(
    r"^(?P<process>\S+)\s+(?P<state>RUNNING|STOPPED)"
    r"(?:\s+pid=(?P<pid>\d+))?\s+log=(?P<log>.+)$"
)


def _selected_processes(process_id: str | None) -> tuple[AiTradeProcess, ...]:
    if process_id is None:
        return AI_TRADE_PROCESSES
    try:
        return (_PROCESS_BY_ID[process_id],)
    except KeyError as exc:
        choices = ", ".join(sorted(_PROCESS_BY_ID))
        raise ValueError(f"unknown ai_trade process {process_id!r}; choose one of: {choices}") from exc


def resolve_repo_root(value: str | Path | None) -> Path:
    candidate = value or os.environ.get("AI_TRADE_ROOT")
    if not candidate:
        raise ValueError("provide --repo or set AI_TRADE_ROOT to the ai_trade checkout")
    root = Path(candidate).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ai_trade checkout not found: {root}")
    return root


def list_live_stack_processes(process_id: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for process in _selected_processes(process_id):
        row = asdict(process)
        row.pop("credential_groups")
        rows.append(row)
    return rows


def _configured_env_names(root: Path) -> set[str]:
    names = {name for name, value in os.environ.items() if value}
    env_path = root / ".env"
    if not env_path.is_file():
        return names
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if separator and name.strip() and value.strip():
            names.add(name.strip())
    return names


def _simulated_contract(root: Path) -> dict[str, bool]:
    start_path = root / _CONTROL_SCRIPTS["start"]
    profile_path = root / "src" / "runtime_profiles.py"
    start_text = (
        start_path.read_text(encoding="utf-8", errors="replace")
        if start_path.is_file()
        else ""
    )
    profile_text = (
        profile_path.read_text(encoding="utf-8", errors="replace")
        if profile_path.is_file()
        else ""
    )
    return {
        "agent_modes_simulated": start_text.count("--mode simulated") >= 2,
        "benchmark_profiles_simulated": profile_text.count('"is_simulated": True') >= 2,
    }


def inspect_live_stack(
    repo_root: str | Path | None,
    process_id: str | None = None,
) -> dict[str, Any]:
    root = resolve_repo_root(repo_root)
    configured_names = _configured_env_names(root)
    required_files = {
        relative: (root / relative).is_file()
        for relative in _REQUIRED_FILES
    }
    simulated_contract = _simulated_contract(root)
    process_rows: list[dict[str, Any]] = []
    for process in _selected_processes(process_id):
        missing_groups = [
            list(group)
            for group in process.credential_groups
            if not any(name in configured_names for name in group)
        ]
        entrypoint_exists = (root / process.entrypoint).is_file()
        process_rows.append(
            {
                **list_live_stack_processes(process.process_id)[0],
                "entrypoint_exists": entrypoint_exists,
                "credentials_present": not missing_groups,
                "missing_credential_groups": missing_groups,
                "normalized_events_exists": (root / process.normalized_events).is_file(),
            }
        )
    valid = all(required_files.values()) and all(
        row["entrypoint_exists"] for row in process_rows
    )
    ready = (
        valid
        and (root / ".env").is_file()
        and shutil.which("bash") is not None
        and all(simulated_contract.values())
        and all(row["credentials_present"] for row in process_rows)
    )
    return {
        "repo_root": str(root),
        "mode": "okx_simulated",
        "valid": valid,
        "ready": ready,
        "env_file_present": (root / ".env").is_file(),
        "bash_available": shutil.which("bash") is not None,
        "required_files": required_files,
        "simulated_contract": simulated_contract,
        "processes": process_rows,
    }


def _run_control_script(root: Path, action: str) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("bash is required to control the ai_trade live stack")
    script = root / _CONTROL_SCRIPTS[action]
    if not script.is_file():
        raise FileNotFoundError(f"ai_trade control script not found: {script}")
    command = [bash, str(script)]
    run_cwd: Path | None = root
    if os.name == "nt" and Path(bash).name.lower() == "bash.exe":
        drive = root.drive.rstrip(":").lower()
        if not drive:
            raise RuntimeError(f"cannot translate ai_trade path for WSL: {root}")
        relative = root.as_posix().split(":", 1)[1].lstrip("/")
        wsl_root = f"/mnt/{drive}/{relative}"
        command = [
            bash,
            "-lc",
            f"cd {shlex.quote(wsl_root)} && bash {shlex.quote(_CONTROL_SCRIPTS[action])}",
        ]
        run_cwd = None
    completed = subprocess.run(
        command,
        cwd=run_cwd,
        check=False,
        capture_output=True,
    )
    return subprocess.CompletedProcess(
        completed.args,
        completed.returncode,
        _decode_external_output(completed.stdout),
        _decode_external_output(completed.stderr),
    )


def _decode_external_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else value.decode("utf-8", errors="replace")
    text = text.replace("\x00", "")
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


def _last_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last_line = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                last_line = line
    if not last_line:
        return None
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _event_summary(path: Path) -> dict[str, Any] | None:
    event = _last_json_object(path)
    if event is None:
        return None
    signal = event.get("signal") or {}
    execution = event.get("execution_result") or {}
    portfolio = event.get("portfolio_after") or event.get("portfolio_before") or {}
    model = event.get("model") or {}
    return {
        "bar_open_time": event.get("bar_open_time"),
        "bar_close_time": event.get("bar_close_time"),
        "processed_at": event.get("processed_at"),
        "status": event.get("status"),
        "action": signal.get("action"),
        "rating": signal.get("rating"),
        "execution_status": execution.get("execution_status"),
        "submitted": execution.get("submitted"),
        "total_equity": portfolio.get("total_equity"),
        "model_provider": model.get("provider"),
        "model_name": model.get("model_name"),
    }


def status_live_stack(
    repo_root: str | Path | None,
    process_id: str | None = None,
) -> dict[str, Any]:
    root = resolve_repo_root(repo_root)
    completed = _run_control_script(root, "status")
    parsed: dict[str, dict[str, Any]] = {}
    for line in completed.stdout.splitlines():
        match = _STATUS_PATTERN.match(line.strip())
        if not match:
            continue
        process = match.group("process")
        parsed[process] = {
            "state": match.group("state").lower(),
            "pid": int(match.group("pid")) if match.group("pid") else None,
            "log": match.group("log"),
        }

    rows: list[dict[str, Any]] = []
    for process in _selected_processes(process_id):
        state = parsed.get(process.process_id, {"state": "unknown", "pid": None, "log": None})
        rows.append(
            {
                **list_live_stack_processes(process.process_id)[0],
                **state,
                "latest_event": _event_summary(root / process.normalized_events),
            }
        )
    healthy = completed.returncode == 0 and all(row["state"] == "running" for row in rows)
    return {
        "repo_root": str(root),
        "mode": "okx_simulated",
        "healthy": healthy,
        "returncode": completed.returncode,
        "processes": rows,
        "stderr": completed.stderr.strip() or None,
    }


def _control_live_stack(
    repo_root: str | Path | None,
    *,
    action: str,
    confirm: str,
) -> dict[str, Any]:
    root = resolve_repo_root(repo_root)
    expected = "SIMULATED_ORDERS" if action == "start" else "STOP_LIVE_STACK"
    if confirm != expected:
        raise PermissionError(f"refusing to {action} ai_trade stack; pass --confirm {expected}")
    if action == "start":
        inspection = inspect_live_stack(root)
        if not inspection["ready"]:
            raise RuntimeError(
                "ai_trade stack is not ready; run runtime ai-trade check and resolve all failures"
            )
    completed = _run_control_script(root, action)
    return {
        "repo_root": str(root),
        "action": action,
        "mode": "okx_simulated",
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip() or None,
        "stderr": completed.stderr.strip() or None,
    }


def start_live_stack(
    repo_root: str | Path | None,
    *,
    confirm: str,
) -> dict[str, Any]:
    return _control_live_stack(repo_root, action="start", confirm=confirm)


def stop_live_stack(
    repo_root: str | Path | None,
    *,
    confirm: str,
) -> dict[str, Any]:
    return _control_live_stack(repo_root, action="stop", confirm=confirm)
