"""Built-in adapters behind the unified Method interface."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from quant_bench.methods.interface import (
    AdapterCheck,
    AdapterOutcome,
    CheckIssue,
    ExecutionMode,
    MethodDescriptor,
    MethodRunSpec,
)


def _path_issue(path: Path | None, label: str) -> list[CheckIssue]:
    if path is None:
        return [CheckIssue(code=f"missing_{label}", message=f"provide --{label.replace('_', '-')}")]
    if not path.exists():
        return [CheckIssue(code=f"invalid_{label}", message=f"{label} not found: {path}")]
    return []


def _completed_outcome(
    returncode: int,
    *,
    command: list[str] | None = None,
    artifacts: dict[str, str] | None = None,
    details: dict[str, Any] | None = None,
) -> AdapterOutcome:
    return AdapterOutcome(
        status="completed" if returncode == 0 else "failed",
        returncode=returncode,
        command=command,
        artifacts=artifacts or {},
        details=details or {},
    )


class CanonicalExperimentAdapter:
    descriptor = MethodDescriptor(
        method_id="canonical",
        display_name="Canonical quant-bench experiment",
        family="predictive_model",
        adapter="quant_bench",
        description="Typed offline experiment using a recipe and ModelPlugin.",
        modes=[ExecutionMode.BACKTEST],
        frequencies=["5m", "15m", "1h", "4h", "1d"],
        resumable_modes=[],
        aliases=["experiment"],
    )

    def check(self, spec: MethodRunSpec) -> AdapterCheck:
        recipe = str(spec.parameters.get("recipe") or spec.config or "crypto_smoke_v1")
        try:
            from quant_bench.config.loader import load_config
            from quant_bench.registry import ModelRegistry

            config = load_config(recipe, list(spec.parameters.get("set", [])))
            available, detail = ModelRegistry().availability(config.model.model_id)
            if not available:
                issue = CheckIssue(code="model_unavailable", message=detail)
                return AdapterCheck(
                    valid=True,
                    ready=False,
                    issues=[issue],
                    details={"recipe": recipe, "model_id": config.model.model_id},
                )
        except Exception as exc:
            issue = CheckIssue(code="invalid_config", message=str(exc))
            return AdapterCheck(valid=False, ready=False, issues=[issue], details={"recipe": recipe})
        return AdapterCheck(details={"recipe": recipe})

    def run(self, spec: MethodRunSpec, run_dir: Path) -> AdapterOutcome:
        from quant_bench.config.loader import load_config
        from quant_bench.experiments import Experiment

        recipe = str(spec.parameters.get("recipe") or spec.config or "crypto_smoke_v1")
        config = load_config(recipe, list(spec.parameters.get("set", [])))
        config.experiment.workspace = run_dir / "native"
        result = Experiment(config).run()
        return _completed_outcome(
            0 if result.status == "completed" else 1,
            artifacts={
                "native_run_dir": str(result.run_dir),
                "run_manifest": str(result.manifest_path),
            },
            details={"metrics": result.metrics},
        )

    def status(self, spec: MethodRunSpec) -> dict[str, Any]:
        del spec
        return {"kind": "durable_run_record"}


class QlibWorkflowAdapter:
    descriptor = MethodDescriptor(
        method_id="qlib",
        display_name="Qlib workflow",
        family="qlib",
        adapter="pyqlib",
        description="Packaged or user-provided Qlib workflow executed as an offline backtest.",
        modes=[ExecutionMode.BACKTEST],
        frequencies=["5m", "15m", "1h", "4h", "1d"],
        aliases=["qlib_workflow"],
    )

    def check(self, spec: MethodRunSpec) -> AdapterCheck:
        issues = _path_issue(spec.config, "config")
        if importlib.util.find_spec("qlib") is None:
            issues.append(
                CheckIssue(
                    code="missing_dependency",
                    message='Qlib Method requires: pip install "quant-bench[qlib]"',
                )
            )
        if spec.config and spec.config.is_file():
            from quant_bench.integrations.qlib.workflow import validate_workflow

            issues.extend(
                CheckIssue(code="invalid_workflow", message=message)
                for message in validate_workflow(spec.config)
            )
        return AdapterCheck(
            valid=not issues,
            ready=not issues,
            issues=issues,
            command=["quant-bench", "qlib", "run", str(spec.config)] if spec.config else None,
        )

    def run(self, spec: MethodRunSpec, run_dir: Path) -> AdapterOutcome:
        from quant_bench.integrations.qlib.workflow import run_workflow

        assert spec.config is not None
        native = run_workflow(
            spec.config,
            run_dir / "native",
            str(spec.parameters.get("experiment_name", "quant-bench")),
            Path(spec.parameters["provider_uri"]).expanduser().resolve()
            if spec.parameters.get("provider_uri")
            else None,
        )
        return _completed_outcome(0, artifacts={"native_run_dir": str(native)})

    def status(self, spec: MethodRunSpec) -> dict[str, Any]:
        del spec
        return {"kind": "durable_run_record"}


class FinGPTNewsAdapter:
    descriptor = MethodDescriptor(
        method_id="fingpt_news",
        display_name="FinGPT news strategy",
        family="llm_sentiment",
        adapter="fingpt_news",
        description="FinGPT sentiment research backtest or local paper cycle.",
        modes=[ExecutionMode.BACKTEST, ExecutionMode.PAPER],
        frequencies=["1d"],
        default_frequency="1d",
        network_optional_modes=[ExecutionMode.PAPER],
        aliases=["fingpt"],
    )

    def check(self, spec: MethodRunSpec) -> AdapterCheck:
        issues = _path_issue(spec.config, "config")
        if spec.config and spec.config.is_file():
            try:
                payload = yaml.safe_load(spec.config.read_text(encoding="utf-8")) or {}
                if spec.mode == ExecutionMode.PAPER:
                    trade_mode = str(payload.get("trade", {}).get("mode", "paper_spot"))
                    if not trade_mode.startswith("paper"):
                        issues.append(
                            CheckIssue(
                                code="paper_config_required",
                                message=f"paper mode requires trade.mode=paper_*; found {trade_mode}",
                            )
                        )
            except (OSError, yaml.YAMLError) as exc:
                issues.append(CheckIssue(code="invalid_config", message=str(exc)))
        return AdapterCheck(valid=not issues, ready=not issues, issues=issues)

    def run(self, spec: MethodRunSpec, run_dir: Path) -> AdapterOutcome:
        assert spec.config is not None
        if spec.mode == ExecutionMode.BACKTEST:
            from quant_bench.methods.fingpt_news.research import run_research_pipeline

            result = run_research_pipeline(
                spec.config,
                run_dir / "native",
                force=bool(spec.parameters.get("force", False)),
            )
            return _completed_outcome(
                0,
                artifacts={"native_run_dir": str(run_dir / "native")},
                details={"result": result},
            )

        from quant_bench.methods.fingpt_news import run_fingpt_news

        run_fingpt_news.WORKSPACE_ROOT = run_dir / "native"
        runner = run_fingpt_news.FinGPTNewsPaperRunner(
            spec.config,
            dry_run=not spec.allow_network,
            allow_network=spec.allow_network,
            execute_orders=False,
        )
        runner.run_once(
            trade=True,
            fetch=True,
            force_trade=bool(spec.parameters.get("force_trade", True)),
            trade_date=spec.parameters.get("trade_date"),
        )
        return _completed_outcome(0, artifacts={"native_run_dir": str(runner.run_dir)})

    def status(self, spec: MethodRunSpec) -> dict[str, Any]:
        del spec
        return {"kind": "durable_run_record"}


class FinMemAdapter:
    descriptor = MethodDescriptor(
        method_id="finmem",
        display_name="FinMem / InvestorBench",
        family="memory_agent",
        adapter="finmem",
        description="InvestorBench historical run or one FinMem paper, simulated, or live cycle.",
        modes=[
            ExecutionMode.BACKTEST,
            ExecutionMode.PAPER,
            ExecutionMode.SIMULATED,
            ExecutionMode.LIVE,
        ],
        frequencies=["1d"],
        default_frequency="1d",
        network_required_modes=[
            ExecutionMode.BACKTEST,
            ExecutionMode.SIMULATED,
            ExecutionMode.LIVE,
        ],
        network_optional_modes=[ExecutionMode.PAPER],
        account_modes=[ExecutionMode.SIMULATED, ExecutionMode.LIVE],
        order_optional_modes=[ExecutionMode.SIMULATED, ExecutionMode.LIVE],
    )

    def check(self, spec: MethodRunSpec) -> AdapterCheck:
        issues = _path_issue(spec.config, "config")
        from quant_bench.methods.finmem.operations import doctor

        details = doctor(spec.workspace, spec.parameters.get("data_dir"))
        if spec.mode in {ExecutionMode.SIMULATED, ExecutionMode.LIVE}:
            credential_key = "okx_demo" if spec.mode == ExecutionMode.SIMULATED else "okx_live"
            if (spec.query_account or spec.execute_orders) and not details["credentials_present"][
                credential_key
            ]:
                issues.append(
                    CheckIssue(
                        code="missing_credentials",
                        message=f"FinMem {credential_key} credentials are not configured",
                    )
                )
        return AdapterCheck(valid=not issues, ready=not issues, issues=issues, details=details)

    def run(self, spec: MethodRunSpec, run_dir: Path) -> AdapterOutcome:
        assert spec.config is not None
        from quant_bench.methods.finmem.operations import run_investorbench, run_live_process

        if spec.mode == ExecutionMode.BACKTEST:
            command = str(spec.parameters.get("investor_command", "test"))
            returncode = run_investorbench(command, spec.config, allow_network=spec.allow_network)
            return _completed_outcome(returncode)

        mode = {
            ExecutionMode.PAPER: "paper",
            ExecutionMode.SIMULATED: "demo",
            ExecutionMode.LIVE: "live",
        }[spec.mode]
        confirmation = (
            "DEMO_ORDERS"
            if spec.mode == ExecutionMode.SIMULATED and spec.execute_orders
            else spec.confirm
        )
        returncode = run_live_process(
            str(spec.parameters.get("process", "cycle")),
            workspace=run_dir / "native",
            symbol=str(spec.parameters.get("symbol", "BTC")),
            mode=mode,
            investor_config=spec.config,
            allow_network=spec.allow_network,
            query_account=spec.query_account,
            execute_orders=spec.execute_orders,
            confirmation=confirmation,
            env_file=spec.parameters.get("env_file"),
        )
        return _completed_outcome(
            returncode,
            artifacts={"native_workspace": str(run_dir / "native")},
        )

    def status(self, spec: MethodRunSpec) -> dict[str, Any]:
        from quant_bench.methods.finmem.operations import doctor

        return doctor(spec.workspace, spec.parameters.get("data_dir"))


@dataclass
class RuntimeProcessAdapter:
    process_id: str
    frequency: str
    family: str

    @property
    def descriptor(self) -> MethodDescriptor:
        return MethodDescriptor(
            method_id=f"runtime:{self.process_id}",
            display_name=self.process_id.replace("_", " ").title(),
            family=self.family,
            adapter="quant_bench.runtime",
            description="Packaged runtime process using reviewed local model artifacts.",
            modes=[ExecutionMode.SIMULATED, ExecutionMode.LIVE],
            frequencies=[self.frequency],
            default_frequency=self.frequency,
            network_required_modes=[ExecutionMode.SIMULATED, ExecutionMode.LIVE],
            account_modes=[ExecutionMode.SIMULATED, ExecutionMode.LIVE],
            order_required_modes=[ExecutionMode.SIMULATED, ExecutionMode.LIVE],
            aliases=[self.process_id],
        )

    def check(self, spec: MethodRunSpec) -> AdapterCheck:
        issues = _path_issue(spec.config, "config")
        rows: list[dict[str, Any]] = []
        if importlib.util.find_spec("okx") is None:
            issues.append(
                CheckIssue(
                    code="missing_dependency",
                    message='runtime Method requires: pip install "quant-bench[runtime-okx]"',
                )
            )
        if spec.config and spec.config.is_dir():
            from quant_bench.runtime.processes import inspect_process_configs

            rows = inspect_process_configs(self.process_id, spec.config)
            expected_simulated = spec.mode == ExecutionMode.SIMULATED
            for row in rows:
                if not row.get("valid"):
                    issues.append(
                        CheckIssue(
                            code="invalid_runtime_config",
                            message=f"{row.get('strategy_id')}: {row.get('error', 'invalid config')}",
                        )
                    )
                elif row.get("is_simulated") != expected_simulated:
                    issues.append(
                        CheckIssue(
                            code="runtime_mode_mismatch",
                            message=(
                                f"{row['config']}: api.is_simulated must be "
                                f"{str(expected_simulated).lower()}"
                            ),
                        )
                    )
                elif not row.get("ready"):
                    issues.append(
                        CheckIssue(
                            code="runtime_not_ready",
                            message=f"{row['strategy_id']}: credentials or model artifact missing",
                        )
                    )
        return AdapterCheck(valid=not issues, ready=not issues, issues=issues, details={"configs": rows})

    def run(self, spec: MethodRunSpec, run_dir: Path) -> AdapterOutcome:
        assert spec.config is not None
        from quant_bench.runtime.processes import start_process

        mode = "demo" if spec.mode == ExecutionMode.SIMULATED else "live"
        confirmation = "DEMO_ORDERS" if spec.mode == ExecutionMode.SIMULATED else spec.confirm
        start_process(
            self.process_id,
            spec.config,
            run_dir / "native",
            mode=mode,
            confirm=confirmation,
        )
        return _completed_outcome(0, artifacts={"native_workspace": str(run_dir / "native")})

    def status(self, spec: MethodRunSpec) -> dict[str, Any]:
        if spec.config is None or not spec.config.is_dir():
            return {"available": False, "reason": "provide --config with the runtime config directory"}
        from quant_bench.runtime.processes import inspect_process_configs

        return {"available": True, "configs": inspect_process_configs(self.process_id, spec.config)}


@dataclass
class AiTradeAdapter:
    process_id: str
    project: str
    profile: str
    entrypoint: str
    frequency: str
    modes: tuple[ExecutionMode, ...]

    @property
    def descriptor(self) -> MethodDescriptor:
        aliases = [self.process_id]
        if self.project in {"finagent", "tradingagents"}:
            aliases.append(self.project)
        frequencies = ["1d", "4h"] if self.project == "finagent" else [self.frequency]
        frequencies_by_mode = (
            {
                ExecutionMode.BACKTEST.value: ["1d"],
                ExecutionMode.PAPER.value: ["4h"],
                ExecutionMode.SIMULATED.value: ["4h"],
                ExecutionMode.LIVE.value: ["4h"],
            }
            if self.project == "finagent"
            else {mode.value: [self.frequency] for mode in self.modes}
        )
        default_frequency_by_mode = {
            mode: values[0] for mode, values in frequencies_by_mode.items()
        }
        return MethodDescriptor(
            method_id=f"ai_trade:{self.process_id}",
            display_name=f"ai_trade {self.process_id}",
            family=f"multi_agent_{self.project}",
            adapter="external.ai_trade",
            description=f"External ai_trade {self.project} agent with native raw artifacts.",
            modes=list(self.modes),
            frequencies=frequencies,
            default_frequency=self.frequency,
            frequencies_by_mode=frequencies_by_mode,
            default_frequency_by_mode=default_frequency_by_mode,
            network_required_modes=list(self.modes),
            account_modes=[
                mode
                for mode in (ExecutionMode.SIMULATED, ExecutionMode.LIVE)
                if mode in self.modes
            ],
            order_required_modes=[
                mode
                for mode in (ExecutionMode.SIMULATED, ExecutionMode.LIVE)
                if mode in self.modes
            ],
            resumable_modes=[ExecutionMode.BACKTEST]
            if ExecutionMode.BACKTEST in self.modes
            else [],
            aliases=aliases,
            external=True,
        )

    @staticmethod
    def _configured_env(root: Path) -> dict[str, str]:
        values = {name: value for name, value in os.environ.items() if value}
        env_file = root / ".env"
        if not env_file.is_file():
            return values
        for raw_line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            name, separator, value = line.partition("=")
            if separator and name.strip() and value.strip():
                values.setdefault(name.strip(), value.strip().strip("'\""))
        return values

    def _live_names(self) -> tuple[str, str, str]:
        process_name = self.process_id.removesuffix("_live").upper()
        prefix = f"AI_TRADE_{process_name}_LIVE"
        return (f"{prefix}_API_KEY", f"{prefix}_SECRET_KEY", f"{prefix}_PASSPHRASE")

    @staticmethod
    def _python(root: Path) -> str:
        candidates = (
            root / ".venv-live-stack" / "bin" / "python",
            root / ".venv-live-stack" / "Scripts" / "python.exe",
        )
        return str(next((path for path in candidates if path.is_file()), Path(sys.executable)))

    def _default_config(self, root: Path) -> Path:
        return (
            root
            / "subprojects"
            / "finagent_dvampire"
            / "configs"
            / "exp"
            / "trading_mi_w_decision"
            / "BTCUSD_deepseek.py"
        )

    def _backtest_entrypoint(self) -> str:
        return {
            "benchmark": "main.py",
            "tradingagents": "tradingagents/scripts/run_btc_4h_backtest.py",
            "finagent": "subprojects/finagent_dvampire/tools/main_mi_w_decision.py",
        }[self.project]

    def _command(self, spec: MethodRunSpec, run_dir: Path) -> list[str]:
        assert spec.repo is not None
        root = spec.repo
        python = self._python(root)
        native = (
            spec.resume_from
            if spec.mode == ExecutionMode.BACKTEST and spec.resume and spec.resume_from
            else run_dir / "native"
        )
        symbol = str(spec.parameters.get("symbol", "BTC-USDT"))
        if spec.mode == ExecutionMode.BACKTEST:
            if self.project == "benchmark":
                command = [
                    python,
                    str(root / "main.py"),
                    "--mode",
                    "backtest",
                    "--coin_list",
                    symbol,
                    "--trade_mode",
                    str(spec.parameters.get("trade_mode", "spot")),
                    "--record_path",
                    str(native),
                    "--time_interval",
                    str(spec.parameters.get("time_interval", 240)),
                    "--backtest_initial_cash",
                    str(spec.initial_capital),
                    "--backtest_fee_rate",
                    str(spec.fee_rate),
                    "--backtest_resume",
                    str(spec.resume).lower(),
                    "--backtest_run_id",
                    str(
                        spec.parameters.get("backtest_run_id")
                        or (native.parent.name if spec.resume else run_dir.name)
                    ),
                    "--runtime_profile",
                    self.profile,
                ]
                if spec.start:
                    command.extend(["--backtest_start", spec.start])
                if spec.end:
                    command.extend(["--backtest_end", spec.end])
                database = spec.parameters.get("local_kline_db_path")
                if database:
                    command.extend(["--local_kline_db_path", str(database)])
                return command
            if self.project == "tradingagents":
                command = [
                    python,
                    str(root / "tradingagents" / "scripts" / "run_btc_4h_backtest.py"),
                    "--ticker",
                    symbol.replace("-", ""),
                    "--interval",
                    spec.frequency or self.frequency,
                    "--initial-capital",
                    str(spec.initial_capital),
                    "--fee-rate",
                    str(spec.fee_rate),
                    "--results-dir",
                    str(native),
                ]
                if spec.start:
                    command.extend(["--start-date", spec.start])
                if spec.end:
                    command.extend(["--end-date", spec.end])
                if spec.resume:
                    command.extend(["--resume-from-dir", str(native)])
                return command

            config = spec.config or self._default_config(root)
            native_tag = native.parent.name if spec.resume else run_dir.name
            overrides = [
                f"workdir={native.as_posix()}",
                f"tag={native_tag}",
                f"dataset.workdir={native.as_posix()}",
                f"dataset.tag={native_tag}",
                f"plots.workdir={native.as_posix()}",
                f"plots.tag={native_tag}",
                f"memory.workdir={native.as_posix()}",
                f"memory.tag={native_tag}",
                f"valid_environment.initial_amount={spec.initial_capital}",
                f"valid_environment.transaction_cost_pct={spec.fee_rate}",
            ]
            if spec.start:
                overrides.extend(
                    [
                        f"valid_start_date={spec.start}",
                        f"valid_environment.start_date={spec.start}",
                    ]
                )
            if spec.end:
                overrides.extend(
                    [
                        f"valid_end_date={spec.end}",
                        f"valid_environment.end_date={spec.end}",
                    ]
                )
            return [
                python,
                str(root / self._backtest_entrypoint()),
                "--config",
                str(config),
                "--root",
                str(root / "subprojects" / "finagent_dvampire"),
                "--cfg-options",
                *overrides,
            ]

        if self.project == "benchmark":
            return [
                python,
                str(root / self.entrypoint),
                "--runtime-profile",
                self.profile,
                "--coin",
                symbol,
                "--is-simulated",
                str(spec.mode == ExecutionMode.SIMULATED).lower(),
                "--record-root",
                str(native),
                "--run-id",
                run_dir.name,
                "--once",
                str(spec.once).lower(),
            ]
        if self.project == "tradingagents":
            return [
                python,
                str(root / self.entrypoint),
                "--ticker",
                symbol.replace("-", ""),
                "--mode",
                spec.mode.value,
                "--execute",
                str(spec.execute_orders).lower(),
                "--once",
                str(spec.once).lower(),
                "--results-dir",
                str(native),
                "--state-path",
                str(run_dir / "state.json"),
                "--data-cache-dir",
                str(run_dir / "cache"),
                "--memory-log-path",
                str(run_dir / "memory.md"),
            ]

        config = spec.config or self._default_config(root)
        command = [
            python,
            str(root / self.entrypoint),
            "--config",
            str(config),
            "--root",
            str(root / "subprojects" / "finagent_dvampire"),
            "--mode",
            spec.mode.value,
            "--inst-id",
            symbol,
            "--price-bar",
            spec.frequency or self.frequency.upper(),
            "--output-workdir",
            str(native),
            "--output-tag",
            run_dir.name,
            "--paper-initial-cash",
            str(spec.initial_capital),
            "--paper-fee-rate",
            str(spec.fee_rate),
        ]
        if spec.once:
            command.append("--once")
        if spec.mode == ExecutionMode.SIMULATED:
            command.extend(["--okx-api-key", "ZI3A", "--okx-secret-key", "ZI3S"])
        elif spec.mode == ExecutionMode.LIVE:
            api_key, secret_key, _ = self._live_names()
            command.extend(["--okx-api-key", api_key, "--okx-secret-key", secret_key])
        return command

    def _credential_issues(self, spec: MethodRunSpec) -> list[CheckIssue]:
        assert spec.repo is not None
        values = self._configured_env(spec.repo)
        issues: list[CheckIssue] = []
        llm_names = (
            ("DASHSCOPE_API_KEY", "DASHSCOPE_APIKEY", "DASHSCOPE_API_KEY_YU")
            if self.profile == "benchmark-qwen"
            else ("DEEPSEEK_API_KEY", "DEEPSEEK")
        )
        if not any(values.get(name) for name in llm_names):
            issues.append(
                CheckIssue(
                    code="missing_llm_credentials",
                    message=f"configure one of: {', '.join(llm_names)}",
                )
            )
        if spec.mode in {ExecutionMode.BACKTEST, ExecutionMode.PAPER}:
            return issues
        if spec.mode == ExecutionMode.LIVE:
            groups = ((name,) for name in self._live_names())
        else:
            from quant_bench.integrations.ai_trade.live_stack import AI_TRADE_PROCESSES

            process = next(item for item in AI_TRADE_PROCESSES if item.process_id == self.process_id)
            groups = process.credential_groups[1:]
        for group in groups:
            if not any(values.get(name) for name in group):
                issues.append(
                    CheckIssue(
                        code="missing_broker_credentials",
                        message=f"configure one of: {', '.join(group)}",
                    )
                )
        return issues

    def check(self, spec: MethodRunSpec) -> AdapterCheck:
        issues = _path_issue(spec.repo, "repo")
        if spec.repo and spec.repo.is_dir():
            entrypoint = spec.repo / (
                self._backtest_entrypoint()
                if spec.mode == ExecutionMode.BACKTEST
                else self.entrypoint
            )
            if not entrypoint.is_file():
                issues.append(
                    CheckIssue(code="missing_entrypoint", message=f"entrypoint not found: {entrypoint}")
                )
            if self.project == "finagent":
                config = spec.config or self._default_config(spec.repo)
                if not config.is_file():
                    issues.append(
                        CheckIssue(code="missing_config", message=f"FinAgent config not found: {config}")
                    )
                required_finagent_files = [
                    spec.repo
                    / "subprojects"
                    / "finagent_dvampire"
                    / "configs"
                    / "deepseek_config.json",
                    spec.repo
                    / "subprojects"
                    / "finagent_dvampire"
                    / "res"
                    / "prompts"
                    / "asset_infos"
                    / "exp_cryptos.json",
                ]
                if spec.mode == ExecutionMode.BACKTEST:
                    required_finagent_files.extend(
                        [
                            spec.repo
                            / "subprojects"
                            / "finagent_dvampire"
                            / "datasets"
                            / "custom_cryptos"
                            / kind
                            / "BTCUSD.parquet"
                            for kind in ("price", "news")
                        ]
                    )
                for required in required_finagent_files:
                    if not required.is_file():
                        issues.append(
                            CheckIssue(
                                code="missing_finagent_asset",
                                message=f"required FinAgent file not found: {required}",
                            )
                        )
            if spec.mode == ExecutionMode.BACKTEST and self.project in {
                "benchmark",
                "tradingagents",
            }:
                database = Path(
                    str(
                        spec.parameters.get("local_kline_db_path")
                        or spec.repo / "data" / "crypto_data.db"
                    )
                ).expanduser().resolve()
                if not database.is_file():
                    issues.append(
                        CheckIssue(
                            code="missing_kline_database",
                            message=f"historical kline database not found: {database}",
                        )
                    )
            issues.extend(self._credential_issues(spec))
        preview_dir = spec.workspace / "method_runs" / "preview"
        command = self._command(spec, preview_dir) if spec.repo and spec.repo.is_dir() else None
        return AdapterCheck(
            valid=not any(
                issue.code.startswith(("invalid_", "missing_repo", "missing_entrypoint"))
                for issue in issues
            ),
            ready=not issues,
            issues=issues,
            command=command,
            details={
                "repo": str(spec.repo) if spec.repo else None,
                "credentials_redacted": True,
                "native_artifacts_preserved": True,
            },
        )

    def _execution_env(self, spec: MethodRunSpec) -> dict[str, str]:
        assert spec.repo is not None
        values = self._configured_env(spec.repo)
        env = os.environ.copy()
        env.update(values)
        if self.profile == "benchmark-qwen":
            qwen_key = values.get("DASHSCOPE_API_KEY") or values.get("DASHSCOPE_API_KEY_YU")
            if qwen_key:
                env["DASHSCOPE_API_KEY"] = qwen_key
        else:
            deepseek_key = values.get("DEEPSEEK_API_KEY") or values.get("DEEPSEEK")
            if deepseek_key:
                env["DEEPSEEK_API_KEY"] = deepseek_key
        if spec.parameters.get("local_kline_db_path"):
            env["POLY_KLINE_DB_PATH"] = str(
                Path(str(spec.parameters["local_kline_db_path"])).expanduser().resolve()
            )
        if spec.mode != ExecutionMode.LIVE:
            return env
        api_key, secret_key, passphrase = self._live_names()
        if self.project == "benchmark":
            env["OKX_API_KEY1"] = values[api_key]
            env["OKX_SECRET_KEY1"] = values[secret_key]
            env["OKX_PASSPHRASE1"] = values[passphrase]
        elif self.project == "tradingagents":
            env["TRADINGAGENTS_OKX_API_KEY"] = values[api_key]
            env["TRADINGAGENTS_OKX_SECRET_KEY"] = values[secret_key]
            env["TRADINGAGENTS_OKX_PASSPHRASE"] = values[passphrase]
        else:
            env["OKX_PASSPHRASE"] = values[passphrase]
        return env

    def run(self, spec: MethodRunSpec, run_dir: Path) -> AdapterOutcome:
        assert spec.repo is not None
        command = self._command(spec, run_dir)
        native = (
            spec.resume_from
            if spec.mode == ExecutionMode.BACKTEST and spec.resume and spec.resume_from
            else run_dir / "native"
        )
        completed = subprocess.run(
            command,
            cwd=spec.repo,
            env=self._execution_env(spec),
            check=False,
        )
        return _completed_outcome(
            int(completed.returncode),
            command=command,
            artifacts={"native_run_dir": str(native)},
            details={"external_repo": str(spec.repo), "command": command},
        )

    def status(self, spec: MethodRunSpec) -> dict[str, Any]:
        if spec.repo is None or not spec.repo.is_dir():
            return {"available": False, "reason": "provide --repo"}
        from quant_bench.integrations.ai_trade.live_stack import (
            AI_TRADE_PROCESSES,
            _event_summary,
        )

        process = next(item for item in AI_TRADE_PROCESSES if item.process_id == self.process_id)
        return {
            "available": True,
            "latest_deployed_event": _event_summary(spec.repo / process.normalized_events),
        }


def builtin_adapters() -> list[Any]:
    """Build every adapter without importing optional heavy backends."""

    from quant_bench.runtime.processes import PROCESSES

    adapters: list[Any] = [
        CanonicalExperimentAdapter(),
        QlibWorkflowAdapter(),
        FinGPTNewsAdapter(),
        FinMemAdapter(),
    ]
    adapters.extend(
        RuntimeProcessAdapter(process_id, process.frequency, process.method_family)
        for process_id, process in PROCESSES.items()
    )
    adapters.extend(
        (
            AiTradeAdapter(
                "benchmark_deepseek",
                "benchmark",
                "benchmark-deepseek",
                "scripts/run_benchmark_4h_live.py",
                "4h",
                (ExecutionMode.BACKTEST, ExecutionMode.SIMULATED, ExecutionMode.LIVE),
            ),
            AiTradeAdapter(
                "benchmark_qwen",
                "benchmark",
                "benchmark-qwen",
                "scripts/run_benchmark_4h_live.py",
                "4h",
                (ExecutionMode.BACKTEST, ExecutionMode.SIMULATED, ExecutionMode.LIVE),
            ),
            AiTradeAdapter(
                "finagent_live",
                "finagent",
                "finagent-deepseek",
                "subprojects/finagent_dvampire/tools/live_mi_decision.py",
                "4h",
                (
                    ExecutionMode.BACKTEST,
                    ExecutionMode.PAPER,
                    ExecutionMode.SIMULATED,
                    ExecutionMode.LIVE,
                ),
            ),
            AiTradeAdapter(
                "tradingagents_live",
                "tradingagents",
                "tradingagents-deepseek",
                "tradingagents/scripts/run_btc_4h_paper_live.py",
                "4h",
                (ExecutionMode.BACKTEST, ExecutionMode.SIMULATED, ExecutionMode.LIVE),
            ),
        )
    )
    return adapters
