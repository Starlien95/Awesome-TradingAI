from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from platformdirs import user_data_path

from quant_bench.contracts.csv import (
    RUNTIME_METRICS_V1,
    RUNTIME_SIGNALS_V1,
    RUNTIME_TRADES_V1,
    RUNTIME_VOLUME_V1,
)
from quant_bench.methods.fingpt_news.execution.okx_spot_broker import PerCoinOKXSpotBroker
from quant_bench.methods.fingpt_news.inference.label_logprob import LabelLogprobSentimentService
from quant_bench.methods.fingpt_news.news.selector import build_candidates, write_candidates
from quant_bench.methods.fingpt_news.news.store import NewsStore
from quant_bench.methods.fingpt_news.strategy.daily_signal import (
    append_daily_outputs,
    build_daily_signals,
    load_per_coin_params,
)
from quant_bench.runtime.core.atomic_io import atomic_write_csv, safe_read_csv
from quant_bench.runtime.core.audit_log import append_csv_rows, append_jsonl_events
from quant_bench.runtime.core.execution import TradeExecutor
from quant_bench.runtime.core.paper_broker import PerCoinPaperBroker
from quant_bench.runtime.core.run_manifest import write_run_manifest
from quant_bench.runtime.core.state_store import JsonStateStore
from quant_bench.trading import (
    MarketType,
    OrderAction,
    OrderRequest,
    RebalanceBrokerBackend,
    TradingService,
)

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = user_data_path("quant-bench", appauthor=False)
logger = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
EMPTY_CSV_CONTRACTS = {
    "metrics": list(RUNTIME_METRICS_V1.required_columns),
    "signals": list(RUNTIME_SIGNALS_V1.required_columns),
    "trades": list(RUNTIME_TRADES_V1.required_columns),
    "volume": list(RUNTIME_VOLUME_V1.required_columns),
}


def local_midnight_timestamp(value: str, tz: ZoneInfo) -> float:
    day = pd.Timestamp(value).date()
    return datetime.combine(day, dt_time.min, tzinfo=tz).timestamp()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def scheduler_timezone(cfg: dict[str, Any]) -> ZoneInfo:
    return ZoneInfo(str(cfg.get("scheduler", {}).get("timezone", "Asia/Shanghai")))


def parse_local_time(value: str) -> dt_time:
    hour, minute = str(value).strip().split(":", 1)
    return dt_time(hour=int(hour), minute=int(minute))


def now_local_date(tz: ZoneInfo) -> str:
    return datetime.now(tz).strftime("%Y-%m-%d")


def trade_schedule_bounds(
    trade_date: str,
    cfg: dict[str, Any],
    tz: ZoneInfo,
) -> tuple[datetime, datetime]:
    trade_day = pd.Timestamp(trade_date).date()
    trade_time = parse_local_time(cfg.get("scheduler", {}).get("trade_time_local", "00:05"))
    cutoff_time = parse_local_time(cfg.get("scheduler", {}).get("trade_catchup_cutoff_local", "01:00"))
    target = datetime.combine(trade_day, trade_time, tzinfo=tz)
    cutoff = datetime.combine(trade_day, cutoff_time, tzinfo=tz)
    if cutoff < target:
        cutoff += timedelta(days=1)
    return target, cutoff


def news_window_for_trade(
    trade_date: str,
    cfg: dict[str, Any],
    tz: ZoneInfo,
) -> tuple[str, str, str]:
    trade_cfg = cfg.get("trade", {})
    mode = str(trade_cfg.get("signal_news_window", "previous_calendar_day"))
    trade_day = pd.Timestamp(trade_date).date()
    trade_time = parse_local_time(cfg.get("scheduler", {}).get("trade_time_local", "00:05"))
    trade_dt = datetime.combine(trade_day, trade_time, tzinfo=tz)
    if mode == "same_calendar_day_until_trade_time":
        start_dt = datetime.combine(trade_day, dt_time(0, 0), tzinfo=tz)
        end_dt = trade_dt
    elif mode == "lookback_hours":
        hours = float(trade_cfg.get("signal_lookback_hours", 24.0) or 24.0)
        end_dt = trade_dt
        start_dt = end_dt - timedelta(hours=hours)
    else:
        start_day = trade_day - timedelta(days=1)
        start_dt = datetime.combine(start_day, dt_time(0, 0), tzinfo=tz)
        end_dt = datetime.combine(trade_day, dt_time(0, 0), tzinfo=tz)
    news_date = start_dt.strftime("%Y-%m-%d")
    return (
        news_date,
        start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        end_dt.strftime("%Y-%m-%d %H:%M:%S"),
    )


def local_epoch_seconds(value: str, tz: ZoneInfo) -> int:
    dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
    return int(dt.timestamp())


def resolve_project_path(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((WORKSPACE_ROOT / path).resolve())


def configure_run_file_logging(log_path: str | Path, level_name: str = "INFO") -> None:
    target = Path(log_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    root.setLevel(min(root.level or level, level))
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == target:
            return
    file_handler = logging.FileHandler(target, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(file_handler)


def local_datetime_from_epoch(epoch: float | int | None, tz: ZoneInfo) -> str:
    if not epoch:
        return "never"
    return datetime.fromtimestamp(float(epoch), tz).strftime("%Y-%m-%d %H:%M:%S")


def time_in_local_window(now: datetime, start: dt_time, end: dt_time) -> bool:
    current = now.time().replace(microsecond=0)
    if start <= end:
        return start <= current < end
    return current >= start or current < end


def local_window_bounds(now: datetime, start: dt_time, end: dt_time) -> tuple[datetime, datetime]:
    today = now.date()
    if start <= end:
        return datetime.combine(today, start, tzinfo=now.tzinfo), datetime.combine(today, end, tzinfo=now.tzinfo)
    if now.time().replace(microsecond=0) >= start:
        return (
            datetime.combine(today, start, tzinfo=now.tzinfo),
            datetime.combine(today + timedelta(days=1), end, tzinfo=now.tzinfo),
        )
    return (
        datetime.combine(today - timedelta(days=1), start, tzinfo=now.tzinfo),
        datetime.combine(today, end, tzinfo=now.tzinfo),
    )


def summarize_counts(counts: dict[str, Any], limit: int = 8) -> str:
    if not counts:
        return "-"
    items = sorted(counts.items(), key=lambda item: str(item[0]))[:limit]
    suffix = "" if len(counts) <= limit else f", +{len(counts) - limit} more"
    return ", ".join(f"{key}={value}" for key, value in items) + suffix


class StaticPriceProvider:
    def __init__(self, prices: dict[str, float] | None = None):
        self.prices = prices or {}

    def get_current_price(self, coin: str) -> float:
        return float(self.prices.get(coin, 100.0))


class OKXPublicPriceProvider:
    """Unauthenticated ticker client used by network-enabled paper mode."""

    def __init__(self, timeout_sec: int = 10):
        self.timeout_sec = max(1, int(timeout_sec))

    def get_current_price(self, coin: str) -> float:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "network paper mode requires the runtime extra: pip install 'quant-bench[runtime]'"
            ) from exc
        response = requests.get(
            "https://www.okx.com/api/v5/market/ticker",
            params={"instId": coin},
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        if str(payload.get("code", "")) != "0" or not rows:
            raise RuntimeError(f"OKX public ticker failed for {coin}: {payload}")
        price = float(rows[0].get("last", 0.0) or 0.0)
        if price <= 0:
            raise RuntimeError(f"OKX public ticker returned invalid price for {coin}")
        return price


class FinGPTNewsPaperRunner:
    def __init__(
        self,
        config_path: str | Path,
        dry_run: bool = False,
        *,
        allow_network: bool = False,
        execute_orders: bool = False,
        confirmation: str = "",
    ):
        self.config_path = Path(config_path)
        self.cfg = load_yaml(self.config_path)
        self.dry_run = bool(dry_run or self.cfg.get("runtime", {}).get("dry_run", False))
        self.allow_network = bool(allow_network)
        self.execute_orders = bool(execute_orders)
        self.confirmation = str(confirmation)
        self.strategy_id = self.cfg["strategy"]["strategy_id"]
        self.method_family = "fingpt_news"
        self.tz = scheduler_timezone(self.cfg)
        self._validate_capabilities()
        self.run_dir = (WORKSPACE_ROOT / self.cfg["runtime"]["run_dir"]).resolve()
        self._init_paths()
        configure_run_file_logging(self.paths["runtime_log"], self.cfg.get("logging", {}).get("level", "INFO"))
        self._write_manifest()
        trade_cfg = self.cfg["trade"]
        self.coins = list(trade_cfg["coins"])
        self.executor = self._make_executor() if self._uses_okx_execution() else None
        if self.dry_run:
            self.price_provider = StaticPriceProvider(self.cfg.get("runtime", {}).get("dry_run_prices", {}))
        elif self.executor is not None:
            self.price_provider = self.executor
        else:
            self.price_provider = OKXPublicPriceProvider(
                timeout_sec=int(self.cfg.get("news", {}).get("request_timeout_sec", 15))
            )
        self.broker = self._make_broker(trade_cfg)
        configured_trade_mode = str(trade_cfg.get("mode", "paper_spot")).strip().lower()
        if self._uses_okx_execution() and self.executor is not None:
            execution_mode = self.executor.execution_mode
        else:
            execution_mode = "paper"
        self.trading_backend = RebalanceBrokerBackend(
            self.broker,
            backend_id=f"fingpt-news:{configured_trade_mode}",
            execution_mode=execution_mode,
        )
        self.trading_service = TradingService(self.trading_backend)
        self.news_store = NewsStore(self.paths["raw_news"], self.paths["news_checkpoint"])
        self.scheduler_state = JsonStateStore(
            self.paths["scheduler_state"],
            {
                "last_news_fetch_at": 0,
                "last_news_prepare_at": 0,
                "last_model_infer_at": 0,
                "last_trade_check_at": 0,
            },
        )
        self.per_coin_params = load_per_coin_params(self.paths["per_coin_params"])
        self.sentiment_service: LabelLogprobSentimentService | None = None
        self._last_heartbeat_log = 0.0
        self._log_startup_summary()

    def _uses_okx_execution(self) -> bool:
        mode = str(self.cfg.get("trade", {}).get("mode", "paper_spot")).strip().lower()
        return (not self.dry_run) and mode in {"okx_spot", "demo_spot", "simulated_spot", "simu_spot"}

    def _validate_capabilities(self) -> None:
        mode = str(self.cfg.get("trade", {}).get("mode", "paper_spot")).strip().lower()
        order_mode = mode in {"okx_spot", "demo_spot", "simulated_spot", "simu_spot"}
        if self.dry_run:
            if self.execute_orders:
                raise PermissionError("dry-run cannot execute orders")
            return
        if not self.allow_network:
            raise PermissionError("FinGPT news, public prices, model services, and OKX require --allow-network")
        if self.execute_orders and not order_mode:
            raise PermissionError("--execute-orders requires an OKX trade mode in the reviewed config")
        if order_mode:
            if not self.execute_orders:
                raise PermissionError(
                    "OKX trade mode is disabled until --execute-orders and an exact confirmation token are supplied"
                )
            api_cfg = self.cfg.get("api", {})
            flag = str(api_cfg.get("okx_flag", "1")).strip()
            expected = "DEMO_ORDERS" if flag == "1" else "LIVE_ORDERS"
            if self.confirmation != expected:
                raise PermissionError(f"refusing OKX order mode; pass --confirm {expected}")

    def _make_executor(self) -> TradeExecutor:
        api_cfg = self.cfg.get("api", {})
        inline = {
            "okx_api_key": api_cfg.get("okx_api_key") or api_cfg.get("api_key") or "",
            "okx_secret_key": api_cfg.get("okx_secret_key") or api_cfg.get("secret_key") or "",
            "okx_passphrase": api_cfg.get("okx_passphrase") or api_cfg.get("passphrase") or "",
        }
        if any(inline.values()):
            raise ValueError("inline OKX credentials are not supported; use environment variables")
        api_key_env = str(api_cfg.get("okx_api_key_env", "OKX_API_KEY_SIMU"))
        secret_key_env = str(api_cfg.get("okx_secret_key_env", "OKX_SECRET_KEY_SIMU"))
        passphrase_env = str(api_cfg.get("okx_passphrase_env", "OKX_PASSPHRASE"))
        api_key = os.getenv(api_key_env, "")
        secret_key = os.getenv(secret_key_env, "")
        passphrase = os.getenv(passphrase_env, "")
        missing = [
            name
            for name, value in {
                api_key_env: api_key,
                secret_key_env: secret_key,
                passphrase_env: passphrase,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "FinGPT OKX credentials are missing from environment variables: " + ", ".join(missing)
            )
        return TradeExecutor(
            identifier=self.strategy_id,
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            flag=str(api_cfg.get("okx_flag", "1")),
            trade_mode="spot",
            allow_env_credentials=False,
        )

    def _make_broker(self, trade_cfg: dict[str, Any]):
        mode = str(trade_cfg.get("mode", "paper_spot")).strip().lower()
        common = (
            self.coins,
            float(trade_cfg["initial_capital_usdt"]),
            float(trade_cfg["per_coin_capital_usdt"]),
            float(trade_cfg["fee_bps"]),
            float(trade_cfg.get("min_position_value_usdt", 2.0)),
        )
        if not self.dry_run and mode in {"okx_spot", "demo_spot", "simulated_spot", "simu_spot"}:
            return PerCoinOKXSpotBroker(
                self.paths["account_state"],
                self.executor,
                *common,
                order_settle_delay_sec=float(trade_cfg.get("order_settle_delay_sec", 1.5)),
                require_full_cash=bool(trade_cfg.get("require_full_cash", True)),
                order_cash_buffer_bps=(
                    float(trade_cfg["order_cash_buffer_bps"])
                    if trade_cfg.get("order_cash_buffer_bps") is not None
                    else None
                ),
                reconcile_account_positions_enabled=bool(
                    trade_cfg.get("reconcile_account_positions", False)
                ),
            )
        return PerCoinPaperBroker(
            self.paths["account_state"],
            common[0],
            common[1],
            common[2],
            common[3],
            float(trade_cfg.get("slippage_bps", 0.0)),
            common[4],
        )

    def _init_paths(self) -> None:
        trade_mode = str(self.cfg.get("trade", {}).get("mode", "paper_spot")).strip().lower()
        use_okx_state = (not self.dry_run) and trade_mode in {"okx_spot", "demo_spot", "simulated_spot", "simu_spot"}
        account_state_name = "okx_account_state.json" if use_okx_state else "paper_account_state.json"
        self.paths = {
            "config_copy": self.run_dir / "config.yaml",
            "raw_news": self.run_dir / "artifacts" / "raw_news_master.csv",
            "news_checkpoint": self.run_dir / "state" / "news_checkpoint.json",
            "scheduler_state": self.run_dir / "state" / "scheduler_state.json",
            "account_state": self.run_dir / "state" / account_state_name,
            "prompt_cache": self.run_dir / "state" / "prompt_score_cache.csv",
            "candidates": self.run_dir / "artifacts" / "news_candidates.csv",
            "sentiment": self.run_dir / "artifacts" / "news_model_sentiment.csv",
            "daily_scores": self.run_dir / "signals" / "daily_sentiment_scores.csv",
            "daily_attribution": self.run_dir / "signals" / "daily_news_attribution.csv",
            "signals": self.run_dir / "signals" / f"{self.cfg['strategy']['strategy_id']}_signals.csv",
            "metrics": self.run_dir / "metrics" / f"{self.cfg['strategy']['strategy_id']}_metrics.csv",
            "trades": self.run_dir / "trades" / f"{self.cfg['strategy']['strategy_id']}_trades.csv",
            "orders": self.run_dir / "trades" / f"{self.cfg['strategy']['strategy_id']}_orders.csv",
            "fills": self.run_dir / "trades" / f"{self.cfg['strategy']['strategy_id']}_fills.csv",
            "failed_orders": self.run_dir / "trades" / f"{self.cfg['strategy']['strategy_id']}_failed_orders.csv",
            "order_events": self.run_dir / "trades" / f"{self.cfg['strategy']['strategy_id']}_order_events.jsonl",
            "runtime_log": self.run_dir / "logs" / "fingpt_news.log",
            "runtime_events": self.run_dir / "logs" / "runtime_events.jsonl",
            "account_snapshots": self.run_dir / "account" / f"{self.cfg['strategy']['strategy_id']}_account_snapshots.csv",
            "rebalance_plan": self.run_dir / "decisions" / f"{self.cfg['strategy']['strategy_id']}_rebalance_plan.csv",
            "volume": self.run_dir / "volume" / f"{self.cfg['strategy']['strategy_id']}_volume.csv",
            "per_coin_params": (CURRENT_DIR / self.cfg["strategy"]["per_coin_params_path"]).resolve(),
        }
        for path in self.paths.values():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        for key, columns in EMPTY_CSV_CONTRACTS.items():
            contract_path = Path(self.paths[key])
            if not contract_path.exists():
                atomic_write_csv(pd.DataFrame(columns=columns), contract_path)
        Path(self.paths["config_copy"]).write_text(yaml.safe_dump(self.cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def _write_manifest(self) -> None:
        sid = self.cfg["strategy"]["strategy_id"]
        trade_mode = str(self.cfg.get("trade", {}).get("mode", "paper_spot"))
        write_run_manifest(
            self.run_dir,
            strategy_id=sid,
            method_family=self.method_family,
            frequency="1d",
            mode=trade_mode,
            paths={
                "metrics_path": f"metrics/{sid}_metrics.csv",
                "signals_path": f"signals/{sid}_signals.csv",
                "trades_path": f"trades/{sid}_trades.csv",
                "orders_path": f"trades/{sid}_orders.csv",
                "fills_path": f"trades/{sid}_fills.csv",
                "failed_orders_path": f"trades/{sid}_failed_orders.csv",
                "order_events_path": f"trades/{sid}_order_events.jsonl",
                "account_snapshots_path": f"account/{sid}_account_snapshots.csv",
                "rebalance_plan_path": f"decisions/{sid}_rebalance_plan.csv",
                "volume_path": f"volume/{sid}_volume.csv",
                "daily_sentiment_path": "signals/daily_sentiment_scores.csv",
                "news_sentiment_path": "artifacts/news_model_sentiment.csv",
            },
        )

    def _emit_event(self, event_type: str, **fields: Any) -> None:
        try:
            now = datetime.now(self.tz)
            append_jsonl_events(
                self.paths["runtime_events"],
                [
                    {
                        "event_type": event_type,
                        "timestamp": time.time(),
                        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "strategy_id": self.strategy_id,
                        "method_family": self.method_family,
                        **fields,
                    }
                ],
            )
        except Exception as exc:
            logger.debug("Failed to write FinGPT runtime event %s: %s", event_type, exc)

    def _log_startup_summary(self) -> None:
        start_time, end_time = self._model_window_times()
        logger.info("[FinGPT][startup] strategy=%s run_dir=%s dry_run=%s", self.strategy_id, self.run_dir, self.dry_run)
        logger.info(
            "[FinGPT][startup] timezone=%s trade_time=%s model_window=%s-%s news_fetch_interval=%ss",
            self.tz,
            self.cfg.get("scheduler", {}).get("trade_time_local", "00:05"),
            start_time.strftime("%H:%M"),
            end_time.strftime("%H:%M"),
            self.cfg.get("scheduler", {}).get("news_fetch_interval_sec", 3600),
        )
        logger.info("[FinGPT][startup] log_file=%s events=%s", self.paths["runtime_log"], self.paths["runtime_events"])
        self._emit_event(
            "startup",
            run_dir=str(self.run_dir),
            dry_run=self.dry_run,
            model_window_start=start_time.strftime("%H:%M"),
            model_window_end=end_time.strftime("%H:%M"),
        )

    def _model_window_times(self) -> tuple[dt_time, dt_time]:
        schedule = self.cfg.get("model_schedule", {})
        start = parse_local_time(schedule.get("load_start_local", "23:55"))
        end = parse_local_time(schedule.get("unload_end_local", "01:00"))
        return start, end

    def is_model_window_open(self, now_dt: datetime | None = None) -> bool:
        now_dt = now_dt or datetime.now(self.tz)
        start, end = self._model_window_times()
        return time_in_local_window(now_dt, start, end)

    def _model_window_bounds(self, now_dt: datetime | None = None) -> tuple[datetime, datetime]:
        now_dt = now_dt or datetime.now(self.tz)
        start, end = self._model_window_times()
        return local_window_bounds(now_dt, start, end)

    def _model_window_deadline_epoch(self, now_dt: datetime | None = None) -> float | None:
        if not self.is_model_window_open(now_dt):
            return None
        _start_dt, end_dt = self._model_window_bounds(now_dt)
        margin = float(self.cfg.get("model_schedule", {}).get("deadline_margin_sec", 5.0) or 0.0)
        return max(time.time(), end_dt.timestamp() - margin)

    def _can_load_model_now(self, now_dt: datetime | None = None) -> bool:
        if self.dry_run:
            return True
        schedule = self.cfg.get("model_schedule", {})
        if bool(schedule.get("allow_model_outside_window", False)):
            return True
        return self.is_model_window_open(now_dt)

    def _use_isolated_model_process(self) -> bool:
        if self.dry_run:
            return False
        model_execution = self.cfg.get("model_execution", {})
        return bool(model_execution.get("isolated_process", True))

    def _trade_date_for_model_window(self, now_dt: datetime | None = None) -> str:
        now_dt = now_dt or datetime.now(self.tz)
        start, end = self._model_window_times()
        trade_day = now_dt.date()
        if start > end and now_dt.time().replace(microsecond=0) >= start:
            trade_day = trade_day + timedelta(days=1)
        return trade_day.strftime("%Y-%m-%d")

    def _get_sentiment_service(self) -> LabelLogprobSentimentService:
        if self.sentiment_service is None:
            model_cfg = self.cfg["model"]
            self.sentiment_service = LabelLogprobSentimentService(
                base_model_id=model_cfg.get("base_model_id", ""),
                adapter_id=resolve_project_path(model_cfg.get("adapter_id", "")),
                cache_path=self.paths["prompt_cache"],
                max_length=int(model_cfg.get("max_length", 512)),
                batch_size=int(model_cfg.get("batch_size", 8)),
                dry_run=self.dry_run,
            )
        return self.sentiment_service

    def _unload_sentiment_model(self, reason: str) -> None:
        if self.sentiment_service is None or not self.sentiment_service.is_model_loaded:
            return
        logger.info("[FinGPT][model] unload requested reason=%s", reason)
        self._emit_event("model_unload_requested", reason=reason)
        self.sentiment_service.unload_model()
        self._emit_event("model_unloaded", reason=reason)

    def _run_inference_worker(
        self,
        candidates: pd.DataFrame,
        reason: str,
        trade_date: str | None,
        deadline_epoch: float | None,
    ) -> tuple[pd.DataFrame, bool]:
        atomic_write_csv(candidates, self.paths["candidates"])
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--infer-worker",
            "--config",
            str(self.config_path),
            "--workspace",
            str(WORKSPACE_ROOT),
            "--reason",
            reason,
            "--candidates-path",
            str(self.paths["candidates"]),
            "--sentiment-path",
            str(self.paths["sentiment"]),
            "--prompt-cache-path",
            str(self.paths["prompt_cache"]),
        ]
        if trade_date:
            cmd.extend(["--trade-date", trade_date])
        if deadline_epoch is not None:
            cmd.extend(["--deadline-epoch", str(float(deadline_epoch))])
        if self.dry_run:
            cmd.append("--dry-run")
        if self.allow_network:
            cmd.append("--allow-network")

        timeout = None
        if deadline_epoch is not None:
            grace = float(self.cfg.get("model_execution", {}).get("subprocess_timeout_grace_sec", 60.0) or 60.0)
            timeout = max(30.0, deadline_epoch - time.time() + grace)
        started = time.time()
        logger.info(
            "[FinGPT][infer-worker] spawn reason=%s trade_date=%s timeout=%s cmd=%s",
            reason,
            trade_date or "-",
            f"{timeout:.1f}s" if timeout is not None else "-",
            " ".join(cmd),
        )
        self._emit_event(
            "inference_worker_spawned",
            reason=reason,
            trade_date=trade_date or "",
            timeout_sec=round(timeout, 3) if timeout is not None else None,
        )
        success = False
        try:
            completed = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), check=False, timeout=timeout)
            if completed.returncode != 0:
                logger.warning("[FinGPT][infer-worker] exited with code=%s", completed.returncode)
                self._emit_event("inference_worker_failed", reason=reason, returncode=int(completed.returncode))
            else:
                success = True
                logger.info("[FinGPT][infer-worker] completed in %.1fs", time.time() - started)
                self._emit_event(
                    "inference_worker_finished",
                    reason=reason,
                    trade_date=trade_date or "",
                    elapsed_sec=round(time.time() - started, 3),
                )
        except subprocess.TimeoutExpired as exc:
            logger.warning("[FinGPT][infer-worker] timed out after %.1fs and was killed.", float(exc.timeout or 0.0))
            self._emit_event("inference_worker_timeout", reason=reason, timeout_sec=float(exc.timeout or 0.0))

        self.sentiment_service = None
        return safe_read_csv(self.paths["sentiment"]), success

    def _prioritize_candidates_for_trade(self, candidates: pd.DataFrame, trade_date: str | None) -> pd.DataFrame:
        if candidates.empty or not trade_date or "datetime" not in candidates.columns:
            return candidates
        _news_date, window_start, window_end = news_window_for_trade(trade_date, self.cfg, self.tz)
        work = candidates.copy()
        dt = pd.to_datetime(work["datetime"], errors="coerce")
        in_window = (dt >= pd.Timestamp(window_start)) & (dt < pd.Timestamp(window_end))
        work["_priority"] = in_window.map(lambda value: 0 if bool(value) else 1)
        work["_datetime_sort"] = dt
        work = work.sort_values(["_priority", "_datetime_sort"], ascending=[True, True])
        return work.drop(columns=["_priority", "_datetime_sort"])

    @staticmethod
    def _candidate_summary(candidates: pd.DataFrame) -> dict[str, Any]:
        if candidates.empty:
            return {"candidate_count": 0, "bucket_counts": {}, "symbol_counts": {}}
        bucket_counts = candidates.get("bucket", pd.Series(dtype=str)).value_counts().to_dict()
        symbol_counts = candidates.get("symbol", pd.Series(dtype=str)).value_counts().to_dict()
        return {
            "candidate_count": len(candidates),
            "bucket_counts": {str(key): int(value) for key, value in bucket_counts.items()},
            "symbol_counts": {str(key): int(value) for key, value in symbol_counts.items()},
        }

    def _log_heartbeat(self, force: bool = False) -> None:
        interval = int(self.cfg.get("logging", {}).get("heartbeat_interval_sec", 300) or 300)
        now = time.time()
        if not force and now - self._last_heartbeat_log < interval:
            return
        self._last_heartbeat_log = now
        candidates = safe_read_csv(self.paths["candidates"])
        sentiment = safe_read_csv(self.paths["sentiment"])
        raw = safe_read_csv(self.paths["raw_news"])
        service = self._get_sentiment_service()
        cache_stats = service.cache_stats(candidates)
        now_dt = datetime.now(self.tz)
        model_window_open = self.is_model_window_open(now_dt)
        logger.info(
            "[FinGPT][heartbeat] now=%s model_window=%s model_loaded=%s raw_news=%d candidates=%d sentiment_rows=%d pending_prompts=%d last_fetch=%s last_infer=%s",
            now_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "open" if model_window_open else "closed",
            service.is_model_loaded,
            len(raw),
            len(candidates),
            len(sentiment),
            cache_stats["missing"],
            local_datetime_from_epoch(self.scheduler_state.get("last_news_fetch_at", 0), self.tz),
            local_datetime_from_epoch(self.scheduler_state.get("last_model_infer_at", 0), self.tz),
        )
        self._emit_event(
            "heartbeat",
            model_window_open=model_window_open,
            model_loaded=service.is_model_loaded,
            raw_news_count=len(raw),
            candidate_count=len(candidates),
            sentiment_rows=len(sentiment),
            pending_prompts=int(cache_stats["missing"]),
        )

    def fetch_and_prepare_news(
        self,
        until_ts: int | None = None,
        pages_override: int | None = None,
        reason: str = "scheduled",
    ) -> pd.DataFrame:
        started = time.time()
        before_raw = self.news_store.load_raw_news()
        if self.dry_run and self.cfg.get("runtime", {}).get("sample_news"):
            reference_end = (
                datetime.fromtimestamp(float(until_ts), self.tz)
                if until_ts
                else datetime.combine(datetime.now(self.tz).date(), dt_time(0, 0), tzinfo=self.tz)
            )
            rows = []
            for index, sample in enumerate(self.cfg["runtime"]["sample_news"]):
                published = reference_end - timedelta(hours=12 - index)
                row = dict(sample)
                row["date"] = published.strftime("%Y-%m-%d %H:%M:%S")
                row["published_on"] = int(published.timestamp())
                rows.append(row)
        else:
            news_cfg = self.cfg["news"]
            pages = int(pages_override or news_cfg.get("pages_per_fetch", 3))
            logger.info(
                "[FinGPT][news] fetch_start reason=%s provider=%s pages=%d until_ts=%s lang=%s",
                reason,
                news_cfg.get("provider", "ccdata"),
                pages,
                until_ts,
                news_cfg.get("language", "EN"),
            )
            provider = str(news_cfg.get("provider", "ccdata")).strip().lower()
            if provider in {"rss", "rss_google", "google_rss"}:
                from quant_bench.methods.fingpt_news.news.rss import RSSGoogleNewsFetcher

                fetcher = RSSGoogleNewsFetcher(
                    history_path=self.run_dir / "artifacts" / "rss_news_history.csv",
                    feeds=news_cfg.get("rss_feeds") or None,
                    timeout_sec=int(news_cfg.get("request_timeout_sec", 20)),
                    max_retries=int(news_cfg.get("max_retries", 2)),
                    sleep_sec=float(news_cfg.get("request_sleep_sec", 1.0)),
                    timezone=str(self.cfg.get("scheduler", {}).get("timezone", "Asia/Shanghai")),
                    recent_days=int(news_cfg.get("rss_recent_days", 10)),
                    target_records=int(news_cfg.get("rss_target_records", 8000)),
                )
            elif provider == "ccdata":
                from quant_bench.methods.fingpt_news.news.fetcher import CCDataNewsFetcher

                fetcher = CCDataNewsFetcher(
                    api_key=news_cfg.get("ccdata_api_key", ""),
                    api_key_env=str(news_cfg.get("ccdata_api_key_env", "CCDATA_API_KEY")),
                    timeout_sec=int(news_cfg.get("request_timeout_sec", 15)),
                    max_retries=int(news_cfg.get("max_retries", 5)),
                    sleep_sec=float(news_cfg.get("request_sleep_sec", 1.0)),
                    timezone=str(self.cfg.get("scheduler", {}).get("timezone", "Asia/Shanghai")),
                )
            else:
                raise ValueError(f"unsupported FinGPT news provider: {provider}")
            rows = fetcher.fetch_pages(
                pages=pages,
                lang=news_cfg.get("language", "EN"),
                until_ts=until_ts,
            )
        raw = self.news_store.append_rows(rows)
        retention_days = int(self.cfg.get("news", {}).get("candidate_retention_days", 7))
        candidate_start = (datetime.now(self.tz) - timedelta(days=retention_days)).strftime("%Y-%m-%d %H:%M:%S")
        candidates = build_candidates(raw, self.cfg, start=candidate_start)
        write_candidates(candidates, self.paths["candidates"])
        self.scheduler_state.set("last_news_fetch_at", time.time())
        self.scheduler_state.set("last_news_prepare_at", time.time())
        self.scheduler_state.save()
        summary = self._candidate_summary(candidates)
        logger.info(
            "[FinGPT][news] fetch_done reason=%s fetched=%d raw_before=%d raw_after=%d new_raw=%d candidates=%d buckets=[%s] elapsed=%.1fs",
            reason,
            len(rows),
            len(before_raw),
            len(raw),
            max(0, len(raw) - len(before_raw)),
            summary["candidate_count"],
            summarize_counts(summary["bucket_counts"]),
            time.time() - started,
        )
        if not raw.empty and "date" in raw.columns:
            raw_dt = pd.to_datetime(raw["date"], errors="coerce").dropna()
            if not raw_dt.empty:
                logger.info(
                    "[FinGPT][news] raw_range first=%s latest=%s",
                    raw_dt.min().strftime("%Y-%m-%d %H:%M:%S"),
                    raw_dt.max().strftime("%Y-%m-%d %H:%M:%S"),
                )
        logger.info("[FinGPT][news] candidate_symbols=[%s]", summarize_counts(summary["symbol_counts"], limit=12))
        self._emit_event(
            "news_prepared",
            reason=reason,
            fetched_rows=len(rows),
            raw_before=len(before_raw),
            raw_after=len(raw),
            candidate_count=int(summary["candidate_count"]),
            bucket_counts=summary["bucket_counts"],
        )
        return candidates

    def infer_pending_candidates(
        self,
        candidates: pd.DataFrame | None = None,
        reason: str = "scheduled",
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        candidates = safe_read_csv(self.paths["candidates"]) if candidates is None else candidates
        if candidates.empty:
            logger.info("[FinGPT][infer] skip reason=%s because there are no candidates.", reason)
            return pd.DataFrame()
        prioritized = self._prioritize_candidates_for_trade(candidates, trade_date)
        service = self._get_sentiment_service()
        stats_before = service.cache_stats(prioritized)
        now_dt = datetime.now(self.tz)
        allow_model_load = self._can_load_model_now(now_dt)
        if stats_before["missing"] and not allow_model_load:
            logger.info(
                "[FinGPT][infer] skip_model_load reason=%s model_window=closed candidates=%d missing_prompts=%d cached_prompts=%d",
                reason,
                stats_before["candidates"],
                stats_before["missing"],
                stats_before["cached"],
            )
            self._emit_event(
                "inference_skipped_outside_window",
                reason=reason,
                candidates=int(stats_before["candidates"]),
                missing_prompts=int(stats_before["missing"]),
                cached_prompts=int(stats_before["cached"]),
            )
            result = service.infer_candidates(
                prioritized,
                self.paths["sentiment"],
                allow_model_load=False,
                allow_partial=True,
                progress_every_batches=int(self.cfg.get("logging", {}).get("progress_every_batches", 10) or 10),
            )
            self._unload_sentiment_model("outside_model_window")
            return result

        deadline_epoch = self._model_window_deadline_epoch(now_dt)
        if stats_before["missing"] and allow_model_load and self._use_isolated_model_process():
            logger.info(
                "[FinGPT][infer] delegating %d missing prompts to isolated worker.",
                stats_before["missing"],
            )
            result, worker_success = self._run_inference_worker(prioritized, reason, trade_date, deadline_epoch)
            self.sentiment_service = None
            service = self._get_sentiment_service()
            stats_after = service.cache_stats(prioritized)
            if not worker_success:
                logger.warning(
                    "[FinGPT][infer] worker_result_failed reason=%s output_rows=%d remaining_missing_prompts=%d",
                    reason,
                    len(result),
                    stats_after["missing"],
                )
                self._emit_event(
                    "inference_failed",
                    reason=reason,
                    trade_date=trade_date or "",
                    output_rows=len(result),
                    remaining_missing_prompts=int(stats_after["missing"]),
                    isolated_process=True,
                )
                return result
            self.scheduler_state.set("last_model_infer_at", time.time())
            self.scheduler_state.save()
            logger.info(
                "[FinGPT][infer] worker_result reason=%s output_rows=%d remaining_missing_prompts=%d",
                reason,
                len(result),
                stats_after["missing"],
            )
            self._emit_event(
                "inference_finished",
                reason=reason,
                trade_date=trade_date or "",
                output_rows=len(result),
                remaining_missing_prompts=int(stats_after["missing"]),
                isolated_process=True,
            )
            return result

        logger.info(
            "[FinGPT][infer] start reason=%s trade_date=%s candidates=%d unique_prompts=%d missing_prompts=%d deadline=%s",
            reason,
            trade_date or "-",
            stats_before["candidates"],
            stats_before["unique_prompts"],
            stats_before["missing"],
            local_datetime_from_epoch(deadline_epoch, self.tz) if deadline_epoch else "-",
        )
        self._emit_event(
            "inference_started",
            reason=reason,
            trade_date=trade_date or "",
            candidates=int(stats_before["candidates"]),
            unique_prompts=int(stats_before["unique_prompts"]),
            missing_prompts=int(stats_before["missing"]),
            allow_model_load=allow_model_load,
        )
        started = time.time()
        result = service.infer_candidates(
            prioritized,
            self.paths["sentiment"],
            allow_model_load=allow_model_load and not self._use_isolated_model_process(),
            allow_partial=True,
            deadline_epoch=deadline_epoch,
            progress_every_batches=int(self.cfg.get("logging", {}).get("progress_every_batches", 10) or 10),
        )
        stats_after = service.cache_stats(prioritized)
        self.scheduler_state.set("last_model_infer_at", time.time())
        self.scheduler_state.save()
        logger.info(
            "[FinGPT][infer] done reason=%s output_rows=%d remaining_missing_prompts=%d elapsed=%.1fs",
            reason,
            len(result),
            stats_after["missing"],
            time.time() - started,
        )
        self._emit_event(
            "inference_finished",
            reason=reason,
            trade_date=trade_date or "",
            output_rows=len(result),
            remaining_missing_prompts=int(stats_after["missing"]),
            elapsed_sec=round(time.time() - started, 3),
        )
        keep_loaded = bool(self.cfg.get("model_schedule", {}).get("keep_loaded_during_window", True))
        if not self.is_model_window_open() or not keep_loaded:
            self._unload_sentiment_model("inference_complete")
        return result

    def fetch_and_infer_news(
        self,
        until_ts: int | None = None,
        pages_override: int | None = None,
        reason: str = "compat",
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        candidates = self.fetch_and_prepare_news(until_ts=until_ts, pages_override=pages_override, reason=reason)
        return self.infer_pending_candidates(candidates, reason=reason, trade_date=trade_date)

    def run_daily_trade(self, trade_date: str | None = None, force: bool = False) -> None:
        trade_date = trade_date or now_local_date(self.tz)
        if self.broker.has_processed_trade_date(trade_date) and not force:
            logger.info("[FinGPT][trade] skip trade_date=%s because it is already processed.", trade_date)
            return
        news_date, window_start, window_end = news_window_for_trade(trade_date, self.cfg, self.tz)
        logger.info(
            "[FinGPT][trade] start trade_date=%s news_window=%s -> %s force=%s",
            trade_date,
            window_start,
            window_end,
            force,
        )
        sentiment = safe_read_csv(self.paths["sentiment"])
        if not self._sentiment_ready_for_window(sentiment, window_start, window_end):
            logger.warning(
                "Skip FinGPT daily trade for %s because sentiment inference is incomplete for %s - %s.",
                trade_date,
                window_start,
                window_end,
            )
            return
        daily_scores, attribution = build_daily_signals(
            sentiment,
            self.per_coin_params,
            self.coins,
            news_date,
            trade_date,
            window_start=window_start,
            window_end=window_end,
        )
        append_daily_outputs(daily_scores, attribution, self.paths["daily_scores"], self.paths["daily_attribution"])
        label_counts = daily_scores.get("model_label", pd.Series(dtype=str)).value_counts().to_dict()
        logger.info(
            "[FinGPT][signal] trade_date=%s labels=[%s] attribution_rows=%d",
            trade_date,
            summarize_counts({str(key): int(value) for key, value in label_counts.items()}),
            len(attribution),
        )
        prices = {coin: self.price_provider.get_current_price(coin) for coin in self.coins}
        missing_prices = [coin for coin, price in prices.items() if float(price or 0.0) <= 0.0]
        if self._uses_okx_execution() and missing_prices:
            logger.warning(
                "Skip FinGPT daily trade for %s because current prices are unavailable: %s",
                trade_date,
                missing_prices,
            )
            return
        reconcile = getattr(self.broker, "reconcile_account_positions", None)
        if self._uses_okx_execution() and callable(reconcile):
            reconcile(prices, timestamp=time.time(), trade_date=trade_date)
        trades = []
        signal_records = []
        plan_records = []
        ts = time.time()
        cycle_id = self._next_cycle_id()
        for row in daily_scores.to_dict("records"):
            coin = row["symbol"]
            model_score = int(row["model_score"])
            current_position = self._current_position(coin, prices.get(coin, 0.0))
            target_position = 1 if model_score > 0 else 0 if model_score < 0 else current_position
            plan_record = self._rebalance_plan_record(
                cycle_id,
                ts,
                trade_date,
                news_date,
                coin,
                row,
                prices.get(coin, 0.0),
                target_position,
            )
            plan_records.append(plan_record)
            action = (
                OrderAction.BUY
                if target_position > current_position
                else OrderAction.SELL
                if target_position < current_position
                else OrderAction.HOLD
            )
            order_request = OrderRequest(
                strategy_id=self.strategy_id,
                decision_id=str(plan_record["decision_id"]),
                instrument_id=coin,
                market_type=MarketType.SPOT,
                action=action,
                allow_backend_sizing=action in {OrderAction.BUY, OrderAction.SELL},
                reference_price=float(prices.get(coin, 0.0)),
                reason=str(plan_record.get("reason", "")),
                metadata={
                    "target_position": target_position,
                    "trade_date": trade_date,
                    "signal_row": row,
                    "timestamp": ts,
                },
            )
            trade_cycle = self.trading_service.execute(order_request)
            trade = (
                trade_cycle.execution.to_legacy_payload()
                if trade_cycle.execution.legacy_payload.get("trade_id")
                else None
            )
            if trade:
                trade["datetime"] = datetime.fromtimestamp(ts, self.tz).strftime("%Y-%m-%d %H:%M:%S")
                trades.append(trade)
            signal_records.append(self._signal_record(cycle_id, ts, trade_date, news_date, coin, row, prices.get(coin, 0.0)))
        self._append_csv(self.paths["signals"], signal_records, ["cycle_id", "coin", "trade_date"])
        if plan_records:
            self._append_csv(self.paths["rebalance_plan"], plan_records, ["decision_id"])
        if trades:
            self._append_csv(self.paths["trades"], trades, ["trade_id"])
        self._write_broker_audit_records()
        self._record_metrics(ts, trade_date, prices)
        self._record_volume(ts, trades)
        plan_sides = pd.Series([row["side"] for row in plan_records], dtype=str).value_counts().to_dict() if plan_records else {}
        successful_trades = [trade for trade in trades if bool(trade.get("success", True))]
        logger.info(
            "[FinGPT][trade] plan_sides=[%s] submitted_trades=%d successful_trades=%d",
            summarize_counts({str(key): int(value) for key, value in plan_sides.items()}),
            len(trades),
            len(successful_trades),
        )
        failed_trades = [trade for trade in trades if not bool(trade.get("success", True))]
        if failed_trades and self._uses_okx_execution() and not bool(self.cfg["trade"].get("mark_done_on_trade_fail", False)):
            logger.warning(
                "FinGPT daily trade for %s has %d failed OKX orders; leaving the date open for retry.",
                trade_date,
                len(failed_trades),
            )
            return
        self.broker.mark_trade_date_done(trade_date)
        logger.info("[FinGPT][trade] done trade_date=%s cycle_id=%d", trade_date, cycle_id)
        self._emit_event(
            "trade_finished",
            trade_date=trade_date,
            cycle_id=int(cycle_id),
            labels={str(key): int(value) for key, value in label_counts.items()},
            submitted_trades=len(trades),
            successful_trades=len(successful_trades),
        )

    def _sentiment_ready_for_window(self, sentiment: pd.DataFrame, window_start: str, window_end: str) -> bool:
        required_cols = {
            "datetime",
            "news_id",
            "model_label",
            "model_score",
            "article_score",
            "logprob_negative",
            "logprob_neutral",
            "logprob_positive",
        }
        missing_cols = sorted(required_cols - set(sentiment.columns))
        if sentiment.empty or missing_cols:
            logger.warning("[FinGPT] sentiment output missing required columns: %s", missing_cols)
            return False

        start_ts = pd.Timestamp(window_start)
        end_ts = pd.Timestamp(window_end)
        sent_dt = pd.to_datetime(sentiment["datetime"], errors="coerce")
        window_sentiment = sentiment[(sent_dt >= start_ts) & (sent_dt < end_ts)].copy()
        if window_sentiment.empty:
            logger.warning("[FinGPT] no sentiment rows found in news window %s - %s", window_start, window_end)
            return False

        value_cols = [
            "model_label",
            "model_score",
            "article_score",
            "logprob_negative",
            "logprob_neutral",
            "logprob_positive",
        ]
        null_counts = window_sentiment[value_cols].isna().sum()
        bad_cols = [col for col, count in null_counts.items() if int(count) > 0]
        if bad_cols:
            logger.warning("[FinGPT] sentiment rows contain null model outputs: %s", bad_cols)
            return False

        candidates = safe_read_csv(self.paths["candidates"])
        if candidates.empty or "news_id" not in candidates.columns or "datetime" not in candidates.columns:
            return True
        cand_dt = pd.to_datetime(candidates["datetime"], errors="coerce")
        window_candidates = candidates[(cand_dt >= start_ts) & (cand_dt < end_ts)].copy()
        missing_ids = set(window_candidates["news_id"].dropna().astype(str)) - set(
            window_sentiment["news_id"].dropna().astype(str)
        )
        if missing_ids:
            logger.warning("[FinGPT] %d candidate news rows are missing sentiment outputs", len(missing_ids))
            return False
        return True

    def _current_position(self, coin: str, price: float) -> int:
        qty = float(self.broker.state["position_qty_by_symbol"].get(coin, 0.0))
        return 1 if qty * price >= float(self.cfg["trade"].get("min_position_value_usdt", 2.0)) else 0

    def _next_cycle_id(self) -> int:
        df = safe_read_csv(self.paths["signals"])
        if df.empty or "cycle_id" not in df:
            return 1
        return int(pd.to_numeric(df["cycle_id"], errors="coerce").max() or 0) + 1

    def _signal_record(self, cycle_id: int, ts: float, trade_date: str, news_date: str, coin: str, row: dict[str, Any], price: float) -> dict[str, Any]:
        return {
            "cycle_id": cycle_id,
            "timestamp": ts,
            "datetime": datetime.fromtimestamp(ts, self.tz).strftime("%Y-%m-%d %H:%M:%S"),
            "bar_timestamp": local_midnight_timestamp(news_date, self.tz),
            "bar_datetime": news_date,
            "trade_date": trade_date,
            "news_date": news_date,
            "news_window_start": row.get("news_window_start", ""),
            "news_window_end": row.get("news_window_end", ""),
            "coin": coin,
            "score": row.get("final_score", 0.0),
            "signal_label": row.get("model_label", ""),
            "signal_score": row.get("model_score", 0),
            "price_at_pred": price,
            "price_future": float("nan"),
            "true_return": float("nan"),
            "labeled": False,
            "asset_score": row.get("asset_score", 0.0),
            "market_score": row.get("market_score", 0.0),
            "asset_news_count": row.get("asset_news_count", 0),
            "market_news_count": row.get("market_news_count", 0),
            "method_family": "fingpt_news",
            "strategy_id": self.strategy_id,
            "frequency": "1d",
        }

    def _rebalance_plan_record(
        self,
        cycle_id: int,
        ts: float,
        trade_date: str,
        news_date: str,
        coin: str,
        row: dict[str, Any],
        price: float,
        target_position: int,
    ) -> dict[str, Any]:
        cash = float(self.broker.state["cash_by_symbol"].get(coin, 0.0))
        qty = float(self.broker.state["position_qty_by_symbol"].get(coin, 0.0))
        ref_price = float(price or 0.0)
        current_value = qty * ref_price if ref_price > 0 else 0.0
        current_position = self._current_position(coin, ref_price)
        if target_position > current_position:
            planned_delta = cash
            side = "buy"
        elif target_position < current_position:
            planned_delta = current_value
            side = "sell"
        else:
            planned_delta = 0.0
            side = "hold"
        return {
            "decision_id": f"{self.strategy_id}-{trade_date}-{coin}",
            "cycle_id": cycle_id,
            "timestamp": ts,
            "datetime": datetime.fromtimestamp(ts, self.tz).strftime("%Y-%m-%d %H:%M:%S"),
            "trade_date": trade_date,
            "news_date": news_date,
            "news_window_start": row.get("news_window_start", ""),
            "news_window_end": row.get("news_window_end", ""),
            "strategy_id": self.strategy_id,
            "method_family": self.method_family,
            "frequency": "1d",
            "coin": coin,
            "side": side,
            "reference_price": ref_price,
            "current_qty": qty,
            "current_value_usdt": current_value,
            "cash_available_for_symbol": cash,
            "target_position": int(target_position),
            "current_position": int(current_position),
            "planned_delta_usdt": planned_delta,
            "signal_label": row.get("model_label", ""),
            "signal_score": row.get("model_score", 0),
            "final_score": row.get("final_score", 0.0),
            "threshold": row.get("threshold", 0.0),
            "asset_score": row.get("asset_score", 0.0),
            "market_score": row.get("market_score", 0.0),
            "asset_news_count": row.get("asset_news_count", 0),
            "market_news_count": row.get("market_news_count", 0),
            "will_submit_order": side in {"buy", "sell"} and planned_delta >= float(self.cfg["trade"].get("min_position_value_usdt", 2.0)),
            "reason": "signal_rebalance",
        }

    def _write_broker_audit_records(self) -> None:
        drain = getattr(self.broker, "drain_audit_records", None)
        if not callable(drain):
            return
        records = drain()
        if records.get("orders"):
            append_csv_rows(self.paths["orders"], records["orders"], ["trade_id"])
        if records.get("fills"):
            append_csv_rows(self.paths["fills"], records["fills"], ["trade_id", "fill_index"])
        if records.get("failed_orders"):
            append_csv_rows(self.paths["failed_orders"], records["failed_orders"], ["trade_id"])
        if records.get("account_snapshots"):
            append_csv_rows(self.paths["account_snapshots"], records["account_snapshots"], ["trade_id", "snapshot_phase"])
        if records.get("order_events"):
            append_jsonl_events(self.paths["order_events"], records["order_events"])

    def _record_metrics(self, ts: float, trade_date: str, prices: dict[str, float]) -> None:
        snapshot = self.broker.equity_snapshot(prices)
        btc_price = float(prices.get("BTC-USDT", 0.0) or 0.0)
        baseline_equity = self._equal_weight_baseline_equity(prices)
        row = {
            "timestamp": ts,
            "datetime": datetime.fromtimestamp(ts, self.tz).strftime("%Y-%m-%d %H:%M:%S"),
            "bar_timestamp": local_midnight_timestamp(trade_date, self.tz),
            "bar_datetime": trade_date,
            "initial_capital_usdt": self.broker.initial_capital_usdt,
            "cash_total_usdt": snapshot["cash_total_usdt"],
            "holdings_value_usdt": snapshot["holdings_value_usdt"],
            "total_equity_usdt": snapshot["total_equity_usdt"],
            "strategy_equity": snapshot["strategy_equity"],
            "strategy_pnl": snapshot["strategy_pnl"],
            "strategy_returns_pct": snapshot["strategy_returns_pct"],
            "baseline_equity": baseline_equity,
            "baseline_pnl": baseline_equity - self.broker.initial_capital_usdt,
            "baseline_returns_pct": (baseline_equity / self.broker.initial_capital_usdt - 1.0) * 100.0,
            "btc_price": btc_price,
            "active_positions": snapshot["active_positions"],
            "gross_exposure": snapshot["gross_exposure"],
            "cash_ratio": snapshot["cash_ratio"],
            "method_family": "fingpt_news",
            "strategy_id": self.strategy_id,
            "frequency": "1d",
        }
        self._append_csv(self.paths["metrics"], [row], ["bar_datetime"])

    def _equal_weight_baseline_equity(self, prices: dict[str, float]) -> float:
        state = self.broker.state.setdefault("baseline", {})
        if not state:
            state["qty_by_symbol"] = {
                coin: (self.broker.per_coin_capital_usdt / price if (price := float(prices.get(coin, 0.0) or 0.0)) > 0 else 0.0)
                for coin in self.coins
            }
            self.broker.store.save()
        return float(sum(float(state["qty_by_symbol"].get(coin, 0.0)) * float(prices.get(coin, 0.0) or 0.0) for coin in self.coins))

    def _record_volume(self, ts: float, trades: list[dict[str, Any]]) -> None:
        old = safe_read_csv(self.paths["volume"])
        cumulative_total = float(pd.to_numeric(old.get("cumulative_total_volume_usdt", pd.Series([0.0])), errors="coerce").iloc[-1]) if not old.empty else 0.0
        dt_str = datetime.fromtimestamp(ts, self.tz).strftime("%Y-%m-%d %H:%M:%S")
        date_str = datetime.fromtimestamp(ts, self.tz).strftime("%Y-%m-%d")
        daily_total_before = 0.0
        per_coin_daily: dict[str, float] = {}
        if not old.empty and "datetime" in old:
            last_dt = str(old["datetime"].iloc[-1])
            if last_dt[:10] == date_str:
                daily_total_before = float(pd.to_numeric(old.get("daily_total_volume_usdt", pd.Series([0.0])), errors="coerce").iloc[-1])
                if "per_coin_daily_json" in old:
                    try:
                        per_coin_daily = json.loads(str(old["per_coin_daily_json"].iloc[-1]))
                    except Exception:
                        per_coin_daily = {}
        successful_trades = [trade for trade in trades if bool(trade.get("success", True))]
        cycle_volume = float(sum(abs(float(t.get("notional_usdt", 0.0))) for t in successful_trades))
        per_coin_cycle: dict[str, float] = {}
        for trade in successful_trades:
            coin = str(trade.get("coin", "UNKNOWN"))
            per_coin_cycle[coin] = per_coin_cycle.get(coin, 0.0) + abs(float(trade.get("notional_usdt", 0.0)))
        per_coin_cumulative: dict[str, float] = {}
        if not old.empty and "per_coin_cumulative_json" in old:
            try:
                per_coin_cumulative = json.loads(str(old["per_coin_cumulative_json"].iloc[-1]))
            except Exception:
                per_coin_cumulative = {}
        for coin, value in per_coin_cycle.items():
            per_coin_cumulative[coin] = float(per_coin_cumulative.get(coin, 0.0)) + value
            per_coin_daily[coin] = float(per_coin_daily.get(coin, 0.0)) + value
        row = {
            "timestamp": ts,
            "datetime": dt_str,
            "cycle_volume_usdt": cycle_volume,
            "cumulative_total_volume_usdt": cumulative_total + cycle_volume,
            "daily_total_volume_usdt": daily_total_before + cycle_volume,
            "per_coin_cumulative_json": json.dumps(per_coin_cumulative, ensure_ascii=False, sort_keys=True),
            "per_coin_daily_json": json.dumps(per_coin_daily, ensure_ascii=False, sort_keys=True),
        }
        self._append_csv(self.paths["volume"], [row], ["timestamp"])

    @staticmethod
    def _append_csv(path: str | Path, rows: list[dict[str, Any]], dedupe_cols: list[str]) -> None:
        if not rows:
            return
        old = safe_read_csv(path)
        combined = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if not old.empty else pd.DataFrame(rows)
        existing_cols = [col for col in dedupe_cols if col in combined.columns]
        if existing_cols:
            combined = combined.drop_duplicates(existing_cols, keep="last")
        atomic_write_csv(combined, path)

    def should_trade_now(self) -> bool:
        now = datetime.now(self.tz)
        trade_date = now.strftime("%Y-%m-%d")
        target, cutoff = trade_schedule_bounds(trade_date, self.cfg, self.tz)
        return target <= now <= cutoff and not self.broker.has_processed_trade_date(trade_date)

    def run_once(self, trade: bool = True, fetch: bool = True, force_trade: bool = False, trade_date: str | None = None) -> None:
        self._log_heartbeat(force=True)
        if trade and not force_trade and trade_date is None:
            current_trade_date = now_local_date(self.tz)
            now = datetime.now(self.tz)
            target, cutoff = trade_schedule_bounds(current_trade_date, self.cfg, self.tz)
            if not (target <= now <= cutoff):
                logger.info(
                    "Skip FinGPT daily trade for %s because current time %s is outside trade window %s-%s.",
                    current_trade_date,
                    now.strftime("%H:%M"),
                    target.strftime("%H:%M"),
                    cutoff.strftime("%H:%M"),
                )
                trade = False
        effective_trade_date = trade_date or now_local_date(self.tz)
        candidates = None
        if fetch:
            until_ts = None
            pages_override = None
            if trade:
                _news_date, _window_start, window_end = news_window_for_trade(
                    effective_trade_date, self.cfg, self.tz
                )
                until_ts = local_epoch_seconds(window_end, self.tz)
                pages_override = int(self.cfg.get("news", {}).get("trade_catchup_pages", self.cfg.get("news", {}).get("pages_per_fetch", 3)))
            candidates = self.fetch_and_prepare_news(
                until_ts=until_ts,
                pages_override=pages_override,
                reason="run_once_trade_catchup" if trade else "run_once",
            )
        if fetch or trade:
            infer_trade_date = effective_trade_date if trade else self._trade_date_for_model_window()
            self.infer_pending_candidates(
                candidates,
                reason="run_once_trade_catchup" if trade else "run_once",
                trade_date=infer_trade_date,
            )
        if trade:
            self.run_daily_trade(trade_date=trade_date, force=force_trade)
        if not self.is_model_window_open():
            self._unload_sentiment_model("run_once_complete_outside_window")

    def start(self) -> None:
        self._log_heartbeat(force=True)
        startup_hours = float(self.cfg.get("scheduler", {}).get("startup_catchup_hours", 0) or 0)
        if startup_hours > 0 and not float(self.scheduler_state.get("last_news_fetch_at", 0)):
            until_dt = datetime.now(self.tz) - timedelta(hours=startup_hours)
            candidates = self.fetch_and_prepare_news(
                until_ts=int(until_dt.timestamp()),
                pages_override=int(self.cfg.get("news", {}).get("startup_catchup_pages", self.cfg.get("news", {}).get("trade_catchup_pages", self.cfg.get("news", {}).get("pages_per_fetch", 3)))),
                reason="startup_catchup",
            )
            if self.is_model_window_open() or self.dry_run:
                self.infer_pending_candidates(candidates, reason="startup_catchup", trade_date=self._trade_date_for_model_window())
        while True:
            try:
                self._log_heartbeat()
                now = time.time()
                now_dt = datetime.now(self.tz)
                interval = int(self.cfg["scheduler"].get("news_fetch_interval_sec", 3600))
                infer_interval = int(self.cfg["scheduler"].get("model_infer_interval_sec", 300))
                candidates = None
                did_fetch = False
                if now - float(self.scheduler_state.get("last_news_fetch_at", 0)) >= interval:
                    candidates = self.fetch_and_prepare_news(reason="scheduled")
                    did_fetch = True

                if self.is_model_window_open(now_dt) or self.dry_run:
                    should_infer = did_fetch or now - float(self.scheduler_state.get("last_model_infer_at", 0)) >= infer_interval
                    if should_infer:
                        self.infer_pending_candidates(
                            candidates,
                            reason="model_window",
                            trade_date=self._trade_date_for_model_window(now_dt),
                        )
                else:
                    self._unload_sentiment_model("outside_model_window")

                if self.should_trade_now():
                    trade_date = now_local_date(self.tz)
                    _news_date, _window_start, window_end = news_window_for_trade(
                        trade_date, self.cfg, self.tz
                    )
                    candidates = self.fetch_and_prepare_news(
                        until_ts=local_epoch_seconds(window_end, self.tz),
                        pages_override=int(self.cfg.get("news", {}).get("trade_catchup_pages", self.cfg.get("news", {}).get("pages_per_fetch", 3))),
                        reason="trade_catchup",
                    )
                    self.infer_pending_candidates(candidates, reason="trade_catchup", trade_date=trade_date)
                    self.scheduler_state.set("last_trade_check_at", time.time())
                    self.scheduler_state.save()
                    self.run_daily_trade(trade_date=trade_date)
            except Exception as exc:
                logger.exception("[FinGPT][loop] unexpected error: %s", exc)
                self._emit_event("loop_error", error=str(exc))
                if not self.is_model_window_open():
                    self._unload_sentiment_model("loop_error_outside_window")
            time.sleep(30)


def run_inference_worker(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config)
    tz = scheduler_timezone(cfg)
    run_dir = (WORKSPACE_ROOT / cfg["runtime"]["run_dir"]).resolve()
    runtime_log = run_dir / "logs" / "fingpt_news.log"
    runtime_events = run_dir / "logs" / "runtime_events.jsonl"
    configure_run_file_logging(runtime_log, cfg.get("logging", {}).get("level", "INFO"))
    started = time.time()
    reason = str(args.reason or "worker")
    trade_date = str(args.trade_date or "")
    candidates = safe_read_csv(args.candidates_path)
    if trade_date and not candidates.empty and "datetime" in candidates.columns:
        _news_date, window_start, window_end = news_window_for_trade(trade_date, cfg, tz)
        work = candidates.copy()
        dt = pd.to_datetime(work["datetime"], errors="coerce")
        in_window = (dt >= pd.Timestamp(window_start)) & (dt < pd.Timestamp(window_end))
        work["_priority"] = in_window.map(lambda value: 0 if bool(value) else 1)
        work["_datetime_sort"] = dt
        candidates = work.sort_values(["_priority", "_datetime_sort"], ascending=[True, True]).drop(columns=["_priority", "_datetime_sort"])

    model_cfg = cfg["model"]
    service = LabelLogprobSentimentService(
        base_model_id=model_cfg.get("base_model_id", ""),
        adapter_id=resolve_project_path(model_cfg.get("adapter_id", "")),
        cache_path=args.prompt_cache_path,
        max_length=int(model_cfg.get("max_length", 512)),
        batch_size=int(model_cfg.get("batch_size", 8)),
        dry_run=bool(args.dry_run or cfg.get("runtime", {}).get("dry_run", False)),
    )
    stats_before = service.cache_stats(candidates)
    logger.info(
        "[FinGPT][infer-worker] start reason=%s trade_date=%s candidates=%d unique_prompts=%d missing_prompts=%d",
        reason,
        trade_date or "-",
        stats_before["candidates"],
        stats_before["unique_prompts"],
        stats_before["missing"],
    )
    append_jsonl_events(
        runtime_events,
        [
            {
                "event_type": "inference_worker_started",
                "timestamp": time.time(),
                "datetime": datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S"),
                "strategy_id": cfg["strategy"]["strategy_id"],
                "method_family": "fingpt_news",
                "reason": reason,
                "trade_date": trade_date,
                "candidates": int(stats_before["candidates"]),
                "missing_prompts": int(stats_before["missing"]),
            }
        ],
    )
    try:
        result = service.infer_candidates(
            candidates,
            args.sentiment_path,
            allow_model_load=True,
            allow_partial=True,
            deadline_epoch=float(args.deadline_epoch) if args.deadline_epoch is not None else None,
            progress_every_batches=int(cfg.get("logging", {}).get("progress_every_batches", 10) or 10),
        )
        stats_after = service.cache_stats(candidates)
        logger.info(
            "[FinGPT][infer-worker] done reason=%s output_rows=%d remaining_missing_prompts=%d elapsed=%.1fs",
            reason,
            len(result),
            stats_after["missing"],
            time.time() - started,
        )
        append_jsonl_events(
            runtime_events,
            [
                {
                    "event_type": "inference_worker_done",
                    "timestamp": time.time(),
                    "datetime": datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S"),
                    "strategy_id": cfg["strategy"]["strategy_id"],
                    "method_family": "fingpt_news",
                    "reason": reason,
                    "trade_date": trade_date,
                    "output_rows": len(result),
                    "remaining_missing_prompts": int(stats_after["missing"]),
                    "elapsed_sec": round(time.time() - started, 3),
                }
            ],
        )
        return 0
    finally:
        service.unload_model()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FinGPT SFT news sentiment paper trader.")
    parser.add_argument("--config", type=Path, default=CURRENT_DIR / "configs" / "sentiment_sft_live.yaml")
    parser.add_argument("--workspace", type=Path, default=WORKSPACE_ROOT)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--trade-only", action="store_true")
    parser.add_argument("--force-trade", action="store_true")
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--execute-orders", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--infer-worker", action="store_true")
    parser.add_argument("--reason", default="worker")
    parser.add_argument("--candidates-path", type=Path, default=None)
    parser.add_argument("--sentiment-path", type=Path, default=None)
    parser.add_argument("--prompt-cache-path", type=Path, default=None)
    parser.add_argument("--deadline-epoch", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    global WORKSPACE_ROOT
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
    )
    args = parse_args()
    WORKSPACE_ROOT = args.workspace.expanduser().resolve()
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    if args.infer_worker:
        if not args.dry_run and not args.allow_network:
            raise PermissionError("model inference worker requires --allow-network")
        missing = [
            name
            for name, value in {
                "candidates_path": args.candidates_path,
                "sentiment_path": args.sentiment_path,
                "prompt_cache_path": args.prompt_cache_path,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(f"infer worker missing required args: {', '.join(missing)}")
        raise SystemExit(run_inference_worker(args))
    runner = FinGPTNewsPaperRunner(
        args.config,
        dry_run=args.dry_run,
        allow_network=args.allow_network,
        execute_orders=args.execute_orders,
        confirmation=args.confirm,
    )
    if args.once:
        runner.run_once(
            trade=not args.fetch_only,
            fetch=not args.trade_only,
            force_trade=args.force_trade,
            trade_date=args.trade_date,
        )
    else:
        runner.start()


if __name__ == "__main__":
    main()
