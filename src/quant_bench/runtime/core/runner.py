import asyncio
import json
import logging
import os
import time
import traceback
from typing import Any

import pandas as pd

from quant_bench.runtime.core.atomic_io import atomic_write_csv, safe_read_csv
from quant_bench.runtime.core.audit_log import append_csv_rows, append_jsonl_events
from quant_bench.runtime.core.config import ConfigManager
from quant_bench.runtime.core.data_manager import DataManager
from quant_bench.runtime.core.execution import TradeExecutor
from quant_bench.runtime.core.metrics import MetricsTracker
from quant_bench.runtime.core.signal_tracker import SignalTracker
from quant_bench.runtime.core.volume_tracker import VolumeTracker
from quant_bench.trading import MarketType, OrderAction, OrderRequest, TradingService

logger = logging.getLogger(__name__)

class TimeframeRunner:
    """
    负责启动同一个时间尺度下的多个不同模型实例，
    集中拉取数据分发，并调用模型决策和交易执行，最终统一计入 Metrics。

    v3 新增：
    - 首周期等待（根据历史 CSV 最后时间戳计算）
    - 配置热加载（每周期开始前 + sleep 期间提前 60 秒）
    - VolumeTracker 集成
    - 持仓价格补全（防止热更新移除币种后仓位卡死）
    - 交易成功/失败跟踪（只统计实际成交量）
    """
    def __init__(
        self,
        timeframe: str,
        update_interval_sec: int,
        method_family: str = "traditional_ml",
        run_root: str = "",
    ):
        self.timeframe = timeframe
        self.update_interval_sec = update_interval_sec
        self.method_family = method_family
        self.run_root = run_root
        self.models: list[dict[str, Any]] = []

        # 共享的数据拉取器，用于拉取 K 线（使用环境变量或首个模型的 API 配置）
        self.shared_api_client = TradeExecutor(identifier="PublicDataFetcher")
        self.data_manager = DataManager(
            api_client=self.shared_api_client, timeframe_str=self.timeframe
        )

    def register_model(self, model_name: str, config: ConfigManager, adapter_instance):
        """
        向 Runner 注册一个新的模型实例。
        每个模型拥有自己的 TradeExecutor、MetricsTracker、SignalTracker、VolumeTracker。
        """
        # 获取 flag (OKX: "0"实盘, "1"模拟盘)
        flag = "1" if config.is_simulated else "0"

        exec_instance = TradeExecutor(
            identifier=model_name,
            api_key=config.api_key,
            secret_key=config.secret_key,
            passphrase=config.passphrase,
            flag=flag,
            trade_mode=config.trade_mode
        )

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if self.run_root:
            log_dir = self.run_root if os.path.isabs(self.run_root) else os.path.join(project_root, self.run_root)
        else:
            log_dir = os.path.join(project_root, "timeframes", self.timeframe, "logs")

        # 每个模型独立统计跑分
        metrics_instance = MetricsTracker(
            identifier=model_name,
            timeframe=self.timeframe,
            api_client=exec_instance,
            log_dir=log_dir,
            initial_capital_usdt=config.initial_capital_usdt,
            method_family=self.method_family,
        )

        # 每个模型独立记录预测信号并追踪 IC
        signal_tracker_instance = SignalTracker(
            identifier=model_name,
            timeframe=self.timeframe,
            log_dir=log_dir,
            method_family=self.method_family,
        )

        # 每个模型独立记录交易量
        volume_tracker_instance = VolumeTracker(
            identifier=model_name,
            timeframe=self.timeframe,
            log_dir=log_dir
        )
        trade_log_path = os.path.join(
            log_dir,
            model_name,
            f"{model_name}_{self.timeframe}_trades.csv",
        )
        audit_paths = {
            "trades": trade_log_path,
            "orders": os.path.join(log_dir, model_name, f"{model_name}_{self.timeframe}_orders.csv"),
            "fills": os.path.join(log_dir, model_name, f"{model_name}_{self.timeframe}_fills.csv"),
            "failed_orders": os.path.join(log_dir, model_name, f"{model_name}_{self.timeframe}_failed_orders.csv"),
            "account_snapshots": os.path.join(log_dir, model_name, f"{model_name}_{self.timeframe}_account_snapshots.csv"),
            "rebalance_plan": os.path.join(log_dir, model_name, f"{model_name}_{self.timeframe}_rebalance_plan.csv"),
            "order_events": os.path.join(log_dir, model_name, f"{model_name}_{self.timeframe}_order_events.jsonl"),
        }

        # 组装结构体
        self.models.append({
            "name": model_name,
            "config": config,
            "adapter": adapter_instance,
            "executor": exec_instance,
            "metrics": metrics_instance,
            "signal_tracker": signal_tracker_instance,
            "volume_tracker": volume_tracker_instance,
            "trade_log_path": trade_log_path,
            "audit_paths": audit_paths,
        })
        if self.run_root:
            try:
                from quant_bench.runtime.core.run_manifest import write_run_manifest

                run_dir = os.path.join(log_dir, model_name)
                write_run_manifest(
                    run_dir,
                    strategy_id=model_name,
                    method_family=self.method_family,
                    frequency=self.timeframe,
                    mode="okx_demo" if config.is_simulated else "live",
                    paths={
                        "metrics_path": f"{model_name}_{self.timeframe}_metrics.csv",
                        "signals_path": f"{model_name}_{self.timeframe}_signals.csv",
                        "ic_path": f"{model_name}_{self.timeframe}_ic_report.csv",
                        "volume_path": f"{model_name}_{self.timeframe}_volume.csv",
                        "trades_path": f"{model_name}_{self.timeframe}_trades.csv",
                        "orders_path": f"{model_name}_{self.timeframe}_orders.csv",
                        "fills_path": f"{model_name}_{self.timeframe}_fills.csv",
                        "failed_orders_path": f"{model_name}_{self.timeframe}_failed_orders.csv",
                        "account_snapshots_path": f"{model_name}_{self.timeframe}_account_snapshots.csv",
                        "rebalance_plan_path": f"{model_name}_{self.timeframe}_rebalance_plan.csv",
                        "order_events_path": f"{model_name}_{self.timeframe}_order_events.jsonl",
                    },
                )
            except Exception as exc:
                logger.warning(f"[Runner-{self.timeframe}] 写 run_manifest 失败: {exc}")
        logger.info(f"[Runner-{self.timeframe}] 已成功拉起模型节点接入: {model_name}")

    async def _run_model_cycle(self, model_pack: dict[str, Any], ts_now: float):
        """单开协程跑一个模型的预测与执行生命周期"""
        name = model_pack['name']
        cfg = model_pack['config']
        adapter = model_pack['adapter']
        executor: TradeExecutor = model_pack['executor']
        metrics: MetricsTracker = model_pack['metrics']
        signal_tracker: SignalTracker = model_pack['signal_tracker']
        volume_tracker: VolumeTracker = model_pack['volume_tracker']
        trade_log_path = model_pack.get('trade_log_path')
        audit_paths = model_pack.get('audit_paths', {})

        try:
            logger.info(f"[{name}] 开始运行周期推理...")
            coins = cfg.coins
            limit = cfg.data_limit
            shared_data = {}
            current_prices = {}

            # 使用公共的 DataManager 拿取本周期的全量数据
            for coin in coins:
                df = await self.data_manager.get_kline_data(coin, limit=limit)
                if df is not None and not df.empty:
                    shared_data[coin] = df
                    current_prices[coin] = df.iloc[-1]['close']

            if not shared_data:
                logger.warning(f"[{name}] 数据拉取全部失败，跳过本次预测循环。")
                return

            missing_coins = [coin for coin in coins if coin not in shared_data]
            if missing_coins:
                logger.warning(
                    f"[{name}] 本轮缺少已收盘K线数据 {missing_coins}，"
                    "为避免缺失预测触发误交易，跳过本次循环。"
                )
                return

            bar_timestamps = []
            for df in shared_data.values():
                if 'ts' in df.columns and not df.empty:
                    bar_timestamps.append(float(df.iloc[-1]['ts']) / 1000.0)
            unique_bar_timestamps = {int(x) for x in bar_timestamps}
            if len(unique_bar_timestamps) > 1:
                logger.warning(
                    f"[{name}] 本周期不同币种的最新已收盘K线时间不完全一致: "
                    f"{sorted(unique_bar_timestamps)}，跳过本次循环。"
                )
                return
            bar_timestamp = bar_timestamps[0] if bar_timestamps else 0.0

            last_bar_timestamp = metrics.get_last_bar_timestamp()
            if (
                bar_timestamp > 0
                and last_bar_timestamp <= 0
                and not self._is_just_after_close(ts_now, bar_timestamp)
            ):
                logger.warning(
                    f"[{name}] 未找到该模型自己的历史K线进度，且当前 bar_ts={bar_timestamp:.0f} "
                    "已不在刚收盘启动窗口内；跳过本轮，等待下一根K线收盘后再启动，"
                    "避免清空CSV/重启后补打一根旧K线。"
                )
                return
            if bar_timestamp > 0 and last_bar_timestamp >= bar_timestamp:
                logger.info(
                    f"[{name}] 已处理过该K线 bar_ts={bar_timestamp:.0f}，"
                    f"跳过以避免重启后重复交易。"
                )
                return

            # 获取该模型的真实资金和仓位
            current_positions = executor.get_holdings()

            # ★ BUG-3 修复：补全已持仓但不在 coins/current_prices 中的币种价格
            for coin, qty in current_positions.items():
                if qty > 0 and coin not in current_prices:
                    price = executor.get_current_price(coin)
                    if price > 0:
                        current_prices[coin] = price
                        logger.info(
                            f"[{name}] 📌 持仓币种 {coin} 不在监控列表中，"
                            f"已补充价格: {price}"
                        )

            # ======= 1. 用本周期价格标注上一周期 pending 信号 =======
            signal_tracker.finalize_pending(current_prices, executor=executor)

            # ======= 2. 预测打分 =======
            predictions = await asyncio.to_thread(adapter.predict, shared_data, cfg)

            # ======= 3. 记录本周期预测到 pending（不写CSV） =======
            signal_tracker.record_signals(
                predictions, current_prices, ts_now, bar_timestamp=bar_timestamp
            )

            # ======= 4. 选币和生成交易动作 =======
            budget_snapshot = metrics.get_budget_snapshot(current_prices)
            budget_snapshot = self._reconcile_strategy_positions_to_account(
                model_name=name,
                timestamp=ts_now,
                bar_timestamp=bar_timestamp,
                cfg=cfg,
                executor=executor,
                metrics=metrics,
                budget_snapshot=budget_snapshot,
                actual_positions=current_positions,
                current_prices=current_prices,
                trade_log_path=trade_log_path,
                audit_paths=audit_paths,
            )
            strategy_positions = budget_snapshot.get("strategy_positions")
            positions_for_signals = (
                strategy_positions
                if isinstance(strategy_positions, dict)
                else current_positions
            )
            trade_signals = adapter.generate_signals(
                predictions,
                positions_for_signals,
                current_prices,
                cfg,
                budget_snapshot=budget_snapshot,
            )

            if trade_signals:
                logger.info(
                    f"[{name}] ====== 🚀 [交易动作] "
                    f"本轮产生执行信号 ({len(trade_signals)}笔) ======"
                )

            # ======= 5. 执行交易并跟踪成功/失败 =======
            if trade_signals and audit_paths.get("rebalance_plan"):
                append_csv_rows(
                    audit_paths["rebalance_plan"],
                    [
                        self._rebalance_plan_record(
                            model_name=name,
                            timestamp=ts_now,
                            bar_timestamp=bar_timestamp,
                            action=action,
                            current_positions=positions_for_signals,
                            current_prices=current_prices,
                            budget_snapshot=budget_snapshot,
                        )
                        for action in trade_signals
                    ],
                    ["decision_id"],
                )

            executed_trades = []
            trade_records = []
            order_records = []
            fill_records = []
            failed_order_records = []
            account_snapshot_records = []
            order_event_records = []
            ordered_trade_signals = (
                [action for action in trade_signals if action.get("side") == "sell"]
                + [action for action in trade_signals if action.get("side") != "sell"]
            )
            actual_buy_budget = max(
                0.0,
                float(budget_snapshot.get("cash_available_for_strategy", 0.0) or 0.0),
            )
            min_order_notional = float(cfg.min_position_value_usdt)
            for action in ordered_trade_signals:
                action = dict(action)
                coin = action['coin']
                side = action['side']
                amount = float(action['amount_usdt'])
                if side == "buy":
                    if actual_buy_budget < min_order_notional:
                        logger.info(
                            f"[{name}] skip buy {coin}: actual buy budget "
                            f"{actual_buy_budget:.2f}U < min_order {min_order_notional:.2f}U"
                        )
                        continue
                    if amount > actual_buy_budget:
                        logger.info(
                            f"[{name}] cap buy {coin}: planned={amount:.2f}U "
                            f"actual_budget={actual_buy_budget:.2f}U"
                        )
                        amount = actual_buy_budget
                        action["amount_usdt"] = amount
                    if amount < min_order_notional:
                        logger.info(
                            f"[{name}] skip buy {coin}: capped amount "
                            f"{amount:.2f}U < min_order {min_order_notional:.2f}U"
                        )
                        continue
                price = float(current_prices.get(coin, 0.0) or 0.0)
                if price <= 0:
                    price = executor.get_current_price(coin)

                market_type = MarketType.SWAP if executor.trade_mode == "swap" else MarketType.SPOT
                instrument_id = (
                    coin
                    if market_type is MarketType.SPOT or str(coin).upper().endswith("-SWAP")
                    else f"{coin}-SWAP"
                )
                order_request = OrderRequest(
                    strategy_id=name,
                    decision_id=(
                        f"{name}-{int(ts_now)}-{action.get('coin', '')}-"
                        f"{action.get('side', '')}"
                    ),
                    instrument_id=instrument_id,
                    market_type=market_type,
                    action=OrderAction(side),
                    notional_usdt=amount,
                    reference_price=price,
                    reason=str(action.get("reason", "")),
                    metadata={
                        "min_notional_usdt": min_order_notional,
                        "max_sell_qty": action.get("max_sell_qty") if side == "sell" else None,
                        "target_value_usdt": action.get("target_value_usdt"),
                        "target_weight": action.get("target_weight"),
                    },
                )
                trade_cycle = TradingService(executor).execute(order_request)
                result = trade_cycle.execution
                account_before = trade_cycle.account_before.raw if trade_cycle.account_before else {}
                account_after = trade_cycle.account_after.raw if trade_cycle.account_after else {}
                trade_record = self._trade_record(
                    model_name=name,
                    timestamp=ts_now,
                    bar_timestamp=bar_timestamp,
                    action=action,
                    price=price,
                    result=result,
                )
                trade_records.append(trade_record)
                context = self._audit_context(trade_record)
                order_row = result.to_order_row(context)
                order_records.append(order_row)
                fill_records.extend(result.to_fill_rows(context))
                order_event_records.extend(result.to_event_records(context))
                if not bool(result):
                    failed_order_records.append(order_row)
                account_snapshot_records.extend(
                    [
                        self._account_snapshot_record(context, "before", account_before),
                        self._account_snapshot_record(context, "after", account_after),
                    ]
                )
                if bool(result):
                    executed_action = dict(action)
                    executed_action["amount_usdt"] = float(result.filled_notional_usdt or amount or 0.0)
                    executed_trades.append(executed_action)
                    if side == "sell":
                        actual_buy_budget += float(result.filled_notional_usdt or amount or 0.0)
                    elif side == "buy":
                        actual_buy_budget = max(
                            0.0,
                            actual_buy_budget - float(result.filled_notional_usdt or amount or 0.0),
                        )

            if trade_records and trade_log_path:
                self._append_trade_records(trade_log_path, trade_records)
            if order_records and audit_paths.get("orders"):
                append_csv_rows(audit_paths["orders"], order_records, ["trade_id"])
            if fill_records and audit_paths.get("fills"):
                append_csv_rows(audit_paths["fills"], fill_records, ["trade_id", "fill_index"])
            if failed_order_records and audit_paths.get("failed_orders"):
                append_csv_rows(audit_paths["failed_orders"], failed_order_records, ["trade_id"])
            if account_snapshot_records and audit_paths.get("account_snapshots"):
                append_csv_rows(audit_paths["account_snapshots"], account_snapshot_records, ["trade_id", "snapshot_phase"])
            if order_event_records and audit_paths.get("order_events"):
                append_jsonl_events(audit_paths["order_events"], order_event_records)

            if trade_signals:
                logger.info(
                    f"[{name}] 📊 成交 {len(executed_trades)}/{len(trade_signals)} 笔"
                )
                logger.info(
                    f"[{name}] ======================================================"
                )
            else:
                logger.info(f"[{name}] 💤 [无交易] 本周期无建仓或平仓动作推荐。")

            # ======= 6. 记录交易量（仅实际成交） =======
            volume_tracker.record_trades(executed_trades, ts_now)
            volume_tracker.export_csv()

            # ======= 7. 记录 metrics =======
            metrics.record_cycle(
                ts_now, current_prices, bar_timestamp=bar_timestamp
            )
            metrics.export_snapshot()

            # ======= 8. 信号追踪: IC 报告 + CSV 落盘 =======
            signal_tracker.report_ic()          # 日志（受 report_interval 节流）
            signal_tracker.export_ic_csv(ts_now) # IC CSV（每周期写）
            signal_tracker.export_csv()          # signals CSV（原子写，含 pending）

        except Exception as e:
            logger.error(f"[{name}] 执行单周期抛出异常: {e}")
            traceback.print_exc()

    def _trade_record(
        self,
        model_name: str,
        timestamp: float,
        bar_timestamp: float,
        action: dict[str, Any],
        price: float,
        result: Any,
    ) -> dict[str, Any]:
        reference_price = float(price or 0.0)
        actual_price = float(getattr(result, "avg_fill_price", 0.0) or 0.0)
        actual_qty = float(getattr(result, "filled_qty", 0.0) or 0.0)
        actual_notional = float(getattr(result, "filled_notional_usdt", 0.0) or 0.0)
        planned_notional = float(action.get("amount_usdt", 0.0) or 0.0)
        planned_qty = planned_notional / reference_price if reference_price > 0 else 0.0
        return {
            "trade_id": f"{model_name}-{int(timestamp)}-{action.get('coin', '')}-{action.get('side', '')}",
            "timestamp": timestamp,
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
            "trade_date": time.strftime("%Y-%m-%d", time.localtime(timestamp)),
            "bar_timestamp": bar_timestamp or 0.0,
            "bar_datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(bar_timestamp)) if bar_timestamp else "",
            "strategy_id": model_name,
            "method_family": self.method_family,
            "frequency": self.timeframe,
            "coin": action.get("coin", ""),
            "side": action.get("side", ""),
            "price": actual_price or reference_price,
            "reference_price": reference_price,
            "actual_avg_price": actual_price,
            "qty": actual_qty,
            "requested_qty": planned_qty,
            "actual_qty": actual_qty,
            "notional_usdt": actual_notional,
            "requested_notional_usdt": planned_notional,
            "actual_notional_usdt": actual_notional,
            "fee_usdt": float(getattr(result, "fee_usdt", 0.0) or 0.0),
            "fee_ccy": getattr(result, "fee_ccy", ""),
            "slippage_bps": float(getattr(result, "slippage_bps", 0.0) or 0.0),
            "order_id": getattr(result, "ord_id", ""),
            "client_order_id": getattr(result, "cl_ord_id", ""),
            "order_status": getattr(result, "status", ""),
            "fill_verified": bool(getattr(result, "fill_verified", False)),
            "partial_fill": bool(getattr(result, "partial_fill", False)),
            "exchange_error_code": getattr(result, "okx_s_code", "") or getattr(result, "okx_code", ""),
            "exchange_error_msg": getattr(result, "okx_s_msg", "") or getattr(result, "okx_msg", ""),
            "target_value_usdt": float(action.get("target_value_usdt", 0.0) or 0.0),
            "target_weight": float(action.get("target_weight", 0.0) or 0.0),
            "reason": action.get("reason", ""),
            "success": bool(result),
        }

    def _reconcile_strategy_positions_to_account(
        self,
        model_name: str,
        timestamp: float,
        bar_timestamp: float,
        cfg: ConfigManager,
        executor: TradeExecutor,
        metrics: MetricsTracker,
        budget_snapshot: dict[str, Any],
        actual_positions: dict[str, float],
        current_prices: dict[str, float],
        trade_log_path: str,
        audit_paths: dict[str, str],
    ) -> dict[str, Any]:
        mode = getattr(cfg, "account_position_reconcile_mode", "remove_unbacked")
        if mode == "off":
            return budget_snapshot
        if budget_snapshot.get("strategy_accounting_mode") != "trade_ledger":
            return budget_snapshot

        strategy_positions = budget_snapshot.get("strategy_positions") or {}
        if not isinstance(strategy_positions, dict):
            strategy_positions = {}

        reconcile_coins = set(strategy_positions)
        if mode == "full_account":
            trade_coins = set(getattr(cfg, "coins", []) or [])
            reconcile_coins.update(
                coin for coin in actual_positions if not trade_coins or coin in trade_coins
            )
        if not reconcile_coins:
            return budget_snapshot

        min_order_notional = float(getattr(cfg, "min_position_value_usdt", 0.0) or 0.0)
        if min_order_notional <= 0:
            return budget_snapshot

        trade_rows = []
        order_rows = []
        for coin in sorted(reconcile_coins):
            raw_qty = strategy_positions.get(coin, 0.0)
            local_qty = float(raw_qty or 0.0)
            price = float(current_prices.get(coin, 0.0) or 0.0)
            if price <= 0:
                price = executor.get_current_price(coin)
                if price > 0:
                    current_prices[coin] = price
            if price <= 0:
                continue

            actual_qty = max(0.0, float(actual_positions.get(coin, 0.0) or 0.0))
            local_value = local_qty * price
            actual_value = actual_qty * price
            target_qty: float | None = None
            reason = ""

            if local_qty < 0 and abs(local_value) >= min_order_notional:
                target_qty = 0.0
                reason = "ledger_negative_position_reset"
            elif local_qty > 0:
                missing_value = (local_qty - actual_qty) * price
                if missing_value >= min_order_notional:
                    target_qty = actual_qty if actual_value >= min_order_notional else 0.0
                    reason = "ledger_position_reconcile_actual_lower"
                elif mode == "full_account":
                    extra_value = (actual_qty - local_qty) * price
                    if extra_value >= min_order_notional:
                        target_qty = actual_qty
                        reason = "ledger_position_reconcile_actual_higher"
            elif mode == "full_account" and actual_value >= min_order_notional:
                target_qty = actual_qty
                reason = "ledger_position_reconcile_actual_untracked"

            if target_qty is None:
                continue

            delta_qty = target_qty - local_qty
            if abs(delta_qty) * price < min_order_notional:
                continue

            trade_id = f"{model_name}-{int(timestamp)}-{coin}-ledger-reconcile"
            row = self._ledger_reconcile_trade_record(
                trade_id=trade_id,
                model_name=model_name,
                timestamp=timestamp,
                bar_timestamp=bar_timestamp,
                coin=coin,
                delta_qty=delta_qty,
                price=price,
                local_value=local_value,
                local_qty=local_qty,
                target_qty=target_qty,
                actual_qty=actual_qty,
                actual_value=actual_value,
                reason=reason,
            )
            trade_rows.append(row)
            order_rows.append(self._ledger_reconcile_order_record(row, cfg))
            logger.warning(
                "[%s] ledger/account reconcile %s: local=%.8f (~%.2fU), "
                "actual=%.8f (~%.2fU), target=%.8f, delta=%.8f; reason=%s",
                model_name,
                coin,
                local_qty,
                local_value,
                actual_qty,
                actual_value,
                target_qty,
                delta_qty,
                reason,
            )

        if not trade_rows:
            return budget_snapshot

        if trade_log_path:
            self._append_trade_records(trade_log_path, trade_rows)
        if audit_paths.get("orders"):
            append_csv_rows(audit_paths["orders"], order_rows, ["trade_id"])
        return metrics.get_budget_snapshot(current_prices)

    def _ledger_reconcile_trade_record(
        self,
        trade_id: str,
        model_name: str,
        timestamp: float,
        bar_timestamp: float,
        coin: str,
        delta_qty: float,
        price: float,
        local_value: float,
        local_qty: float,
        target_qty: float,
        actual_qty: float,
        actual_value: float,
        reason: str,
    ) -> dict[str, Any]:
        qty = abs(delta_qty)
        side = "buy" if delta_qty > 0 else "sell"
        requested_notional = qty * price
        cash_delta = (
            -requested_notional
            if reason in {
                "ledger_position_reconcile_actual_higher",
                "ledger_position_reconcile_actual_untracked",
            }
            else 0.0
        )
        return {
            "trade_id": trade_id,
            "timestamp": timestamp,
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
            "trade_date": time.strftime("%Y-%m-%d", time.localtime(timestamp)),
            "bar_timestamp": bar_timestamp or 0.0,
            "bar_datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(bar_timestamp)) if bar_timestamp else "",
            "strategy_id": model_name,
            "method_family": self.method_family,
            "frequency": self.timeframe,
            "coin": coin,
            "side": side,
            "price": price,
            "reference_price": price,
            "actual_avg_price": price,
            "qty": qty,
            "requested_qty": qty,
            "actual_qty": qty,
            "notional_usdt": 0.0,
            "requested_notional_usdt": requested_notional,
            "actual_notional_usdt": 0.0,
            "fee_usdt": 0.0,
            "fee_ccy": "",
            "slippage_bps": 0.0,
            "order_id": "",
            "client_order_id": "",
            "order_status": "synthetic",
            "fill_verified": False,
            "partial_fill": False,
            "exchange_error_code": "",
            "exchange_error_msg": (
                "ledger/account reconcile: "
                f"local_qty={local_qty:.12g}, target_qty={target_qty:.12g}, "
                f"actual_qty={actual_qty:.12g}, actual_value_usdt={actual_value:.4f}"
            ),
            "target_value_usdt": 0.0,
            "target_weight": 0.0,
            "reason": reason,
            "success": True,
            "ledger_position_delta_qty": delta_qty,
            "ledger_cash_delta_usdt": cash_delta,
        }

    @staticmethod
    def _ledger_reconcile_order_record(trade_row: dict[str, Any], cfg: ConfigManager) -> dict[str, Any]:
        row = {
            key: trade_row.get(key, "")
            for key in [
                "trade_id",
                "timestamp",
                "datetime",
                "trade_date",
                "bar_timestamp",
                "bar_datetime",
                "strategy_id",
                "method_family",
                "frequency",
                "coin",
            ]
        }
        row.update(
            {
                "exchange": "local_ledger",
                "account_mode": "ledger",
                "trade_mode": getattr(cfg, "trade_mode", "spot"),
                "inst_id": trade_row.get("coin", ""),
                "side": trade_row.get("side", ""),
                "ord_type": "synthetic",
                "td_mode": "",
                "tgt_ccy": "base_ccy",
                "requested_notional_usdt": trade_row.get("requested_notional_usdt", 0.0),
                "requested_qty": trade_row.get("requested_qty", 0.0),
                "submitted_sz": f"{float(trade_row.get('requested_qty', 0.0) or 0.0):.8f}".rstrip("0").rstrip("."),
                "reference_price": trade_row.get("reference_price", 0.0),
                "submit_ts": trade_row.get("timestamp", 0.0),
                "ack_ts": trade_row.get("timestamp", 0.0),
                "settle_ts": trade_row.get("timestamp", 0.0),
                "order_id": "",
                "client_order_id": "",
                "order_status": "synthetic",
                "okx_code": "",
                "okx_msg": "",
                "okx_s_code": "",
                "okx_s_msg": "",
                "avg_fill_price": trade_row.get("actual_avg_price", 0.0),
                "filled_qty": trade_row.get("actual_qty", 0.0),
                "filled_notional_usdt": 0.0,
                "fee_usdt": 0.0,
                "fee_ccy": "",
                "slippage_bps": 0.0,
                "fill_verified": False,
                "partial_fill": False,
                "success": True,
                "error_type": "",
                "error_message": trade_row.get("exchange_error_msg", ""),
                "ledger_position_delta_qty": trade_row.get("ledger_position_delta_qty", 0.0),
                "ledger_cash_delta_usdt": trade_row.get("ledger_cash_delta_usdt", 0.0),
            }
        )
        return row

    def _rebalance_plan_record(
        self,
        model_name: str,
        timestamp: float,
        bar_timestamp: float,
        action: dict[str, Any],
        current_positions: dict[str, float],
        current_prices: dict[str, float],
        budget_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        coin = action.get("coin", "")
        price = float(current_prices.get(coin, 0.0) or 0.0)
        current_qty = float(current_positions.get(coin, 0.0) or 0.0)
        current_value = current_qty * price if price > 0 else 0.0
        target_value = float(action.get("target_value_usdt", 0.0) or 0.0)
        amount = float(action.get("amount_usdt", 0.0) or 0.0)
        return {
            "decision_id": f"{model_name}-{int(timestamp)}-{coin}-{action.get('side', '')}",
            "timestamp": timestamp,
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
            "bar_timestamp": bar_timestamp or 0.0,
            "bar_datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(bar_timestamp)) if bar_timestamp else "",
            "strategy_id": model_name,
            "method_family": self.method_family,
            "frequency": self.timeframe,
            "coin": coin,
            "side": action.get("side", ""),
            "reference_price": price,
            "current_qty": current_qty,
            "current_value_usdt": current_value,
            "target_value_usdt": target_value,
            "target_weight": float(action.get("target_weight", 0.0) or 0.0),
            "planned_delta_usdt": amount,
            "strategy_equity": float(budget_snapshot.get("strategy_equity", 0.0) or 0.0),
            "cash_available_for_strategy": float(budget_snapshot.get("cash_available_for_strategy", 0.0) or 0.0),
            "reason": action.get("reason", ""),
            "will_submit_order": amount > 0,
        }

    @staticmethod
    def _audit_context(trade_record: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "trade_id",
            "timestamp",
            "datetime",
            "trade_date",
            "bar_timestamp",
            "bar_datetime",
            "strategy_id",
            "method_family",
            "frequency",
            "coin",
        ]
        return {key: trade_record.get(key, "") for key in keys}

    @staticmethod
    def _account_snapshot_record(context: dict[str, Any], phase: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        row = dict(context)
        row.update(
            {
                "snapshot_phase": phase,
                "account_mode": snapshot.get("account_mode", ""),
                "trade_mode": snapshot.get("trade_mode", ""),
                "usdt_available": snapshot.get("usdt_available", 0.0),
                "holdings_value_usdt": snapshot.get("holdings_value_usdt", 0.0),
                "estimated_total_equity_usdt": snapshot.get("estimated_total_equity_usdt", 0.0),
                "holdings_json": json.dumps(snapshot.get("holdings_json", {}), ensure_ascii=False, sort_keys=True),
            }
        )
        return row

    @staticmethod
    def _append_trade_records(path: str, rows: list[dict[str, Any]]) -> None:
        old = safe_read_csv(path)
        combined = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if not old.empty else pd.DataFrame(rows)
        if "trade_id" in combined:
            combined = combined.drop_duplicates(["trade_id"], keep="last")
        atomic_write_csv(combined, path)

    async def run_single_cycle(self):
        """同时启动该周期内所有的注册模型并行预测和交易"""
        ts_now = time.time()

        # 通知数据管理器这是新的一个周期，旧缓存清空
        self.data_manager.clear_cycle_cache()

        # 并发启动所有的模型管线
        tasks = []
        for p in self.models:
            tasks.append(asyncio.create_task(self._run_model_cycle(p, ts_now)))

        await asyncio.gather(*tasks)

    # ==========================================================
    # 配置热加载
    # ==========================================================

    def _reload_all_configs(self):
        """重新从磁盘加载所有已注册模型的配置文件"""
        for model_pack in self.models:
            cfg: ConfigManager = model_pack['config']
            cfg.reload()

    # ==========================================================
    # K线收盘边界调度
    # ==========================================================

    def _timeframe_to_seconds(self) -> int:
        tf = str(self.timeframe).strip().lower()
        if not tf:
            return int(self.update_interval_sec)
        unit = tf[-1]
        try:
            value = int(tf[:-1])
        except ValueError:
            logger.warning(
                f"[{self.timeframe}] 无法解析 timeframe，退回 update_interval_sec"
            )
            return int(self.update_interval_sec)
        if unit == "m":
            return value * 60
        if unit == "h":
            return value * 60 * 60
        if unit == "d":
            return value * 24 * 60 * 60
        return int(self.update_interval_sec)

    def _close_delay_sec(self) -> float:
        delays = []
        for model_pack in self.models:
            cfg: ConfigManager = model_pack['config']
            delays.append(cfg.scheduler_close_delay_sec)
        return max(delays) if delays else 8.0

    def _latest_closed_bar_start(self, now: float) -> float:
        """
        返回当前时刻已确认应该可用的最新已收盘K线起始时间戳。

        例：1h 周期在 13:00:08 后，最新已收盘K线起点是 12:00:00。
        """
        interval = self._timeframe_to_seconds()
        delay = self._close_delay_sec()
        settled_now = max(0.0, now - delay)
        current_boundary = int(settled_now // interval) * interval
        return max(0.0, current_boundary - interval)

    def _next_close_run_time(self, now: float) -> float:
        interval = self._timeframe_to_seconds()
        delay = self._close_delay_sec()
        next_boundary = (int(now // interval) + 1) * interval
        return next_boundary + delay

    def _is_just_after_close(self, now: float, bar_start: float) -> bool:
        if bar_start <= 0:
            return False
        interval = self._timeframe_to_seconds()
        close_time = bar_start + interval
        grace = max(60.0, self._close_delay_sec() * 4)
        return 0 <= (now - close_time) <= grace

    def _get_last_processed_bar_timestamp(self) -> float:
        bar_timestamps = []
        for model_pack in self.models:
            metrics: MetricsTracker = model_pack['metrics']
            ts = metrics.get_last_bar_timestamp()
            if ts <= 0:
                return 0.0
            bar_timestamps.append(ts)
        # 多模型时用最小值：只要有一个模型还没处理最新K线，就允许本轮启动。
        return min(bar_timestamps) if bar_timestamps else 0.0

    def _has_processed_bar_history(self) -> bool:
        return any(
            model_pack['metrics'].get_last_bar_timestamp() > 0
            for model_pack in self.models
        )

    def _sleep_with_reload(self, sleep_time: float):
        """睡眠期间保留原先“提前60秒热加载配置”的行为。"""
        sleep_time = max(0.0, sleep_time)
        if sleep_time > 60:
            time.sleep(sleep_time - 60)
            self._reload_all_configs()
            time.sleep(60)
        else:
            time.sleep(sleep_time)

    # ==========================================================
    # 主循环
    # ==========================================================

    def start(self):
        """阻塞当前进程，卡死在时间周期调度里"""
        logger.info("🚀 ==========================================")
        logger.info(f"🚀 [{self.timeframe}] 开始自动并发跟单主循环")
        logger.info(
            f"🚀 [登记在册模型数：{len(self.models)}，"
            f" 唤醒频率：{self.update_interval_sec}秒]"
        )
        logger.info("🚀 ==========================================")

        # 初始化事件循环
        loop = asyncio.get_event_loop()

        while True:
            now = time.time()
            latest_closed_bar = self._latest_closed_bar_start(now)
            last_processed_bar = self._get_last_processed_bar_timestamp()
            has_history = self._has_processed_bar_history()
            in_close_window = self._is_just_after_close(now, latest_closed_bar)
            should_run_now = (
                latest_closed_bar > last_processed_bar
                and in_close_window
            )

            if not should_run_now:
                run_at = self._next_close_run_time(now)
                sleep_time = max(0.0, run_at - now)
                if has_history and latest_closed_bar > last_processed_bar and not in_close_window:
                    logger.warning(
                        f"⏭️ [{self.timeframe}] 最新已收盘K线 "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_closed_bar))} "
                        "已错过刚收盘交易窗口，本轮不补交易，等待下一根K线。"
                    )
                logger.info(
                    f"💤 [{self.timeframe}] 等待下一根K线收盘确认，"
                    f"将在 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run_at))} "
                    f"唤醒，剩余 {sleep_time:.0f} 秒。"
                )
                self._sleep_with_reload(sleep_time)
                continue

            cycle_start = time.time()
            before_bar = self._get_last_processed_bar_timestamp()

            # ★ 每周期开始前热加载配置
            self._reload_all_configs()

            # 执行周期
            loop.run_until_complete(self.run_single_cycle())

            cycle_end = time.time()
            elapsed = cycle_end - cycle_start
            after_bar = self._get_last_processed_bar_timestamp()

            logger.info(
                f"✅ [{self.timeframe}] 本周期耗时 {elapsed:.2f} s，"
                f"已处理K线进度: {after_bar:.0f}"
            )

            if after_bar <= before_bar and latest_closed_bar > before_bar:
                retry_sleep = max(5.0, min(30.0, self._close_delay_sec() * 2))
                logger.warning(
                    f"⏳ [{self.timeframe}] 本轮未推进到最新收盘K线，"
                    f"{retry_sleep:.0f} 秒后重试，避免交易所K线确认延迟。"
                )
                self._sleep_with_reload(retry_sleep)
