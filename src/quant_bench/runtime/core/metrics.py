import json
import logging
import math
import os
import time
from datetime import datetime
from typing import Any

import pandas as pd

from quant_bench.contracts.csv import RUNTIME_METRICS_V1
from quant_bench.runtime.core.execution import TradeExecutor

logger = logging.getLogger(__name__)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
        return default if math.isnan(out) or math.isinf(out) else out
    except (TypeError, ValueError):
        return default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _safe_read_csv(path: str) -> pd.DataFrame | None:
    """安全读取CSV，损坏或不存在时返回 None"""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception as e:
        logger.warning(f"CSV 读取失败（{path}），从空状态启动: {e}")
        # 备份损坏文件以便排查
        backup = path + f'.corrupted.{int(time.time())}'
        try:
            os.rename(path, backup)
            logger.info(f"已备份损坏文件到: {backup}")
        except OSError:
            pass
        return None


def _atomic_write_csv(df: pd.DataFrame, target_path: str):
    """先写临时文件，再原子替换，避免崩溃导致文件损坏"""
    tmp_path = target_path + '.tmp'
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, target_path)


class MetricsTracker:
    """
    负责记录某个模型的交易表现（本金、盈亏、回撤等），
    同时管理一个特权对齐的"全仓BTC买入并持有"基准线。
    支持跨重启的 CSV 续写和 PnL 连续性（通过 _metadata.json 持久化基准）。
    """
    def __init__(
        self,
        identifier: str,
        timeframe: str,
        api_client: TradeExecutor,
        log_dir: str,
        initial_capital_usdt: float | None = None,
        method_family: str = "traditional_ml",
    ):
        self.identifier = identifier
        self.timeframe = timeframe
        self.method_family = method_family
        self.api = api_client  # 用于查询持仓和算账

        self.log_dir = os.path.join(log_dir, identifier)
        os.makedirs(self.log_dir, exist_ok=True)

        self._csv_path = os.path.join(
            self.log_dir, f"{identifier}_{timeframe}_metrics.csv"
        )
        self._meta_path = os.path.join(
            self.log_dir, f"{identifier}_metadata.json"
        )
        self._trade_log_path = os.path.join(
            self.log_dir, f"{identifier}_{timeframe}_trades.csv"
        )

        # 1. 从 CSV 恢复历史记录
        self.history_records: list[dict] = []
        self._load_existing_csv()

        # 2. 从元数据恢复 initial_usdt & baseline，保证跨重启连续
        meta = self._load_or_create_metadata(initial_capital_usdt)
        self.initial_usdt = meta['initial_usdt']
        self.external_cash_buffer_usdt = meta.get('external_cash_buffer_usdt', 0.0)
        self.capital_mode = meta.get('capital_mode', 'account_equity')
        self.baseline_btc_qty = meta['baseline_btc_qty']
        self.btc_instId = meta['btc_instId']

        if self.initial_usdt <= 0:
            logger.warning(
                f"[{self.identifier}] 初始总资产为 {self.initial_usdt:.4f}，"
                "收益率将启用除零保护并在后续周期尝试自动重置基准。"
            )

    # ==========================================================
    # 启动时恢复
    # ==========================================================

    def _load_existing_csv(self):
        """启动时从磁盘恢复历史 metrics 记录"""
        df = _safe_read_csv(self._csv_path)
        if df is not None:
            self.history_records = df.to_dict('records')
            logger.info(
                f"[{self.identifier}] 📂 从 CSV 恢复 {len(self.history_records)} 条 metrics 记录"
            )

    def _write_metadata(self, meta: dict[str, Any]):
        with open(self._meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def _build_metadata(
        self,
        initial_usdt: float,
        account_total_equity: float,
        capital_mode: str,
        previous_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """构造资金池/基准元数据。"""
        btc_instId = "BTC-USDT-SWAP" if self.api.trade_mode == "swap" else "BTC-USDT"
        btc_price = self.api.get_current_price(btc_instId)
        baseline_btc_qty = (initial_usdt / btc_price * 0.999) if btc_price > 0 else 0.0

        meta: dict[str, Any] = {
            'initial_usdt': initial_usdt,
            'capital_base_usdt': initial_usdt,
            'external_cash_buffer_usdt': max(0.0, account_total_equity - initial_usdt),
            'account_total_equity_at_start': account_total_equity,
            'capital_mode': capital_mode,
            'strategy_accounting_mode': (
                'trade_ledger' if capital_mode == 'configured_budget' else 'account_equity'
            ),
            'baseline_btc_qty': baseline_btc_qty,
            'btc_instId': btc_instId,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if previous_meta:
            meta['migrated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            meta['previous_initial_usdt'] = previous_meta.get('initial_usdt')
            meta['previous_capital_mode'] = previous_meta.get('capital_mode')
        return meta

    def _load_or_create_metadata(self, configured_initial_usdt: float | None) -> dict:
        """加载元数据；首次运行时计算并保存"""
        if os.path.exists(self._meta_path):
            try:
                with open(self._meta_path) as f:
                    meta = json.load(f)
                # 防止加载到 initial_usdt=0 的损坏元数据
                if meta.get('initial_usdt', 0) > 0:
                    if (
                        configured_initial_usdt is not None
                        and meta.get('capital_mode') != 'configured_budget'
                    ):
                        account_total = self._calculate_total_equity()
                        new_meta = self._build_metadata(
                            initial_usdt=configured_initial_usdt,
                            account_total_equity=account_total,
                            capital_mode='configured_budget',
                            previous_meta=meta,
                        )
                        self._write_metadata(new_meta)
                        logger.warning(
                            f"[{self.identifier}] 检测到新的 initial_capital_usdt="
                            f"{configured_initial_usdt:.2f}，已将统计基准迁移为资金池模式；"
                            f"外部资金缓冲={new_meta['external_cash_buffer_usdt']:.2f}"
                        )
                        return new_meta

                    changed = False
                    if 'external_cash_buffer_usdt' not in meta:
                        meta['external_cash_buffer_usdt'] = 0.0
                        changed = True
                    if 'capital_base_usdt' not in meta:
                        meta['capital_base_usdt'] = meta['initial_usdt']
                        changed = True
                    if 'capital_mode' not in meta:
                        meta['capital_mode'] = 'account_equity'
                        changed = True
                    expected_accounting_mode = (
                        'trade_ledger'
                        if meta.get('capital_mode') == 'configured_budget'
                        else 'account_equity'
                    )
                    if meta.get('strategy_accounting_mode') != expected_accounting_mode:
                        meta['strategy_accounting_mode'] = expected_accounting_mode
                        changed = True
                    if changed:
                        self._write_metadata(meta)

                    if (
                        configured_initial_usdt is not None
                        and meta.get('capital_mode') == 'configured_budget'
                        and abs(meta.get('initial_usdt', 0.0) - configured_initial_usdt) > 0.01
                    ):
                        logger.warning(
                            f"[{self.identifier}] metadata 已有资金池本金 "
                            f"{meta['initial_usdt']:.2f}，本次配置 "
                            f"{configured_initial_usdt:.2f} 不会覆盖，以保证重启连续性。"
                        )
                    logger.info(
                        f"[{self.identifier}] 📂 恢复基准: "
                        f"initial_usdt={meta['initial_usdt']:.2f}, "
                        f"external_buffer={meta.get('external_cash_buffer_usdt', 0.0):.2f}, "
                        f"accounting={meta.get('strategy_accounting_mode', 'account_equity')}, "
                        f"baseline_btc_qty={meta['baseline_btc_qty']:.6f}"
                    )
                    return meta
                else:
                    logger.warning(
                        f"[{self.identifier}] 元数据中 initial_usdt=0，重新计算"
                    )
            except Exception as e:
                logger.warning(f"[{self.identifier}] 元数据读取失败，重新创建: {e}")

        # 首次启动：计算并保存。若配置了 initial_capital_usdt，则只把该资金池作为本金。
        account_total_equity = self._calculate_total_equity()
        initial_usdt = (
            configured_initial_usdt
            if configured_initial_usdt is not None
            else account_total_equity
        )
        capital_mode = (
            'configured_budget'
            if configured_initial_usdt is not None
            else 'account_equity'
        )
        meta = self._build_metadata(initial_usdt, account_total_equity, capital_mode)

        # 只有 initial_usdt > 0 时才持久化（防止 API 首次失败时存入 0）
        if initial_usdt > 0:
            self._write_metadata(meta)
            logger.info(
                f"[{self.identifier}] 📈 基准线已建立并保存: "
                f"initial_usdt={initial_usdt:.2f}, "
                f"external_buffer={meta['external_cash_buffer_usdt']:.2f}, "
                f"baseline_btc_qty={meta['baseline_btc_qty']:.6f} BTC"
            )
        else:
            logger.warning(
                f"[{self.identifier}] initial_usdt=0, 暂不保存 metadata，等首周期修复"
            )

        return meta

    def _calculate_total_equity(self) -> float:
        """计算当前账户的总资产 = USDT现金 + 所有持仓的市值"""
        usdt_cash = self.api.get_usdt_balance()
        holdings = self.api.get_holdings()

        holdings_value = 0.0
        for coin, qty in holdings.items():
            if qty > 0:
                price = self.api.get_current_price(coin)
                holdings_value += qty * price

        total = usdt_cash + holdings_value
        logger.info(
            f"[{self.identifier}] 📊 启动总资产快照: "
            f"USDT现金={usdt_cash:.2f}, 持仓市值={holdings_value:.2f}, 总计={total:.2f}"
        )
        return total

    def _initialize_baseline(self):
        """重新计算 BTC 基准线（用于延迟初始化场景）"""
        current_btc_price = self.api.get_current_price(self.btc_instId)
        if current_btc_price > 0:
            self.baseline_btc_qty = (self.initial_usdt / current_btc_price) * (1 - 0.001)
            logger.info(
                f"[{self.identifier}] 📈 重建 BTC 基准线: "
                f"虚拟买入 {self.baseline_btc_qty:.6f} 个 BTC @ {current_btc_price}"
            )

    # ==========================================================
    # 运行时
    # ==========================================================

    def _safe_returns_pct(self, equity: float) -> float:
        """避免 initial_usdt 为 0 时发生除零"""
        if self.initial_usdt <= 0:
            return 0.0
        return (equity / self.initial_usdt - 1) * 100

    def _account_portfolio_snapshot(self, current_prices: dict[str, float]) -> dict[str, Any]:
        """计算 OKX 账户整体快照。"""
        current_usdt = self.api.get_usdt_balance()
        holdings = self.api.get_holdings()

        holdings_value = 0.0
        priced_holdings: dict[str, float] = {}
        for coin, qty in holdings.items():
            if qty <= 0:
                continue
            price = current_prices.get(coin, 0.0)
            if price <= 0:
                price = self.api.get_current_price(coin)
            value = qty * price if price > 0 else 0.0
            holdings_value += value
            priced_holdings[coin] = value

        total_equity = current_usdt + holdings_value
        max(0.0, total_equity - self.external_cash_buffer_usdt)
        max(
            0.0, current_usdt - self.external_cash_buffer_usdt
        )

        return {
            'cash_total_usdt': current_usdt,
            'holdings_value_usdt': holdings_value,
            'total_equity_usdt': total_equity,
            'external_cash_buffer_usdt': self.external_cash_buffer_usdt,
            'priced_holdings_usdt': priced_holdings,
            'positions': holdings,
        }

    def _load_trade_ledger_rows(self) -> list[dict[str, Any]]:
        df = _safe_read_csv(self._trade_log_path)
        if df is None:
            return []
        if 'success' in df.columns:
            df = df[df['success'].map(_truthy)]
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        return df.to_dict('records')

    @staticmethod
    def _trade_notional(row: dict[str, Any], price: float, qty: float) -> float:
        actual_notional = _to_float(row.get('actual_notional_usdt'), 0.0)
        if actual_notional > 0:
            return actual_notional
        if 'actual_notional_usdt' in row:
            return price * qty if price > 0 and qty > 0 else 0.0
        for key in ('notional_usdt', 'requested_notional_usdt'):
            value = _to_float(row.get(key), 0.0)
            if value > 0:
                return value
        return price * qty if price > 0 and qty > 0 else 0.0

    @staticmethod
    def _trade_qty(row: dict[str, Any], price: float) -> float:
        actual_qty = _to_float(row.get('actual_qty'), 0.0)
        if actual_qty > 0:
            return actual_qty
        if 'actual_qty' in row:
            notional = _to_float(row.get('actual_notional_usdt'), 0.0)
            return notional / price if price > 0 and notional > 0 else 0.0
        for key in ('qty', 'requested_qty'):
            value = _to_float(row.get(key), 0.0)
            if value > 0:
                return value
        notional = _to_float(row.get('notional_usdt'), 0.0)
        return notional / price if price > 0 and notional > 0 else 0.0

    def _strategy_ledger_snapshot(self, current_prices: dict[str, float]) -> dict[str, Any]:
        """
        用本策略自己的成交台账核算资金池。

        共享 OKX demo 账户里，账户总权益和现货仓位会被其它策略/手工资金污染。
        配置了 initial_capital_usdt 后，策略 PnL 必须从自己的成交记录推导。
        """
        cash = float(self.initial_usdt or 0.0)
        positions: dict[str, float] = {}

        for row in self._load_trade_ledger_rows():
            coin = str(row.get('coin', '') or '')
            if not coin:
                continue
            ledger_position_delta = _to_float(row.get('ledger_position_delta_qty'), 0.0)
            ledger_cash_delta = _to_float(row.get('ledger_cash_delta_usdt'), 0.0)
            if ledger_position_delta or ledger_cash_delta:
                positions[coin] = positions.get(coin, 0.0) + ledger_position_delta
                cash += ledger_cash_delta
                continue

            side = str(row.get('side', '') or '').lower()
            price = _to_float(row.get('actual_avg_price'), 0.0) or _to_float(row.get('price'), 0.0)
            qty = self._trade_qty(row, price)
            notional = self._trade_notional(row, price, qty)
            if qty <= 0 or notional <= 0:
                continue
            fee_usdt = _to_float(row.get('fee_usdt'), 0.0)
            fee_ccy = str(row.get('fee_ccy', '') or '').upper()
            base_ccy = coin.split('-')[0].upper()
            fee_qty = fee_usdt / price if price > 0 and fee_usdt > 0 else 0.0

            if side == 'buy':
                cash -= notional
                net_qty = qty
                if fee_ccy == 'USDT':
                    cash -= fee_usdt
                elif fee_ccy == base_ccy:
                    net_qty = max(0.0, qty - fee_qty)
                positions[coin] = positions.get(coin, 0.0) + net_qty
            elif side == 'sell':
                cash += notional
                sold_qty = qty
                if fee_ccy == 'USDT':
                    cash -= fee_usdt
                elif fee_ccy == base_ccy:
                    sold_qty += fee_qty
                positions[coin] = positions.get(coin, 0.0) - sold_qty

        positions = {
            coin: qty
            for coin, qty in positions.items()
            if abs(qty) > 1e-12
        }
        holdings_value = 0.0
        priced_holdings: dict[str, float] = {}
        strategy_prices: dict[str, float] = {}
        for coin, qty in positions.items():
            if qty <= 0:
                continue
            price = current_prices.get(coin, 0.0)
            if price <= 0:
                price = self.api.get_current_price(coin)
            if price <= 0:
                continue
            strategy_prices[coin] = price
            value = qty * price
            holdings_value += value
            priced_holdings[coin] = value

        total_equity = cash + holdings_value
        return {
            'strategy_cash_ledger': cash,
            'strategy_holdings_value': holdings_value,
            'strategy_equity': total_equity,
            'strategy_positions': positions,
            'strategy_prices': strategy_prices,
            'priced_holdings_usdt': priced_holdings,
        }

    def _portfolio_snapshot(self, current_prices: dict[str, float]) -> dict[str, Any]:
        """
        计算当前快照，并拆分出策略资金池权益。

        旧的 external buffer 只适合单策略独占账户。资金池模式下改用本策略
        trades.csv 台账，避免共享账户中其它资金的波动被计入本策略 PnL。
        """
        account_snapshot = self._account_portfolio_snapshot(current_prices)

        if self.capital_mode == 'configured_budget':
            ledger = self._strategy_ledger_snapshot(current_prices)
            strategy_cash = ledger['strategy_cash_ledger']
            cash_available_for_strategy = min(
                max(0.0, strategy_cash),
                max(0.0, account_snapshot['cash_total_usdt']),
            )
            return {
                'cash_total_usdt': account_snapshot['cash_total_usdt'],
                'account_holdings_value_usdt': account_snapshot['holdings_value_usdt'],
                'total_equity_usdt': account_snapshot['total_equity_usdt'],
                'strategy_cash_ledger': strategy_cash,
                'holdings_value_usdt': ledger['strategy_holdings_value'],
                'strategy_equity': ledger['strategy_equity'],
                'cash_available_for_strategy': cash_available_for_strategy,
                'external_cash_buffer_usdt': self.external_cash_buffer_usdt,
                'priced_holdings_usdt': ledger['priced_holdings_usdt'],
                'strategy_positions': ledger['strategy_positions'],
                'strategy_prices': ledger['strategy_prices'],
                'strategy_accounting_mode': 'trade_ledger',
            }

        current_usdt = account_snapshot['cash_total_usdt']
        total_equity = account_snapshot['total_equity_usdt']
        strategy_equity = max(0.0, total_equity - self.external_cash_buffer_usdt)
        cash_available_for_strategy = max(
            0.0, current_usdt - self.external_cash_buffer_usdt
        )
        return {
            'cash_total_usdt': current_usdt,
            'account_holdings_value_usdt': account_snapshot['holdings_value_usdt'],
            'holdings_value_usdt': account_snapshot['holdings_value_usdt'],
            'total_equity_usdt': total_equity,
            'strategy_equity': strategy_equity,
            'cash_available_for_strategy': cash_available_for_strategy,
            'external_cash_buffer_usdt': self.external_cash_buffer_usdt,
            'priced_holdings_usdt': account_snapshot['priced_holdings_usdt'],
            'strategy_positions': account_snapshot['positions'],
            'strategy_prices': {},
            'strategy_accounting_mode': 'account_equity',
        }

    def get_budget_snapshot(self, current_prices: dict[str, float]) -> dict[str, Any]:
        """供策略层读取资金池权益和可用现金。"""
        return self._portfolio_snapshot(current_prices)

    def get_last_bar_timestamp(self) -> float:
        """返回最近一次已处理的K线起始时间戳，用于重启后避免重复交易。"""
        for record in reversed(self.history_records):
            try:
                bar_ts = float(record.get('bar_timestamp', 0.0))
            except (TypeError, ValueError):
                bar_ts = 0.0
            if bar_ts > 0:
                return bar_ts
        return 0.0

    def _refresh_initial_usdt_if_needed(self, current_usdt: float):
        """
        若初始化时本金未正确获取（=0），在后续周期发现有效余额时自动修复基准，
        并立即持久化到 metadata.json。
        """
        if self.initial_usdt > 0 or current_usdt <= 0:
            return
        self.initial_usdt = self._calculate_total_equity()
        if self.initial_usdt <= 0:
            self.initial_usdt = current_usdt  # 兜底：至少用 USDT 现金
        self._initialize_baseline()

        # 修复后立即持久化
        meta = {
            'initial_usdt': self.initial_usdt,
            'capital_base_usdt': self.initial_usdt,
            'external_cash_buffer_usdt': self.external_cash_buffer_usdt,
            'account_total_equity_at_start': self.initial_usdt,
            'capital_mode': self.capital_mode,
            'strategy_accounting_mode': (
                'trade_ledger' if self.capital_mode == 'configured_budget' else 'account_equity'
            ),
            'baseline_btc_qty': self.baseline_btc_qty,
            'btc_instId': self.btc_instId,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self._write_metadata(meta)

        logger.warning(
            f"[{self.identifier}] 检测到有效余额，"
            f"已自动重置 initial_usdt={self.initial_usdt:.2f} 与 BTC 基准线起点，"
            f"并保存到 metadata.json"
        )

    def record_cycle(
        self,
        timestamp: float,
        current_prices: dict[str, float],
        bar_timestamp: float | None = None,
    ):
        """在每一轮交易周期末尾收集并快照整个账户状态"""
        # ====================
        # 1. 计算模型策略资金池的总权益
        # ====================
        snapshot = self._portfolio_snapshot(current_prices)
        current_usdt = snapshot['cash_total_usdt']
        self._refresh_initial_usdt_if_needed(current_usdt)
        holdings = snapshot.get('strategy_positions') or {}

        strategy_equity = snapshot['strategy_equity']
        strategy_pnl = strategy_equity - self.initial_usdt

        # ====================
        # 2. 计算同期全仓 BTC 策略的总权益 (Benchmark)
        # ====================
        btc_price_now = current_prices.get(self.btc_instId)
        if not btc_price_now:
            btc_price_now = self.api.get_current_price(self.btc_instId)

        if btc_price_now > 0 and self.baseline_btc_qty > 0:
            baseline_equity = self.baseline_btc_qty * btc_price_now
            baseline_pnl = baseline_equity - self.initial_usdt
        else:
            baseline_equity = self.initial_usdt
            baseline_pnl = 0.0

        # ====================
        # 3. 压入记录列表
        # ====================
        dt_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        bar_dt_str = (
            datetime.fromtimestamp(bar_timestamp).strftime("%Y-%m-%d %H:%M:%S")
            if bar_timestamp
            else ""
        )
        record = {
            "timestamp": timestamp,
            "datetime": dt_str,
            "bar_timestamp": bar_timestamp or 0.0,
            "bar_datetime": bar_dt_str,
            "initial_capital_usdt": self.initial_usdt,
            "external_cash_buffer_usdt": self.external_cash_buffer_usdt,
            "strategy_accounting_mode": snapshot.get('strategy_accounting_mode', ''),
            "account_cash_total": snapshot['cash_total_usdt'],
            "account_holdings_value": snapshot.get('account_holdings_value_usdt', 0.0),
            "account_total_equity": snapshot['total_equity_usdt'],
            "strategy_cash_available": snapshot['cash_available_for_strategy'],
            "strategy_cash_ledger": snapshot.get('strategy_cash_ledger', snapshot['cash_available_for_strategy']),
            "strategy_holdings_value": snapshot['holdings_value_usdt'],
            "cash_total_usdt": snapshot.get('strategy_cash_ledger', snapshot['cash_available_for_strategy']),
            "holdings_value_usdt": snapshot['holdings_value_usdt'],
            "total_equity_usdt": strategy_equity,
            "strategy_equity": strategy_equity,
            "strategy_pnl": strategy_pnl,
            "baseline_equity": baseline_equity,
            "baseline_pnl": baseline_pnl,
            "strategy_returns_pct": self._safe_returns_pct(strategy_equity),
            "baseline_returns_pct": self._safe_returns_pct(baseline_equity),
            "btc_price": btc_price_now,
            "active_positions": sum(float(qty) > 0 for qty in holdings.values()),
            "gross_exposure": (
                snapshot['holdings_value_usdt'] / strategy_equity
                if strategy_equity > 0
                else 0.0
            ),
            "cash_ratio": (
                snapshot.get('strategy_cash_ledger', snapshot['cash_available_for_strategy'])
                / strategy_equity
                if strategy_equity > 0
                else 0.0
            ),
            "method_family": self.method_family,
            "strategy_id": self.identifier,
            "frequency": self.timeframe,
        }

        self.history_records.append(record)

        # 详细分类输出日志
        icon = "🟢" if strategy_pnl >= 0 else "🔴"
        holdings_str = ", ".join(
            [f"{k}: {v:.4f}" for k, v in holdings.items() if v > 0]
        ) or "空仓 (全量为USDT)"

        logger.info(f"[{self.identifier}] ====== 账户快照回顾 ({dt_str}) ======")
        logger.info(
            f"[{self.identifier}] 💰 [收益统计] 累计 PnL: {strategy_pnl:+.2f} U {icon} "
            f"| 策略资金池权益: {strategy_equity:.2f} U "
            f"| 账户总权益: {snapshot['total_equity_usdt']:.2f} U"
        )
        logger.info(
            f"[{self.identifier}] 🧮 [资金池约束] 初始本金={self.initial_usdt:.2f} U, "
            f"外部资金缓冲={self.external_cash_buffer_usdt:.2f} U, "
            f"可用策略现金={snapshot['cash_available_for_strategy']:.2f} U"
        )
        holdings_label = "策略持仓(本地台账)" if self.capital_mode == 'configured_budget' else "实际持仓"
        logger.info(f"[{self.identifier}] 📦 [{holdings_label}] {holdings_str}")
        logger.info(
            f"[{self.identifier}] 📈 [宏观基准] BTC同期 PnL: {baseline_pnl:+.2f} U    "
            f"| 满仓BTC资产: {baseline_equity:.2f} U"
        )
        logger.info(f"[{self.identifier}] ============================================\n")

    def export_snapshot(self):
        """原子写：将内存中的时间序列记录保存至本地"""
        if not self.history_records:
            return
        df = pd.DataFrame(self.history_records)
        df = df.reindex(columns=RUNTIME_METRICS_V1.ordered_columns(df.columns))
        _atomic_write_csv(df, self._csv_path)
        logger.debug(f"[{self.identifier}] 图表数据快照已保存: {self._csv_path}")
