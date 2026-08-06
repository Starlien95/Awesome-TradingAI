"""
SignalTracker: 实时预测信号记录 & 延迟 IC 计算模块

核心设计 (v3.1):
1. 每个交易周期结束后，将模型对各币种的预测分记录到 _pending_buffer（仅内存缓冲）
2. 在下一周期到来时，用新的收盘价回填 _pending_buffer 中的 true_return，
   标记 labeled=True 后移入 _records
3. export_csv() 同时写出 _records（已标注）和 _pending_buffer（待标注），
   确保 Dashboard 能实时看到最新预测
4. 启动时从 CSV 恢复：labeled=True 的进 _records，labeled=False 的进 _pending_buffer
   → 首个周期的 finalize_pending() 会标注它们 → 跨重启无永久 False
5. IC 计算拆分：_compute_ic() 每周期可调用，report_ic() 日志按 interval 节流
6. IC 报告落盘到独立的 _ic_report.csv（每周期追加一行）
"""

import logging
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from quant_bench.contracts.csv import RUNTIME_SIGNALS_V1

logger = logging.getLogger(__name__)


def _safe_read_csv(path: str) -> pd.DataFrame | None:
    """安全读取CSV，损坏或不存在时返回 None"""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception as e:
        logger.warning(f"CSV 读取失败（{path}），从空状态启动: {e}")
        backup = path + f'.corrupted.{int(time.time())}'
        try:
            os.rename(path, backup)
            logger.info(f"已备份损坏文件到: {backup}")
        except OSError:
            pass
        return None


class SignalTracker:
    """
    每个模型实例持有一个 SignalTracker。

    生命周期:
      1. runner 每周期调用 finalize_pending(current_prices, executor) 标注上一周期的 pending
      2. runner 每周期调用 record_signals(predictions, prices, timestamp) 记录新预测到 pending
      3. runner 每周期调用 report_ic() 输出 IC 日志（按 interval 节流）
      4. runner 每周期调用 export_ic_csv(timestamp) 写入 IC CSV
      5. runner 每周期调用 export_csv() 原子写 signals CSV
    """

    def __init__(
        self,
        identifier: str,
        timeframe: str,
        log_dir: str,
        method_family: str = "traditional_ml",
    ):
        self.identifier = identifier
        self.timeframe = timeframe
        self.method_family = method_family

        self.log_dir = os.path.join(log_dir, identifier)
        os.makedirs(self.log_dir, exist_ok=True)

        self._csv_path = os.path.join(
            self.log_dir, f"{identifier}_{timeframe}_signals.csv"
        )
        self._ic_csv_path = os.path.join(
            self.log_dir, f"{identifier}_{timeframe}_ic_report.csv"
        )

        # 全量已标注信号记录
        self._records: list[dict[str, Any]] = []
        self._cycle_count: int = 0

        # 最新一个周期的预测缓冲（等下一周期标注后入库）
        self._pending_buffer: list[dict[str, Any]] = []

        # 从磁盘恢复
        self._load_existing_csv()

        # IC 报告日志输出的节流间隔（每隔多少个周期输出一次日志）
        self.report_interval: int = 20

    # ==========================================================
    # 启动时恢复
    # ==========================================================

    def _load_existing_csv(self):
        """
        启动时从 CSV 恢复：
        - labeled=True 的 → _records
        - labeled=False 的 → _pending_buffer（首周期 finalize_pending 会标注它们）
        """
        df = _safe_read_csv(self._csv_path)
        if df is None:
            return

        records = df.to_dict('records')
        for rec in records:
            if rec.get('labeled', False):
                self._records.append(rec)
            else:
                self._pending_buffer.append(rec)

        if not df.empty:
            self._cycle_count = int(df['cycle_id'].max())

        logger.info(
            f"[{self.identifier}] 📂 从 CSV 恢复 {len(self._records)} 条已标注 + "
            f"{len(self._pending_buffer)} 条待标注信号, cycle_count={self._cycle_count}"
        )

    # ==========================================================
    # 每周期调用
    # ==========================================================

    def finalize_pending(self, current_prices: dict[str, float], executor=None):
        """
        用本周期的最新价格标注上一周期 pending 中的信号。
        标注完成后移入 _records。

        :param current_prices: 当前各币种价格
        :param executor: 可选的 TradeExecutor，用于兜底拉取 current_prices 中缺失的价格
        """
        if not self._pending_buffer:
            return

        filled = 0
        for rec in self._pending_buffer:
            coin = rec['coin']
            future_price = current_prices.get(coin, 0.0)
            # 兜底：current_prices 中没有 → 通过 API 直接获取
            if future_price <= 0 and executor is not None:
                future_price = executor.get_current_price(coin)

            if future_price > 0 and rec.get('price_at_pred', 0) > 0:
                rec['price_future'] = float(future_price)
                rec['true_return'] = (future_price / rec['price_at_pred']) - 1.0
                rec['labeled'] = True
                filled += 1
            else:
                # 真的拿不到价格 → 仍入库，标注 NaN（保留审计迹但不参与 IC 计算）
                rec['price_future'] = float('nan')
                rec['true_return'] = float('nan')
                rec['labeled'] = True

            self._records.append(rec)

        logger.debug(
            f"[{self.identifier}] ✅ Pending finalized: "
            f"{filled}/{len(self._pending_buffer)} records labeled with returns"
        )
        self._pending_buffer = []

    def record_signals(
        self,
        predictions: list[dict[str, Any]],
        current_prices: dict[str, float],
        timestamp: float,
        bar_timestamp: float | None = None,
    ):
        """
        记录本周期预测到 pending 缓冲区（不写入 _records）。
        下一周期的 finalize_pending() 会标注并入库。

        :param predictions: 模型返回的预测列表, 每个元素 {'coin': str, 'score': float, ...}
        :param current_prices: 当前各币种价格
        :param timestamp: 当前周期的 unix 时间戳
        :param bar_timestamp: 本次预测使用的已收盘K线起始时间戳
        """
        self._cycle_count += 1
        cycle_id = self._cycle_count
        bar_dt = (
            datetime.fromtimestamp(bar_timestamp).strftime("%Y-%m-%d %H:%M:%S")
            if bar_timestamp
            else ""
        )

        self._pending_buffer = []
        recorded_count = 0
        for pred in predictions:
            coin = pred.get('coin', '')
            score = pred.get('score', float('nan'))
            price = current_prices.get(coin, float('nan'))

            if not coin or np.isnan(price) or price <= 0:
                continue

            self._pending_buffer.append({
                'cycle_id': cycle_id,
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                'bar_timestamp': bar_timestamp or 0.0,
                'bar_datetime': bar_dt,
                'coin': coin,
                'score': float(score),
                'signal_label': pred.get('signal_label', ''),
                'signal_score': float(pred.get('signal_score', score)),
                'price_at_pred': float(price),
                'price_future': float('nan'),
                'true_return': float('nan'),
                'labeled': False,
                'method_family': self.method_family,
                'strategy_id': self.identifier,
                'frequency': self.timeframe,
            })
            recorded_count += 1

        logger.debug(
            f"[{self.identifier}] 📝 Signal recorded to pending: "
            f"cycle={cycle_id}, {recorded_count} coins"
        )

    # ==========================================================
    # IC 计算与报告
    # ==========================================================

    def _compute_ic(self) -> dict[str, Any] | None:
        """
        纯 IC 计算函数，不受节流控制。每周期可调用。
        仅使用 _records 中 labeled=True 且 true_return 非 NaN 的记录。
        """
        labeled = [
            r for r in self._records
            if r.get('labeled') and not np.isnan(r.get('true_return', float('nan')))
        ]

        if len(labeled) < 10:
            return None

        df = pd.DataFrame(labeled)
        result = {}

        # 1. 整体时序 IC (全局 Pearson)
        overall_ic = df['score'].corr(df['true_return'])
        result['overall_ts_ic'] = float(overall_ic) if not np.isnan(overall_ic) else 0.0

        # 2. 横截面 IC (每个 cycle 内跨标的算 IC，取均值)
        n_coins_per_cycle = df.groupby('cycle_id')['coin'].nunique()
        multi_coin_cycles = n_coins_per_cycle[n_coins_per_cycle >= 2].index

        if len(multi_coin_cycles) > 0:
            cs_ic_series = (
                df[df['cycle_id'].isin(multi_coin_cycles)]
                .groupby('cycle_id')
                .apply(
                    lambda g: g['score'].corr(g['true_return'])
                    if len(g) >= 2 else float('nan')
                )
                .dropna()
            )
            if len(cs_ic_series) > 0:
                result['cross_sec_ic_mean'] = float(cs_ic_series.mean())
                result['cross_sec_ic_std'] = float(cs_ic_series.std())
                icir = (
                    result['cross_sec_ic_mean'] / result['cross_sec_ic_std']
                    if result['cross_sec_ic_std'] > 0 else 0.0
                )
                result['icir'] = float(icir)
            else:
                result['cross_sec_ic_mean'] = 0.0
                result['cross_sec_ic_std'] = 0.0
                result['icir'] = 0.0
        else:
            result['cross_sec_ic_mean'] = float('nan')
            result['cross_sec_ic_std'] = float('nan')
            result['icir'] = float('nan')

        # 3. 各标的的时序 IC
        per_coin_ic = {}
        for coin in df['coin'].unique():
            coin_df = df[df['coin'] == coin]
            if len(coin_df) >= 5:
                ic = coin_df['score'].corr(coin_df['true_return'])
                per_coin_ic[coin] = float(ic) if not np.isnan(ic) else 0.0
        result['per_coin_ts_ic'] = per_coin_ic

        # 4. 方向准确率
        df_dir = df.copy()
        df_dir['pred_up'] = (df_dir['score'] > 0).astype(int)
        df_dir['real_up'] = (df_dir['true_return'] > 0).astype(int)
        acc = (df_dir['pred_up'] == df_dir['real_up']).mean()
        result['direction_accuracy'] = float(acc)

        # 元信息
        result['labeled_samples'] = len(labeled)
        result['total_cycles'] = int(df['cycle_id'].nunique())

        return result

    def report_ic(self, force: bool = False) -> dict[str, Any] | None:
        """
        计算并输出 IC 统计日志。受 report_interval 节流。
        """
        if not force and self._cycle_count % self.report_interval != 0:
            return None

        result = self._compute_ic()
        if result is None:
            logger.info(
                f"[{self.identifier}] 📊 IC 报告: 样本不足，暂不输出"
            )
            return None

        # 输出日志
        logger.info(f"[{self.identifier}] ════════════ 实盘 IC 监控报告 ════════════")
        logger.info(
            f"[{self.identifier}] 📊 已标注样本: {result['labeled_samples']} 条, "
            f"涵盖 {result['total_cycles']} 个周期"
        )
        logger.info(
            f"[{self.identifier}] 📈 整体时序 IC       : {result['overall_ts_ic']:+.4f}"
        )

        if not np.isnan(result.get('cross_sec_ic_mean', float('nan'))):
            logger.info(
                f"[{self.identifier}] 📈 横截面 IC (mean)  : "
                f"{result['cross_sec_ic_mean']:+.4f}"
            )
            logger.info(
                f"[{self.identifier}] 📈 横截面 IC (std)   : "
                f"{result['cross_sec_ic_std']:.4f}"
            )
            logger.info(
                f"[{self.identifier}] 📈 ICIR             : {result['icir']:+.3f}"
            )

        logger.info(
            f"[{self.identifier}] 🎯 方向准确率        : "
            f"{result['direction_accuracy']:.2%}"
        )

        per_coin_ic = result.get('per_coin_ts_ic', {})
        if per_coin_ic:
            logger.info(f"[{self.identifier}] ── 各标的时序 IC ──")
            for coin, ic in sorted(per_coin_ic.items()):
                logger.info(f"[{self.identifier}]     {coin:<18} IC={ic:+.4f}")

        logger.info(
            f"[{self.identifier}] ═══════════════════════════════════════════"
        )

        return result

    def export_ic_csv(self, timestamp: float):
        """
        每周期调用：将 IC 报告追加到独立 CSV（不受节流限制）。
        """
        result = self._compute_ic()
        if result is None:
            return

        dt_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        row = {
            'timestamp': timestamp,
            'datetime': dt_str,
            'overall_ts_ic': result['overall_ts_ic'],
            'cross_sec_ic_mean': result.get('cross_sec_ic_mean', float('nan')),
            'cross_sec_ic_std': result.get('cross_sec_ic_std', float('nan')),
            'icir': result.get('icir', float('nan')),
            'direction_accuracy': result['direction_accuracy'],
            'labeled_samples': result['labeled_samples'],
            'total_cycles': result['total_cycles'],
        }

        file_exists = os.path.exists(self._ic_csv_path)
        pd.DataFrame([row]).to_csv(
            self._ic_csv_path, mode='a', header=not file_exists, index=False
        )

    # ==========================================================
    # CSV 导出
    # ==========================================================

    def export_csv(self):
        """
        原子写 signals.csv。
        同时写出 _records（已标注）和 _pending_buffer（待标注），
        确保 Dashboard 能实时看到最新预测。
        """
        all_records = self._records + self._pending_buffer
        if not all_records:
            return
        tmp_path = self._csv_path + '.tmp'
        df = pd.DataFrame(all_records)
        df = df.reindex(columns=RUNTIME_SIGNALS_V1.ordered_columns(df.columns))
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, self._csv_path)
        logger.debug(f"[{self.identifier}] 💾 Signal log saved: {self._csv_path}")

    # ==========================================================
    # 辅助查询
    # ==========================================================

    def get_summary(self) -> dict[str, Any]:
        """返回当前的简要统计（供外部查询）"""
        total = len(self._records) + len(self._pending_buffer)
        labeled = sum(
            1 for r in self._records
            if r.get('labeled') and not np.isnan(r.get('true_return', float('nan')))
        )
        pending = len(self._pending_buffer)
        return {
            'total_signals': total,
            'labeled_signals': labeled,
            'pending_signals': pending,
            'cycles_completed': self._cycle_count
        }
