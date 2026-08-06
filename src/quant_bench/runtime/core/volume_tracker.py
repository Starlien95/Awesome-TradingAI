"""
VolumeTracker: 交易量统计追踪模块

记录每个周期的交易量统计：
- 累计总交易量（自跟随开始）
- 各币种累计交易量
- 当天交易量（每天 00:00 本地时间重置）
- 各币种当天交易量

支持跨重启 CSV 续接：从最后一行恢复累计和当天计数器。
"""

import contextlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

import pandas as pd

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
        with contextlib.suppress(OSError):
            os.rename(path, backup)
        return None


def _atomic_write_csv(df: pd.DataFrame, target_path: str):
    """先写临时文件，再原子替换"""
    tmp_path = target_path + '.tmp'
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, target_path)


class VolumeTracker:
    """
    记录每个周期的交易量统计。
    所有数据通过 CSV 持久化，支持跨重启恢复。
    """

    def __init__(self, identifier: str, timeframe: str, log_dir: str):
        self.identifier = identifier
        self.timeframe = timeframe

        self.log_dir = os.path.join(log_dir, identifier)
        os.makedirs(self.log_dir, exist_ok=True)

        self._csv_path = os.path.join(
            self.log_dir, f"{identifier}_{timeframe}_volume.csv"
        )

        # 累计计数器
        self.cumulative_total: float = 0.0
        self.cumulative_per_coin: dict[str, float] = {}

        # 当天计数器（每天 00:00 重置）
        self.daily_total: float = 0.0
        self.daily_per_coin: dict[str, float] = {}
        self.current_date: str | None = None  # "YYYY-MM-DD"

        # 历史记录
        self.history_records: list[dict] = []

        # 从 CSV 恢复
        self._load_from_csv()

    def _load_from_csv(self):
        """启动时从 CSV 恢复累计计数器和历史记录"""
        df = _safe_read_csv(self._csv_path)
        if df is None:
            return

        self.history_records = df.to_dict('records')
        last = df.iloc[-1]

        # 恢复累计量
        self.cumulative_total = float(last.get('cumulative_total_volume_usdt', 0.0))
        try:
            self.cumulative_per_coin = json.loads(
                last.get('per_coin_cumulative_json', '{}')
            )
        except (json.JSONDecodeError, TypeError):
            self.cumulative_per_coin = {}

        # 恢复当天量：对比最后一条记录的日期与今天
        last_dt_str = last.get('datetime', '')
        try:
            last_date = pd.to_datetime(last_dt_str).strftime('%Y-%m-%d')
        except Exception:
            last_date = ''

        today = datetime.now().strftime('%Y-%m-%d')
        if last_date == today:
            self.daily_total = float(last.get('daily_total_volume_usdt', 0.0))
            try:
                self.daily_per_coin = json.loads(
                    last.get('per_coin_daily_json', '{}')
                )
            except (json.JSONDecodeError, TypeError):
                self.daily_per_coin = {}
        # 不同天则 daily 保持 0（已在 __init__ 初始化）

        self.current_date = today

        logger.info(
            f"[{self.identifier}] 📂 从 CSV 恢复交易量: "
            f"累计={self.cumulative_total:.2f} USDT, "
            f"当天={self.daily_total:.2f} USDT, "
            f"历史记录={len(self.history_records)} 条"
        )

    def record_trades(self, executed_trades: list[dict[str, Any]], timestamp: float):
        """
        每周期交易执行后调用，记录实际成交的交易量。

        :param executed_trades: 实际成交的交易列表（仅 execute_trade 返回 True 的）
        :param timestamp: 当前周期的时间戳
        """
        today = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

        # 判断是否跨天，重置 daily 计数器
        if self.current_date is not None and today != self.current_date:
            self.daily_total = 0.0
            self.daily_per_coin = {}
            logger.info(
                f"[{self.identifier}] 📅 跨天重置日交易量 "
                f"({self.current_date} → {today})"
            )
        self.current_date = today

        # 累加本周期的交易量
        cycle_volume = 0.0
        for trade in executed_trades:
            vol = abs(trade.get('amount_usdt', 0.0))
            coin = trade.get('coin', 'UNKNOWN')
            cycle_volume += vol
            self.cumulative_total += vol
            self.daily_total += vol
            self.cumulative_per_coin[coin] = (
                self.cumulative_per_coin.get(coin, 0.0) + vol
            )
            self.daily_per_coin[coin] = (
                self.daily_per_coin.get(coin, 0.0) + vol
            )

        # 追加一条记录
        dt_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        record = {
            'timestamp': timestamp,
            'datetime': dt_str,
            'cycle_volume_usdt': cycle_volume,
            'cumulative_total_volume_usdt': self.cumulative_total,
            'daily_total_volume_usdt': self.daily_total,
            'per_coin_cumulative_json': json.dumps(self.cumulative_per_coin),
            'per_coin_daily_json': json.dumps(self.daily_per_coin),
        }
        self.history_records.append(record)

        if cycle_volume > 0:
            logger.info(
                f"[{self.identifier}] 📊 本周期成交量: {cycle_volume:.2f} USDT | "
                f"累计: {self.cumulative_total:.2f} USDT | "
                f"当天: {self.daily_total:.2f} USDT"
            )

    def export_csv(self):
        """原子写交易量 CSV"""
        if not self.history_records:
            return
        df = pd.DataFrame(self.history_records)
        _atomic_write_csv(df, self._csv_path)
        logger.debug(f"[{self.identifier}] 💾 Volume log saved: {self._csv_path}")
