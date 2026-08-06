import asyncio
import logging

import pandas as pd

from quant_bench.runtime.core.execution import TradeExecutor

logger = logging.getLogger(__name__)

class DataManager:
    """
    数据管理器 (DataManager)
    统一负责某个特定时间尺度 (Timeframe) 下的所有数据拉取。
    保证在一个执行周期内，同一个币种只向 API 发送一次获取历史K线数据的请求。
    取得的数据存储在内存中，并分发给本周期所有并发执行的策略模型。
    """
    def __init__(self, api_client: TradeExecutor, timeframe_str: str):
        """
        :param api_client: 具备 get_coin_kline 能力的执行器
        :param timeframe_str: 时间尺度，如 '15m', '1h', '4h'
        """
        self.api = api_client
        self.timeframe_str = timeframe_str

        # 缓存：记录 "本周期" 拉取下来的数据
        # 结构: { "BTC-USDT-SWAP": DataFrame, ... }
        self._current_cycle_cache: dict[str, pd.DataFrame] = {}

        # 用于防止并发下的重复拉取 (异步锁)
        self._fetch_locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, coin: str) -> asyncio.Lock:
        if coin not in self._fetch_locks:
            self._fetch_locks[coin] = asyncio.Lock()
        return self._fetch_locks[coin]

    def clear_cycle_cache(self):
        """在新的周期开始时，清空老数据缓存"""
        self._current_cycle_cache.clear()

    def _cache_satisfies_limit(self, df: pd.DataFrame, limit: int) -> bool:
        # OKX candlesticks limit is capped at 300 and usually includes one unclosed bar.
        # After confirm=1 filtering, a 300-bar request often has 299 usable rows.
        effective_limit = min(max(1, limit), 299)
        return len(df) >= effective_limit

    async def get_kline_data(self, coin: str, limit: int = 200) -> pd.DataFrame | None:
        """
        获取K线数据。供各个模型并发调用。
        :param coin: 交易对如 "BTC-USDT-SWAP"
        :param limit: K线根数
        """
        # 第一层检查：如果已经被本周期某个模型拉取过了，直接返回内存数据的副本
        if coin in self._current_cycle_cache:
            df = self._current_cycle_cache[coin]
            # 根据模型要求的长度进行裁剪
            if self._cache_satisfies_limit(df, limit):
                return df.tail(limit).copy()
            else:
                # 只有当已缓存的长度不足以满足当前模型胃口时，才允许往下走 (一般来说初始设计使得最大的limit涵盖所有)
                pass

        # 第二层：异步锁机制避免并发击穿
        async with self._get_lock(coin):
            # 获取锁后再次检查（可能被刚才拿锁的人拉完了）
            if coin in self._current_cycle_cache and self._cache_satisfies_limit(self._current_cycle_cache[coin], limit):
                return self._current_cycle_cache[coin].tail(limit).copy()

            # 真正向 API 发起请求（带重试，保持同一周期内时间基准不变）
            max_retries = 3
            retry_delay = 2.0  # 秒
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    # OKX candlesticks limit 通常最高 300；封顶避免因多取未收盘K线缓冲而请求失败。
                    request_limit = min(max(limit + 2, limit), 300)
                    raw_data = await asyncio.to_thread(
                        self.api.get_coin_kline,
                        coin,
                        self.timeframe_str,
                        request_limit,
                    )
                    if raw_data:
                        df = self._parse_to_df(raw_data)
                        self._current_cycle_cache[coin] = df
                        return df.tail(limit).copy()

                    logger.warning(f"[{coin}] {self.timeframe_str} API返回数据为空 (attempt {attempt}/{max_retries})")
                except Exception as e:
                    last_exc = e
                    logger.warning(f"[{coin}] 拉取数据发生异常 (attempt {attempt}/{max_retries}): {e}")

                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)

            if last_exc is not None:
                logger.error(f"[{coin}] 拉取数据最终失败: {last_exc}")
            return None

    def _parse_to_df(self, raw_data_list: list) -> pd.DataFrame:
        """解析 OKX 格式的数据列表为标准的 DataFrame"""
        columns = ['ts', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm']
        df = pd.DataFrame(raw_data_list, columns=columns)

        # 转换数据类型
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['confirm'] = pd.to_numeric(df['confirm'], errors='coerce').fillna(0).astype(int)

        # OKX candlesticks 的最后一根通常是未收盘K线(confirm=0)。
        # 实盘预测/交易必须只使用 confirm=1 的已收盘K线，否则会产生未来不稳定信号。
        df = df[df['confirm'] == 1].copy()
        df = df.dropna(subset=['ts', 'open', 'high', 'low', 'close'])

        # 按照时间从小到大排序（最老的在前，最新的在最后）
        df = df.sort_values(by='ts', ascending=True)
        # 转换为 datetime 索引
        df.index = pd.to_datetime(df['ts'], unit='ms')

        return df
