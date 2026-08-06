import asyncio
import logging
import math

import pandas as pd
import requests

from quant_bench.runtime.core.data_manager import DataManager
from quant_bench.runtime.core.execution import TradeExecutor

logger = logging.getLogger(__name__)


class MacroHFTDataManager(DataManager):
    """
    Data manager used only by MacroHFT runners.

    The shared DataManager intentionally caps OKX /market/candles at 300 rows.
    MacroHFT needs 360+ closed bars for slope_360/vol_360, so this class keeps
    the original cache/parse contract but backfills older candles through OKX's
    public history endpoint when a model asks for more than 300 bars.
    """

    BASE_URL = "https://www.okx.com"
    CURRENT_CANDLES_PATH = "/api/v5/market/candles"
    HISTORY_CANDLES_PATH = "/api/v5/market/history-candles"
    CURRENT_PAGE_LIMIT = 300
    HISTORY_PAGE_LIMIT = 100

    def __init__(
        self,
        api_client: TradeExecutor,
        timeframe_str: str,
        request_timeout_sec: float = 10.0,
    ):
        super().__init__(api_client=api_client, timeframe_str=timeframe_str)
        self.request_timeout_sec = request_timeout_sec

    async def get_kline_data(self, coin: str, limit: int = 200) -> pd.DataFrame | None:
        if limit <= self.CURRENT_PAGE_LIMIT:
            return await super().get_kline_data(coin, limit=limit)

        if coin in self._current_cycle_cache:
            cached = self._current_cycle_cache[coin]
            if len(cached) >= limit:
                return cached.tail(limit).copy()

        async with self._get_lock(coin):
            if coin in self._current_cycle_cache and len(self._current_cycle_cache[coin]) >= limit:
                return self._current_cycle_cache[coin].tail(limit).copy()

            df = await asyncio.to_thread(self._fetch_extended_kline_data, coin, limit)
            if df is None or df.empty:
                logger.warning(
                    f"[MacroHFTDataManager] {coin} {self.timeframe_str} extended K-line fetch failed"
                )
                return None

            self._current_cycle_cache[coin] = df
            if len(df) < limit:
                logger.warning(
                    f"[MacroHFTDataManager] {coin} only got {len(df)}/{limit} closed bars"
                )
            return df.tail(limit).copy()

    def _fetch_extended_kline_data(self, coin: str, limit: int) -> pd.DataFrame | None:
        raw_rows: list[list[str]] = []

        first_page = self._fetch_page(
            self.CURRENT_CANDLES_PATH,
            coin,
            limit=self.CURRENT_PAGE_LIMIT,
        )
        raw_rows.extend(first_page)

        df = self._raw_rows_to_df(raw_rows)
        if len(df) >= limit:
            return df.tail(limit).copy()

        after_ts = self._oldest_ts(raw_rows)
        max_pages = max(3, math.ceil(limit / self.HISTORY_PAGE_LIMIT) + 3)

        for _ in range(max_pages):
            if after_ts is None:
                break

            page = self._fetch_page(
                self.HISTORY_CANDLES_PATH,
                coin,
                limit=self.HISTORY_PAGE_LIMIT,
                after=after_ts,
            )
            if not page:
                break

            previous_count = len(raw_rows)
            raw_rows.extend(page)
            raw_rows = self._dedupe_raw_rows(raw_rows)

            df = self._raw_rows_to_df(raw_rows)
            if len(df) >= limit:
                return df.tail(limit).copy()

            new_after_ts = self._oldest_ts(raw_rows)
            if new_after_ts is None or new_after_ts == after_ts or len(raw_rows) == previous_count:
                break
            after_ts = new_after_ts

        return df.tail(limit).copy() if not df.empty else None

    def _fetch_page(
        self,
        path: str,
        coin: str,
        limit: int,
        after: float | None = None,
    ) -> list[list[str]]:
        params: dict[str, str] = {
            "instId": coin,
            "bar": self._normalize_bar(self.timeframe_str),
            "limit": str(limit),
        }
        if after is not None:
            params["after"] = str(int(after))

        try:
            response = requests.get(
                f"{self.BASE_URL}{path}",
                params=params,
                timeout=self.request_timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning(
                f"[MacroHFTDataManager] request failed {path} {coin} params={params}: {exc}"
            )
            return []

        if payload.get("code") != "0":
            logger.warning(
                f"[MacroHFTDataManager] OKX returned code={payload.get('code')} "
                f"msg={payload.get('msg')} path={path} coin={coin}"
            )
            return []

        data = payload.get("data") or []
        return data if isinstance(data, list) else []

    def _raw_rows_to_df(self, raw_rows: list[list[str]]) -> pd.DataFrame:
        rows = self._dedupe_raw_rows(raw_rows)
        if not rows:
            return pd.DataFrame()
        return self._parse_to_df(rows)

    @staticmethod
    def _dedupe_raw_rows(raw_rows: list[list[str]]) -> list[list[str]]:
        by_ts: dict[str, list[str]] = {}
        for row in raw_rows:
            if not row:
                continue
            by_ts[str(row[0])] = row
        return list(by_ts.values())

    @staticmethod
    def _oldest_ts(raw_rows: list[list[str]]) -> float | None:
        timestamps = []
        for row in raw_rows:
            try:
                timestamps.append(float(row[0]))
            except (TypeError, ValueError, IndexError):
                continue
        return min(timestamps) if timestamps else None

    @staticmethod
    def _normalize_bar(bar: str) -> str:
        if bar.endswith("h"):
            return bar.replace("h", "H")
        if bar.endswith("d"):
            return bar.replace("d", "D")
        if bar.endswith("w"):
            return bar.replace("w", "W")
        return bar
