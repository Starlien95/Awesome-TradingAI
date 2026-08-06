"""OKX K-line collection used by the FinMem live cycle."""

import time

import pandas as pd


class TradingDataCollector:
    """Maintain a bounded, chronological K-line cache for one instrument."""

    HISTORY_BATCH_LIMIT = 100
    MAX_HISTORY_BATCHES = 30

    def __init__(self, okxbot, inst_id):
        self.inst_id = inst_id
        self.okxbot = okxbot
        self.kline_cache = pd.DataFrame()
        self.min_history_len = 1000
        self.is_initialized = False

    def get_price_data(self, bar="5m", limit=50):
        """Warm the cache once, merge the latest candles, and return records."""
        if not self.is_initialized:
            if not self._fetch_initial_history(bar):
                raise RuntimeError(f"Failed to warm K-line history for {self.inst_id}")
            self.is_initialized = True

        try:
            response = self.okxbot.get_coin_kline(self.inst_id, bar, limit=limit)
            if response and response.get("data"):
                self._update_cache(self._parse_to_df(response["data"]))
            else:
                print(f"[WARNING] {self.inst_id}: latest K-line response is empty")
        except Exception as exc:
            print(f"[WARNING] {self.inst_id}: latest K-line update failed: {exc}")

        return self.kline_cache.to_dict("records")

    def _fetch_initial_history(self, bar):
        frames = []
        cursor = None

        for batch_index in range(self.MAX_HISTORY_BATCHES):
            response = self.okxbot.get_history_kline(
                self.inst_id,
                bar,
                limit=self.HISTORY_BATCH_LIMIT,
                after=cursor,
            )
            batch = (response or {}).get("data") or []
            if not batch:
                break

            frame = self._parse_to_df(batch)
            if frame.empty:
                break
            frames.append(frame)

            next_cursor = str(batch[-1][0])
            if next_cursor == cursor:
                break
            cursor = next_cursor

            row_count = sum(len(item) for item in frames)
            print(
                f"[KLINE WARMUP] {self.inst_id}: "
                f"batch={batch_index + 1}, rows={row_count}"
            )
            if row_count >= self.min_history_len:
                break
            time.sleep(0.1)

        if not frames:
            return False

        self._update_cache(pd.concat(frames, ignore_index=True))
        print(f"[KLINE WARMUP DONE] {self.inst_id}: rows={len(self.kline_cache)}")
        return True

    @staticmethod
    def _parse_to_df(raw_rows):
        columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "volCcy",
            "volCcyQuote",
            "confirm",
        ]
        normalized_rows = [list(row[: len(columns)]) for row in raw_rows if len(row) >= 6]
        if not normalized_rows:
            return pd.DataFrame(columns=columns)

        # Older responses may omit the last optional fields.
        width = max(len(row) for row in normalized_rows)
        frame = pd.DataFrame(normalized_rows, columns=columns[:width])
        for column in columns[width:]:
            frame[column] = ""

        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        return frame.dropna(subset=["timestamp", "close"]).astype(
            {"timestamp": "int64"}
        )

    def _update_cache(self, new_frame):
        if new_frame.empty:
            return

        self.kline_cache = pd.concat(
            [self.kline_cache, new_frame],
            ignore_index=True,
        )
        self.kline_cache = (
            self.kline_cache.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .tail(self.min_history_len + 500)
            .reset_index(drop=True)
        )
