# live_json_builder.py

from pathlib import Path

import pandas as pd

from quant_bench.methods.finmem.live.news import NewsProcessor
from quant_bench.runtime.core.atomic_io import atomic_write_json


class LiveJsonBuilder:
    def __init__(self, output_dir="live_data", news_processor=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.news_processor = news_processor or NewsProcessor()

    @staticmethod
    def inst_id_to_symbol(inst_id):
        """
        BTC-USDT -> BTC
        ETH-USDT -> ETH
        """
        return str(inst_id).upper().split("-")[0]

    @staticmethod
    def normalize_kline_records(kline_records, include_unconfirmed=False):
        df = pd.DataFrame(kline_records)

        if df.empty:
            raise ValueError("Empty kline_records")

        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise ValueError(f"Missing kline columns: {missing}")

        df["timestamp"] = df["timestamp"].astype("int64")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        df = df.dropna(subset=["close"])

        # 默认只使用已确认K线；
        # 如果 include_unconfirmed=True，则保留当前正在形成的K线。
        if "confirm" in df.columns and not include_unconfirmed:
            df = df[df["confirm"].astype(str) == "1"].copy()

        df = df.sort_values("timestamp")
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        df = df.reset_index(drop=True)

        # OKX 1D 是 UTC+8 日线，所以这里转 Asia/Shanghai 取 date 是合理的。
        df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["datetime"] = df["datetime_utc"].dt.tz_convert("Asia/Shanghai")
        df["date"] = df["datetime"].dt.date.astype(str)

        return df

    @staticmethod
    def select_rolling_window(kline_df, previous_days=10):
        """
        保留：当前决策日 + 前 previous_days 根K线。

        previous_days=3 时，最终最多保留 4 行：
            t-3, t-2, t-1, t
        """
        if previous_days is None:
            return kline_df.copy()

        previous_days = int(previous_days)
        if previous_days < 0:
            raise ValueError("previous_days must be >= 0")

        window_rows = previous_days + 1

        if len(kline_df) <= window_rows:
            return kline_df.copy()

        return kline_df.tail(window_rows).copy()

    @staticmethod
    def filter_news_by_date(news_df, cur_date):
        """
        如果 news_df 里有 date 字段，可以按日期过滤历史新闻。
        如果没有 date 字段，就返回原 news_df。

        注意：
        - 实时抓取的 news_df 通常是“当前新闻”，不应该复制到历史日期。
        - 只有当你的 news_df 确实包含历史多日新闻时，才建议对历史日期挂新闻。
        """
        if news_df is None or news_df.empty:
            return news_df

        if "date" not in news_df.columns:
            return news_df

        tmp = news_df.copy()
        tmp["__date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.date.astype(str)
        return tmp[tmp["__date"] == str(cur_date)].drop(columns=["__date"])

    def build_for_symbol(
        self,
        inst_id,
        kline_records,
        news_df=None,
        max_rows=300,
        output_filename=None,
        include_unconfirmed=False,
        previous_days=10,
        news_mode="latest",
    ):
        """
        生成单个币种的 live json。

        参数说明
        ----------
        inst_id:
            例如 BTC-USDT。

        kline_records:
            OKX 拉取到的 K 线记录。

        news_df:
            新闻 DataFrame。

        max_rows:
            对原始 K 线最多保留多少行，防止文件过大。
            这一步在 rolling window 之前做。

        include_unconfirmed:
            True 时保留 OKX 当前未完成日K。

        previous_days:
            生成 rolling JSON 时保留当前日之前多少根 K 线。
            例如 previous_days=10，则 JSON 最多包含 11 天。

        news_mode:
            "latest":
                只把当前抓到的新闻挂到最新日期。推荐 live 模拟盘使用。
            "by_date":
                如果 news_df 里有 date 字段，则按日期给每一天挂对应新闻。
                适合历史 replay，但要求 news_df 是历史多日新闻。
            "none":
                不挂新闻，只保留价格。
        """
        symbol = self.inst_id_to_symbol(inst_id)

        kline_df = self.normalize_kline_records(
            kline_records,
            include_unconfirmed=include_unconfirmed,
        )

        # 先限制总体大小
        if max_rows is not None and len(kline_df) > max_rows:
            kline_df = kline_df.tail(max_rows).copy()

        # 再取 rolling window：当前日 + 前 previous_days 天
        kline_df = self.select_rolling_window(
            kline_df,
            previous_days=previous_days,
        )

        if kline_df.empty:
            raise ValueError("No kline data after rolling-window selection.")

        latest_date = kline_df["date"].iloc[-1]
        earliest_date = kline_df["date"].iloc[0]

        result = {}

        for _, row in kline_df.iterrows():
            cur_date = row["date"]
            close_price = float(row["close"])
            confirm = str(row.get("confirm", "1"))

            news_items = []

            if news_df is not None and not news_df.empty and news_mode != "none":
                if news_mode == "latest":
                    # 实时模拟盘推荐：
                    # 只把当前抓到的新闻挂到最新K线日期，不复制给历史日期。
                    if cur_date == latest_date:
                        news_items = self.news_processor.build_news_for_symbol(
                            news_df,
                            symbol,
                        )

                elif news_mode == "by_date":
                    # 历史 replay 可用：
                    # 只有 news_df 本身包含 date 字段时，才按日期挂新闻。
                    date_news_df = self.filter_news_by_date(news_df, cur_date)
                    if date_news_df is not None and not date_news_df.empty:
                        news_items = self.news_processor.build_news_for_symbol(
                            date_news_df,
                            symbol,
                        )

                else:
                    raise ValueError(
                        "news_mode must be one of: 'latest', 'by_date', 'none'"
                    )

            result[cur_date] = {
                "prices": close_price,
                "news": news_items,
                "is_unconfirmed": confirm == "0",
                "kline_time": row["datetime"].strftime("%Y-%m-%d %H:%M:%S%z"),
            }

        if output_filename is None:
            output_filename = f"{symbol.lower()}_live.json"

        output_path = self.output_dir / output_filename

        atomic_write_json(output_path, result)

        print("=" * 80)
        print("[LIVE JSON BUILDER]")
        print(f"symbol:             {symbol}")
        print(f"output_path:        {output_path}")
        print(f"earliest_date:      {earliest_date}")
        print(f"latest_date:        {latest_date}")
        print(f"rows:               {len(result)}")
        print(f"include_unconfirmed:{include_unconfirmed}")
        print(f"previous_days:      {previous_days}")
        print(f"news_mode:          {news_mode}")
        print("=" * 80)

        return str(output_path), result
