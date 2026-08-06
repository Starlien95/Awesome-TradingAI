# news_processor.py

import ast
import re
import pandas as pd


COIN_SUBJECT_TO_SYMBOL = {
    "ada": "ADA",
    "btc": "BTC",
    "bitcoin": "BTC",
    "doge": "DOGE",
    "dogecoin": "DOGE",
    "eth": "ETH",
    "ethereum": "ETH",
    "hbar": "HBAR",
    "link": "LINK",
    "ltc": "LTC",
    "okb": "OKB",
    "sol": "SOL",
    "solana": "SOL",
    "trx": "TRX",
    "xrp": "XRP",
}

GLOBAL_SUBJECTS = {
    "blockchain",
    "altcoin",
    "business",
    "exchange",
    "market",
    "regulation",
    "fiat",
    "trading",
    "technology",
    "mining",
    "wallet",
    "cryptocurrency",
    "macroeconomics",
    "commodity",
    "asia",
    "security incidents",
    "token listing and delisting",
    "administrative events",
    "digital asset treasury",
    "defi",
    "stablecoin",
    "web3",
    "general",
    "artificial intelligence",
    "cryptocurrency market news",
    "latest news",
    "regulation & legal",
}

DROP_SUBJECTS = {
    "sponsored",
    "airdrop",
    "token sale",
}

SUBJECT_ALIASES = {
    "bitcoin": "btc",
    "ethereum": "eth",
    "solana": "sol",
    "dogecoin": "doge",
    "macro/regulation": "regulation",
    "macro/oil": "regulation",
    "regulation & legal": "regulation",
    "cryptocurrency market news": "market",
    "latest news": "market",
    "aa news": "market",
}


class NewsCollector:
    def fetch_latest_news(self, limit=80, days=3, max_pages=2):
        """
        获取最新新闻，返回 DataFrame。

        默认使用 historical_news_collector.py 内置的 list.md RSS/Google News
        来源，避免 fallback latest 模式重新依赖 CCData。
        """
        from quant_bench.methods.finmem.live.historical_news import fetch_historical_news_for_window

        end_date = pd.Timestamp.now(tz="Asia/Shanghai").date()
        start_date = end_date - pd.Timedelta(days=int(days))

        df = fetch_historical_news_for_window(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            target_records=limit,
            max_pages=max_pages,
            verbose=False,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        return df.sort_values("datetime", ascending=False).reset_index(drop=True)

class NewsProcessor:
    def __init__(
        self,
        max_coin_news=5,
        max_global_news=5,
        max_total_news=10,
    ):
        self.max_coin_news = max_coin_news
        self.max_global_news = max_global_news
        self.max_total_news = max_total_news

    @staticmethod
    def normalize_subject(subject):
        subject = str(subject or "general").lower().strip()
        return SUBJECT_ALIASES.get(subject, subject)

    @staticmethod
    def clean_text(s):
        if pd.isna(s):
            return ""

        s = str(s)
        s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        s = re.sub(r"\s+", " ", s)
        return s.strip().lower()

    @staticmethod
    def parse_sentiment_class(sentiment):
        if isinstance(sentiment, dict):
            cls = sentiment.get("class", "neutral")
        else:
            try:
                obj = ast.literal_eval(str(sentiment))
                cls = obj.get("class", "neutral")
            except Exception:
                cls = str(sentiment)

        cls = str(cls).lower().strip()

        if cls not in {"positive", "negative", "neutral"}:
            cls = "neutral"

        return cls

    def build_news_text(self, row, prefix):
        title = self.clean_text(row.get("title", ""))
        text = self.clean_text(row.get("text", ""))
        sentiment = self.parse_sentiment_class(row.get("sentiment", "neutral"))

        if title and text:
            content = f"{title}{text}"
        elif title:
            content = title
        else:
            content = text

        content = content.strip()

        if not content:
            return ""

        return f"{prefix} {content} (sentiment:{sentiment})"

    @staticmethod
    def deduplicate_news(news_list):
        seen = set()
        result = []

        for item in news_list:
            key = re.sub(r"\s+", " ", item.lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            result.append(item)

        return result

    def build_news_for_symbol(self, news_df, symbol):
        """
        为某个币种构造当天新闻列表。
        实时交易建议按最新时间选，而不是随机选。
        """
        if news_df is None or news_df.empty:
            return []

        symbol = symbol.upper()
        coin_subject = symbol.lower()

        df = news_df.copy()

        if "subject" not in df.columns:
            df["subject"] = "general"

        df["subject"] = df["subject"].map(self.normalize_subject)
        df = df[~df["subject"].isin(DROP_SUBJECTS)].copy()

        if "datetime" not in df.columns and "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"], errors="coerce")

        if "datetime" in df.columns:
            df = df.sort_values("datetime", ascending=False)

        # 币种直接新闻
        coin_news_df = df[df["subject"] == coin_subject].copy()

        # 全市场新闻
        global_news_df = df[df["subject"].isin(GLOBAL_SUBJECTS)].copy()

        news_items = []

        for _, row in coin_news_df.head(self.max_coin_news).iterrows():
            item = self.build_news_text(row, prefix=f"[{symbol}][coin]")
            if item:
                news_items.append(item)

        for _, row in global_news_df.head(self.max_global_news).iterrows():
            subject = row.get("subject", "market")
            item = self.build_news_text(row, prefix=f"[MARKET][{subject}]")
            if item:
                news_items.append(item)

        news_items = self.deduplicate_news(news_items)
        return news_items[:self.max_total_news]
