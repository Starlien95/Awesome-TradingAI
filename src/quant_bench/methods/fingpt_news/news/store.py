from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_bench.methods.fingpt_news.news.cleaner import normalize_text, stable_id
from quant_bench.runtime.core.atomic_io import atomic_write_csv, safe_read_csv
from quant_bench.runtime.core.state_store import JsonStateStore

RAW_NEWS_COLUMNS = ["date", "source", "subject", "text", "title", "url", "published_on", "raw_news_id"]


class NewsStore:
    def __init__(self, raw_news_path: str | Path, checkpoint_path: str | Path):
        self.raw_news_path = Path(raw_news_path)
        self.checkpoint = JsonStateStore(
            checkpoint_path,
            {"last_success_at": "", "last_fetch_ts": 0, "seen_news_ids": []},
        )

    def load_raw_news(self) -> pd.DataFrame:
        df = safe_read_csv(self.raw_news_path)
        if df.empty:
            return pd.DataFrame(columns=RAW_NEWS_COLUMNS)
        return self._dedupe(df)

    def append_rows(self, rows: list[dict]) -> pd.DataFrame:
        existing = self.load_raw_news()
        if rows:
            incoming = pd.DataFrame(rows)
            incoming["raw_news_id"] = [
                stable_id(row.get("date", ""), row.get("url", ""), normalize_text(row.get("title", "")), prefix="raw_")
                for row in incoming.to_dict("records")
            ]
            combined = pd.concat([existing, incoming], ignore_index=True)
        else:
            combined = existing
        combined = self._dedupe(combined)
        atomic_write_csv(combined[RAW_NEWS_COLUMNS], self.raw_news_path)
        if not combined.empty:
            self.checkpoint.update(
                {
                    "last_success_at": pd.Timestamp.utcnow().isoformat(),
                    "last_fetch_ts": int(pd.to_numeric(combined.get("published_on"), errors="coerce").max() or 0),
                    "seen_news_ids": combined["raw_news_id"].tail(5000).tolist(),
                }
            )
            self.checkpoint.save()
        return combined

    @staticmethod
    def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=RAW_NEWS_COLUMNS)
        work = df.copy()
        for col in RAW_NEWS_COLUMNS:
            if col not in work:
                work[col] = ""
        work["title_norm"] = work["title"].map(normalize_text)
        work["text_norm"] = work["text"].map(lambda x: normalize_text(x)[:500])
        work = work.sort_values("date", ascending=True)
        work = work.drop_duplicates("raw_news_id", keep="first")
        work = work.drop_duplicates("url", keep="first")
        work = work.drop_duplicates("title_norm", keep="first")
        work = work.drop_duplicates("text_norm", keep="first")
        work = work.sort_values("date", ascending=True)
        return work[RAW_NEWS_COLUMNS].reset_index(drop=True)
