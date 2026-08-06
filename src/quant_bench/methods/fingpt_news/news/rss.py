"""Optional RSS and Google News collector for FinGPT paper workflows."""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import pandas as pd

from quant_bench.methods.fingpt_news.news.cleaner import clean_html_text
from quant_bench.runtime.core.atomic_io import atomic_write_csv

HISTORY_COLUMNS = ["date", "source", "subject", "text", "title", "url", "published_on"]
DEFAULT_FEEDS = [
    {"name": "CoinDesk", "url": "https://feeds.feedburner.com/CoinDesk"},
    {"name": "Cointelegraph", "url": "https://cointelegraph.com/rss"},
    {"name": "Decrypt", "url": "https://decrypt.co/feed"},
    {"name": "Bitcoin.com", "url": "https://news.bitcoin.com/feed/"},
    {
        "name": "Google News Crypto",
        "url": "https://news.google.com/rss/search?q="
        + quote_plus("crypto OR bitcoin OR ethereum OR digital assets")
        + "&hl=en-US&gl=US&ceid=US:en",
    },
]
SUBJECT_PATTERNS = {
    "ada": r"\b(?:cardano|ada)\b",
    "btc": r"\b(?:bitcoin|btc)\b",
    "doge": r"\b(?:dogecoin|doge)\b",
    "eth": r"\b(?:ethereum|ether|eth)\b",
    "hbar": r"\b(?:hedera|hbar)\b",
    "link": r"\b(?:chainlink|link token)\b",
    "ltc": r"\b(?:litecoin|ltc)\b",
    "okb": r"\b(?:okb|okx token)\b",
    "trx": r"\b(?:tron|trx)\b",
    "xrp": r"\b(?:xrp|ripple)\b",
    "regulation": r"\b(?:sec|cftc|regulation|regulatory|lawsuit|court|etf)\b",
    "defi": r"\b(?:defi|decentralized finance)\b",
}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", clean_html_text(str(value or ""))).strip()


def _normalize_history(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    work = frame.copy()
    for column in HISTORY_COLUMNS:
        if column not in work:
            work[column] = ""
    parsed = pd.to_datetime(work["date"], errors="coerce")
    work = work.loc[parsed.notna()].copy()
    work["date"] = parsed.loc[work.index].dt.strftime("%Y-%m-%d %H:%M:%S")
    work["published_on"] = pd.to_numeric(work["published_on"], errors="coerce").fillna(0).astype(int)
    for column in ["source", "subject", "text", "title", "url"]:
        work[column] = work[column].fillna("").astype(str)
    work = work.drop_duplicates(["title", "url"], keep="last")
    return work[HISTORY_COLUMNS].sort_values("date").reset_index(drop=True)


class RSSGoogleNewsFetcher:
    """Fetch bounded RSS windows and maintain an atomic local dedupe cache."""

    def __init__(
        self,
        *,
        history_path: str | Path,
        feeds: list[dict[str, str]] | None = None,
        timeout_sec: int = 20,
        max_retries: int = 2,
        sleep_sec: float = 1.0,
        timezone: str = "Asia/Shanghai",
        recent_days: int = 10,
        target_records: int = 8000,
    ) -> None:
        self.history_path = Path(history_path)
        self.feeds = feeds or DEFAULT_FEEDS
        self.timeout_sec = max(1, int(timeout_sec))
        self.max_retries = max(1, int(max_retries))
        self.sleep_sec = max(0.0, float(sleep_sec))
        self.timezone = ZoneInfo(timezone)
        self.recent_days = max(1, int(recent_days))
        self.target_records = max(1, int(target_records))
        for feed in self.feeds:
            if not feed.get("name") or not feed.get("url"):
                raise ValueError("every RSS feed requires non-empty name and url")

    @staticmethod
    def _dependencies() -> tuple[Any, Any, Any]:
        try:
            import feedparser
            import requests
            from dateutil import parser as date_parser
        except ImportError as exc:
            raise RuntimeError(
                "RSS news requires the 'fingpt-live' extra: pip install 'quant-bench[fingpt-live]'"
            ) from exc
        return feedparser, requests, date_parser

    def _load_history(self) -> pd.DataFrame:
        if not self.history_path.is_file():
            return _normalize_history(None)
        try:
            return _normalize_history(pd.read_csv(self.history_path))
        except Exception as exc:
            raise ValueError(f"cannot read RSS history {self.history_path}: {exc}") from exc

    def _save_history(self, frame: pd.DataFrame) -> None:
        atomic_write_csv(_normalize_history(frame), self.history_path)

    def _request(self, url: str) -> Any | None:
        feedparser, requests, _ = self._dependencies()
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    headers={"User-Agent": "quant-bench/0.1 RSS research client"},
                    timeout=self.timeout_sec,
                )
                response.raise_for_status()
                return feedparser.parse(response.content)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(min(30.0, self.sleep_sec * (attempt + 1)))
        if last_error:
            return None
        return None

    def _subject(self, title: str, text: str) -> str:
        content = f"{title} {text}".lower()
        for subject, pattern in SUBJECT_PATTERNS.items():
            if re.search(pattern, content, flags=re.IGNORECASE):
                return subject
        return "market"

    def _entry(self, entry: Any, source: str) -> dict[str, Any]:
        _, _, date_parser = self._dependencies()
        title = _clean(entry.get("title", ""))
        summary = _clean(entry.get("summary") or entry.get("description") or "") or title
        raw_date = entry.get("published") or entry.get("updated") or entry.get("created")
        try:
            parsed = date_parser.parse(str(raw_date))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=self.timezone)
            else:
                parsed = parsed.astimezone(self.timezone)
        except Exception:
            parsed = datetime.now(self.timezone)
        return {
            "date": parsed.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "subject": self._subject(title, summary),
            "text": summary[:600],
            "title": title,
            "url": str(entry.get("link", "") or "").strip(),
            "published_on": int(parsed.timestamp()),
        }

    def fetch_pages(
        self,
        pages: int = 1,
        lang: str = "EN",
        lts: int | None = None,
        until_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        del pages, lang, lts
        end = datetime.fromtimestamp(until_ts, self.timezone) if until_ts else datetime.now(self.timezone)
        start = end - timedelta(days=self.recent_days)
        history = self._load_history()
        fetched: list[dict[str, Any]] = []
        for feed in self.feeds:
            parsed = self._request(feed["url"])
            for entry in getattr(parsed, "entries", []) if parsed is not None else []:
                fetched.append(self._entry(entry, feed["name"]))
            if self.sleep_sec:
                time.sleep(self.sleep_sec)
        combined = _normalize_history(pd.concat([history, pd.DataFrame(fetched)], ignore_index=True))
        self._save_history(combined)
        dates = pd.to_datetime(combined["date"], errors="coerce")
        window = combined[(dates >= start.replace(tzinfo=None)) & (dates < end.replace(tzinfo=None))].copy()
        if len(window) > self.target_records:
            window = window.sort_values("date", ascending=False).head(self.target_records).sort_values("date")
        return window.to_dict("records")
