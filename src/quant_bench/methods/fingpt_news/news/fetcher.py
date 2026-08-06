from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests

from quant_bench.methods.fingpt_news.news.cleaner import clean_html_text


class CCDataNewsFetcher:
    def __init__(
        self,
        api_key: str = "",
        api_key_env: str = "CCDATA_API_KEY",
        timeout_sec: int = 15,
        max_retries: int = 5,
        sleep_sec: float = 1.0,
        timezone: str = "Asia/Shanghai",
    ):
        self.api_key = api_key or (os.getenv(api_key_env, "") if api_key_env else "")
        self.timeout_sec = int(timeout_sec)
        self.max_retries = int(max_retries)
        self.sleep_sec = float(sleep_sec)
        self.timezone = ZoneInfo(timezone)

    def _headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["authorization"] = f"Apikey {self.api_key}"
        return headers

    def fetch_pages(
        self,
        pages: int = 3,
        lang: str = "EN",
        lts: int | None = None,
        until_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        last_ts = lts
        for _page in range(max(1, int(pages))):
            url = f"https://min-api.cryptocompare.com/data/v2/news/?lang={lang}"
            if last_ts:
                url += f"&lTs={int(last_ts)}"
            data = self._request_json(url)
            if not data or data.get("Type") != 100 or not data.get("Data"):
                break
            batch = data["Data"]
            for item in batch:
                rows.append(self._normalize_item(item))
            last_ts = batch[-1].get("published_on")
            if until_ts and last_ts and int(last_ts) <= int(until_ts):
                break
            time.sleep(self.sleep_sec)
        return rows

    def _request_json(self, url: str) -> dict[str, Any] | None:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(url, headers=self._headers(), timeout=self.timeout_sec)
                return response.json()
            except Exception:
                if attempt >= self.max_retries:
                    return None
                time.sleep(min(30.0, 2.0 * attempt))
        return None

    def _normalize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        published_on = int(item.get("published_on", 0) or 0)
        date = datetime.fromtimestamp(published_on, tz=self.timezone).strftime("%Y-%m-%d %H:%M:%S") if published_on > 0 else ""
        categories = str(item.get("categories", "") or "").split("|")
        source_info = item.get("source_info") or {}
        return {
            "date": date,
            "source": source_info.get("name", item.get("source", "")),
            "subject": categories[0].lower() if categories and categories[0] else "general",
            "text": clean_html_text(item.get("body", "")),
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "published_on": published_on,
        }
