# historical_news_collector.py

from datetime import datetime
from pathlib import Path
import ast
import time
from urllib.parse import quote_plus, urlparse

import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from textblob import TextBlob


DEFAULT_LIST_FILE = Path(__file__).resolve().parents[1] / "resources" / "news_sources.md"
REQUEST_DELAY_SECONDS = 1.0
DEFAULT_MAX_PAGES = 2

feedparser.USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def google_news_rss(query):
    encoded = quote_plus(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"


SOURCE_FEEDS = {
    "CoinDesk": "https://feeds.feedburner.com/CoinDesk",
    "Bloomberg (Crypto)": [
        "https://feeds.bloomberg.com/crypto/news.rss",
        google_news_rss("site:bloomberg.com crypto OR bitcoin OR ethereum"),
    ],
    "Financial Times (Crypto)": [
        "https://www.ft.com/cryptocurrencies?format=rss",
        google_news_rss("site:ft.com crypto OR bitcoin OR ethereum"),
    ],
    "Decrypt": [
        "https://decrypt.co/feed",
        "https://decrypt.co/feed?category=defi",
        google_news_rss("site:decrypt.co bitcoin OR ethereum OR solana"),
    ],
    "Cointelegraph": [
        "https://cointelegraph.com/rss",
        "https://cointelegraph.com/rss/tag/bitcoin",
        "https://cointelegraph.com/rss/tag/ethereum",
        "https://cointelegraph.com/rss/tag/blockchain",
        "https://cointelegraph.com/rss/tag/defi",
    ],
    "Forbes Digital Assets": [
        "https://www.forbes.com/digital-assets/feed/",
        google_news_rss("site:forbes.com/sites/digital-assets crypto OR bitcoin OR ethereum"),
    ],
    "Investing.com Crypto News": [
        "https://www.investing.com/rss/news_301.rss",
        google_news_rss("site:investing.com/news/cryptocurrency-news crypto OR bitcoin OR ethereum"),
    ],
    "Crypto Potato": [
        "https://cryptopotato.com/feed/",
        google_news_rss("site:cryptopotato.com bitcoin OR ethereum OR solana"),
    ],
    "Bitcoin.com": [
        "https://news.bitcoin.com/feed/",
        "https://news.bitcoin.com/category/crypto-news/feed/",
        "https://news.bitcoin.com/category/market-updates/feed/",
        "https://news.bitcoin.com/category/finance/feed/",
        "https://news.bitcoin.com/category/regulation-and-legal/feed/",
        "https://news.bitcoin.com/category/blockchain/feed/",
        "https://news.bitcoin.com/category/mining/feed/",
    ],
    "U.Today": [
        "https://u.today/rss",
        google_news_rss("site:u.today bitcoin OR ethereum OR xrp"),
    ],
    "NewsBTC": [
        "https://www.newsbtc.com/feed/",
        "https://www.newsbtc.com/news/feed/",
        "https://www.newsbtc.com/news/bitcoin/feed/",
        "https://www.newsbtc.com/news/ethereum/feed/",
        "https://www.newsbtc.com/analysis/feed/",
    ],
}

NEWS_SUBJECT_MAP = {
    "bitcoin": "btc",
    "ethereum": "eth",
    "solana": "sol",
    "dogecoin": "doge",
    "macro/regulation": "regulation",
    "stablecoin": "stablecoin",
    "defi": "defi",
    "blockchain": "blockchain",
    "altcoin": "altcoin",
    "general": "general",
}


def load_high_quality_sources(list_file=DEFAULT_LIST_FILE):
    text = Path(list_file).read_text(encoding="utf-8")
    module = ast.parse(text, filename=str(list_file))
    for node in module.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "HIGH_QUALITY_SOURCES" in names:
                value = ast.literal_eval(node.value)
                return {str(k): float(v) for k, v in value.items()}
    raise ValueError(f"未在 {list_file} 中找到 HIGH_QUALITY_SOURCES")


def paged_feed_url(url, page):
    if page <= 1:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}paged={page}"


def should_try_pagination(url):
    host = urlparse(url).netloc.lower()
    return any(
        domain in host
        for domain in ["news.bitcoin.com", "newsbtc.com", "cryptopotato.com", "decrypt.co"]
    )


def _normalize_news_subject(subject):
    subject = str(subject or "general").lower().strip()
    return NEWS_SUBJECT_MAP.get(subject, subject)


def clean_html_text(raw_html):
    if not raw_html:
        return ""
    return BeautifulSoup(str(raw_html), "html.parser").get_text(
        separator=" ",
        strip=True,
    )


def clean_text(text):
    if not text:
        return ""
    return " ".join(str(text).split())


def get_sentiment(text):
    if not text:
        return "{'class': 'neutral', 'polarity': 0.0, 'subjectivity': 0.0}"

    blob = TextBlob(str(text))
    polarity = round(blob.sentiment.polarity, 2)
    subjectivity = round(blob.sentiment.subjectivity, 2)

    if polarity > 0.05:
        sentiment_class = "positive"
    elif polarity < -0.05:
        sentiment_class = "negative"
    else:
        sentiment_class = "neutral"

    return (
        "{"
        f"'class': '{sentiment_class}', "
        f"'polarity': {polarity}, "
        f"'subjectivity': {subjectivity}"
        "}"
    )


def get_subject(title, text, entry_tags=None):
    target_coins = {
        "bitcoin": ["bitcoin", "btc"],
        "ethereum": ["ethereum", "eth"],
        "solana": ["solana", "sol"],
        "xrp": ["xrp", "ripple"],
        "dogecoin": ["dogecoin", "doge"],
        "stablecoin": ["stablecoin", "usdt", "usdc", "tether", "circle"],
        "altcoin": ["altcoin", "altcoins"],
        "defi": ["defi", "decentralized finance"],
        "blockchain": ["blockchain", "web3"],
        "macro/regulation": [
            "usoil",
            "oil",
            "fed",
            "interest rate",
            "sec",
            "cftc",
            "etf",
            "regulation",
            "regulatory",
            "lawsuit",
        ],
    }
    content = f" {title} {text} ".lower()
    for subject, keywords in target_coins.items():
        if any(keyword in content for keyword in keywords):
            return subject

    if entry_tags:
        blacklist = {
            "news",
            "aa news",
            "market",
            "markets",
            "analysis",
            "crypto",
            "cryptocurrency",
            "crypto news",
            "press release",
            "daily",
        }
        for tag in entry_tags:
            term = str(getattr(tag, "term", "") or tag.get("term", "")).strip().lower()
            if term and term not in blacklist:
                return term

    return "general"


def format_date(date_string):
    try:
        parsed = dateparser.parse(str(date_string))
        if parsed is None:
            raise ValueError("empty date")
        if parsed.tzinfo is not None:
            parsed = parsed.tz_convert(None) if hasattr(parsed, "tz_convert") else parsed.replace(tzinfo=None)
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fetch_rss(url, source_name, retries=2):
    for attempt in range(1, retries + 1):
        try:
            feed = feedparser.parse(url)
            if feed.entries:
                return feed
            error = getattr(feed, "bozo_exception", None)
            if error:
                print(f"  [!] {source_name}: {url} 解析异常: {error}")
        except Exception as exc:
            print(f"  [!] {source_name}: {url} 请求失败 ({attempt}/{retries}): {exc}")
        if attempt < retries:
            time.sleep(REQUEST_DELAY_SECONDS)
    return None


def parse_rss_entries(feed, source_name):
    rows = []
    for entry in feed.entries:
        title = clean_html_text(entry.get("title", ""))
        url = str(entry.get("link", "")).strip()
        raw_date = (
            entry.get("published")
            or entry.get("updated")
            or entry.get("created")
            or entry.get("pubDate")
            or ""
        )
        raw_summary = entry.get("summary") or entry.get("description") or entry.get("content", "")
        clean_txt = clean_text(clean_html_text(raw_summary))
        tags = entry.get("tags", None)
        rows.append(
            {
                "date": format_date(raw_date),
                "sentiment": get_sentiment(clean_txt),
                "source": source_name,
                "subject": get_subject(title, clean_txt, tags),
                "text": clean_txt if clean_txt else title,
                "title": title,
                "url": url,
            }
        )
    return rows


def crawl_source(source_name, since_date, max_pages=DEFAULT_MAX_PAGES, verbose=True):
    feed_urls = SOURCE_FEEDS.get(source_name)
    if not feed_urls:
        if verbose:
            print(f"[!] {source_name}: 未配置 RSS/Google News 源，跳过。")
        return []

    if isinstance(feed_urls, str):
        feed_urls = [feed_urls]

    rows = []
    seen = set()
    since_key = pd.Timestamp(since_date).strftime("%Y-%m-%d")

    for feed_url in feed_urls:
        pages = max_pages if should_try_pagination(feed_url) else 1
        for page in range(1, pages + 1):
            url = paged_feed_url(feed_url, page)
            feed = fetch_rss(url, source_name)
            if not feed:
                if verbose and page == 1:
                    print(f"  [!] 无法获取: {url}")
                break

            articles = parse_rss_entries(feed, source_name)
            page_rows = []
            for article in articles:
                key = (
                    str(article.get("title", "")).strip(),
                    str(article.get("url", "")).strip(),
                )
                if key in seen:
                    continue
                seen.add(key)
                if article["date"] >= since_key:
                    page_rows.append(article)

            rows.extend(page_rows)
            oldest = min((r["date"] for r in page_rows), default="无")
            if verbose:
                print(
                    f"  使用: {url} | RSS {len(feed.entries)} 条 | "
                    f"新增 {len(page_rows)} 条 | 最旧 {oldest}"
                )

            if page > 1 and not page_rows:
                break
            if page > 1 and oldest != "无" and oldest < since_key:
                break
            time.sleep(REQUEST_DELAY_SECONDS)

    if verbose:
        print(f"  {source_name}: 合计 {len(rows)} 条")
    return rows


def crawl_all_sources(since_date, max_pages=DEFAULT_MAX_PAGES, verbose=True):
    sources = load_high_quality_sources()
    rows = []

    if verbose:
        print("=" * 80)
        print("[CRAWL NEWS SOURCES]")
        print("sources:", list(sources.keys()))
        print("since_date:", pd.Timestamp(since_date).strftime("%Y-%m-%d"))
        print("max_pages:", max_pages)
        print("=" * 80)

    for source_name in sources.keys():
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"数据源: {source_name}")
        rows.extend(crawl_source(source_name, since_date, max_pages=max_pages, verbose=verbose))
        time.sleep(REQUEST_DELAY_SECONDS)

    return rows


def _normalize_date(date_like):
    return pd.to_datetime(date_like).date()


def fetch_historical_news_for_window(
    start_date,
    end_date,
    target_records=5000,
    max_pages=200,
    api_key=None,
    sleep_seconds=0.5,
    lang="EN",
    timeout=20,
    verbose=True,
):
    """
    按日期窗口抓取历史新闻。

    参数
    ----
    start_date, end_date:
        'YYYY-MM-DD' 或 date-like。
        返回结果只保留 [start_date, end_date] 内的新闻。
        日期按 Asia/Shanghai 计算，用来对齐 OKX 1D K线。

    target_records:
        窗口内最多保留多少条新闻。注意：这个值如果太小，可能还没回溯到
        start_date 就停止，所以默认给 5000。

    max_pages:
        对支持分页的网站最多回溯页数，防止极端情况下请求过多。

    返回
    ----
    pd.DataFrame，字段兼容你的 NewsProcessor:
        date, sentiment, source, subject, text, title, url

    注意
    ----
    不会把某一天的新闻复制给其他日期。现在默认使用本模块内置的
    list.md 驱动 RSS/Google News 来源，不再使用 CCData/CryptoCompare。
    """
    start_date = _normalize_date(start_date)
    end_date = _normalize_date(end_date)

    if start_date > end_date:
        raise ValueError(f"start_date {start_date} > end_date {end_date}")

    if verbose:
        print("=" * 80)
        print("[FETCH HISTORICAL NEWS FOR WINDOW]")
        print("start_date:", start_date)
        print("end_date:", end_date)
        print("target_records:", target_records)
        print("max_pages:", max_pages)
        print("collector: list.md RSS/Google News")
        print("=" * 80)

    columns_order = ["date", "sentiment", "source", "subject", "text", "title", "url"]

    start_dt = pd.Timestamp(start_date).to_pydatetime()

    rows = crawl_all_sources(
        since_date=start_dt,
        max_pages=int(max_pages),
        verbose=verbose,
    )

    if not rows:
        if verbose:
            print("[WARNING] No historical news found for this window.")
        return pd.DataFrame(columns=columns_order)

    df = pd.DataFrame(rows)

    for col in columns_order:
        if col not in df.columns:
            df[col] = ""

    parsed_date = pd.to_datetime(df["date"], errors="coerce")
    df = df.loc[~parsed_date.isna()].copy()
    parsed_date = parsed_date.loc[df.index]

    date_mask = (parsed_date.dt.date >= start_date) & (parsed_date.dt.date <= end_date)
    df = df.loc[date_mask].copy()
    parsed_date = parsed_date.loc[df.index]

    if df.empty:
        if verbose:
            print("[WARNING] No historical news found for this window.")
        return pd.DataFrame(columns=columns_order)

    df["date"] = parsed_date.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["subject"] = df["subject"].map(_normalize_news_subject)
    df["text"] = df["text"].fillna("").astype(str)
    df["text"] = df["text"].map(lambda x: x[:300] + "..." if len(x) > 300 else x)

    dedup_cols = [c for c in ["title", "url"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep="last")
    else:
        df = df.drop_duplicates(keep="last")

    if target_records is not None and len(df) > int(target_records):
        max_records = int(target_records)
        required_dates = _date_range_strings(start_date, end_date)
        per_day_quota = max(1, max_records // max(len(required_dates), 1))

        tmp = df.sort_values("date", ascending=False).copy()
        tmp["__date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        balanced = (
            tmp.groupby("__date", group_keys=False)
            .head(per_day_quota)
            .copy()
        )

        if len(balanced) < max_records:
            remaining = tmp.drop(index=balanced.index, errors="ignore")
            fill = remaining.head(max_records - len(balanced))
            balanced = pd.concat([balanced, fill], ignore_index=False)

        df = balanced.drop(columns=["__date"]).head(max_records)

    df = df[columns_order].copy()
    df = df.sort_values("date").reset_index(drop=True)

    if verbose:
        print("=" * 80)
        print("[NEWS WINDOW RESULT]")
        print("rows:", len(df))
        print("date range:", df["date"].min(), "->", df["date"].max())

        counts = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").value_counts().sort_index()
        print("[NEWS COUNT BY DATE]")
        print(counts.to_string())

        show_cols = ["date", "subject", "title"]
        print(df[show_cols].head(10).to_string(index=False))
        print("=" * 80)

    return df


# ============================================================
# News history cache helpers
# ============================================================

NEWS_HISTORY_COLUMNS = ["date", "sentiment", "source", "subject", "text", "title", "url"]


def normalize_news_history_df(df):
    """
    Normalize a news DataFrame so it can be safely stored/reused as a local cache.

    Required output columns:
        date, sentiment, source, subject, text, title, url

    date is kept as "YYYY-MM-DD HH:MM:SS" when possible.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=NEWS_HISTORY_COLUMNS)

    df = df.copy()

    for col in NEWS_HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Parse and normalize date.
    parsed_date = pd.to_datetime(df["date"], errors="coerce")
    df = df.loc[~parsed_date.isna()].copy()
    parsed_date = parsed_date.loc[df.index]
    df["date"] = parsed_date.dt.strftime("%Y-%m-%d %H:%M:%S")

    # Normalize text-like columns.
    for col in ["sentiment", "source", "subject", "text", "title", "url"]:
        df[col] = df[col].fillna("").astype(str)

    # Deduplicate by title + url when available.
    if not df.empty:
        df = df.drop_duplicates(subset=["title", "url"], keep="last")
        df = df.sort_values("date").reset_index(drop=True)

    return df[NEWS_HISTORY_COLUMNS].copy()


def load_news_history(history_path):
    history_path = Path(history_path)

    if not history_path.exists():
        return pd.DataFrame(columns=NEWS_HISTORY_COLUMNS)

    try:
        df = pd.read_csv(history_path)
        return normalize_news_history_df(df)
    except Exception as e:
        print(f"[WARNING] Failed to read news history cache: {history_path}")
        print(e)
        return pd.DataFrame(columns=NEWS_HISTORY_COLUMNS)


def save_news_history(df, history_path):
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    df = normalize_news_history_df(df)
    df.to_csv(history_path, index=False, encoding="utf-8-sig")

    return str(history_path)


def _date_range_strings(start_date, end_date):
    start_date = _normalize_date(start_date)
    end_date = _normalize_date(end_date)

    if start_date > end_date:
        raise ValueError(f"start_date {start_date} > end_date {end_date}")

    return [
        d.strftime("%Y-%m-%d")
        for d in pd.date_range(start=start_date, end=end_date, freq="D")
    ]


def _news_count_by_date(df):
    if df is None or df.empty:
        return {}

    tmp = normalize_news_history_df(df)

    if tmp.empty:
        return {}

    dates = pd.to_datetime(tmp["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return dates.value_counts().sort_index().astype(int).to_dict()


def _filter_news_window(df, start_date, end_date):
    df = normalize_news_history_df(df)

    if df.empty:
        return df

    start_date = _normalize_date(start_date)
    end_date = _normalize_date(end_date)

    dates = pd.to_datetime(df["date"], errors="coerce").dt.date
    mask = (dates >= start_date) & (dates <= end_date)

    return df.loc[mask].copy().reset_index(drop=True)


def merge_news_history(old_df, new_df):
    old_df = normalize_news_history_df(old_df)
    new_df = normalize_news_history_df(new_df)

    merged = pd.concat([old_df, new_df], ignore_index=True)
    merged = normalize_news_history_df(merged)

    return merged


def get_news_for_window_with_cache(
    start_date,
    end_date,
    history_path="live_data/news_history.csv",
    target_records=8000,
    max_pages=400,
    min_news_per_day=1,
    fetch_if_missing=True,
    sleep_seconds=0.5,
    verbose=True,
    return_summary=False,
):
    """
    Return historical news for [start_date, end_date] using a local CSV cache.

    Workflow:
      1. Load local news_history.csv.
      2. Check whether each date in the rolling window has at least
         min_news_per_day cached records.
      3. If some dates are missing, fetch only the missing date range from
         this module's list.md-driven RSS/Google News collector.
      4. Merge, deduplicate, save back to news_history.csv.
      5. Return only the requested window.

    This prevents re-fetching the entire 10-day window every time.
    After the cache is filled, daily live trading usually only needs to fetch
    the newly confirmed latest date.
    """
    history_path = Path(history_path)
    required_dates = _date_range_strings(start_date, end_date)

    history_df = load_news_history(history_path)
    counts_before = _news_count_by_date(_filter_news_window(history_df, start_date, end_date))

    missing_dates = [
        d for d in required_dates
        if int(counts_before.get(d, 0)) < int(min_news_per_day)
    ]

    fetched_df = pd.DataFrame(columns=NEWS_HISTORY_COLUMNS)
    fetched_range = None

    if verbose:
        print("=" * 80)
        print("[NEWS CACHE]")
        print("history_path:", history_path)
        print("window:", f"{start_date} -> {end_date}")
        print("required_dates:", required_dates)
        print("counts_before:", counts_before)
        print("missing_dates:", missing_dates)

    if missing_dates and fetch_if_missing:
        fetch_start = min(missing_dates)
        fetch_end = max(missing_dates)
        fetched_range = [fetch_start, fetch_end]

        if verbose:
            print("=" * 80)
            print("[NEWS CACHE MISS]")
            print("fetch_start:", fetch_start)
            print("fetch_end:", fetch_end)

        fetched_df = fetch_historical_news_for_window(
            start_date=fetch_start,
            end_date=fetch_end,
            target_records=target_records,
            max_pages=max_pages,
            sleep_seconds=sleep_seconds,
            verbose=verbose,
        )

        history_df = merge_news_history(history_df, fetched_df)
        save_news_history(history_df, history_path)

    elif missing_dates and not fetch_if_missing:
        if verbose:
            print("[NEWS CACHE] missing dates exist, but fetch_if_missing=False")

    else:
        if verbose:
            print("[NEWS CACHE HIT] No API fetch required for this window.")

    window_df = _filter_news_window(history_df, start_date, end_date)
    counts_after = _news_count_by_date(window_df)

    summary = {
        "history_path": str(history_path),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "required_dates": required_dates,
        "missing_dates_before_fetch": missing_dates,
        "fetched": bool(missing_dates and fetch_if_missing),
        "fetched_range": fetched_range,
        "fetched_rows": int(len(fetched_df)) if fetched_df is not None else 0,
        "window_rows": int(len(window_df)),
        "counts_before": counts_before,
        "counts_after": counts_after,
    }

    if verbose:
        print("=" * 80)
        print("[NEWS CACHE RESULT]")
        print(json_dump_pretty(summary))
        print("=" * 80)

    if return_summary:
        return window_df, summary

    return window_df


def json_dump_pretty(obj):
    import json
    return json.dumps(obj, ensure_ascii=False, indent=2)
