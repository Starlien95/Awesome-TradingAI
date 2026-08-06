from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from quant_bench.methods.fingpt_news.news.cleaner import compact_text, normalize_text, stable_id
from quant_bench.runtime.core.atomic_io import atomic_write_csv

COIN_PATTERNS = {
    "ADA-USDT": r"\b(?:cardano|ada)\b",
    "BTC-USDT": r"\b(?:bitcoin|btc)\b",
    "DOGE-USDT": r"\b(?:dogecoin|doge)\b",
    "ETH-USDT": r"\b(?:ethereum|ether|eth)\b",
    "HBAR-USDT": r"\b(?:hedera|hbar)\b",
    "LINK-USDT": r"\b(?:chainlink|\$link|link token|link-usdt|link/usdt)\b",
    "LTC-USDT": r"\b(?:litecoin|ltc)\b",
    "OKB-USDT": r"\b(?:okb|okx token)\b",
    "TRX-USDT": r"\b(?:tron|trx)\b",
    "XRP-USDT": r"\b(?:xrp|ripple)\b",
}

MARKET_SUBJECTS = {
    "general",
    "market",
    "markets",
    "blockchain",
    "regulation",
    "exchange",
    "exchanges",
    "defi",
    "stablecoin",
    "stablecoins",
    "security",
}

MARKET_PATTERN = re.compile(
    r"\b(?:crypto|cryptocurrency|digital asset|blockchain|defi|stablecoin|stablecoins|"
    r"tether|usdt|usdc|sec|cftc|regulation|regulatory|lawsuit|court|etf|spot etf|"
    r"federal reserve|fed|interest rate|inflation|macro|binance|coinbase|kraken|okx|"
    r"hack|hacked|exploit|security breach|blackrock|grayscale|microstrategy)\b",
    re.IGNORECASE,
)
MAJOR_ASSET_PATTERN = re.compile(r"\b(?:bitcoin|btc|ethereum|ether|eth)\b", re.IGNORECASE)
NEWS_SENTIMENT_INSTRUCTION = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}"


def prepare_raw(raw: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    news = raw.copy()
    news["datetime"] = pd.to_datetime(news["date"], errors="coerce")
    news = news.dropna(subset=["datetime"]).copy()
    if start:
        news = news[news["datetime"] >= pd.Timestamp(start)]
    if end:
        news = news[news["datetime"] < pd.Timestamp(end)]
    news["date"] = news["datetime"].dt.floor("D")
    news["subject_norm"] = news["subject"].fillna("").astype(str).str.lower().str.strip()
    news["title_text"] = news["title"].fillna("").astype(str)
    news["body_text"] = news["text"].fillna("").astype(str)
    news["combined_text"] = (news["subject_norm"] + " " + news["title_text"] + " " + news["body_text"]).str.lower()
    news["headline_norm"] = news["title_text"].map(normalize_text)
    return news[news["headline_norm"].str.len() > 0].copy()


def source_cap(df: pd.DataFrame, group_cols: list[str], max_source_news_per_day: int) -> pd.DataFrame:
    if df.empty or max_source_news_per_day <= 0:
        return df
    work = df.copy()
    work["source_norm"] = work["source"].fillna("").astype(str).str.lower()
    work["source_rank"] = work.groupby([*group_cols, "source_norm"]).cumcount()
    return work[work["source_rank"] < max_source_news_per_day].drop(columns=["source_norm", "source_rank"])


def input_text(row: pd.Series, news_text_chars: int) -> str:
    title = compact_text(row.get("title", ""), 280)
    body = compact_text(row.get("text", ""), news_text_chars)
    if body and body.lower() not in title.lower():
        return compact_text(f"{title} {body}", news_text_chars + 320)
    return compact_text(title or body, news_text_chars + 320)


def build_candidates(raw: pd.DataFrame, cfg: dict, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    news = prepare_raw(raw, start=start, end=end)
    if news.empty:
        return pd.DataFrame()
    rows = []
    asset_weights = cfg["news"]["asset_relevance_weight"]
    for symbol in cfg["trade"]["coins"]:
        asset = symbol.split("-")[0]
        pattern = re.compile(COIN_PATTERNS[symbol], re.IGNORECASE)
        subject_mask = news["subject_norm"].eq(asset.lower())
        title_mask = news["title_text"].str.contains(pattern, regex=True, na=False)
        text_mask = news["body_text"].str.contains(pattern, regex=True, na=False)
        df = news[subject_mask | title_mask | text_mask].copy()
        if df.empty:
            continue
        df["bucket"] = "asset"
        df["symbol"] = symbol
        df["asset"] = asset
        df["match_level"] = "text_keyword"
        df.loc[title_mask.loc[df.index], "match_level"] = "title_keyword"
        df.loc[subject_mask.loc[df.index], "match_level"] = "subject"
        df["relevance_weight"] = df["match_level"].map(asset_weights).astype(float)
        df = df.sort_values(["symbol", "date", "relevance_weight", "datetime"], ascending=[True, True, False, False])
        df = source_cap(df, ["symbol", "date"], int(cfg["news"]["max_source_news_per_day"]))
        rows.append(df)

    market_weights = cfg["news"]["market_relevance_weight"]
    subject_mask = news["subject_norm"].isin(MARKET_SUBJECTS)
    title_mask = news["title_text"].str.contains(MARKET_PATTERN, regex=True, na=False)
    text_mask = news["body_text"].str.contains(MARKET_PATTERN, regex=True, na=False)
    major_mask = news["combined_text"].str.contains(MAJOR_ASSET_PATTERN, regex=True, na=False)
    market = news[subject_mask | title_mask | text_mask | major_mask].copy()
    if not market.empty:
        market["bucket"] = "market"
        market["symbol"] = "MARKET"
        market["asset"] = "MARKET"
        market["match_level"] = "major_asset"
        market.loc[text_mask.loc[market.index], "match_level"] = "market_text"
        market.loc[title_mask.loc[market.index], "match_level"] = "market_title"
        market.loc[subject_mask.loc[market.index], "match_level"] = "market_subject"
        market["relevance_weight"] = market["match_level"].map(market_weights).astype(float)
        market = market.sort_values(["date", "relevance_weight", "datetime"], ascending=[True, False, False])
        market = source_cap(market, ["date"], int(cfg["news"]["max_source_news_per_day"]))
        rows.append(market)

    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if combined.empty:
        return combined
    combined["input"] = combined.apply(lambda row: input_text(row, int(cfg["news"]["news_text_chars"])), axis=1)
    combined = combined[combined["input"].str.len() > 0].copy()
    combined["instruction"] = NEWS_SENTIMENT_INSTRUCTION
    combined["news_id"] = [
        stable_id(row.bucket, row.symbol, row.datetime, row.raw_news_id, row.match_level, prefix="live_")
        for row in combined.itertuples(index=False)
    ]
    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
    combined["datetime"] = pd.to_datetime(combined["datetime"]).astype(str)
    columns = [
        "bucket",
        "symbol",
        "asset",
        "date",
        "datetime",
        "news_id",
        "raw_news_id",
        "source",
        "subject",
        "title",
        "url",
        "instruction",
        "input",
        "relevance_weight",
        "match_level",
    ]
    return combined[columns].drop_duplicates("news_id").sort_values(["date", "bucket", "symbol", "datetime"])


def write_candidates(candidates: pd.DataFrame, path: str | Path) -> None:
    atomic_write_csv(candidates, path)
