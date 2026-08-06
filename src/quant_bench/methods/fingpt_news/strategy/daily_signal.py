from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_bench.runtime.core.atomic_io import atomic_write_csv

LABEL_SCORE = {"negative": -1, "neutral": 0, "positive": 1}


def weighted_score(group: pd.DataFrame, weight_col: str = "effective_weight") -> float:
    weights = pd.to_numeric(group[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    scores = pd.to_numeric(group["article_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(scores) == 0:
        return 0.0
    if weights.sum() <= 0:
        return float(np.mean(scores))
    return float(np.average(scores, weights=weights))


def label_from_score(score: float, threshold: float) -> str:
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"


def load_per_coin_params(path: str | Path) -> dict[str, dict[str, Any]]:
    df = pd.read_csv(path)
    return {str(row["symbol"]): row for row in df.to_dict("records")}


def build_daily_signals(
    sentiment: pd.DataFrame,
    per_coin_params: dict[str, dict[str, Any]],
    coins: list[str],
    news_date: str,
    trade_date: str,
    window_start: str | None = None,
    window_end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = sentiment.copy()
    if work.empty:
        return _empty_signals(coins, news_date, trade_date, window_start, window_end), pd.DataFrame()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    work = work.dropna(subset=["datetime"]).copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if window_start and window_end:
        start_ts = pd.Timestamp(window_start)
        end_ts = pd.Timestamp(window_end)
        work = work[(work["datetime"] >= start_ts) & (work["datetime"] < end_ts)].copy()
    else:
        work = work[work["date"] == news_date].copy()
    work["article_score"] = pd.to_numeric(work["article_score"], errors="coerce").fillna(0.0)
    work["relevance_weight"] = pd.to_numeric(work["relevance_weight"], errors="coerce").fillna(1.0)
    work["abs_article_score"] = work["article_score"].abs()

    signal_rows = []
    attribution_rows = []
    market_all = work[work["bucket"] == "market"].copy()
    for symbol in coins:
        params = per_coin_params[symbol]
        min_abs = float(params.get("agg_min_abs_article_score", 0.0) or 0.0)
        score_power = float(params.get("agg_score_power", 0.0) or 0.0)
        asset_top_n = int(params.get("agg_asset_top_n", 0) or 0)
        market_top_n = int(params.get("agg_market_top_n", 0) or 0)
        asset = work[(work["bucket"] == "asset") & (work["symbol"] == symbol)].copy()
        market = market_all.copy()
        if min_abs > 0:
            asset = asset[asset["abs_article_score"] >= min_abs].copy()
            market = market[market["abs_article_score"] >= min_abs].copy()
        for df in (asset, market):
            if df.empty:
                continue
            if score_power != 1.0:
                df["effective_weight"] = df["relevance_weight"] * np.power(df["abs_article_score"].clip(lower=1e-9), score_power)
            else:
                df["effective_weight"] = df["relevance_weight"]
        if not asset.empty:
            asset = asset.sort_values(["relevance_weight", "abs_article_score", "datetime"], ascending=[False, False, False]).head(asset_top_n)
        if not market.empty:
            market = market.sort_values(["relevance_weight", "abs_article_score", "datetime"], ascending=[False, False, False]).head(market_top_n)
        asset_score = weighted_score(asset) if not asset.empty else 0.0
        market_score = weighted_score(market) if not market.empty else 0.0
        asset_weight_sum = float(asset["effective_weight"].sum()) if not asset.empty else 0.0
        market_weight_sum = float(market["effective_weight"].sum()) if not market.empty else 0.0
        variant = params.get("signal_variant")
        if variant == "asset_only":
            alpha = 1.0
            final_score = asset_score
        elif variant == "market_only":
            alpha = 0.0
            final_score = market_score
        elif variant == "hybrid_adaptive":
            k = float(params.get("signal_adaptive_k", 4.0) or 4.0)
            min_alpha = float(params.get("signal_min_alpha", 0.65) or 0.65)
            max_alpha = float(params.get("signal_max_alpha", 0.95) or 0.95)
            alpha = float(np.clip(asset_weight_sum / (asset_weight_sum + k), min_alpha, max_alpha))
            final_score = alpha * asset_score + (1.0 - alpha) * market_score
        else:
            alpha = float(params.get("signal_alpha", 0.8) or 0.8)
            final_score = alpha * asset_score + (1.0 - alpha) * market_score
        threshold = float(params.get("signal_threshold", 0.08) or 0.08)
        conflict_policy = str(params.get("signal_conflict_policy", "none") or "none")
        conflict_threshold = float(params.get("signal_conflict_threshold", 0.08) or 0.08)
        if (
            conflict_policy == "neutral"
            and np.sign(asset_score) != np.sign(market_score)
            and abs(asset_score) >= conflict_threshold
            and abs(market_score) >= conflict_threshold
        ):
            label = "neutral"
        else:
            label = label_from_score(final_score, threshold)
        signal_rows.append(
            {
                "trade_date": trade_date,
                "news_date": news_date,
                "news_window_start": window_start or news_date,
                "news_window_end": window_end or news_date,
                "symbol": symbol,
                "asset_score": asset_score,
                "market_score": market_score,
                "final_score": final_score,
                "threshold": threshold,
                "model_label": label,
                "model_score": LABEL_SCORE[label],
                "alpha": alpha,
                "asset_news_count": len(asset),
                "market_news_count": len(market),
                "asset_weight_sum": asset_weight_sum,
                "market_weight_sum": market_weight_sum,
                "strategy_variant": variant,
                "created_at": pd.Timestamp.utcnow().isoformat(),
            }
        )
        for bucket_df in (asset, market):
            for rank, row in enumerate(bucket_df.to_dict("records"), start=1):
                attribution_rows.append(
                    {
                        "trade_date": trade_date,
                        "news_date": news_date,
                        "news_window_start": window_start or news_date,
                        "news_window_end": window_end or news_date,
                        "symbol": symbol,
                        "bucket": row.get("bucket", ""),
                        "news_id": row.get("news_id", ""),
                        "datetime": row.get("datetime", ""),
                        "source": row.get("source", ""),
                        "title": row.get("title", ""),
                        "url": row.get("url", ""),
                        "article_score": row.get("article_score", 0.0),
                        "relevance_weight": row.get("relevance_weight", 0.0),
                        "effective_weight": row.get("effective_weight", 0.0),
                        "used_in_signal": True,
                        "rank_within_bucket": rank,
                    }
                )
    return pd.DataFrame(signal_rows), pd.DataFrame(attribution_rows)


def _empty_signals(
    coins: list[str],
    news_date: str,
    trade_date: str,
    window_start: str | None = None,
    window_end: str | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "news_date": news_date,
                "news_window_start": window_start or news_date,
                "news_window_end": window_end or news_date,
                "symbol": symbol,
                "asset_score": 0.0,
                "market_score": 0.0,
                "final_score": 0.0,
                "threshold": 0.0,
                "model_label": "neutral",
                "model_score": 0,
                "alpha": 0.0,
                "asset_news_count": 0,
                "market_news_count": 0,
                "asset_weight_sum": 0.0,
                "market_weight_sum": 0.0,
                "strategy_variant": "empty",
                "created_at": pd.Timestamp.utcnow().isoformat(),
            }
            for symbol in coins
        ]
    )


def append_daily_outputs(signals: pd.DataFrame, attribution: pd.DataFrame, signals_path: str | Path, attribution_path: str | Path) -> None:
    existing = pd.read_csv(signals_path) if Path(signals_path).exists() else pd.DataFrame()
    combined = pd.concat([existing, signals], ignore_index=True)
    combined = combined.drop_duplicates(["trade_date", "symbol"], keep="last")
    atomic_write_csv(combined, signals_path)
    if not attribution.empty:
        old_attr = pd.read_csv(attribution_path) if Path(attribution_path).exists() else pd.DataFrame()
        attr = pd.concat([old_attr, attribution], ignore_index=True)
        attr = attr.drop_duplicates(["trade_date", "symbol", "bucket", "news_id"], keep="last")
        atomic_write_csv(attr, attribution_path)
