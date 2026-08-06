"""Versioned metadata for the 52-feature crypto Qlib handler."""

from __future__ import annotations

import hashlib
import json


class CryptoAlpha52V1:
    feature_set_id = "crypto_alpha52_v1"
    version = "1"
    max_lookback_bars = 288

    EXPRESSIONS = (
        "($close / Ref($close, 6) - 1) * 10",
        "($close / Ref($close, 12) - 1) * 10",
        "($close / Ref($close, 24) - 1) * 5",
        "($close / Ref($close, 48) - 1) * 5",
        "($close / Ref($close, 96) - 1) * 3",
        "($close / Ref($close, 144) - 1) * 3",
        "($close / Ref($close, 288) - 1) * 2",
        "(($close / Ref($close, 6) - 1) - (Ref($close, 6) / Ref($close, 12) - 1)) * 20",
        "(Mean($close, 12) / Ref(Mean($close, 12), 6) - 1) * 20",
        "(Mean($close, 48) / Ref(Mean($close, 48), 12) - 1) * 10",
        "(Mean($close, 144) / Ref(Mean($close, 144), 24) - 1) * 5",
        "($close / (Mean($close, 12) + 1e-12) - 1) * 10",
        "($close / (Mean($close, 48) + 1e-12) - 1) * 5",
        "($close / (Mean($close, 144) + 1e-12) - 1) * 3",
        "($close / (Mean($close, 288) + 1e-12) - 1) * 2",
        "(Mean($close, 12) - Ref(Mean($close, 12), 6)) / (Mean($close, 12) + 1e-12) * 20",
        "(Mean($close, 48) - Ref(Mean($close, 48), 12)) / (Mean($close, 48) + 1e-12) * 10",
        "(Mean($close, 12) - Mean($close, 48)) / (Mean($close, 48) + 1e-12) * 10",
        "(EMA($close, 24) - EMA($close, 52)) / ($close + 1e-12) * 100",
        "EMA(EMA($close, 24) - EMA($close, 52), 18) / ($close + 1e-12) * 100",
        "((EMA($close, 24) - EMA($close, 52)) - EMA(EMA($close, 24) - EMA($close, 52), 18)) / ($close + 1e-12) * 100",
        "Std($close, 12) / (Mean(Std($close, 12), 96) + 1e-12) - 1",
        "Std($close, 48) / (Mean(Std($close, 48), 96) + 1e-12) - 1",
        "Std($close, 144) / (Mean(Std($close, 144), 96) + 1e-12) - 1",
        "Std($close, 12) / (Std($close, 48) + 1e-12) - 1",
        "Std($close, 48) / (Std($close, 144) + 1e-12) - 1",
        "Std($close, 24) / (Mean(Std($close, 24), 96) + 1e-12) - 1",
        "(2 * Std($close, 48) / (Mean($close, 48) + 1e-12)) / (Mean(2 * Std($close, 48) / (Mean($close, 48) + 1e-12), 96) + 1e-12) - 1",
        "($close - (Mean($close, 48) - 2 * Std($close, 48))) / (4 * Std($close, 48) + 1e-12) - 0.5",
        "Std($close, 24) / (Std($close, 96) + 1e-12) - 1",
        "Log($volume / (Mean($volume, 24) + 1e-12) + 1e-12)",
        "Log($volume / (Mean($volume, 96) + 1e-12) + 1e-12)",
        "Corr($close, $volume, 48)",
        "Corr($close, $volume, 144)",
        "(Std($volume, 24) / (Mean($volume, 24) + 1e-12)) / (Mean(Std($volume, 24) / (Mean($volume, 24) + 1e-12), 96) + 1e-12) - 1",
        "Mean($volume, 12) / (Mean($volume, 48) + 1e-12) - 1",
        "($close - $open) / ($high - $low + 1e-12)",
        "(($high - $low) / ($close + 1e-12)) / (Mean(($high - $low) / ($close + 1e-12), 96) + 1e-12) - 1",
        "($high - If($open > $close, $open, $close)) / ($high - $low + 1e-12)",
        "(If($open < $close, $open, $close) - $low) / ($high - $low + 1e-12)",
        "(2*$close - $high - $low) / ($high - $low + 1e-12)",
        "Sum(If($close > Ref($close, 1), 1, 0), 24) / 24 - 0.5",
        "Sum(If($close > Ref($close, 1), 1, 0), 96) / 96 - 0.5",
        "Sum(If($close > Ref($close, 1), $close - Ref($close, 1), 0), 24) / (Sum(Abs($close - Ref($close, 1)), 24) + 1e-12) - 0.5",
        "Sum(If($close > Ref($close, 1), 1, -1), 6) / 6",
        "($close - Min($low, 48)) / (Max($high, 48) - Min($low, 48) + 1e-12) - 0.5",
        "($close - Min($low, 144)) / (Max($high, 144) - Min($low, 144) + 1e-12) - 0.5",
        "($close - Min($low, 288)) / (Max($high, 288) - Min($low, 288) + 1e-12) - 0.5",
        "($close - Max($high, 48)) / (Max($high, 48) + 1e-12) * 10",
        "(Mean($high - $low, 12) / ($close + 1e-12)) / (Mean(Mean($high - $low, 12) / ($close + 1e-12), 96) + 1e-12) - 1",
        "(Mean($high - $low, 24) / ($close + 1e-12)) / (Mean(Mean($high - $low, 24) / ($close + 1e-12), 96) + 1e-12) - 1",
        "Mean($high - $low, 12) / (Mean($high - $low, 48) + 1e-12) - 1",
    )
    NAMES = tuple(f"feat_{index:02d}" for index in range(52))

    @classmethod
    def qlib_config(cls) -> tuple[list[str], list[str]]:
        return list(cls.EXPRESSIONS), list(cls.NAMES)

    @classmethod
    def schema_sha256(cls) -> str:
        payload = json.dumps(
            {"feature_set_id": cls.feature_set_id, "version": cls.version, "expressions": cls.EXPRESSIONS},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
