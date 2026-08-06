"""Canonical Qlib handlers with versioned feature identities."""

from __future__ import annotations

import copy
from typing import Any

from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.handler import DataHandlerLP

from quant_bench.features import CryptoAlpha52V1, CryptoOHLCV6V1


class CryptoAlpha52Handler(Alpha158):
    """Compatibility handler for the versioned CryptoAlpha52V1 expressions."""

    feature_set_id = CryptoAlpha52V1.feature_set_id

    def get_feature_config(self) -> tuple[list[str], list[str]]:
        return CryptoAlpha52V1.qlib_config()


class CryptoOHLCV6Handler(DataHandlerLP):
    """Six causal OHLCV features with a correctly named typical-price feature."""

    feature_set_id = CryptoOHLCV6V1.feature_set_id

    def __init__(
        self,
        instruments: str = "all",
        start_time: str | None = None,
        end_time: str | None = None,
        freq: str = "60min",
        fit_start_time: str | None = None,
        fit_end_time: str | None = None,
        label: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        fields, names = CryptoOHLCV6V1.qlib_config()
        qlib_config: dict[str, Any] = {"feature": (fields, names)}
        if label:
            qlib_config["label"] = (label, [f"LABEL{index}" for index in range(len(label))])

        if "infer_processors" in kwargs:
            processors = copy.deepcopy(kwargs["infer_processors"])
            for processor in processors:
                if processor.get("class") == "RobustZScoreNorm":
                    processor_kwargs = processor.setdefault("kwargs", {})
                    if fit_start_time:
                        processor_kwargs["fit_start_time"] = fit_start_time
                    if fit_end_time:
                        processor_kwargs["fit_end_time"] = fit_end_time
            kwargs["infer_processors"] = processors

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader={
                "class": "QlibDataLoader",
                "kwargs": {"config": qlib_config, "freq": freq},
            },
            **kwargs,
        )


CustomHandler158 = CryptoAlpha52Handler
CustomHandler360 = CryptoOHLCV6Handler
