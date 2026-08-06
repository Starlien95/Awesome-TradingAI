from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_bench.artifacts import LocalArtifactStore
from quant_bench.contracts import DatasetView
from quant_bench.models import NumpyLinearPlugin


def _dataset() -> DatasetView:
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC")], ["A", "B"]],
        names=["timestamp", "symbol"],
    )
    features = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "z": [1.0, 1.0, 2.0, 2.0]}, index=index)
    labels = pd.Series([0.0, 1.0, 2.0, 3.0], index=index, name="label")
    return DatasetView(features, labels, features, labels, features, labels)


def test_fit_save_load_predict_roundtrip(tmp_path: Path) -> None:
    plugin = NumpyLinearPlugin(ridge=1e-8)
    dataset = _dataset()
    model = plugin.fit(dataset)
    expected = plugin.predict(model, dataset).frame["score"].to_numpy()
    store = LocalArtifactStore(tmp_path / "run")
    plugin.save(model, store)
    loaded = plugin.load(store)
    actual = plugin.predict(loaded, dataset).frame["score"].to_numpy()
    np.testing.assert_allclose(actual, expected)
