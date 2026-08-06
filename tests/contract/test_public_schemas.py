from __future__ import annotations

from quant_bench.contracts import DatasetManifest, RunManifest
from quant_bench.methods import MethodDescriptor, MethodRunResult, MethodRunSpec
from quant_bench.trading import (
    AccountSnapshot,
    ExecutionReport,
    OrderRequest,
    TradingCycleResult,
)


def test_public_schemas_are_versioned_and_closed() -> None:
    dataset_schema = DatasetManifest.model_json_schema()
    run_schema = RunManifest.model_json_schema()
    assert dataset_schema["properties"]["schema_version"]["const"] == "1"
    assert run_schema["properties"]["schema_version"]["const"] == "1"
    assert dataset_schema["additionalProperties"] is False
    assert run_schema["additionalProperties"] is False

    for contract in (
        MethodDescriptor,
        MethodRunSpec,
        MethodRunResult,
        OrderRequest,
        AccountSnapshot,
        ExecutionReport,
        TradingCycleResult,
    ):
        schema = contract.model_json_schema()
        assert schema["properties"]["schema_version"]["const"] == "1"
        assert schema["additionalProperties"] is False
