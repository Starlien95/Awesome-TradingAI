from __future__ import annotations

from quant_bench.integrations.qlib.catalog import list_workflows, resolve_workflow


def test_packaged_workflow_catalog_is_complete_and_filterable() -> None:
    assert len(list_workflows()) == 202
    lgb_1h = list_workflows(feature_set="158", model="lgb", frequency="1h")
    assert len(lgb_1h) == 1
    assert lgb_1h[0]["resource"].endswith("workflow_config_lgb_1h.yaml")
    assert resolve_workflow("158", "lgb", "1h").is_file()
