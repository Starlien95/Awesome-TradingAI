"""Create editable FinMem workspaces without mutating package resources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_bench.methods.finmem.data import inspect_dataset
from quant_bench.runtime.core.atomic_io import atomic_write_json, atomic_write_text

SYSTEM_PROMPT = (
    "You are a financial trading assistant. Respond only with a valid JSON object containing "
    "investment_decision, summary_reason, short_memory_ids, mid_memory_ids, long_memory_ids, "
    "and reflection_memory_ids. investment_decision must be buy, sell, or hold. Memory id "
    "fields must be arrays of integers. Use only information available on the simulation date."
)


def _date_intersection(paths: dict[str, Path]) -> list[str]:
    common: set[str] | None = None
    for path in paths.values():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"dataset root must be an object: {path}")
        dates = set(payload)
        common = dates if common is None else common.intersection(dates)
    result = sorted(common or [])
    if len(result) < 4:
        raise ValueError("selected datasets need at least four common dates")
    return result


def _split_dates(dates: list[str]) -> tuple[str, str, str, str]:
    split = min(len(dates) - 2, max(2, int(len(dates) * 0.7)))
    return dates[0], dates[split - 1], dates[split], dates[-1]


def _memory_config(symbols: list[str], qdrant_endpoint: str) -> dict[str, Any]:
    shared = {
        "clean_up_recency_threshold": 0.05,
        "clean_up_importance_threshold": 5.0,
    }
    return {
        "memory_db_endpoint": qdrant_endpoint,
        "memory_importance_upper_bound": 100.0,
        "memory_importance_score_update_step": 18.0,
        "trading_symbols": symbols,
        "short": {
            "db_name": "short",
            "importance_init_val": 50.0,
            "decay_recency_factor": 3.0,
            "decay_importance_factor": 0.92,
            "jump_upper_threshold": 55.0,
            **shared,
        },
        "mid": {
            "db_name": "mid",
            "importance_init_val": 60.0,
            "decay_recency_factor": 90.0,
            "decay_importance_factor": 0.96,
            "jump_lower_threshold": 55.0,
            "jump_upper_threshold": 85.0,
            **shared,
        },
        "long": {
            "db_name": "long",
            "importance_init_val": 90.0,
            "decay_recency_factor": 365.0,
            "decay_importance_factor": 0.96,
            "jump_lower_threshold": 85.0,
            **shared,
        },
        "reflection": {
            "db_name": "reflection",
            "importance_init_val": 80.0,
            "decay_recency_factor": 365.0,
            "decay_importance_factor": 0.98,
            "similarity_threshold": 0.95,
            **shared,
        },
    }


def build_investorbench_config(
    workspace: Path,
    data_paths: dict[str, Path],
    *,
    chat_model: str = "Qwen/Qwen2.5-7B-Instruct",
    chat_endpoint: str = "http://127.0.0.1:8000",
    embedding_model: str = "text-embedding-3-small",
    embedding_endpoint: str = "https://api.openai.com/v1/embeddings",
    embedding_size: int = 1536,
    qdrant_endpoint: str = "http://127.0.0.1:6333",
) -> dict[str, Any]:
    symbols = sorted(data_paths)
    warmup_start, warmup_end, test_start, test_end = _split_dates(_date_intersection(data_paths))
    result_root = workspace / "results"
    run_id = "_".join(symbol.lower() for symbol in symbols)
    characters = {
        symbol: (
            f"You manage a benchmark portfolio for {symbol}. Use only prices, news, filings, "
            "momentum, and retrieved memories dated no later than the current simulation date. "
            "Explain uncertainty and never assume future information."
        )
        for symbol in symbols
    }
    return {
        "chat_config": {
            "chat_model": chat_model,
            "lora": False,
            "chat_model_type": "instruction",
            "chat_model_inference_engine": "vllm",
            "chat_system_message": SYSTEM_PROMPT,
            "chat_parameters": {"temperature": 0.2},
            "chat_max_new_token": 2048,
            "chat_request_timeout": 300,
            "chat_vllm_endpoint": chat_endpoint.rstrip("/"),
        },
        "emb_config": {
            "emb_model_name": embedding_model,
            "request_endpoint": embedding_endpoint,
            "emb_size": int(embedding_size),
            "embedding_timeout": 120,
            "embedding_batch_size": 10,
            "embedding_max_retries": 5,
        },
        "env_config": {
            "trading_symbols": symbols,
            "warmup_start_time": warmup_start,
            "warmup_end_time": warmup_end,
            "test_start_time": test_start,
            "test_end_time": test_end,
            "momentum_window_size": 3,
            "env_data_path": {symbol: str(data_paths[symbol]) for symbol in symbols},
        },
        "portfolio_config": {
            "trading_symbols": symbols,
            "type": "single-asset" if len(symbols) == 1 else "multi-assets",
            "cash": None if len(symbols) == 1 else 100_000.0,
            "look_back_window_size": 3,
        },
        "agent_config": {
            "agent_name": f"finmem_{run_id}",
            "trading_symbols": symbols,
            "character_string": characters,
            "top_k": 5,
            "memory_db_config": _memory_config(symbols, qdrant_endpoint),
        },
        "meta_config": {
            "run_name": f"finmem_{run_id}",
            "momentum_window_size": 3,
            "warmup_checkpoint_save_path": str(result_root / "warmup_checkpoint"),
            "warmup_output_save_path": str(result_root / "warmup_output"),
            "test_checkpoint_save_path": str(result_root / "test_checkpoint"),
            "test_output_save_path": str(result_root / "test_output"),
            "result_save_path": str(result_root / "final_result"),
            "log_save_path": str(result_root / "logs"),
        },
    }


def initialize_workspace(
    workspace: str | Path,
    data_dir: str | Path,
    *,
    symbols: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create a standalone, editable InvestorBench workspace."""

    root = Path(workspace).expanduser().resolve()
    source = Path(data_dir).expanduser().resolve()
    selected = [symbol.upper().strip() for symbol in (symbols or ["BTC"])]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("provide one or more unique symbols")
    data_paths = {symbol: source / f"{symbol.lower()}.json" for symbol in selected}
    for path in data_paths.values():
        inspect_dataset(path, verify_entries=False)

    config_path = root / "configs" / "investorbench.json"
    env_path = root / "finmem.env.example"
    descriptor_path = root / "workspace.json"
    existing = [path for path in (config_path, env_path, descriptor_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to replace existing workspace files: {existing}")

    config = build_investorbench_config(root, data_paths)
    atomic_write_json(config_path, config)
    env_template = (Path(__file__).with_name("resources") / "env.example").read_text(encoding="utf-8")
    atomic_write_text(env_path, env_template)
    descriptor = {
        "schema_version": 1,
        "method": "finmem",
        "config": str(config_path),
        "data_dir": str(source),
        "symbols": selected,
        "network_default": "disabled",
        "order_default": "disabled",
    }
    atomic_write_json(descriptor_path, descriptor)
    return {"workspace": str(root), "written": [str(config_path), str(env_path), str(descriptor_path)]}
