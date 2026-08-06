"""Offline FinGPT sentiment research, tuning, and paper-backtest tools."""

from quant_bench.methods.fingpt_news.research.config import FinGPTResearchConfig
from quant_bench.methods.fingpt_news.research.pipeline import (
    initialize_research_workspace,
    run_research_pipeline,
    validate_research_inputs,
)

__all__ = [
    "FinGPTResearchConfig",
    "initialize_research_workspace",
    "run_research_pipeline",
    "validate_research_inputs",
]
