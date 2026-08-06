# from .anthropic import CluadeGuardRailStructureGeneration
from .base import (
    SingleAssetStructuredGenerationChatEndPoint,
    MultiAssetsStructuredGenerationChatEndPoint,
    SingleAssetStructureGenerationFailure,
    MultiAssetsStructureGenerationFailure,
    SingleAssetStructureOutputResponse,
    MultiAssetsStructureOutputResponse,
)

from .vllm import SingleAssetVLLMStructureGeneration, MultiAssetsVLLMStructureGeneration
try:
    from .guardrails import (
        ClaudeGuardRailStructureGeneration,
        GPTGuardRailStructureGeneration,
    )
except ModuleNotFoundError as exc:
    if exc.name != "guardrails":
        raise
    ClaudeGuardRailStructureGeneration = None
    GPTGuardRailStructureGeneration = None
