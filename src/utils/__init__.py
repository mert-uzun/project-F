"""
Utility modules for the Conflict Detector.

Provides LLM/embedding factory and logging utilities.
"""

from src.utils.llm_factory import (
    LLMFactoryError,
    configure_llama_index,
    get_embedding_model,
    get_llm,
)
from src.utils.logger import (
    LogContext,
    get_logger,
    setup_logging,
)

__all__ = [
    # LLM Factory
    "get_llm",
    "get_embedding_model",
    "configure_llama_index",
    "LLMFactoryError",
    # Logging
    "get_logger",
    "setup_logging",
    "LogContext",
]
