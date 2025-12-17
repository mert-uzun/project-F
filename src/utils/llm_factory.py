"""
LLM Factory - Privacy-First Backend Abstraction.

Provides a unified interface for LLM backends (OpenAI, Ollama).
Enables one-line swapping between cloud and local LLMs.
"""

from functools import lru_cache
from typing import TYPE_CHECKING

from llama_index.core import Settings as LlamaSettings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM

from app.config import EmbeddingBackend, LLMBackend, get_settings
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from app.config import Settings

logger = get_logger(__name__)


class LLMFactoryError(Exception):
    """Raised when LLM factory encounters an error."""

    pass


def _create_openai_llm(settings: "Settings") -> LLM:
    """Create OpenAI LLM instance."""
    try:
        from llama_index.llms.openai import OpenAI
    except ImportError as e:
        raise LLMFactoryError(
            "OpenAI LLM not installed. Run: pip install llama-index-llms-openai"
        ) from e

    if not settings.openai_api_key:
        raise LLMFactoryError(
            "OPENAI_API_KEY not set. Required when llm_backend='openai'"
        )

    logger.info(f"Initializing OpenAI LLM with model: {settings.openai_model}")
    return OpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
    )


def _create_ollama_llm(settings: "Settings") -> LLM:
    """Create Ollama LLM instance for local inference."""
    try:
        from llama_index.llms.ollama import Ollama
    except ImportError as e:
        raise LLMFactoryError(
            "Ollama LLM not installed. Run: pip install llama-index-llms-ollama"
        ) from e

    logger.info(
        f"Initializing Ollama LLM with model: {settings.ollama_model} "
        f"at {settings.ollama_base_url}"
    )
    return Ollama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        request_timeout=settings.ollama_request_timeout,
    )


def _create_huggingface_embedding(settings: "Settings") -> BaseEmbedding:
    """Create HuggingFace embedding model for local embeddings."""
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as e:
        raise LLMFactoryError(
            "HuggingFace embeddings not installed. "
            "Run: pip install llama-index-embeddings-huggingface"
        ) from e

    logger.info(f"Initializing HuggingFace embeddings with model: {settings.embedding_model}")
    return HuggingFaceEmbedding(model_name=settings.embedding_model)


def _create_openai_embedding(settings: "Settings") -> BaseEmbedding:
    """Create OpenAI embedding model."""
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
    except ImportError as e:
        raise LLMFactoryError(
            "OpenAI embeddings not installed. "
            "Run: pip install llama-index-embeddings-openai"
        ) from e

    if not settings.openai_api_key:
        raise LLMFactoryError(
            "OPENAI_API_KEY not set. Required when embedding_backend='openai'"
        )

    logger.info(f"Initializing OpenAI embeddings with model: {settings.embedding_model}")
    return OpenAIEmbedding(
        model_name=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


@lru_cache
def get_llm() -> LLM:
    """
    Get configured LLM instance based on settings.

    Returns:
        LLM instance (OpenAI or Ollama)

    Raises:
        LLMFactoryError: If configuration is invalid or dependencies missing
    """
    settings = get_settings()

    match settings.llm_backend:
        case LLMBackend.OPENAI:
            return _create_openai_llm(settings)
        case LLMBackend.OLLAMA:
            return _create_ollama_llm(settings)
        case _:
            raise LLMFactoryError(f"Unsupported LLM backend: {settings.llm_backend}")


@lru_cache
def get_embedding_model() -> BaseEmbedding:
    """
    Get configured embedding model based on settings.

    Returns:
        Embedding model instance (HuggingFace or OpenAI)

    Raises:
        LLMFactoryError: If configuration is invalid or dependencies missing
    """
    settings = get_settings()

    match settings.embedding_backend:
        case EmbeddingBackend.HUGGINGFACE:
            return _create_huggingface_embedding(settings)
        case EmbeddingBackend.OPENAI:
            return _create_openai_embedding(settings)
        case _:
            raise LLMFactoryError(
                f"Unsupported embedding backend: {settings.embedding_backend}"
            )


def configure_llama_index() -> None:
    """
    Configure LlamaIndex global settings with our LLM and embedding model.

    Call this at application startup to set default LLM/embeddings.
    """
    settings = get_settings()

    logger.info("Configuring LlamaIndex global settings...")

    try:
        LlamaSettings.llm = get_llm()
        LlamaSettings.embed_model = get_embedding_model()

        logger.info(
            f"LlamaIndex configured: LLM={settings.llm_backend.value}, "
            f"Embeddings={settings.embedding_backend.value}"
        )
    except LLMFactoryError as e:
        logger.error(f"Failed to configure LlamaIndex: {e}")
        raise
