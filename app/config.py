"""
Application Configuration.

Pydantic settings for type-safe environment configuration.
Supports swappable LLM backends (OpenAI, Ollama) for privacy-first deployment.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMBackend(str, Enum):
    """Supported LLM backends."""

    OPENAI = "openai"
    OLLAMA = "ollama"


class EmbeddingBackend(str, Enum):
    """Supported embedding backends."""

    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === LLM Backend Selection ===
    llm_backend: LLMBackend = Field(
        default=LLMBackend.OLLAMA,
        description="LLM backend to use: 'openai' or 'ollama'",
    )

    # === OpenAI Configuration ===
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (required if llm_backend='openai')",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model to use",
    )

    # === Ollama Configuration ===
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    ollama_model: str = Field(
        default="llama3",
        description="Ollama model to use",
    )
    ollama_request_timeout: float = Field(
        default=120.0,
        description="Request timeout for Ollama in seconds",
    )

    # === LlamaParse Configuration ===
    llama_cloud_api_key: str = Field(
        default="",
        description="LlamaCloud API key for LlamaParse",
    )

    # === Embedding Configuration ===
    embedding_backend: EmbeddingBackend = Field(
        default=EmbeddingBackend.HUGGINGFACE,
        description="Embedding backend: 'huggingface' or 'openai'",
    )
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Embedding model to use",
    )

    # === ChromaDB Configuration ===
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma"),
        description="Directory for ChromaDB persistence",
    )

    # === Data Directories ===
    data_uploads_dir: Path = Field(
        default=Path("./data/uploads"),
        description="Directory for uploaded files",
    )
    data_processed_dir: Path = Field(
        default=Path("./data/processed"),
        description="Directory for processed documents",
    )
    data_graphs_dir: Path = Field(
        default=Path("./data/graphs"),
        description="Directory for graph data",
    )

    # === Logging ===
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    # === API Configuration ===
    api_host: str = Field(
        default="0.0.0.0",
        description="API host to bind to",
    )
    api_port: int = Field(
        default=8000,
        description="API port to bind to",
    )

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for dir_path in [
            self.chroma_persist_dir,
            self.data_uploads_dir,
            self.data_processed_dir,
            self.data_graphs_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
