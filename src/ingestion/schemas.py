"""
Pydantic Schemas for Ingestion Layer.

Type-safe models for document parsing and chunking.
All LLM outputs from ingestion must conform to these schemas.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class DocumentType(str, Enum):
    """Supported document types."""

    PDF = "pdf"
    CSV = "csv"
    XLSX = "xlsx"
    UNKNOWN = "unknown"


class TableCell(BaseModel):
    """A single cell in a table."""

    row: int = Field(..., ge=0, description="Row index (0-based)")
    col: int = Field(..., ge=0, description="Column index (0-based)")
    value: str = Field(..., description="Cell content")
    is_header: bool = Field(default=False, description="Whether this cell is a header")


class ParsedTable(BaseModel):
    """
    A table extracted from a document.

    Tables are extracted as markdown for LLM consumption,
    but we also store structured data for programmatic access.
    """

    table_id: UUID = Field(default_factory=uuid4, description="Unique table identifier")
    page_number: int = Field(..., ge=1, description="Page number where table appears")
    markdown: str = Field(..., description="Table as markdown for LLM consumption")
    headers: list[str] = Field(default_factory=list, description="Column headers")
    rows: list[list[str]] = Field(default_factory=list, description="Table rows as 2D list")
    caption: str | None = Field(default=None, description="Table caption if detected")

    @property
    def num_rows(self) -> int:
        """Number of data rows (excluding header)."""
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        """Number of columns."""
        return len(self.headers) if self.headers else (len(self.rows[0]) if self.rows else 0)


class DocumentMetadata(BaseModel):
    """Metadata about a parsed document."""

    document_id: UUID = Field(default_factory=uuid4, description="Unique document ID")
    filename: str = Field(..., description="Original filename")
    file_path: Path = Field(..., description="Path to original file")
    document_type: DocumentType = Field(..., description="Document type")
    num_pages: int = Field(..., ge=1, description="Total number of pages")
    num_tables: int = Field(default=0, ge=0, description="Number of tables extracted")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Parse timestamp")
    parser_used: str = Field(..., description="Parser used (llamaparse, unstructured)")
    parse_duration_seconds: float = Field(default=0.0, ge=0, description="Time to parse")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""

    chunk_id: UUID = Field(default_factory=uuid4, description="Unique chunk ID")
    document_id: UUID = Field(..., description="Parent document ID")
    page_number: int = Field(..., ge=1, description="Page number of chunk start")
    page_end: int | None = Field(default=None, description="Page number of chunk end if spans pages")
    section_title: str | None = Field(default=None, description="Section title if detected")
    chunk_index: int = Field(..., ge=0, description="Index of chunk in document")
    char_start: int = Field(..., ge=0, description="Character start position in full text")
    char_end: int = Field(..., ge=0, description="Character end position in full text")
    contains_table: bool = Field(default=False, description="Whether chunk contains table data")
    table_id: UUID | None = Field(default=None, description="Associated table ID if contains_table")


class DocumentChunk(BaseModel):
    """
    A chunk of document text ready for embedding.

    Chunks preserve semantic boundaries and include rich metadata
    for citation and source tracking.
    """

    content: str = Field(..., min_length=1, description="Chunk text content")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(self.content) // 4


class ParseResult(BaseModel):
    """Complete result of document parsing."""

    metadata: DocumentMetadata = Field(..., description="Document metadata")
    full_text: str = Field(..., description="Full extracted text (markdown)")
    tables: list[ParsedTable] = Field(default_factory=list, description="Extracted tables")
    chunks: list[DocumentChunk] = Field(default_factory=list, description="Semantic chunks")
    errors: list[str] = Field(default_factory=list, description="Any parsing errors/warnings")

    @property
    def success(self) -> bool:
        """Whether parsing completed without critical errors."""
        return len(self.full_text) > 0


class EntityExtraction(BaseModel):
    """
    Structured entity extracted from document.

    This is the foundation of our Knowledge Graph.
    Every assertion must have a source citation.
    """

    entity_type: str = Field(..., description="Entity type (Person, Organization, Amount, Date, etc)")
    entity_value: str = Field(..., description="The actual entity value")
    normalized_value: Any = Field(default=None, description="Normalized form for comparison")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    source_chunk_id: UUID = Field(..., description="Source chunk ID for citation")
    source_page: int = Field(..., ge=1, description="Page number for citation")
    context: str = Field(..., description="Surrounding context for verification")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
