"""
Ingestion Layer - The Moat.

Document parsing, table extraction, and semantic chunking.
"""

from src.ingestion.pdf_parser import PDFParser, ParsedDocument
from src.ingestion.chunker import SemanticChunker
from src.ingestion.schemas import (
    DocumentMetadata,
    ParsedTable,
    DocumentChunk,
    ChunkMetadata,
)

__all__ = [
    "PDFParser",
    "ParsedDocument",
    "SemanticChunker",
    "DocumentMetadata",
    "ParsedTable",
    "DocumentChunk",
    "ChunkMetadata",
]
