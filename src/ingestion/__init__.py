"""
Ingestion Layer - The Moat.

Document parsing, table extraction, and semantic chunking.
"""

from src.ingestion.chunker import SemanticChunker
from src.ingestion.pdf_parser import ParsedDocument, PDFParser
from src.ingestion.schemas import (
    ChunkMetadata,
    DocumentChunk,
    DocumentMetadata,
    ParsedTable,
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
