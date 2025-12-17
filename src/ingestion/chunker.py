"""
Semantic Chunker - Clause-Aware Document Splitting.

Key principle: NEVER split a legal clause or sentence mid-way.
Uses semantic boundaries (paragraphs, sections) instead of token counts.
"""

import re

from llama_index.core.node_parser import SentenceSplitter

from src.ingestion.schemas import (
    ChunkMetadata,
    DocumentChunk,
    DocumentMetadata,
    ParsedTable,
    ParseResult,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SemanticChunkerError(Exception):
    """Raised when chunking fails."""

    pass


class SemanticChunker:
    """
    Semantic document chunker that respects clause boundaries.

    Features:
    - Sentence-aware splitting (never breaks mid-sentence)
    - Table preservation (tables kept as single chunks)
    - Overlap for context continuity
    - Page number tracking for citations

    Usage:
        chunker = SemanticChunker(chunk_size=1024, chunk_overlap=128)
        chunks = chunker.chunk(parse_result)
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        paragraph_separator: str = "\n\n",
    ) -> None:
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks for context
            paragraph_separator: Separator between paragraphs
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.paragraph_separator = paragraph_separator

        # LlamaIndex sentence splitter for intelligent splitting
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator=paragraph_separator,
            secondary_chunking_regex="[.!?。]",  # Split on sentence boundaries
        )

    def chunk(self, parse_result: ParseResult) -> list[DocumentChunk]:
        """
        Chunk a parsed document into semantic units.

        Args:
            parse_result: Result from PDFParser

        Returns:
            List of DocumentChunk with metadata
        """
        if not parse_result.full_text:
            logger.warning(f"Empty document: {parse_result.metadata.filename}")
            return []

        logger.info(f"Chunking document: {parse_result.metadata.filename}")

        chunks: list[DocumentChunk] = []

        # First, handle tables as special chunks
        table_chunks = self._chunk_tables(parse_result.tables, parse_result.metadata)
        chunks.extend(table_chunks)

        # Remove table content from text to avoid duplication
        text_without_tables = self._remove_tables_from_text(
            parse_result.full_text, parse_result.tables
        )

        # Chunk the remaining text
        text_chunks = self._chunk_text(
            text_without_tables,
            parse_result.metadata,
            start_index=len(table_chunks),
        )
        chunks.extend(text_chunks)

        logger.info(
            f"Created {len(chunks)} chunks ({len(table_chunks)} tables, "
            f"{len(text_chunks)} text) from {parse_result.metadata.filename}"
        )

        return chunks

    def _chunk_tables(
        self,
        tables: list[ParsedTable],
        doc_metadata: DocumentMetadata,
    ) -> list[DocumentChunk]:
        """
        Create chunks for tables.

        Tables are kept as single chunks to preserve structure.
        """
        chunks: list[DocumentChunk] = []

        for i, table in enumerate(tables):
            # Create a descriptive chunk with table context
            content = f"[TABLE {i + 1}]\n{table.markdown}\n[/TABLE]"

            if table.caption:
                content = f"Table Caption: {table.caption}\n\n{content}"

            metadata = ChunkMetadata(
                document_id=doc_metadata.document_id,
                page_number=table.page_number,
                chunk_index=i,
                char_start=0,  # Tables don't have char positions
                char_end=len(content),
                contains_table=True,
                table_id=table.table_id,
            )

            chunks.append(DocumentChunk(content=content, metadata=metadata))

        return chunks

    def _chunk_text(
        self,
        text: str,
        doc_metadata: DocumentMetadata,
        start_index: int = 0,
    ) -> list[DocumentChunk]:
        """
        Chunk text content using semantic boundaries.
        """
        if not text.strip():
            return []

        chunks: list[DocumentChunk] = []

        # Use LlamaIndex splitter for intelligent chunking
        try:
            text_splits = self._splitter.split_text(text)
        except Exception as e:
            logger.error(f"Splitter failed: {e}, falling back to paragraph split")
            text_splits = self._fallback_split(text)

        char_position = 0

        for i, chunk_text in enumerate(text_splits):
            if not chunk_text.strip():
                continue

            # Find actual position in original text
            try:
                char_start = text.index(chunk_text[:50], char_position)
            except ValueError:
                char_start = char_position

            char_end = char_start + len(chunk_text)
            char_position = char_end

            # Estimate page number from character position
            page_number = self._estimate_page_number(char_start, len(text), doc_metadata.num_pages)

            # Detect section titles
            section_title = self._detect_section_title(chunk_text)

            metadata = ChunkMetadata(
                document_id=doc_metadata.document_id,
                page_number=page_number,
                section_title=section_title,
                chunk_index=start_index + i,
                char_start=char_start,
                char_end=char_end,
                contains_table=False,
            )

            chunks.append(DocumentChunk(content=chunk_text, metadata=metadata))

        return chunks

    def _remove_tables_from_text(self, text: str, tables: list[ParsedTable]) -> str:
        """Remove table markdown from text to avoid duplication."""
        result = text

        for table in tables:
            # Try to find and remove the table markdown
            if table.markdown in result:
                result = result.replace(table.markdown, "[TABLE_REMOVED]")

        # Also remove any [TABLE]...[/TABLE] blocks
        result = re.sub(
            r"\[TABLE\].*?\[/TABLE\]",
            "[TABLE_REMOVED]",
            result,
            flags=re.DOTALL,
        )

        return result

    def _fallback_split(self, text: str) -> list[str]:
        """Simple paragraph-based fallback splitter."""
        paragraphs = text.split(self.paragraph_separator)

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append(self.paragraph_separator.join(current_chunk))
                # Keep last paragraph for overlap
                current_chunk = [current_chunk[-1]] if current_chunk else []
                current_size = len(current_chunk[0]) if current_chunk else 0

            current_chunk.append(para)
            current_size += para_size

        if current_chunk:
            chunks.append(self.paragraph_separator.join(current_chunk))

        return chunks

    def _estimate_page_number(self, char_position: int, total_chars: int, num_pages: int) -> int:
        """Estimate page number from character position."""
        if total_chars == 0 or num_pages == 0:
            return 1

        chars_per_page = total_chars / num_pages
        estimated_page = int(char_position / chars_per_page) + 1

        return min(max(estimated_page, 1), num_pages)

    def _detect_section_title(self, text: str) -> str | None:
        """
        Detect section title from chunk text.

        Looks for common patterns:
        - Lines in ALL CAPS
        - Lines starting with numbers (1. Section, 2.1 Subsection)
        - Lines ending with :
        """
        lines = text.strip().split("\n")

        if not lines:
            return None

        first_line = lines[0].strip()

        # Check for ALL CAPS header (common in legal docs)
        if first_line.isupper() and len(first_line) > 3 and len(first_line) < 100:
            return first_line

        # Check for numbered section
        section_pattern = r"^(\d+\.?\d*\.?\s+[A-Z].*?)(?:\n|$)"
        match = re.match(section_pattern, text.strip())
        if match:
            return match.group(1).strip()

        # Check for bold/emphasized headers (markdown)
        bold_pattern = r"^\*\*(.+?)\*\*"
        match = re.match(bold_pattern, first_line)
        if match:
            return match.group(1).strip()

        return None


def chunk_document(
    parse_result: ParseResult,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
) -> ParseResult:
    """
    Convenience function to chunk a parsed document.

    Args:
        parse_result: Result from PDFParser
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        ParseResult with chunks populated
    """
    chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk(parse_result)

    # Update parse result with chunks
    parse_result.chunks = chunks

    return parse_result
