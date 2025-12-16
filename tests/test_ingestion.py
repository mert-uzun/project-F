"""
Tests for Ingestion Layer.

Uses real components - no mocks.
"""

import pytest
from pathlib import Path
from uuid import uuid4

from src.ingestion.chunker import SemanticChunker, chunk_document
from src.ingestion.schemas import (
    ChunkMetadata,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    ParsedTable,
    ParseResult,
)


# ============================================================================
# SemanticChunker Tests
# ============================================================================

class TestSemanticChunker:
    """Tests for SemanticChunker."""
    
    def test_chunker_initialization(self) -> None:
        """Test SemanticChunker can be initialized with defaults."""
        chunker = SemanticChunker()
        
        assert chunker.chunk_size == 1024
        assert chunker.chunk_overlap == 128
    
    def test_chunker_custom_params(self) -> None:
        """Test SemanticChunker with custom parameters."""
        chunker = SemanticChunker(
            chunk_size=512,
            chunk_overlap=64,
            paragraph_separator="\n\n\n",
        )
        
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 64
        assert chunker.paragraph_separator == "\n\n\n"
    
    def test_chunk_empty_document(self, sample_parse_result: ParseResult) -> None:
        """Test chunking an empty document."""
        sample_parse_result.full_text = ""
        sample_parse_result.tables = []
        
        chunker = SemanticChunker()
        chunks = chunker.chunk(sample_parse_result)
        
        assert len(chunks) == 0
    
    def test_chunk_preserves_tables(self, sample_parse_result: ParseResult) -> None:
        """Test that tables are preserved as single chunks."""
        chunker = SemanticChunker()
        chunks = chunker.chunk(sample_parse_result)
        
        table_chunks = [c for c in chunks if c.metadata.contains_table]
        
        assert len(table_chunks) == 1
        assert "[TABLE" in table_chunks[0].content
        assert table_chunks[0].metadata.table_id is not None
    
    def test_chunk_has_metadata(self, sample_parse_result: ParseResult) -> None:
        """Test that chunks have proper metadata."""
        chunker = SemanticChunker()
        chunks = chunker.chunk(sample_parse_result)
        
        for chunk in chunks:
            assert chunk.metadata.document_id == sample_parse_result.metadata.document_id
            assert chunk.metadata.page_number >= 1
            assert chunk.metadata.chunk_index >= 0
    
    def test_chunk_size_respected(self, sample_parse_result: ParseResult) -> None:
        """Test that chunk size is approximately respected."""
        chunk_size = 256
        chunker = SemanticChunker(chunk_size=chunk_size)
        chunks = chunker.chunk(sample_parse_result)
        
        # Text chunks (not tables) should be within reasonable range
        # Note: Semantic chunking respects sentence/paragraph boundaries
        # so some chunks may exceed target size
        text_chunks = [c for c in chunks if not c.metadata.contains_table]
        
        for chunk in text_chunks:
            # Allow larger variance due to semantic boundary preservation
            assert len(chunk.content) <= chunk_size * 4, (
                f"Chunk too large: {len(chunk.content)} chars"
            )
    
    def test_detect_section_title_all_caps(self) -> None:
        """Test section title detection for ALL CAPS headers."""
        chunker = SemanticChunker()
        
        title = chunker._detect_section_title("EMPLOYMENT AGREEMENT\n\nThis is the content...")
        assert title == "EMPLOYMENT AGREEMENT"
    
    def test_detect_section_title_numbered(self) -> None:
        """Test section title detection for numbered sections."""
        chunker = SemanticChunker()
        
        title = chunker._detect_section_title("1. PARTIES\n\nThis Agreement is between...")
        assert title is not None
        assert "PARTIES" in title
    
    def test_detect_section_title_bold_markdown(self) -> None:
        """Test section title detection for bold markdown."""
        chunker = SemanticChunker()
        
        title = chunker._detect_section_title("**Compensation Details**\n\nThe employee will...")
        assert title == "Compensation Details"
    
    def test_detect_section_title_no_title(self) -> None:
        """Test section title detection returns None for regular content."""
        chunker = SemanticChunker()
        
        title = chunker._detect_section_title("This is just regular content without a title.")
        assert title is None
    
    def test_chunk_document_function(self, sample_parse_result: ParseResult) -> None:
        """Test the convenience chunk_document function."""
        result = chunk_document(sample_parse_result, chunk_size=512, chunk_overlap=64)
        
        assert result is sample_parse_result  # Returns same object
        assert len(result.chunks) > 0
    
    def test_table_chunk_has_caption(self, sample_parse_result: ParseResult) -> None:
        """Test that table chunks include caption if available."""
        chunker = SemanticChunker()
        chunks = chunker.chunk(sample_parse_result)
        
        table_chunks = [c for c in chunks if c.metadata.contains_table]
        
        assert len(table_chunks) == 1
        # Caption was "Cap Table"
        assert "Cap Table" in table_chunks[0].content
    
    def test_chunk_indices_are_sequential(self, sample_parse_result: ParseResult) -> None:
        """Test that chunk indices are sequential."""
        chunker = SemanticChunker()
        chunks = chunker.chunk(sample_parse_result)
        
        indices = [c.metadata.chunk_index for c in chunks]
        
        # Should start from 0 and be sequential
        for i, idx in enumerate(sorted(indices)):
            assert idx == i
    
    def test_estimate_page_number(self) -> None:
        """Test page number estimation from character position."""
        chunker = SemanticChunker()
        
        # Test at start
        assert chunker._estimate_page_number(0, 10000, 10) == 1
        
        # Test at middle
        assert chunker._estimate_page_number(5000, 10000, 10) == 6
        
        # Test at end
        assert chunker._estimate_page_number(9500, 10000, 10) == 10
        
        # Test edge case - zero total
        assert chunker._estimate_page_number(0, 0, 0) == 1


# ============================================================================
# Schema Tests
# ============================================================================

class TestParsedTable:
    """Tests for ParsedTable schema."""
    
    def test_table_properties(self, sample_parse_result: ParseResult) -> None:
        """Test ParsedTable computed properties."""
        table = sample_parse_result.tables[0]
        
        assert table.num_rows == 2
        assert table.num_cols == 3
        assert "John Doe" in table.markdown
    
    def test_table_num_cols_from_headers(self) -> None:
        """Test num_cols from headers."""
        table = ParsedTable(
            page_number=1,
            markdown="| A | B | C |\n|---|---|---|",
            headers=["A", "B", "C"],
            rows=[],
        )
        
        assert table.num_cols == 3
    
    def test_table_num_cols_from_rows(self) -> None:
        """Test num_cols from rows when headers empty."""
        table = ParsedTable(
            page_number=1,
            markdown="| 1 | 2 |\n| 3 | 4 |",
            headers=[],
            rows=[["1", "2"], ["3", "4"]],
        )
        
        assert table.num_cols == 2


class TestDocumentChunk:
    """Tests for DocumentChunk schema."""
    
    def test_chunk_creation(self) -> None:
        """Test DocumentChunk creation."""
        chunk = DocumentChunk(
            content="This is test content for the chunk.",
            metadata=ChunkMetadata(
                chunk_id=uuid4(),
                document_id=uuid4(),
                page_number=1,
                chunk_index=0,
                char_start=0,
                char_end=35,
            ),
        )
        
        assert len(chunk.content) == 35
        assert chunk.metadata.page_number == 1
    
    def test_token_estimate(self) -> None:
        """Test token estimate property."""
        chunk = DocumentChunk(
            content="a" * 400,  # 400 characters
            metadata=ChunkMetadata(
                chunk_id=uuid4(),
                document_id=uuid4(),
                page_number=1,
                chunk_index=0,
                char_start=0,
                char_end=400,
            ),
        )
        
        # 400 chars / 4 = 100 tokens
        assert chunk.token_estimate == 100


class TestParseResult:
    """Tests for ParseResult schema."""
    
    def test_success_property(self, sample_parse_result: ParseResult) -> None:
        """Test success property."""
        assert sample_parse_result.success is True
    
    def test_success_false_when_no_text(self) -> None:
        """Test success is False when no text extracted."""
        result = ParseResult(
            metadata=DocumentMetadata(
                document_id=uuid4(),
                filename="empty.pdf",
                file_path=Path("/tmp/empty.pdf"),
                document_type=DocumentType.PDF,
                num_pages=1,
                file_size_bytes=100,
                parser_used="test",
            ),
            full_text="",
            tables=[],
            chunks=[],
            errors=["No content extracted"],
        )
        
        assert result.success is False


class TestDocumentMetadata:
    """Tests for DocumentMetadata schema."""
    
    def test_metadata_creation(self) -> None:
        """Test DocumentMetadata creation with all fields."""
        metadata = DocumentMetadata(
            document_id=uuid4(),
            filename="contract.pdf",
            file_path=Path("/documents/contract.pdf"),
            document_type=DocumentType.PDF,
            num_pages=10,
            num_tables=3,
            file_size_bytes=1024000,
            parser_used="llamaparse",
            parse_duration_seconds=5.5,
        )
        
        assert metadata.filename == "contract.pdf"
        assert metadata.num_pages == 10
        assert metadata.num_tables == 3
        assert metadata.parser_used == "llamaparse"
    
    def test_metadata_defaults(self) -> None:
        """Test DocumentMetadata default values."""
        metadata = DocumentMetadata(
            filename="test.pdf",
            file_path=Path("/test.pdf"),
            document_type=DocumentType.PDF,
            num_pages=1,
            file_size_bytes=100,
            parser_used="test",
        )
        
        assert metadata.num_tables == 0
        assert metadata.parse_duration_seconds == 0.0
