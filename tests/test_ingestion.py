"""
Tests for Ingestion Layer.
"""

import pytest

from src.ingestion.chunker import SemanticChunker
from src.ingestion.schemas import ParseResult


class TestSemanticChunker:
    """Tests for SemanticChunker."""
    
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
        
        # Text chunks (not tables) should be close to chunk_size
        text_chunks = [c for c in chunks if not c.metadata.contains_table]
        
        for chunk in text_chunks:
            # Allow some variance due to sentence boundaries
            assert len(chunk.content) <= chunk_size * 2, (
                f"Chunk too large: {len(chunk.content)} chars"
            )
    
    def test_detect_section_title(self) -> None:
        """Test section title detection."""
        chunker = SemanticChunker()
        
        # ALL CAPS title
        title = chunker._detect_section_title("EMPLOYMENT AGREEMENT\n\nThis is the content...")
        assert title == "EMPLOYMENT AGREEMENT"
        
        # Numbered section
        title = chunker._detect_section_title("1. PARTIES\n\nThis Agreement is between...")
        assert title is not None
        assert "PARTIES" in title
        
        # No title
        title = chunker._detect_section_title("This is just regular content without a title.")
        assert title is None


class TestParsedTable:
    """Tests for ParsedTable schema."""
    
    def test_table_properties(self, sample_parse_result: ParseResult) -> None:
        """Test ParsedTable computed properties."""
        table = sample_parse_result.tables[0]
        
        assert table.num_rows == 2
        assert table.num_cols == 3
        assert "John Doe" in table.markdown
