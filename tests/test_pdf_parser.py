"""
Tests for PDF Parser - Integration Tests.

These tests require actual PDF files to test parsing.
Some tests require LlamaParse API key or Ollama for OCR fallback.
"""

import pytest
from pathlib import Path
from uuid import uuid4

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.schemas import ParseResult


# ============================================================================
# Parser Initialization Tests
# ============================================================================

class TestPDFParserInit:
    """Tests for PDFParser initialization."""
    
    def test_parser_initialization(self) -> None:
        """Test PDFParser can be initialized."""
        parser = PDFParser()
        
        assert parser is not None
        assert parser.prefer_llamaparse is False  # Default when no API key
    
    def test_parser_with_llamaparse_preference(self) -> None:
        """Test PDFParser with LlamaParse preference."""
        parser = PDFParser(
            llama_api_key="test_key",
            prefer_llamaparse=True,
        )
        
        assert parser.prefer_llamaparse is True
    
    def test_parser_fallback_when_no_key(self) -> None:
        """Test parser falls back when no LlamaParse key."""
        parser = PDFParser(
            llama_api_key=None,
            prefer_llamaparse=True,
        )
        
        # Should still initialize but will use fallback
        assert parser is not None


# ============================================================================
# Table Parsing Tests
# ============================================================================

class TestTableParsing:
    """Tests for table extraction from markdown."""
    
    def test_extract_tables_from_markdown(self) -> None:
        """Test extracting tables from markdown text."""
        parser = PDFParser()
        
        markdown_text = """
# Document Title

Some text before the table.

| Column A | Column B | Column C |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |

Some text after the table.
"""
        
        tables = parser._extract_tables_from_markdown(markdown_text, page_number=1)
        
        assert len(tables) == 1
        assert tables[0].num_rows == 2
        assert tables[0].num_cols == 3
        assert "Value 1" in tables[0].markdown
    
    def test_extract_multiple_tables(self) -> None:
        """Test extracting multiple tables from markdown."""
        parser = PDFParser()
        
        markdown_text = """
# Table 1

| A | B |
|---|---|
| 1 | 2 |

# Table 2

| X | Y | Z |
|---|---|---|
| a | b | c |
| d | e | f |
"""
        
        tables = parser._extract_tables_from_markdown(markdown_text, page_number=1)
        
        assert len(tables) == 2
        assert tables[0].num_cols == 2
        assert tables[1].num_cols == 3
    
    def test_extract_no_tables(self) -> None:
        """Test behavior when no tables in markdown."""
        parser = PDFParser()
        
        markdown_text = """
# Just Text

This document has no tables.

Only paragraphs of text.
"""
        
        tables = parser._extract_tables_from_markdown(markdown_text, page_number=1)
        
        assert len(tables) == 0
    
    def test_parse_table_row(self) -> None:
        """Test parsing a single markdown table row."""
        parser = PDFParser()
        
        row = "| Value A | Value B | Value C |"
        cells = parser._parse_table_row(row)
        
        assert cells == ["Value A", "Value B", "Value C"]
    
    def test_parse_table_row_with_empty_cells(self) -> None:
        """Test parsing row with empty cells."""
        parser = PDFParser()
        
        row = "| Value |  | Another |"
        cells = parser._parse_table_row(row)
        
        assert len(cells) == 3
        assert cells[1] == ""  # Empty cell


# ============================================================================
# Parse Result Tests
# ============================================================================

class TestParseResultHandling:
    """Tests for handling parse results."""
    
    def test_create_parse_result(self) -> None:
        """Test creating a ParseResult object."""
        parser = PDFParser()
        
        result = parser._create_result(
            file_path=Path("/tmp/test.pdf"),
            full_text="This is the document content.",
            tables=[],
            parser_used="unstructured",
            duration=1.5,
            errors=[],
        )
        
        assert isinstance(result, ParseResult)
        assert result.metadata.filename == "test.pdf"
        assert result.full_text == "This is the document content."
        assert result.metadata.parser_used == "unstructured"
        assert result.metadata.parse_duration_seconds == 1.5
    
    def test_create_parse_result_with_errors(self) -> None:
        """Test creating a ParseResult with errors."""
        parser = PDFParser()
        
        result = parser._create_result(
            file_path=Path("/tmp/failed.pdf"),
            full_text="",
            tables=[],
            parser_used="unstructured",
            duration=0.5,
            errors=["Parsing failed", "No content"],
        )
        
        assert len(result.errors) == 2
        assert result.success is False


# ============================================================================
# Integration Tests (Require Files)
# ============================================================================

class TestPDFParserIntegration:
    """Integration tests that require actual PDF files."""
    
    @pytest.fixture
    def sample_pdf_exists(self, sample_pdf_path: Path) -> bool:
        """Check if sample PDF exists."""
        return sample_pdf_path.exists()
    
    @pytest.mark.asyncio
    async def test_parse_sample_pdf(self, sample_pdf_path: Path) -> None:
        """
        Test parsing an actual PDF file.
        Skip if sample.pdf doesn't exist.
        """
        if not sample_pdf_path.exists():
            pytest.skip("Sample PDF not found at tests/fixtures/sample.pdf")
        
        parser = PDFParser()
        result = await parser.parse(sample_pdf_path)
        
        assert isinstance(result, ParseResult)
        assert result.metadata.filename == "sample.pdf"
        # Should have extracted some text
        assert len(result.full_text) > 0
    
    @pytest.mark.asyncio
    async def test_parse_nonexistent_file(self) -> None:
        """Test parsing a file that doesn't exist."""
        parser = PDFParser()
        
        with pytest.raises(FileNotFoundError):
            await parser.parse(Path("/nonexistent/path/file.pdf"))
    
    @pytest.mark.asyncio
    async def test_parse_non_pdf_file(self, tmp_path: Path) -> None:
        """Test parsing a non-PDF file."""
        # Create a text file
        text_file = tmp_path / "test.txt"
        text_file.write_text("This is not a PDF")
        
        parser = PDFParser()
        
        # Should raise an error or return failure
        with pytest.raises(Exception):
            await parser.parse(text_file)
