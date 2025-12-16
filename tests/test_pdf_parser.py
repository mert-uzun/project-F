"""
Tests for PDF Parser - Integration Tests.

These tests require actual PDF files to test parsing.
Some tests require LlamaParse API key.
"""

import pytest
from pathlib import Path
from uuid import uuid4

from src.ingestion.pdf_parser import PDFParser, PDFParserError, LlamaParseClient, UnstructuredClient
from src.ingestion.schemas import ParseResult


# ============================================================================
# Parser Initialization Tests
# ============================================================================

class TestPDFParserInit:
    """Tests for PDFParser initialization."""
    
    def test_parser_initialization_without_key(self) -> None:
        """Test PDFParser can be initialized without API key."""
        parser = PDFParser()
        
        assert parser is not None
        assert parser.llama_client is None  # No API key provided
        assert parser.unstructured_client is not None
    
    def test_parser_with_llamaparse_key(self) -> None:
        """Test PDFParser with LlamaParse API key."""
        parser = PDFParser(
            llama_api_key="test_key",
            prefer_llamaparse=True,
        )
        
        assert parser.prefer_llamaparse is True
        assert parser.llama_client is not None
    
    def test_parser_prefer_unstructured(self) -> None:
        """Test parser falls back when prefer_llamaparse is False."""
        parser = PDFParser(
            llama_api_key="test_key",
            prefer_llamaparse=False,
        )
        
        assert parser.prefer_llamaparse is False


# ============================================================================
# LlamaParseClient Table Parsing Tests
# ============================================================================

class TestLlamaParseClientTables:
    """Tests for table extraction from LlamaParseClient."""
    
    @pytest.fixture
    def client(self) -> LlamaParseClient:
        """Create a LlamaParseClient for testing."""
        return LlamaParseClient(api_key="test_key")
    
    def test_extract_tables_from_markdown(self, client: LlamaParseClient) -> None:
        """Test extracting tables from markdown text."""
        markdown_text = """
# Document Title

Some text before the table.

| Column A | Column B | Column C |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |

Some text after the table.
"""
        
        tables = client._extract_tables_from_markdown(markdown_text)
        
        assert len(tables) == 1
        assert tables[0].num_rows == 2
        assert tables[0].num_cols == 3
        assert "Value 1" in tables[0].markdown
    
    def test_extract_multiple_tables(self, client: LlamaParseClient) -> None:
        """Test extracting multiple tables from markdown."""
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
        
        tables = client._extract_tables_from_markdown(markdown_text)
        
        assert len(tables) == 2
        assert tables[0].num_cols == 2
        assert tables[1].num_cols == 3
    
    def test_extract_no_tables(self, client: LlamaParseClient) -> None:
        """Test behavior when no tables in markdown."""
        markdown_text = """
# Just Text

This document has no tables.

Only paragraphs of text.
"""
        
        tables = client._extract_tables_from_markdown(markdown_text)
        
        assert len(tables) == 0
    
    def test_parse_markdown_table(self, client: LlamaParseClient) -> None:
        """Test parsing a markdown table into ParsedTable."""
        lines = [
            "| Name | Role | Equity |",
            "|------|------|--------|",
            "| John | CEO  | 5%     |",
            "| Jane | CFO  | 3%     |",
        ]
        
        table = client._parse_markdown_table(lines, page_number=1)
        
        assert table is not None
        assert table.num_rows == 2
        assert table.num_cols == 3
        assert table.headers == ["Name", "Role", "Equity"]
        assert table.page_number == 1


# ============================================================================
# UnstructuredClient Tests
# ============================================================================

class TestUnstructuredClient:
    """Tests for UnstructuredClient."""
    
    @pytest.fixture
    def client(self) -> UnstructuredClient:
        """Create an UnstructuredClient for testing."""
        return UnstructuredClient()
    
    def test_html_table_to_markdown(self, client: UnstructuredClient) -> None:
        """Test HTML table to markdown conversion."""
        html = """
        <table>
            <tr><td>A</td><td>B</td></tr>
            <tr><td>1</td><td>2</td></tr>
        </table>
        """
        
        markdown = client._html_table_to_markdown(html)
        
        assert "|" in markdown
        assert "A" in markdown or "1" in markdown


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
        
        with pytest.raises(PDFParserError, match="File not found"):
            await parser.parse(Path("/nonexistent/path/file.pdf"))
    
    @pytest.mark.asyncio
    async def test_parse_non_pdf_file(self, tmp_path: Path) -> None:
        """Test parsing a non-PDF file."""
        # Create a text file
        text_file = tmp_path / "test.txt"
        text_file.write_text("This is not a PDF")
        
        parser = PDFParser()
        
        with pytest.raises(PDFParserError, match="Not a PDF"):
            await parser.parse(text_file)
