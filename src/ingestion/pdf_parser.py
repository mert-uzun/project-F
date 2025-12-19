"""
PDF Parser - Table-Aware Document Extraction.

The Moat: Uses LlamaParse for premium table extraction with Unstructured fallback.
Tables are converted to markdown format for LLM consumption.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from src.ingestion.schemas import (
    DocumentMetadata,
    DocumentType,
    ParsedTable,
    ParseResult,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PDFParserError(Exception):
    """Raised when PDF parsing fails."""

    pass


@dataclass
class ParsedDocument:
    """Intermediate representation of parsed document."""

    text: str
    tables: list[ParsedTable]
    num_pages: int
    parser_used: str


class LlamaParseClient:
    """
    LlamaParse client for premium table extraction.

    LlamaParse excels at:
    - Complex table structures
    - Multi-page tables
    - Merged cells
    - Financial data tables
    """

    def __init__(self, api_key: str) -> None:
        """
        Initialize LlamaParse client.

        Args:
            api_key: LlamaCloud API key
        """
        self.api_key = api_key
        self._parser: Any = None

    def _get_parser(self) -> Any:
        """Lazy-load LlamaParse to avoid import errors if not installed."""
        if self._parser is None:
            try:
                from llama_parse import LlamaParse
            except ImportError as e:
                raise PDFParserError(
                    "LlamaParse not installed. Run: pip install llama-parse"
                ) from e

            self._parser = LlamaParse(
                api_key=self.api_key,
                # Support EU region via env var
                base_url=os.getenv("LLAMA_CLOUD_API_BASE_URL", "https://api.cloud.eu.llamaindex.ai"),
                result_type="markdown",
                # Use system_prompt instead of deprecated parsing_instruction
                system_prompt=(
                    "Extract all tables with their full structure. "
                    "Preserve column headers and row relationships. "
                    "For financial data, maintain number formatting."
                ),
                # Premium features for table extraction (costs extra credits!)
                # Uncomment if you have credits:
                # use_vendor_multimodal_model=True,
                # vendor_multimodal_model_name="anthropic-sonnet-4.0",
            )

        return self._parser

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    async def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse PDF with LlamaParse.

        Args:
            file_path: Path to PDF file

        Returns:
            ParsedDocument with extracted text and tables
        """
        logger.info(f"Parsing with LlamaParse: {file_path.name}")

        parser = self._get_parser()

        try:
            # LlamaParse returns list of Document objects
            documents = await parser.aload_data(str(file_path))

            if not documents:
                raise PDFParserError(f"LlamaParse returned no documents for {file_path}")

            # Combine all document text
            full_text = "\n\n".join(doc.text for doc in documents)

            # Extract tables from markdown
            tables = self._extract_tables_from_markdown(full_text)

            logger.info(f"LlamaParse extracted {len(tables)} tables from {file_path.name}")

            return ParsedDocument(
                text=full_text,
                tables=tables,
                num_pages=len(documents),
                parser_used="llamaparse",
            )

        except Exception as e:
            logger.error(f"LlamaParse failed for {file_path.name}: {e}")
            raise PDFParserError(f"LlamaParse failed: {e}") from e

    def _extract_tables_from_markdown(self, markdown_text: str) -> list[ParsedTable]:
        """
        Extract table structures from markdown text.

        Args:
            markdown_text: Full markdown text from LlamaParse

        Returns:
            List of ParsedTable objects
        """
        tables: list[ParsedTable] = []
        lines = markdown_text.split("\n")

        current_table_lines: list[str] = []
        in_table = False
        current_page = 1

        for i, line in enumerate(lines):
            # Track page numbers (LlamaParse often includes page markers)
            if line.startswith("---") and "page" in line.lower():
                try:
                    current_page = int("".join(filter(str.isdigit, line)))
                except ValueError:
                    pass
                continue

            # Detect table rows (markdown tables have | separators)
            is_table_row = "|" in line and line.strip().startswith("|")
            is_separator = line.strip().startswith("|") and set(line.strip()) <= {
                "|",
                "-",
                ":",
                " ",
            }

            if is_table_row or is_separator:
                if not in_table:
                    in_table = True
                current_table_lines.append(line)
            else:
                if in_table and current_table_lines:
                    # End of table, parse it
                    table = self._parse_markdown_table(current_table_lines, current_page)
                    if table:
                        tables.append(table)
                    current_table_lines = []
                in_table = False

        # Handle table at end of document
        if current_table_lines:
            table = self._parse_markdown_table(current_table_lines, current_page)
            if table:
                tables.append(table)

        return tables

    def _parse_markdown_table(self, lines: list[str], page_number: int) -> ParsedTable | None:
        """
        Parse markdown table lines into ParsedTable.

        Args:
            lines: Lines of markdown table
            page_number: Page where table appears

        Returns:
            ParsedTable or None if invalid
        """
        if len(lines) < 2:
            return None

        # Filter out separator lines for data extraction
        data_lines = [line for line in lines if not (set(line.strip()) <= {"|", "-", ":", " "})]

        if not data_lines:
            return None

        # Parse headers (first data line)
        headers = [cell.strip() for cell in data_lines[0].split("|") if cell.strip()]

        # Parse rows
        rows: list[list[str]] = []
        for line in data_lines[1:]:
            row = [cell.strip() for cell in line.split("|") if cell.strip()]
            if row:
                rows.append(row)

        return ParsedTable(
            page_number=page_number,
            markdown="\n".join(lines),
            headers=headers,
            rows=rows,
        )


class UnstructuredClient:
    """
    Unstructured.io client as fallback parser.

    Used when:
    - LlamaParse API is unavailable
    - Document has no tables
    - Simpler documents that don't need premium parsing
    """

    def __init__(self) -> None:
        """Initialize Unstructured client."""
        self._available: bool | None = None

    def _check_availability(self) -> bool:
        """Check if unstructured is installed."""
        if self._available is None:
            import importlib.util

            self._available = importlib.util.find_spec("unstructured") is not None
        return self._available

    async def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse PDF with Unstructured.

        Args:
            file_path: Path to PDF file

        Returns:
            ParsedDocument with extracted text and tables
        """
        if not self._check_availability():
            raise PDFParserError("Unstructured not installed. Run: pip install unstructured[pdf]")

        logger.info(f"Parsing with Unstructured: {file_path.name}")

        try:
            from unstructured.partition.pdf import partition_pdf

            # Partition PDF into elements
            elements = partition_pdf(
                filename=str(file_path),
                strategy="hi_res",  # Best quality extraction
                infer_table_structure=True,  # Extract table structure
            )

            # Separate tables from text
            tables: list[ParsedTable] = []
            text_parts: list[str] = []

            for element in elements:
                element_type = type(element).__name__

                if element_type == "Table":
                    # Extract table
                    table_html = getattr(element, "metadata", {}).get("text_as_html", "")
                    table_md = (
                        self._html_table_to_markdown(table_html) if table_html else str(element)
                    )

                    page_num = getattr(element.metadata, "page_number", 1) or 1

                    tables.append(
                        ParsedTable(
                            page_number=page_num,
                            markdown=table_md,
                            headers=[],  # Unstructured doesn't always give us clean headers
                            rows=[],
                        )
                    )
                    text_parts.append(f"\n\n[TABLE]\n{table_md}\n[/TABLE]\n\n")
                else:
                    text_parts.append(str(element))

            full_text = "\n".join(text_parts)
            num_pages = max(
                (getattr(e.metadata, "page_number", 1) or 1 for e in elements),
                default=1,
            )

            logger.info(f"Unstructured extracted {len(tables)} tables from {file_path.name}")

            return ParsedDocument(
                text=full_text,
                tables=tables,
                num_pages=num_pages,
                parser_used="unstructured",
            )

        except Exception as e:
            logger.error(f"Unstructured failed for {file_path.name}: {e}")
            raise PDFParserError(f"Unstructured failed: {e}") from e

    def _html_table_to_markdown(self, html: str) -> str:
        """
        Convert HTML table to markdown.

        Basic conversion - complex tables may need more sophisticated handling.
        """
        try:
            from html.parser import HTMLParser

            class TableHTMLParser(HTMLParser):
                def __init__(self) -> None:
                    super().__init__()
                    self.rows: list[list[str]] = []
                    self.current_row: list[str] = []
                    self.current_cell: str = ""
                    self.in_cell = False

                def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                    if tag in ("td", "th"):
                        self.in_cell = True
                        self.current_cell = ""
                    elif tag == "tr":
                        self.current_row = []

                def handle_endtag(self, tag: str) -> None:
                    if tag in ("td", "th"):
                        self.in_cell = False
                        self.current_row.append(self.current_cell.strip())
                    elif tag == "tr" and self.current_row:
                        self.rows.append(self.current_row)

                def handle_data(self, data: str) -> None:
                    if self.in_cell:
                        self.current_cell += data

            parser = TableHTMLParser()
            parser.feed(html)

            if not parser.rows:
                return html

            # Build markdown table
            md_lines: list[str] = []
            for i, row in enumerate(parser.rows):
                md_lines.append("| " + " | ".join(row) + " |")
                if i == 0:
                    md_lines.append("| " + " | ".join(["---"] * len(row)) + " |")

            return "\n".join(md_lines)

        except Exception:
            return html


class PDFParser:
    """
    Main PDF parser with fallback chain.

    Pipeline:
    1. Try LlamaParse (best table extraction)
    2. Fallback to Unstructured if LlamaParse fails

    Usage:
        parser = PDFParser(llama_api_key="...")
        result = await parser.parse(Path("contract.pdf"))
    """

    def __init__(
        self,
        llama_api_key: str | None = None,
        prefer_llamaparse: bool = True,
    ) -> None:
        """
        Initialize PDF parser.

        Args:
            llama_api_key: LlamaCloud API key (optional, falls back to Unstructured)
            prefer_llamaparse: Whether to prefer LlamaParse over Unstructured
        """
        self.llama_client: LlamaParseClient | None = None
        self.unstructured_client = UnstructuredClient()
        self.prefer_llamaparse = prefer_llamaparse

        if llama_api_key:
            self.llama_client = LlamaParseClient(llama_api_key)

    async def parse(self, file_path: Path) -> ParseResult:
        """
        Parse a PDF document.

        Args:
            file_path: Path to PDF file

        Returns:
            ParseResult with metadata, text, tables, and chunks
        """
        if not file_path.exists():
            raise PDFParserError(f"File not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise PDFParserError(f"Not a PDF file: {file_path}")

        start_time = time.time()
        errors: list[str] = []

        # Try parsers in order
        parsed: ParsedDocument | None = None

        if self.prefer_llamaparse and self.llama_client:
            try:
                parsed = await self.llama_client.parse(file_path)
            except PDFParserError as e:
                errors.append(f"LlamaParse failed: {e}")
                logger.warning(f"LlamaParse failed, falling back to Unstructured: {e}")

        if parsed is None:
            try:
                parsed = await self.unstructured_client.parse(file_path)
            except PDFParserError as e:
                errors.append(f"Unstructured failed: {e}")
                raise PDFParserError(
                    f"All parsers failed for {file_path.name}. Errors: {errors}"
                ) from e

        parse_duration = time.time() - start_time

        # Build metadata
        metadata = DocumentMetadata(
            filename=file_path.name,
            file_path=file_path,
            document_type=DocumentType.PDF,
            num_pages=parsed.num_pages,
            num_tables=len(parsed.tables),
            file_size_bytes=file_path.stat().st_size,
            parser_used=parsed.parser_used,
            parse_duration_seconds=parse_duration,
        )

        logger.info(
            f"Parsed {file_path.name}: {parsed.num_pages} pages, "
            f"{len(parsed.tables)} tables in {parse_duration:.2f}s"
        )

        return ParseResult(
            metadata=metadata,
            full_text=parsed.text,
            tables=parsed.tables,
            chunks=[],  # Chunks populated by SemanticChunker
            errors=errors,
        )
