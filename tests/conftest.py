"""
Pytest Configuration and Fixtures.
"""

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator
from uuid import uuid4

import pytest

from src.ingestion.schemas import (
    DocumentMetadata,
    DocumentType,
    ParsedTable,
    ParseResult,
)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_pdf_path() -> Path:
    """Path to sample PDF fixture."""
    return Path(__file__).parent / "fixtures" / "sample.pdf"


@pytest.fixture
def sample_parse_result() -> ParseResult:
    """Create a sample ParseResult for testing."""
    doc_id = uuid4()
    
    metadata = DocumentMetadata(
        document_id=doc_id,
        filename="test_contract.pdf",
        file_path=Path("/tmp/test_contract.pdf"),
        document_type=DocumentType.PDF,
        num_pages=3,
        num_tables=1,
        file_size_bytes=50000,
        parser_used="llamaparse",
        parse_duration_seconds=2.5,
    )
    
    table = ParsedTable(
        page_number=2,
        markdown="""| Name | Role | Equity |
| --- | --- | --- |
| John Doe | CEO | 5% |
| Jane Smith | CFO | 3% |""",
        headers=["Name", "Role", "Equity"],
        rows=[
            ["John Doe", "CEO", "5%"],
            ["Jane Smith", "CFO", "3%"],
        ],
        caption="Cap Table",
    )
    
    full_text = """
# EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is entered into as of January 1, 2024.

## 1. PARTIES

This Agreement is between ABC Corp ("Company") and John Doe ("Employee").

## 2. POSITION AND DUTIES

Employee shall serve as Chief Executive Officer (CEO) of the Company.

## 3. COMPENSATION

### 3.1 Base Salary

Employee shall receive an annual base salary of $500,000.

### 3.2 Equity

[TABLE]
| Name | Role | Equity |
| --- | --- | --- |
| John Doe | CEO | 5% |
| Jane Smith | CFO | 3% |
[/TABLE]

## 4. TERMINATION

Either party may terminate this Agreement with 90 days written notice.

## 5. CONFIDENTIALITY

Employee agrees to maintain strict confidentiality of all Company information.
"""
    
    return ParseResult(
        metadata=metadata,
        full_text=full_text,
        tables=[table],
        chunks=[],
        errors=[],
    )
