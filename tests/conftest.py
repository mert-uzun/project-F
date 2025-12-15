"""
Pytest Configuration and Fixtures.

All fixtures use REAL components - no mocks.
Tests require:
- Ollama running locally (or OPENAI_API_KEY set)
- Sufficient disk space for ChromaDB
"""

import asyncio
import os
from pathlib import Path
from typing import AsyncGenerator, Generator
from uuid import uuid4

import pytest

from src.ingestion.schemas import (
    ChunkMetadata,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    ParsedTable,
    ParseResult,
)
from src.knowledge.schemas import Entity, EntityType, Relationship, RelationshipType


# ============================================================================
# Event Loop
# ============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Skip Markers
# ============================================================================

def requires_llm():
    """Skip test if LLM is not available."""
    # Check if Ollama is running or OpenAI key is set
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    # Try to check Ollama availability
    has_ollama = False
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        has_ollama = response.status_code == 200
    except Exception:
        pass
    
    return pytest.mark.skipif(
        not (has_openai or has_ollama),
        reason="Requires LLM backend (Ollama or OpenAI)"
    )


def requires_ollama():
    """Skip test if Ollama is not running."""
    has_ollama = False
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        has_ollama = response.status_code == 200
    except Exception:
        pass
    
    return pytest.mark.skipif(
        not has_ollama,
        reason="Requires Ollama running on localhost:11434"
    )


# ============================================================================
# Path Fixtures
# ============================================================================

@pytest.fixture
def sample_pdf_path() -> Path:
    """Path to sample PDF fixture."""
    return Path(__file__).parent / "fixtures" / "sample.pdf"


@pytest.fixture
def temp_chroma_dir(tmp_path: Path) -> Path:
    """Temporary directory for ChromaDB."""
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    return chroma_dir


@pytest.fixture
def temp_graph_path(tmp_path: Path) -> Path:
    """Temporary path for graph storage."""
    return tmp_path / "graph.json"


# ============================================================================
# Document Fixtures
# ============================================================================

@pytest.fixture
def sample_document_id() -> uuid4:
    """A consistent document ID for testing."""
    return uuid4()


@pytest.fixture
def sample_parse_result(sample_document_id) -> ParseResult:
    """Create a sample ParseResult for testing."""
    metadata = DocumentMetadata(
        document_id=sample_document_id,
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


@pytest.fixture
def sample_chunks(sample_document_id) -> list[DocumentChunk]:
    """Sample document chunks for testing."""
    chunks = [
        DocumentChunk(
            content="Employee shall receive an annual base salary of $500,000 per year. This salary is subject to annual review.",
            metadata=ChunkMetadata(
                chunk_id=uuid4(),
                document_id=sample_document_id,
                page_number=1,
                chunk_index=0,
                char_start=0,
                char_end=100,
            ),
        ),
        DocumentChunk(
            content="John Doe is granted 5% equity in the Company, vesting over 4 years with a 1-year cliff.",
            metadata=ChunkMetadata(
                chunk_id=uuid4(),
                document_id=sample_document_id,
                page_number=2,
                chunk_index=1,
                char_start=100,
                char_end=200,
            ),
        ),
        DocumentChunk(
            content="The effective date of this agreement is January 1, 2024. The agreement terminates on December 31, 2028.",
            metadata=ChunkMetadata(
                chunk_id=uuid4(),
                document_id=sample_document_id,
                page_number=3,
                chunk_index=2,
                char_start=200,
                char_end=300,
            ),
        ),
    ]
    return chunks


# ============================================================================
# Entity Fixtures
# ============================================================================

@pytest.fixture
def sample_entities(sample_document_id) -> list[Entity]:
    """Sample entities for testing conflict detection."""
    chunk_id = uuid4()
    
    return [
        Entity(
            entity_type=EntityType.PERSON,
            value="John Doe",
            normalized_value="john doe",
            source_document_id=sample_document_id,
            source_chunk_id=chunk_id,
            source_page=1,
            source_text="John Doe is the CEO of the company.",
            confidence=0.95,
        ),
        Entity(
            entity_type=EntityType.MONETARY_AMOUNT,
            value="$500,000",
            normalized_value="500000",
            source_document_id=sample_document_id,
            source_chunk_id=chunk_id,
            source_page=1,
            source_text="Employee shall receive an annual base salary of $500,000.",
            confidence=0.9,
        ),
        Entity(
            entity_type=EntityType.PERCENTAGE,
            value="5%",
            normalized_value="5",
            source_document_id=sample_document_id,
            source_chunk_id=chunk_id,
            source_page=2,
            source_text="John Doe is granted 5% equity.",
            confidence=0.92,
        ),
        Entity(
            entity_type=EntityType.DATE,
            value="January 1, 2024",
            normalized_value="2024-01-01",
            source_document_id=sample_document_id,
            source_chunk_id=chunk_id,
            source_page=1,
            source_text="The effective date is January 1, 2024.",
            confidence=0.95,
        ),
    ]


@pytest.fixture
def conflicting_entities() -> tuple[list[Entity], list[Entity]]:
    """
    Two sets of entities with conflicts for testing.
    
    Document A says: $500,000 salary, 5% equity, January 1, 2024
    Document B says: $450,000 salary, 8% equity, March 15, 2024
    """
    doc_a_id = uuid4()
    doc_b_id = uuid4()
    chunk_a_id = uuid4()
    chunk_b_id = uuid4()
    
    entities_a = [
        Entity(
            entity_type=EntityType.SALARY,
            value="$500,000",
            normalized_value="500000",
            source_document_id=doc_a_id,
            source_chunk_id=chunk_a_id,
            source_page=1,
            source_text="Base salary of $500,000 per year.",
            confidence=0.95,
        ),
        Entity(
            entity_type=EntityType.EQUITY,
            value="5%",
            normalized_value="5",
            source_document_id=doc_a_id,
            source_chunk_id=chunk_a_id,
            source_page=2,
            source_text="Employee receives 5% equity stake.",
            confidence=0.9,
        ),
        Entity(
            entity_type=EntityType.DATE,
            value="January 1, 2024",
            normalized_value="2024-01-01",
            source_document_id=doc_a_id,
            source_chunk_id=chunk_a_id,
            source_page=1,
            source_text="Effective date: January 1, 2024.",
            confidence=0.95,
        ),
    ]
    
    entities_b = [
        Entity(
            entity_type=EntityType.SALARY,
            value="$450,000",
            normalized_value="450000",
            source_document_id=doc_b_id,
            source_chunk_id=chunk_b_id,
            source_page=1,
            source_text="Annual compensation of $450,000.",
            confidence=0.92,
        ),
        Entity(
            entity_type=EntityType.EQUITY,
            value="8%",
            normalized_value="8",
            source_document_id=doc_b_id,
            source_chunk_id=chunk_b_id,
            source_page=2,
            source_text="Equity grant of 8%.",
            confidence=0.88,
        ),
        Entity(
            entity_type=EntityType.DATE,
            value="March 15, 2024",
            normalized_value="2024-03-15",
            source_document_id=doc_b_id,
            source_chunk_id=chunk_b_id,
            source_page=1,
            source_text="Start date: March 15, 2024.",
            confidence=0.93,
        ),
    ]
    
    return entities_a, entities_b


# ============================================================================
# Store Fixtures (Real, not mocked)
# ============================================================================

@pytest.fixture
def vector_store(temp_chroma_dir):
    """Create a real VectorStore for testing."""
    from src.knowledge.vector_store import VectorStore, VectorStoreConfig
    
    config = VectorStoreConfig(
        persist_directory=temp_chroma_dir,
        collection_name="test_collection",
    )
    return VectorStore(config)


@pytest.fixture
def graph_store(temp_graph_path):
    """Create a real GraphStore for testing."""
    from src.knowledge.graph_store import GraphStore
    
    return GraphStore(persist_path=temp_graph_path)


@pytest.fixture
def populated_stores(vector_store, graph_store, sample_chunks, sample_entities):
    """Stores pre-populated with test data."""
    # Add chunks to vector store
    vector_store.add_chunks(sample_chunks)
    
    # Add entities to graph store
    graph_store.add_entities(sample_entities)
    
    return vector_store, graph_store
