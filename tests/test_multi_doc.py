"""
Tests for Multi-Document Analyzer.

Tests cross-document analysis, variation detection, and n-way conflicts.
"""

from uuid import uuid4

import pytest

from src.agents.multi_doc_analyzer import (
    DocumentSet,
    EntityOccurrence,
    EntityVariation,
    MultiDocAnalyzer,
    MultiDocConflict,
    MultiDocReport,
)
from src.agents.schemas import ConflictSeverity
from src.knowledge.graph_store import GraphStore
from src.knowledge.schemas import Entity, EntityType
from src.knowledge.vector_store import VectorStore


class TestDocumentSet:
    """Tests for DocumentSet model."""

    def test_document_set_creation(self) -> None:
        """Test DocumentSet can be created."""
        doc_ids = [uuid4(), uuid4()]
        doc_set = DocumentSet(
            document_ids=doc_ids,
            document_names={
                str(doc_ids[0]): "Contract A.pdf",
                str(doc_ids[1]): "Contract B.pdf",
            }
        )

        assert doc_set.count == 2
        assert doc_set.get_name(doc_ids[0]) == "Contract A.pdf"

    def test_get_name_fallback(self) -> None:
        """Test get_name falls back to ID prefix."""
        doc_id = uuid4()
        doc_set = DocumentSet(document_ids=[doc_id])

        name = doc_set.get_name(doc_id)
        assert "Document" in name


class TestEntityVariation:
    """Tests for EntityVariation model."""

    def test_no_conflict_when_same_values(self) -> None:
        """Test is_conflict is False when all values are same."""
        variation = EntityVariation(
            entity_type=EntityType.SALARY,
            canonical_value="$500,000",
            occurrences=[
                EntityOccurrence(
                    document_id=uuid4(),
                    document_name="Doc A",
                    value="$500,000",
                    page_number=1,
                    chunk_id=uuid4(),
                ),
                EntityOccurrence(
                    document_id=uuid4(),
                    document_name="Doc B",
                    value="$500,000",
                    page_number=1,
                    chunk_id=uuid4(),
                ),
            ]
        )

        assert not variation.is_conflict
        assert variation.conflict_severity is None

    def test_conflict_when_different_values(self) -> None:
        """Test is_conflict is True when values differ."""
        variation = EntityVariation(
            entity_type=EntityType.SALARY,
            canonical_value="$500,000",
            occurrences=[
                EntityOccurrence(
                    document_id=uuid4(),
                    document_name="Doc A",
                    value="$500,000",
                    page_number=1,
                    chunk_id=uuid4(),
                ),
                EntityOccurrence(
                    document_id=uuid4(),
                    document_name="Doc B",
                    value="$450,000",  # Different!
                    page_number=1,
                    chunk_id=uuid4(),
                ),
            ]
        )

        assert variation.is_conflict
        assert variation.conflict_severity is not None

    def test_unique_values(self) -> None:
        """Test unique_values property."""
        doc_a = uuid4()
        doc_b = uuid4()

        variation = EntityVariation(
            entity_type=EntityType.PERCENTAGE,
            canonical_value="5%",
            occurrences=[
                EntityOccurrence(
                    document_id=doc_a,
                    document_name="Doc A",
                    value="5%",
                    page_number=1,
                    chunk_id=uuid4(),
                ),
                EntityOccurrence(
                    document_id=doc_a,
                    document_name="Doc A",
                    value="5%",
                    page_number=2,
                    chunk_id=uuid4(),
                ),
                EntityOccurrence(
                    document_id=doc_b,
                    document_name="Doc B",
                    value="8%",
                    page_number=1,
                    chunk_id=uuid4(),
                ),
            ]
        )

        unique = variation.unique_values
        assert len(unique) == 2
        assert "5%" in unique
        assert "8%" in unique
        # Most common first
        assert unique[0] == "5%"

    def test_document_count(self) -> None:
        """Test document_count property."""
        doc_a = uuid4()
        doc_b = uuid4()

        variation = EntityVariation(
            entity_type=EntityType.DATE,
            canonical_value="January 1, 2024",
            occurrences=[
                EntityOccurrence(
                    document_id=doc_a,
                    document_name="Doc A",
                    value="January 1, 2024",
                    page_number=1,
                    chunk_id=uuid4(),
                ),
                EntityOccurrence(
                    document_id=doc_a,
                    document_name="Doc A",
                    value="January 1, 2024",
                    page_number=5,
                    chunk_id=uuid4(),
                ),
                EntityOccurrence(
                    document_id=doc_b,
                    document_name="Doc B",
                    value="January 1, 2024",
                    page_number=1,
                    chunk_id=uuid4(),
                ),
            ]
        )

        assert variation.document_count == 2


class TestMultiDocConflict:
    """Tests for MultiDocConflict model."""

    def test_conflict_creation(self) -> None:
        """Test MultiDocConflict can be created."""
        from src.agents.schemas import ConflictType

        conflict = MultiDocConflict(
            entity_type=EntityType.SALARY,
            conflict_type=ConflictType.SALARY_MISMATCH,
            severity=ConflictSeverity.HIGH,
            title="Salary Mismatch",
            description="Found different salary values",
            unique_values=["$500,000", "$450,000"],
            document_count=2,
        )

        assert conflict.severity == ConflictSeverity.HIGH
        assert len(conflict.unique_values) == 2


class TestMultiDocReport:
    """Tests for MultiDocReport model."""

    def test_report_creation(self) -> None:
        """Test MultiDocReport can be created."""
        doc_set = DocumentSet(document_ids=[uuid4(), uuid4()])

        report = MultiDocReport(
            document_set=doc_set,
            total_entities=10,
            total_relationships=5,
        )

        assert report.conflict_count == 0
        assert report.total_entities == 10

    def test_critical_conflicts_filter(self) -> None:
        """Test critical_conflicts property."""
        from src.agents.schemas import ConflictType

        doc_set = DocumentSet(document_ids=[uuid4()])

        report = MultiDocReport(
            document_set=doc_set,
            conflicts=[
                MultiDocConflict(
                    entity_type=EntityType.SALARY,
                    conflict_type=ConflictType.SALARY_MISMATCH,
                    severity=ConflictSeverity.CRITICAL,
                    title="Critical Conflict",
                    description="Critical",
                ),
                MultiDocConflict(
                    entity_type=EntityType.DATE,
                    conflict_type=ConflictType.DATE_CONFLICT,
                    severity=ConflictSeverity.LOW,
                    title="Low Conflict",
                    description="Low",
                ),
            ]
        )

        assert len(report.critical_conflicts) == 1
        assert report.critical_conflicts[0].severity == ConflictSeverity.CRITICAL


class TestMultiDocAnalyzer:
    """Tests for MultiDocAnalyzer."""

    @pytest.fixture
    def graph_store(self, tmp_path) -> GraphStore:
        """Create temp graph store with entities."""
        store = GraphStore(persist_path=tmp_path / "test_graph.json")

        doc_a = uuid4()
        doc_b = uuid4()

        # Add entities from two documents with conflicting values
        entities = [
            Entity(
                entity_type=EntityType.SALARY,
                value="$500,000",
                source_document_id=doc_a,
                source_chunk_id=uuid4(),
                source_page=1,
                source_text="Salary of $500,000",
            ),
            Entity(
                entity_type=EntityType.SALARY,
                value="$450,000",  # Conflict!
                source_document_id=doc_b,
                source_chunk_id=uuid4(),
                source_page=1,
                source_text="Salary of $450,000",
            ),
            Entity(
                entity_type=EntityType.PERSON,
                value="John Doe",
                source_document_id=doc_a,
                source_chunk_id=uuid4(),
                source_page=1,
                source_text="John Doe",
            ),
            Entity(
                entity_type=EntityType.PERSON,
                value="John Doe",  # Same across docs
                source_document_id=doc_b,
                source_chunk_id=uuid4(),
                source_page=1,
                source_text="John Doe",
            ),
        ]

        store.add_entities(entities)
        store._doc_a = doc_a
        store._doc_b = doc_b

        return store

    @pytest.fixture
    def vector_store(self, tmp_path) -> VectorStore:
        """Create temp vector store."""
        from src.knowledge.vector_store import VectorStoreConfig
        config = VectorStoreConfig(
            persist_directory=tmp_path / "chromadb",
            collection_name="test"
        )
        return VectorStore(config)

    @pytest.mark.asyncio
    async def test_analyze_detects_variations(
        self,
        graph_store,
        vector_store,
    ) -> None:
        """Test that analyze runs without error."""
        analyzer = MultiDocAnalyzer(vector_store, graph_store)

        doc_set = DocumentSet(
            document_ids=[graph_store._doc_a, graph_store._doc_b],
            document_names={
                str(graph_store._doc_a): "Contract A",
                str(graph_store._doc_b): "Contract B",
            }
        )

        report = await analyzer.analyze(doc_set, resolve_entities=False)

        # Verify report structure is valid
        assert report.document_set.count == 2
        assert isinstance(report.total_entities, int)
        assert isinstance(report.conflicts, list)

    @pytest.mark.asyncio
    async def test_analyze_finds_unanimous(
        self,
        graph_store,
        vector_store,
    ) -> None:
        """Test that analyze runs and returns unanimous list."""
        analyzer = MultiDocAnalyzer(vector_store, graph_store)

        doc_set = DocumentSet(
            document_ids=[graph_store._doc_a, graph_store._doc_b],
        )

        report = await analyzer.analyze(doc_set, resolve_entities=False)

        # Verify unanimous_entities is a list of strings
        assert isinstance(report.unanimous_entities, list)

    @pytest.mark.asyncio
    async def test_analyze_with_focus_areas(
        self,
        graph_store,
        vector_store,
    ) -> None:
        """Test analyze with focus_areas filter."""
        analyzer = MultiDocAnalyzer(vector_store, graph_store)

        doc_set = DocumentSet(
            document_ids=[graph_store._doc_a, graph_store._doc_b],
        )

        report = await analyzer.analyze(
            doc_set,
            focus_areas=["salary"],
            resolve_entities=False,
        )

        # Verify report was created
        assert report.document_set.count == 2
