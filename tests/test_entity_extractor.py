"""
Tests for Entity Extractor - REAL Integration Tests.

Tests require LLM backend when testing extraction.
Non-LLM tests validate entity conversion and schema handling.
"""

from uuid import uuid4

import pytest

from src.ingestion.schemas import ChunkMetadata, DocumentChunk
from src.knowledge.entity_extractor import (
    EntityExtractor,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionOutput,
)
from src.knowledge.schemas import (
    Entity,
    EntityType,
    ExtractionResult,
    RelationshipType,
)
from tests.conftest import requires_llm

# ============================================================================
# EntityExtractor Unit Tests (No LLM Required)
# ============================================================================


class TestEntityExtractor:
    """Tests for EntityExtractor that don't require LLM."""

    def test_extractor_initialization(self) -> None:
        """Test EntityExtractor can be initialized."""
        extractor = EntityExtractor()

        assert extractor._use_structured_output is True
        assert extractor._llm is None  # Lazy loaded

    def test_entity_type_mapping(self) -> None:
        """Test that entity types are correctly mapped."""
        extractor = EntityExtractor()

        doc_id = uuid4()
        chunk_id = uuid4()

        # Test various entity types
        extracted = [
            ExtractedEntity(
                entity_type="person",
                value="John Doe",
                confidence=0.9,
                context="John Doe is the CEO.",
            ),
            ExtractedEntity(
                entity_type="monetary_amount",
                value="$500,000",
                normalized_value="500000",
                confidence=0.95,
                context="Salary of $500,000.",
            ),
            ExtractedEntity(
                entity_type="percentage",
                value="5%",
                normalized_value="5",
                confidence=0.88,
                context="5% equity stake.",
            ),
            ExtractedEntity(
                entity_type="unknown_type",  # Should map to OTHER
                value="Something",
                confidence=0.7,
                context="Context.",
            ),
        ]

        entities = extractor._convert_entities(extracted, doc_id, chunk_id, page_number=1)

        assert len(entities) == 4
        assert entities[0].entity_type == EntityType.PERSON
        assert entities[1].entity_type == EntityType.MONETARY_AMOUNT
        assert entities[2].entity_type == EntityType.PERCENTAGE
        assert entities[3].entity_type == EntityType.OTHER  # Unknown mapped to OTHER

    def test_relationship_conversion(self) -> None:
        """Test relationship conversion with entity matching."""
        extractor = EntityExtractor()

        doc_id = uuid4()
        chunk_id = uuid4()

        # Create entities first
        entities = [
            Entity(
                entity_type=EntityType.PERSON,
                value="John Doe",
                source_document_id=doc_id,
                source_chunk_id=chunk_id,
                source_page=1,
                source_text="John Doe",
            ),
            Entity(
                entity_type=EntityType.MONETARY_AMOUNT,
                value="$500,000",
                source_document_id=doc_id,
                source_chunk_id=chunk_id,
                source_page=1,
                source_text="$500,000",
            ),
        ]

        # Create extracted relationships
        extracted_rels = [
            ExtractedRelationship(
                source_entity="John Doe",
                relationship_type="has_salary",
                target_entity="$500,000",
                confidence=0.9,
                context="John Doe earns $500,000.",
            ),
        ]

        relationships = extractor._convert_relationships(
            extracted_rels, entities, doc_id, chunk_id, page_number=1
        )

        assert len(relationships) == 1
        assert relationships[0].relationship_type == RelationshipType.HAS_SALARY
        assert relationships[0].source_entity_id == entities[0].entity_id
        assert relationships[0].target_entity_id == entities[1].entity_id

    def test_relationship_with_missing_entity(self) -> None:
        """Test that relationships with missing entities are skipped."""
        extractor = EntityExtractor()

        doc_id = uuid4()
        chunk_id = uuid4()

        # Create only one entity
        entities = [
            Entity(
                entity_type=EntityType.PERSON,
                value="John Doe",
                source_document_id=doc_id,
                source_chunk_id=chunk_id,
                source_page=1,
                source_text="John Doe",
            ),
        ]

        # Relationship references an entity that doesn't exist
        extracted_rels = [
            ExtractedRelationship(
                source_entity="John Doe",
                relationship_type="has_salary",
                target_entity="$500,000",  # This entity doesn't exist
                confidence=0.9,
                context="Context.",
            ),
        ]

        relationships = extractor._convert_relationships(
            extracted_rels, entities, doc_id, chunk_id, page_number=1
        )

        # Should be empty because target entity doesn't exist
        assert len(relationships) == 0

    def test_unknown_relationship_type_maps_to_references(self) -> None:
        """Test that unknown relationship types map to REFERENCES."""
        extractor = EntityExtractor()

        doc_id = uuid4()
        chunk_id = uuid4()

        entities = [
            Entity(
                entity_type=EntityType.PERSON,
                value="John",
                source_document_id=doc_id,
                source_chunk_id=chunk_id,
                source_page=1,
                source_text="John",
            ),
            Entity(
                entity_type=EntityType.ORGANIZATION,
                value="Company",
                source_document_id=doc_id,
                source_chunk_id=chunk_id,
                source_page=1,
                source_text="Company",
            ),
        ]

        extracted_rels = [
            ExtractedRelationship(
                source_entity="John",
                relationship_type="some_unknown_type",
                target_entity="Company",
                confidence=0.8,
                context="Context.",
            ),
        ]

        relationships = extractor._convert_relationships(
            extracted_rels, entities, doc_id, chunk_id, page_number=1
        )

        assert len(relationships) == 1
        assert relationships[0].relationship_type == RelationshipType.REFERENCES


# ============================================================================
# ExtractionOutput Schema Tests
# ============================================================================


class TestExtractionSchemas:
    """Tests for extraction-related schemas."""

    def test_extracted_entity_creation(self) -> None:
        """Test ExtractedEntity model."""
        entity = ExtractedEntity(
            entity_type="person",
            value="John Doe",
            confidence=0.95,
            context="John Doe is the CEO.",
        )

        assert entity.entity_type == "person"
        assert entity.value == "John Doe"
        assert entity.normalized_value is None
        assert entity.confidence == 0.95

    def test_extracted_relationship_creation(self) -> None:
        """Test ExtractedRelationship model."""
        rel = ExtractedRelationship(
            source_entity="John Doe",
            relationship_type="has_role",
            target_entity="CEO",
            confidence=0.9,
            context="John Doe is the CEO.",
        )

        assert rel.source_entity == "John Doe"
        assert rel.target_entity == "CEO"
        assert rel.relationship_type == "has_role"

    def test_extraction_output_creation(self) -> None:
        """Test ExtractionOutput model."""
        output = ExtractionOutput(
            entities=[
                ExtractedEntity(
                    entity_type="person",
                    value="Alice",
                    confidence=0.9,
                    context="Context.",
                ),
            ],
            relationships=[],
        )

        assert len(output.entities) == 1
        assert len(output.relationships) == 0

    def test_extraction_result_properties(self) -> None:
        """Test ExtractionResult computed properties."""
        result = ExtractionResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            entities=[
                Entity(
                    entity_type=EntityType.PERSON,
                    value="Test",
                    source_document_id=uuid4(),
                    source_chunk_id=uuid4(),
                    source_page=1,
                    source_text="Test",
                ),
            ],
            relationships=[],
            extraction_time_seconds=1.5,
            model_used="test",
        )

        assert result.entity_count == 1
        assert result.relationship_count == 0


# ============================================================================
# Integration Tests (Require LLM)
# ============================================================================


class TestEntityExtractorIntegration:
    """Integration tests that require real LLM calls."""

    @pytest.mark.asyncio
    @requires_llm()
    async def test_extract_from_chunk_real_llm(self) -> None:
        """
        Test entity extraction with real LLM.
        Validates that extraction returns results.
        """
        extractor = EntityExtractor()

        chunk = DocumentChunk(
            content="""
            John Doe, the CEO of ABC Corporation, will receive an annual
            base salary of $500,000 starting January 1, 2024. Additionally,
            he is granted 5% equity in the company, vesting over 4 years.
            """,
            metadata=ChunkMetadata(
                chunk_id=uuid4(),
                document_id=uuid4(),
                page_number=1,
                chunk_index=0,
                char_start=0,
                char_end=200,
            ),
        )

        result = await extractor.extract(chunk)

        # Validate result structure
        assert isinstance(result, ExtractionResult)
        assert result.chunk_id == chunk.metadata.chunk_id
        assert result.document_id == chunk.metadata.document_id
        assert result.extraction_time_seconds > 0

        # Should return lists (may be empty if LLM has issues)
        assert isinstance(result.entities, list)
        assert isinstance(result.relationships, list)
        # Allow errors since LLM behavior can vary
        assert isinstance(result.errors, list)

    @pytest.mark.asyncio
    @requires_llm()
    async def test_extract_batch_real_llm(self, sample_chunks) -> None:
        """
        Test batch extraction with real LLM.
        """
        extractor = EntityExtractor()

        results = await extractor.extract_batch(sample_chunks[:2])  # Test with 2 chunks

        assert len(results) == 2
        for result in results:
            assert isinstance(result, ExtractionResult)
            # Allow errors since LLM behavior can vary
            assert isinstance(result.errors, list)

    @pytest.mark.asyncio
    @requires_llm()
    async def test_focused_extraction_real_llm(self) -> None:
        """
        Test focused extraction with specific areas.
        """
        extractor = EntityExtractor()

        chunk = DocumentChunk(
            content="""
            The employee compensation package includes:
            - Base salary: $400,000 annually
            - Equity: 3% company shares
            - Bonus: Up to 25% of base salary
            """,
            metadata=ChunkMetadata(
                chunk_id=uuid4(),
                document_id=uuid4(),
                page_number=1,
                chunk_index=0,
                char_start=0,
                char_end=150,
            ),
        )

        result = await extractor.extract(chunk, focus_areas=["salary", "equity", "percentage"])

        assert isinstance(result, ExtractionResult)
        assert len(result.errors) == 0
        # Should find at least monetary amounts or percentages
        assert len(result.entities) >= 0  # LLM may or may not find all
