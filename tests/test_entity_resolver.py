"""
Tests for Entity Resolver.

Tests entity matching, merging, and resolution without LLM.
"""

import pytest
from uuid import uuid4

from src.knowledge.entity_resolver import EntityResolver, ResolvedEntity, EntityMatch
from src.knowledge.schemas import Entity, EntityType
from src.knowledge.graph_store import GraphStore


class TestEntityResolver:
    """Unit tests for EntityResolver."""
    
    @pytest.fixture
    def graph_store(self, tmp_path) -> GraphStore:
        """Create a temp graph store."""
        return GraphStore(persist_path=tmp_path / "test_graph.json")
    
    @pytest.fixture
    def resolver(self, graph_store) -> EntityResolver:
        """Create resolver with graph store."""
        return EntityResolver(graph_store, similarity_threshold=0.85)
    
    def test_resolver_initialization(self, resolver) -> None:
        """Test EntityResolver can be initialized."""
        assert resolver.similarity_threshold == 0.85
        assert resolver.graph_store is not None
    
    def test_exact_match_case_insensitive(self, resolver) -> None:
        """Test exact match (case insensitive)."""
        entity_a = Entity(
            entity_type=EntityType.PERSON,
            value="John Doe",
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=1,
            source_text="John Doe is CEO",
        )
        entity_b = Entity(
            entity_type=EntityType.PERSON,
            value="john doe",
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=2,
            source_text="john doe signed",
        )
        
        score, match_type, _ = resolver._compute_similarity(entity_a, entity_b)
        
        assert score >= 0.95
        assert match_type == "exact"
    
    def test_initial_matching(self, resolver) -> None:
        """Test initial/abbreviation matching J. Doe = John Doe."""
        entity_a = Entity(
            entity_type=EntityType.PERSON,
            value="John Doe",
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=1,
            source_text="John Doe",
        )
        entity_b = Entity(
            entity_type=EntityType.PERSON,
            value="J. Doe",
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=2,
            source_text="J. Doe",
        )
        
        score, match_type, _ = resolver._compute_similarity(entity_a, entity_b)
        
        assert score >= 0.85
        assert match_type == "initial"
    
    def test_title_stripping(self, resolver) -> None:
        """Test title stripping Mr. Doe = Doe."""
        entity_a = Entity(
            entity_type=EntityType.PERSON,
            value="Mr. John Smith",
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=1,
            source_text="Mr. John Smith",
        )
        entity_b = Entity(
            entity_type=EntityType.PERSON,
            value="John Smith",
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=2,
            source_text="John Smith",
        )
        
        score, match_type, _ = resolver._compute_similarity(entity_a, entity_b)
        
        assert score >= 0.90  # After normalization
    
    def test_organization_suffix_matching(self, resolver) -> None:
        """Test organization suffix matching ABC Corp = ABC Corporation."""
        entity_a = Entity(
            entity_type=EntityType.ORGANIZATION,
            value="ABC Corporation",
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=1,
            source_text="ABC Corporation",
        )
        entity_b = Entity(
            entity_type=EntityType.ORGANIZATION,
            value="ABC Corp",
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=2,
            source_text="ABC Corp",
        )
        
        score, match_type, _ = resolver._compute_similarity(entity_a, entity_b)
        
        assert score >= 0.85
    
    def test_amount_matching(self, resolver) -> None:
        """Test monetary amount matching $500k = $500,000."""
        entity_a = Entity(
            entity_type=EntityType.MONETARY_AMOUNT,
            value="$500,000",
            normalized_value="500000",
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=1,
            source_text="$500,000",
        )
        entity_b = Entity(
            entity_type=EntityType.MONETARY_AMOUNT,
            value="$500k",
            normalized_value="500000",
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=2,
            source_text="$500k",
        )
        
        score, match_type, _ = resolver._compute_similarity(entity_a, entity_b)
        
        assert score >= 0.99
        assert match_type == "amount"
    
    def test_no_match_different_names(self, resolver) -> None:
        """Test that different names don't match."""
        entity_a = Entity(
            entity_type=EntityType.PERSON,
            value="John Doe",
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=1,
            source_text="John Doe",
        )
        entity_b = Entity(
            entity_type=EntityType.PERSON,
            value="Jane Smith",
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=2,
            source_text="Jane Smith",
        )
        
        score, _, _ = resolver._compute_similarity(entity_a, entity_b)
        
        assert score < 0.85  # Should not match
    
    def test_resolve_entities_groups_correctly(self, resolver) -> None:
        """Test that resolve_entities correctly groups matching entities."""
        doc_id = uuid4()
        chunk_id = uuid4()
        
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
                entity_type=EntityType.PERSON,
                value="john doe",  # Same person, different case
                source_document_id=doc_id,
                source_chunk_id=chunk_id,
                source_page=2,
                source_text="john doe",
            ),
            Entity(
                entity_type=EntityType.PERSON,
                value="Jane Smith",  # Different person
                source_document_id=doc_id,
                source_chunk_id=chunk_id,
                source_page=3,
                source_text="Jane Smith",
            ),
        ]
        
        resolved = resolver.resolve_entities(entities)
        
        # Should have 2 resolved entities: John Doe group and Jane Smith
        assert len(resolved) == 2
        
        # Check John Doe was merged
        john_group = next(
            (r for r in resolved if "john" in r.canonical_value.lower()),
            None
        )
        assert john_group is not None
        assert len(john_group.source_entity_ids) == 2
    
    def test_create_resolved_entity_uses_most_common(self, resolver) -> None:
        """Test that canonical value uses most common form."""
        doc_id = uuid4()
        chunk_id = uuid4()
        
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
                entity_type=EntityType.PERSON,
                value="John Doe",
                source_document_id=doc_id,
                source_chunk_id=chunk_id,
                source_page=2,
                source_text="John Doe",
            ),
            Entity(
                entity_type=EntityType.PERSON,
                value="J. Doe",  # Less common
                source_document_id=doc_id,
                source_chunk_id=chunk_id,
                source_page=3,
                source_text="J. Doe",
            ),
        ]
        
        resolved = resolver._create_resolved_entity(entities)
        
        # Canonical should be "John Doe" (most common)
        assert resolved.canonical_value == "John Doe"
        assert "J. Doe" in resolved.aliases
    
    def test_find_matches(self, resolver) -> None:
        """Test find_matches returns sorted results."""
        doc_id = uuid4()
        chunk_id = uuid4()
        
        entity = Entity(
            entity_type=EntityType.PERSON,
            value="John Doe",
            source_document_id=doc_id,
            source_chunk_id=chunk_id,
            source_page=1,
            source_text="John Doe",
        )
        
        candidates = [
            Entity(
                entity_type=EntityType.PERSON,
                value="john doe",  # Exact match
                source_document_id=doc_id,
                source_chunk_id=chunk_id,
                source_page=2,
                source_text="john doe",
            ),
            Entity(
                entity_type=EntityType.PERSON,
                value="J. Doe",  # Initial match
                source_document_id=doc_id,
                source_chunk_id=chunk_id,
                source_page=3,
                source_text="J. Doe",
            ),
            Entity(
                entity_type=EntityType.PERSON,
                value="Jane Smith",  # No match
                source_document_id=doc_id,
                source_chunk_id=chunk_id,
                source_page=4,
                source_text="Jane Smith",
            ),
        ]
        
        matches = resolver.find_matches(entity, candidates)
        
        # Should have 2 matches (John doe and J. Doe)
        assert len(matches) == 2
        
        # First should be exact match (higher score)
        assert matches[0].similarity_score > matches[1].similarity_score
    
    def test_parse_amount(self, resolver) -> None:
        """Test amount parsing."""
        assert resolver._parse_amount("$500,000") == 500000
        assert resolver._parse_amount("500k") == 500000
        assert resolver._parse_amount("1.5M") == 1500000
        assert resolver._parse_amount("$1B") == 1000000000
        assert resolver._parse_amount("100") == 100
    
    def test_token_similarity(self, resolver) -> None:
        """Test token (Jaccard) similarity."""
        # Identical
        assert resolver._token_similarity("john doe", "john doe") == 1.0
        
        # Partial overlap
        similarity = resolver._token_similarity("john smith", "john doe")
        assert 0 < similarity < 1
        
        # No overlap
        assert resolver._token_similarity("abc", "xyz") == 0.0
