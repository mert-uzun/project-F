"""
Tests for Knowledge Layer.
"""

from pathlib import Path
from uuid import uuid4

import pytest

from src.knowledge.schemas import (
    Entity,
    EntityType,
    Relationship,
    RelationshipType,
)
from src.knowledge.vector_store import VectorStore, VectorStoreConfig, SearchResult
from src.knowledge.graph_store import GraphStore, GraphNode, GraphEdge


class TestVectorStore:
    """Tests for VectorStore."""
    
    @pytest.fixture
    def temp_store(self, tmp_path: Path) -> VectorStore:
        """Create a temporary vector store."""
        config = VectorStoreConfig(
            persist_directory=tmp_path / "chroma",
            collection_name="test_collection",
        )
        return VectorStore(config)
    
    def test_count_empty_store(self, temp_store: VectorStore) -> None:
        """Test counting an empty store."""
        assert temp_store.count() == 0
    
    def test_search_empty_store(self, temp_store: VectorStore) -> None:
        """Test searching an empty store."""
        results = temp_store.search("test query")
        assert len(results) == 0


class TestGraphStore:
    """Tests for GraphStore."""
    
    @pytest.fixture
    def store(self) -> GraphStore:
        """Create a graph store."""
        return GraphStore()
    
    @pytest.fixture
    def sample_entity(self) -> Entity:
        """Create a sample entity."""
        return Entity(
            entity_type=EntityType.PERSON,
            value="John Doe",
            normalized_value="john doe",
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=1,
            source_text="John Doe is the CEO of the company.",
            confidence=0.95,
        )
    
    @pytest.fixture
    def sample_entity_2(self) -> Entity:
        """Create a second sample entity."""
        return Entity(
            entity_type=EntityType.MONETARY_AMOUNT,
            value="$500,000",
            normalized_value="500000",
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=2,
            source_text="The salary is $500,000 per year.",
            confidence=0.9,
        )
    
    def test_add_entity(self, store: GraphStore, sample_entity: Entity) -> None:
        """Test adding an entity."""
        node = store.add_entity(sample_entity)
        
        assert node.id == str(sample_entity.entity_id)
        assert node.label == "John Doe"
        assert node.type == EntityType.PERSON
        assert store.node_count() == 1
    
    def test_add_relationship(
        self, 
        store: GraphStore, 
        sample_entity: Entity, 
        sample_entity_2: Entity,
    ) -> None:
        """Test adding a relationship."""
        store.add_entity(sample_entity)
        store.add_entity(sample_entity_2)
        
        relationship = Relationship(
            relationship_type=RelationshipType.HAS_SALARY,
            source_entity_id=sample_entity.entity_id,
            target_entity_id=sample_entity_2.entity_id,
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=2,
            source_text="John Doe earns $500,000",
            confidence=0.85,
        )
        
        edge = store.add_relationship(relationship)
        
        assert edge.source == str(sample_entity.entity_id)
        assert edge.target == str(sample_entity_2.entity_id)
        assert store.edge_count() == 1
    
    def test_get_entity(self, store: GraphStore, sample_entity: Entity) -> None:
        """Test retrieving an entity."""
        store.add_entity(sample_entity)
        
        retrieved = store.get_entity(sample_entity.entity_id)
        
        assert retrieved is not None
        assert retrieved.entity.value == "John Doe"
    
    def test_find_entities_by_type(
        self, 
        store: GraphStore, 
        sample_entity: Entity, 
        sample_entity_2: Entity,
    ) -> None:
        """Test finding entities by type."""
        store.add_entity(sample_entity)
        store.add_entity(sample_entity_2)
        
        persons = store.find_entities_by_type(EntityType.PERSON)
        amounts = store.find_entities_by_type(EntityType.MONETARY_AMOUNT)
        
        assert len(persons) == 1
        assert len(amounts) == 1
        assert persons[0].entity.value == "John Doe"
    
    def test_find_entities_by_value(
        self, 
        store: GraphStore, 
        sample_entity: Entity,
    ) -> None:
        """Test finding entities by value."""
        store.add_entity(sample_entity)
        
        # Exact match
        results = store.find_entities_by_value("john doe")
        assert len(results) == 1
        
        # Fuzzy match
        results = store.find_entities_by_value("john", fuzzy=True)
        assert len(results) == 1
    
    def test_get_entity_neighborhood(
        self,
        store: GraphStore,
        sample_entity: Entity,
        sample_entity_2: Entity,
    ) -> None:
        """Test getting entity neighborhood (subgraph)."""
        store.add_entity(sample_entity)
        store.add_entity(sample_entity_2)
        
        relationship = Relationship(
            relationship_type=RelationshipType.HAS_SALARY,
            source_entity_id=sample_entity.entity_id,
            target_entity_id=sample_entity_2.entity_id,
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=2,
            source_text="Context",
            confidence=0.85,
        )
        store.add_relationship(relationship)
        
        subgraph = store.get_entity_neighborhood(sample_entity.entity_id, hops=1)
        
        assert subgraph.node_count == 2
        assert subgraph.edge_count == 1
    
    def test_save_and_load(self, store: GraphStore, sample_entity: Entity, tmp_path: Path) -> None:
        """Test saving and loading the graph."""
        store.add_entity(sample_entity)
        
        save_path = tmp_path / "graph.json"
        store.save(save_path)
        
        assert save_path.exists()
        
        # Load in new store
        new_store = GraphStore(persist_path=save_path)
        
        assert new_store.node_count() == 1
        retrieved = new_store.get_entity(sample_entity.entity_id)
        assert retrieved is not None
        assert retrieved.entity.value == "John Doe"


class TestSchemas:
    """Tests for Knowledge schemas."""
    
    def test_entity_creation(self) -> None:
        """Test Entity model creation."""
        entity = Entity(
            entity_type=EntityType.ORGANIZATION,
            value="Acme Corp",
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=1,
            source_text="Acme Corp is a company.",
        )
        
        assert entity.entity_type == EntityType.ORGANIZATION
        assert entity.value == "Acme Corp"
        assert entity.confidence == 1.0  # default
    
    def test_relationship_creation(self) -> None:
        """Test Relationship model creation."""
        relationship = Relationship(
            relationship_type=RelationshipType.EMPLOYS,
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=1,
            source_text="Company employs person.",
        )
        
        assert relationship.relationship_type == RelationshipType.EMPLOYS
        assert relationship.confidence == 1.0  # default
