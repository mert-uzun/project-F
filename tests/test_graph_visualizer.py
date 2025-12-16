"""
Tests for Graph Visualizer.

Tests PyVis graph generation and HTML export.
"""

import pytest
from pathlib import Path
from uuid import uuid4

from src.knowledge.graph_visualizer import (
    GraphVisualizer,
    visualize_knowledge_graph,
    ENTITY_COLORS,
    SEVERITY_COLORS,
    RELATIONSHIP_COLORS,
)
from src.knowledge.schemas import Entity, EntityType, RelationshipType
from src.knowledge.graph_store import GraphStore
from src.agents.schemas import ConflictSeverity


class TestColorSchemes:
    """Tests for color scheme definitions."""
    
    def test_severity_colors_defined(self) -> None:
        """Test all severity levels have colors."""
        assert ConflictSeverity.CRITICAL in SEVERITY_COLORS
        assert ConflictSeverity.HIGH in SEVERITY_COLORS
        assert ConflictSeverity.MEDIUM in SEVERITY_COLORS
        assert ConflictSeverity.LOW in SEVERITY_COLORS
    
    def test_entity_colors_defined(self) -> None:
        """Test main entity types have colors."""
        assert EntityType.PERSON in ENTITY_COLORS
        assert EntityType.ORGANIZATION in ENTITY_COLORS
        assert EntityType.MONETARY_AMOUNT in ENTITY_COLORS
        assert EntityType.DATE in ENTITY_COLORS
    
    def test_relationship_colors_defined(self) -> None:
        """Test key relationship types have colors."""
        assert RelationshipType.CONFLICTS_WITH in RELATIONSHIP_COLORS
        assert RelationshipType.HAS_SALARY in RELATIONSHIP_COLORS
        assert RelationshipType.REFERENCES in RELATIONSHIP_COLORS
    
    def test_colors_are_valid_hex(self) -> None:
        """Test all colors are valid hex codes."""
        import re
        hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')
        
        for color in SEVERITY_COLORS.values():
            assert hex_pattern.match(color), f"Invalid color: {color}"
        
        for color in ENTITY_COLORS.values():
            assert hex_pattern.match(color), f"Invalid color: {color}"
        
        for color in RELATIONSHIP_COLORS.values():
            assert hex_pattern.match(color), f"Invalid color: {color}"


class TestGraphVisualizer:
    """Tests for GraphVisualizer."""
    
    @pytest.fixture
    def graph_store_with_entities(self, tmp_path) -> GraphStore:
        """Create graph store with sample entities."""
        store = GraphStore(persist_path=tmp_path / "test_graph.json")
        
        doc_id = uuid4()
        
        entities = [
            Entity(
                entity_type=EntityType.PERSON,
                value="John Doe",
                source_document_id=doc_id,
                source_chunk_id=uuid4(),
                source_page=1,
                source_text="John Doe is the CEO",
            ),
            Entity(
                entity_type=EntityType.ORGANIZATION,
                value="ABC Corporation",
                source_document_id=doc_id,
                source_chunk_id=uuid4(),
                source_page=1,
                source_text="ABC Corporation",
            ),
            Entity(
                entity_type=EntityType.MONETARY_AMOUNT,
                value="$500,000",
                source_document_id=doc_id,
                source_chunk_id=uuid4(),
                source_page=2,
                source_text="Salary of $500,000",
            ),
        ]
        
        store.add_entities(entities)
        store._doc_id = doc_id
        store._entities = entities
        
        return store
    
    @pytest.fixture
    def visualizer(self, graph_store_with_entities) -> GraphVisualizer:
        """Create visualizer with graph store."""
        return GraphVisualizer(graph_store_with_entities)
    
    def test_visualizer_initialization(self, visualizer) -> None:
        """Test GraphVisualizer can be initialized."""
        assert visualizer.graph_store is not None
        assert visualizer._network is None
    
    def test_generate_network(self, visualizer, graph_store_with_entities) -> None:
        """Test network generation."""
        network = visualizer.generate_network()
        
        assert network is not None
        assert visualizer._network is not None
    
    def test_generate_network_with_document_filter(
        self,
        visualizer,
        graph_store_with_entities,
    ) -> None:
        """Test network generation with document filter."""
        network = visualizer.generate_network(
            document_ids=[graph_store_with_entities._doc_id]
        )
        
        assert network is not None
    
    def test_generate_network_with_entity_type_filter(
        self,
        visualizer,
    ) -> None:
        """Test network generation with entity type filter."""
        network = visualizer.generate_network(
            entity_types=[EntityType.PERSON]
        )
        
        assert network is not None
    
    def test_get_node_size(self, visualizer) -> None:
        """Test node size calculation."""
        person_size = visualizer._get_node_size(EntityType.PERSON, is_conflict=False)
        org_size = visualizer._get_node_size(EntityType.ORGANIZATION, is_conflict=False)
        conflict_size = visualizer._get_node_size(EntityType.PERSON, is_conflict=True)
        
        assert person_size > 0
        assert org_size > 0
        assert conflict_size > person_size  # Conflicts are larger
    
    def test_build_tooltip(self, visualizer, graph_store_with_entities) -> None:
        """Test tooltip generation."""
        entity = graph_store_with_entities._entities[0]
        
        tooltip = visualizer._build_tooltip(entity)
        
        assert "<b>" in tooltip
        assert entity.value in tooltip
        assert "Type:" in tooltip
    
    def test_truncate_label(self, visualizer) -> None:
        """Test label truncation."""
        short = visualizer._truncate_label("Short")
        assert short == "Short"
        
        long = visualizer._truncate_label("This is a very long entity name that should be truncated")
        assert len(long) <= 25
        assert long.endswith("...")
    
    def test_get_node_color(self, visualizer, graph_store_with_entities) -> None:
        """Test node color retrieval."""
        entity = graph_store_with_entities._entities[0]  # PERSON
        
        color = visualizer.get_node_color(entity)
        
        assert color == ENTITY_COLORS[EntityType.PERSON]
    
    def test_get_node_color_conflict(
        self,
        graph_store_with_entities,
    ) -> None:
        """Test node color for conflict entity."""
        entity = graph_store_with_entities._entities[0]
        
        # Create visualizer with this entity marked as conflict
        visualizer = GraphVisualizer(
            graph_store_with_entities,
            conflict_entities={entity.entity_id},
        )
        
        color = visualizer.get_node_color(entity)
        
        assert color == SEVERITY_COLORS[ConflictSeverity.CRITICAL]
    
    def test_save_html(self, visualizer, tmp_path) -> None:
        """Test HTML export."""
        visualizer.generate_network()
        
        output_path = tmp_path / "test_graph.html"
        result_path = visualizer.save_html(output_path)
        
        assert result_path.exists()
        assert result_path.suffix == ".html"
        
        # Check content
        content = result_path.read_text()
        assert "<html>" in content or "<!DOCTYPE" in content
    
    def test_save_html_with_legend(self, visualizer, tmp_path) -> None:
        """Test HTML export includes legend."""
        visualizer.generate_network()
        
        output_path = tmp_path / "test_graph_legend.html"
        result_path = visualizer.save_html(output_path, include_legend=True)
        
        content = result_path.read_text()
        assert "graph-legend" in content
        assert "Legend" in content


class TestVisualizeKnowledgeGraph:
    """Tests for convenience function."""
    
    def test_visualize_knowledge_graph(self, tmp_path) -> None:
        """Test the convenience function."""
        store = GraphStore(persist_path=tmp_path / "test_graph.json")
        
        doc_id = uuid4()
        entity = Entity(
            entity_type=EntityType.PERSON,
            value="Test Person",
            source_document_id=doc_id,
            source_chunk_id=uuid4(),
            source_page=1,
            source_text="Test",
        )
        store.add_entity(entity)
        
        output_path = tmp_path / "output.html"
        result = visualize_knowledge_graph(
            store,
            output_path,
            document_ids=[doc_id],
        )
        
        assert result.exists()
        assert result == output_path


class TestNetworkPhysics:
    """Tests for network physics configuration."""
    
    @pytest.fixture
    def graph_store(self, tmp_path) -> GraphStore:
        store = GraphStore(persist_path=tmp_path / "test.json")
        entity = Entity(
            entity_type=EntityType.PERSON,
            value="Test",
            source_document_id=uuid4(),
            source_chunk_id=uuid4(),
            source_page=1,
            source_text="Test",
        )
        store.add_entity(entity)
        return store
    
    def test_physics_enabled(self, graph_store) -> None:
        """Test network with physics enabled."""
        viz = GraphVisualizer(graph_store)
        network = viz.generate_network(physics_enabled=True)
        
        assert network is not None
    
    def test_physics_disabled(self, graph_store) -> None:
        """Test network with physics disabled."""
        viz = GraphVisualizer(graph_store)
        network = viz.generate_network(physics_enabled=False)
        
        assert network is not None
