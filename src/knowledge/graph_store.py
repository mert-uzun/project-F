"""
Graph Store - NetworkX Integration.

Provides entity-relationship graph storage and traversal.
Part of the GraphRAG architecture - works alongside the Vector Store.

Key principle: Entities are nodes, Relationships are edges.
This enables questions like "What entities are connected to CEO?"
"""

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import networkx as nx
from pyvis.network import Network

from src.knowledge.schemas import (
    Entity,
    EntityType,
    GraphSearchResult,
    KnowledgeGraphStats,
    Relationship,
    RelationType,
    SubGraph,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphStoreError(Exception):
    """Raised when graph store operations fail."""

    pass


class GraphStore:
    """
    NetworkX-backed graph store for entity-relationship storage.
    
    Stores entities as nodes and relationships as edges.
    Supports graph traversal, subgraph extraction, and persistence.
    
    Usage:
        store = GraphStore(persist_path=Path("./data/graphs/kg.json"))
        store.add_entity(entity)
        store.add_relationship(relationship)
        subgraph = store.get_entity_subgraph(entity_id, max_hops=2)
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        """
        Initialize graph store.
        
        Args:
            persist_path: Path for JSON persistence (None for in-memory only)
        """
        self.persist_path = persist_path
        self._graph: nx.DiGraph = nx.DiGraph()
        self._entities: dict[UUID, Entity] = {}
        self._relationships: dict[UUID, Relationship] = {}
        
        # Load existing graph if persist_path exists
        if persist_path and persist_path.exists():
            self._load()
    
    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to the graph.
        
        Args:
            entity: Entity to add
        """
        self._entities[entity.entity_id] = entity
        
        # Add node with attributes
        self._graph.add_node(
            str(entity.entity_id),
            entity_type=entity.entity_type.value,
            name=entity.name,
            value=str(entity.value) if entity.value else None,
            normalized_value=str(entity.normalized_value) if entity.normalized_value else None,
            source_document_id=str(entity.source_document_id),
            source_page=entity.source_page,
            confidence=entity.confidence,
        )
        
        logger.debug(f"Added entity: {entity.name} ({entity.entity_type.value})")
    
    def add_entities(self, entities: list[Entity]) -> int:
        """Add multiple entities."""
        for entity in entities:
            self.add_entity(entity)
        logger.info(f"Added {len(entities)} entities to graph")
        return len(entities)
    
    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship (edge) to the graph.
        
        Args:
            relationship: Relationship to add
        """
        self._relationships[relationship.relationship_id] = relationship
        
        # Add edge with attributes
        self._graph.add_edge(
            str(relationship.source_entity_id),
            str(relationship.target_entity_id),
            relationship_id=str(relationship.relationship_id),
            relationship_type=relationship.relationship_type.value,
            properties=relationship.properties,
            weight=relationship.weight,
            source_document_id=str(relationship.source_document_id),
            source_page=relationship.source_page,
            confidence=relationship.confidence,
        )
        
        logger.debug(
            f"Added relationship: {relationship.relationship_type.value} "
            f"from {relationship.source_entity_id} to {relationship.target_entity_id}"
        )
    
    def add_relationships(self, relationships: list[Relationship]) -> int:
        """Add multiple relationships."""
        for rel in relationships:
            self.add_relationship(rel)
        logger.info(f"Added {len(relationships)} relationships to graph")
        return len(relationships)
    
    def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get entity by ID."""
        return self._entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self._entities.values() if e.entity_type == entity_type]
    
    def get_entities_by_document(self, document_id: UUID) -> list[Entity]:
        """Get all entities from a specific document."""
        return [
            e for e in self._entities.values()
            if e.source_document_id == document_id
        ]
    
    def get_entity_relationships(
        self,
        entity_id: UUID,
        direction: str = "both",
    ) -> list[Relationship]:
        """
        Get relationships for an entity.
        
        Args:
            entity_id: Entity ID
            direction: "in", "out", or "both"
            
        Returns:
            List of relationships
        """
        relationships: list[Relationship] = []
        entity_id_str = str(entity_id)
        
        if direction in ("out", "both"):
            for _, target, data in self._graph.out_edges(entity_id_str, data=True):
                rel_id = UUID(data.get("relationship_id", ""))
                if rel_id in self._relationships:
                    relationships.append(self._relationships[rel_id])
        
        if direction in ("in", "both"):
            for source, _, data in self._graph.in_edges(entity_id_str, data=True):
                rel_id = UUID(data.get("relationship_id", ""))
                if rel_id in self._relationships:
                    relationships.append(self._relationships[rel_id])
        
        return relationships
    
    def get_entity_subgraph(
        self,
        entity_id: UUID,
        max_hops: int = 2,
    ) -> SubGraph:
        """
        Get a subgraph centered on an entity.
        
        Args:
            entity_id: Center entity ID
            max_hops: Maximum number of hops from center
            
        Returns:
            SubGraph with connected entities and relationships
        """
        entity_id_str = str(entity_id)
        
        if entity_id_str not in self._graph:
            return SubGraph(entities=[], relationships=[], query=f"subgraph:{entity_id}")
        
        # BFS to find all nodes within max_hops
        visited: set[str] = set()
        current_level = {entity_id_str}
        
        for _ in range(max_hops + 1):
            visited.update(current_level)
            next_level: set[str] = set()
            
            for node in current_level:
                # Get neighbors (both directions)
                next_level.update(self._graph.successors(node))
                next_level.update(self._graph.predecessors(node))
            
            current_level = next_level - visited
        
        # Build subgraph
        entities = [
            self._entities[UUID(node_id)]
            for node_id in visited
            if UUID(node_id) in self._entities
        ]
        
        relationships = [
            rel for rel in self._relationships.values()
            if (
                str(rel.source_entity_id) in visited
                and str(rel.target_entity_id) in visited
            )
        ]
        
        return SubGraph(
            entities=entities,
            relationships=relationships,
            query=f"subgraph:{entity_id}:hops={max_hops}",
        )
    
    def find_entities_by_name(
        self,
        name_query: str,
        entity_type: EntityType | None = None,
    ) -> list[Entity]:
        """
        Find entities by name (case-insensitive substring match).
        
        Args:
            name_query: Name to search for
            entity_type: Optional type filter
            
        Returns:
            List of matching entities
        """
        query_lower = name_query.lower()
        results = []
        
        for entity in self._entities.values():
            if query_lower in entity.name.lower():
                if entity_type is None or entity.entity_type == entity_type:
                    results.append(entity)
        
        return results
    
    def find_path(
        self,
        source_id: UUID,
        target_id: UUID,
    ) -> list[Entity] | None:
        """
        Find shortest path between two entities.
        
        Returns:
            List of entities in path, or None if no path exists
        """
        try:
            path = nx.shortest_path(
                self._graph,
                str(source_id),
                str(target_id),
            )
            return [
                self._entities[UUID(node_id)]
                for node_id in path
                if UUID(node_id) in self._entities
            ]
        except nx.NetworkXNoPath:
            return None
    
    def get_stats(self) -> KnowledgeGraphStats:
        """Get graph statistics."""
        entities_by_type: dict[str, int] = {}
        for entity in self._entities.values():
            type_name = entity.entity_type.value
            entities_by_type[type_name] = entities_by_type.get(type_name, 0) + 1
        
        relationships_by_type: dict[str, int] = {}
        for rel in self._relationships.values():
            type_name = rel.relationship_type.value
            relationships_by_type[type_name] = relationships_by_type.get(type_name, 0) + 1
        
        # Count unique documents
        doc_ids = {e.source_document_id for e in self._entities.values()}
        
        return KnowledgeGraphStats(
            total_entities=len(self._entities),
            total_relationships=len(self._relationships),
            entities_by_type=entities_by_type,
            relationships_by_type=relationships_by_type,
            documents_indexed=len(doc_ids),
        )
    
    def save(self) -> None:
        """Save graph to disk."""
        if not self.persist_path:
            logger.warning("No persist_path set, skipping save")
            return
        
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "entities": [e.model_dump(mode="json") for e in self._entities.values()],
            "relationships": [r.model_dump(mode="json") for r in self._relationships.values()],
        }
        
        self.persist_path.write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"Saved graph to {self.persist_path}")
    
    def _load(self) -> None:
        """Load graph from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        try:
            data = json.loads(self.persist_path.read_text())
            
            for entity_data in data.get("entities", []):
                entity = Entity.model_validate(entity_data)
                self.add_entity(entity)
            
            for rel_data in data.get("relationships", []):
                rel = Relationship.model_validate(rel_data)
                self.add_relationship(rel)
            
            logger.info(
                f"Loaded graph from {self.persist_path}: "
                f"{len(self._entities)} entities, {len(self._relationships)} relationships"
            )
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
    
    def visualize(
        self,
        output_path: Path,
        height: str = "800px",
        width: str = "100%",
    ) -> Path:
        """
        Generate interactive HTML visualization of the graph.
        
        Args:
            output_path: Path for HTML output
            height: Height of visualization
            width: Width of visualization
            
        Returns:
            Path to generated HTML file
        """
        net = Network(height=height, width=width, directed=True, notebook=False)
        
        # Color map for entity types
        color_map = {
            EntityType.PERSON: "#4CAF50",
            EntityType.ORGANIZATION: "#2196F3",
            EntityType.MONETARY_AMOUNT: "#FFC107",
            EntityType.PERCENTAGE: "#FF9800",
            EntityType.DATE: "#9C27B0",
            EntityType.ROLE: "#00BCD4",
            EntityType.DOCUMENT: "#607D8B",
        }
        
        # Add nodes
        for entity in self._entities.values():
            color = color_map.get(entity.entity_type, "#9E9E9E")
            label = f"{entity.name}"
            if entity.value:
                label += f"\n({entity.value})"
            
            net.add_node(
                str(entity.entity_id),
                label=label,
                color=color,
                title=f"{entity.entity_type.value}: {entity.name}\nPage: {entity.source_page}",
            )
        
        # Add edges
        for rel in self._relationships.values():
            net.add_edge(
                str(rel.source_entity_id),
                str(rel.target_entity_id),
                label=rel.relationship_type.value,
                title=f"{rel.relationship_type.value}\nPage: {rel.source_page}",
            )
        
        # Configure physics
        net.set_options("""
        {
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -3000,
                    "springLength": 150
                }
            }
        }
        """)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(output_path))
        logger.info(f"Generated visualization at {output_path}")
        
        return output_path
    
    def reset(self) -> None:
        """Clear all data."""
        self._graph.clear()
        self._entities.clear()
        self._relationships.clear()
        logger.warning("Graph store reset")
