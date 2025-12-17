"""
Graph Store - NetworkX Integration.

Stores entities and relationships as a Knowledge Graph.
This is the "Graph" part of GraphRAG - enables relationship-aware retrieval.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import networkx as nx
from pyvis.network import Network

from src.knowledge.schemas import Entity, EntityType, Relationship, RelationshipType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphStoreError(Exception):
    """Raised when graph store operations fail."""
    pass


@dataclass
class GraphNode:
    """A node in the knowledge graph (wrapper around Entity)."""

    entity: Entity

    @property
    def id(self) -> str:
        return str(self.entity.entity_id)

    @property
    def label(self) -> str:
        return self.entity.value

    @property
    def type(self) -> EntityType:
        return self.entity.entity_type

    def to_dict(self) -> dict[str, Any]:
        return self.entity.model_dump(mode="json")


@dataclass
class GraphEdge:
    """An edge in the knowledge graph (wrapper around Relationship)."""

    relationship: Relationship

    @property
    def source(self) -> str:
        return str(self.relationship.source_entity_id)

    @property
    def target(self) -> str:
        return str(self.relationship.target_entity_id)

    @property
    def type(self) -> RelationshipType:
        return self.relationship.relationship_type

    def to_dict(self) -> dict[str, Any]:
        return self.relationship.model_dump(mode="json")


@dataclass
class SubGraph:
    """A subgraph extracted from the main graph."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


class GraphStore:
    """
    NetworkX-based graph store for entity-relationship storage.

    Provides:
    - Entity and relationship storage
    - Graph traversal and queries
    - Subgraph extraction for RAG context
    - Visualization with pyvis
    - Persistence to disk

    Usage:
        store = GraphStore()
        store.add_entity(entity)
        store.add_relationship(relationship)
        subgraph = store.get_entity_neighborhood(entity_id, hops=2)
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        """
        Initialize the graph store.

        Args:
            persist_path: Optional path to persist the graph
        """
        self.persist_path = persist_path
        self._graph: nx.DiGraph = nx.DiGraph()

        # Load existing graph if path exists
        if persist_path and persist_path.exists():
            self._load()

    @property
    def graph(self) -> nx.DiGraph:
        """Get the underlying NetworkX graph."""
        return self._graph

    def add_entity(self, entity: Entity) -> GraphNode:
        """
        Add an entity as a node in the graph.

        Args:
            entity: The entity to add

        Returns:
            GraphNode wrapper
        """
        node = GraphNode(entity=entity)

        self._graph.add_node(
            node.id,
            entity_type=entity.entity_type.value,
            value=entity.value,
            normalized_value=entity.normalized_value,
            source_document_id=str(entity.source_document_id),
            source_page=entity.source_page,
            confidence=entity.confidence,
            data=node.to_dict(),
        )

        logger.debug(f"Added entity node: {entity.value} ({entity.entity_type.value})")
        return node

    def add_relationship(self, relationship: Relationship) -> GraphEdge:
        """
        Add a relationship as an edge in the graph.

        Args:
            relationship: The relationship to add

        Returns:
            GraphEdge wrapper
        """
        edge = GraphEdge(relationship=relationship)

        # Ensure both nodes exist
        if not self._graph.has_node(edge.source):
            logger.warning(f"Source node {edge.source} not found, creating placeholder")
            self._graph.add_node(edge.source, placeholder=True)

        if not self._graph.has_node(edge.target):
            logger.warning(f"Target node {edge.target} not found, creating placeholder")
            self._graph.add_node(edge.target, placeholder=True)

        self._graph.add_edge(
            edge.source,
            edge.target,
            relationship_type=relationship.relationship_type.value,
            source_document_id=str(relationship.source_document_id),
            confidence=relationship.confidence,
            properties=relationship.properties,
            data=edge.to_dict(),
        )

        logger.debug(
            f"Added relationship: {edge.source} --[{relationship.relationship_type.value}]--> {edge.target}"
        )
        return edge

    def add_entities(self, entities: list[Entity]) -> list[GraphNode]:
        """Add multiple entities."""
        return [self.add_entity(e) for e in entities]

    def add_relationships(self, relationships: list[Relationship]) -> list[GraphEdge]:
        """Add multiple relationships."""
        return [self.add_relationship(r) for r in relationships]

    def get_entity(self, entity_id: UUID) -> GraphNode | None:
        """
        Get an entity by ID.

        Args:
            entity_id: The entity UUID

        Returns:
            GraphNode or None if not found
        """
        node_id = str(entity_id)

        if not self._graph.has_node(node_id):
            return None

        node_data = self._graph.nodes[node_id]

        if node_data.get("placeholder"):
            return None

        # Reconstruct entity from stored data
        entity_data = node_data.get("data", {})
        if entity_data:
            entity = Entity.model_validate(entity_data)
            return GraphNode(entity=entity)

        return None

    def get_entity_neighborhood(
        self,
        entity_id: UUID,
        hops: int = 2,
        max_nodes: int = 50,
    ) -> SubGraph:
        """
        Get a subgraph around an entity (for RAG context).

        Args:
            entity_id: Center entity
            hops: Number of relationship hops to include
            max_nodes: Maximum nodes to return

        Returns:
            SubGraph with connected entities and relationships
        """
        node_id = str(entity_id)

        if not self._graph.has_node(node_id):
            return SubGraph()

        # Get nodes within N hops using BFS
        visited_nodes: set[str] = set()
        current_layer = {node_id}

        for _ in range(hops):
            next_layer: set[str] = set()
            for node in current_layer:
                if len(visited_nodes) >= max_nodes:
                    break
                visited_nodes.add(node)
                # Add neighbors
                next_layer.update(self._graph.predecessors(node))
                next_layer.update(self._graph.successors(node))
            current_layer = next_layer - visited_nodes

        visited_nodes.update(current_layer)

        # Build subgraph
        nodes: list[GraphNode] = []
        for nid in visited_nodes:
            node_data = self._graph.nodes[nid]
            if not node_data.get("placeholder") and "data" in node_data:
                entity = Entity.model_validate(node_data["data"])
                nodes.append(GraphNode(entity=entity))

        # Get edges within the subgraph
        edges: list[GraphEdge] = []
        for source, target, edge_data in self._graph.edges(data=True):
            if source in visited_nodes and target in visited_nodes:
                if "data" in edge_data:
                    rel = Relationship.model_validate(edge_data["data"])
                    edges.append(GraphEdge(relationship=rel))

        logger.debug(f"Extracted subgraph: {len(nodes)} nodes, {len(edges)} edges")
        return SubGraph(nodes=nodes, edges=edges)

    def find_entities_by_type(
        self,
        entity_type: EntityType,
        document_id: UUID | None = None,
    ) -> list[GraphNode]:
        """
        Find all entities of a given type.

        Args:
            entity_type: Type of entities to find
            document_id: Optional filter by document

        Returns:
            List of matching GraphNodes
        """
        results: list[GraphNode] = []

        for node_id in self._graph.nodes():
            node_data = self._graph.nodes[node_id]

            if node_data.get("placeholder"):
                continue

            if node_data.get("entity_type") != entity_type.value:
                continue

            if document_id and node_data.get("source_document_id") != str(document_id):
                continue

            if "data" in node_data:
                entity = Entity.model_validate(node_data["data"])
                results.append(GraphNode(entity=entity))

        return results

    def find_entities_by_value(
        self,
        value: str,
        fuzzy: bool = False,
    ) -> list[GraphNode]:
        """
        Find entities by value.

        Args:
            value: Value to search for
            fuzzy: Whether to do substring matching

        Returns:
            List of matching GraphNodes
        """
        results: list[GraphNode] = []
        value_lower = value.lower()

        for node_id in self._graph.nodes():
            node_data = self._graph.nodes[node_id]

            if node_data.get("placeholder"):
                continue

            node_value = node_data.get("value", "").lower()
            normalized = (node_data.get("normalized_value") or "").lower()

            match = False
            if fuzzy:
                match = value_lower in node_value or value_lower in normalized
            else:
                match = value_lower == node_value or value_lower == normalized

            if match and "data" in node_data:
                entity = Entity.model_validate(node_data["data"])
                results.append(GraphNode(entity=entity))

        return results

    def get_related_entities(
        self,
        entity_id: UUID,
        relationship_type: RelationshipType | None = None,
        direction: str = "both",  # "outgoing", "incoming", "both"
    ) -> list[tuple[GraphNode, GraphEdge]]:
        """
        Get entities related to a given entity.

        Args:
            entity_id: Source entity
            relationship_type: Optional filter by relationship type
            direction: Direction of relationships

        Returns:
            List of (related_entity, relationship) tuples
        """
        node_id = str(entity_id)
        results: list[tuple[GraphNode, GraphEdge]] = []

        if not self._graph.has_node(node_id):
            return results

        # Get outgoing relationships
        if direction in ("outgoing", "both"):
            for _, target, edge_data in self._graph.out_edges(node_id, data=True):
                if relationship_type and edge_data.get("relationship_type") != relationship_type.value:
                    continue

                target_data = self._graph.nodes[target]
                if target_data.get("placeholder") or "data" not in target_data:
                    continue

                entity = Entity.model_validate(target_data["data"])
                rel = Relationship.model_validate(edge_data["data"])
                results.append((GraphNode(entity=entity), GraphEdge(relationship=rel)))

        # Get incoming relationships
        if direction in ("incoming", "both"):
            for source, _, edge_data in self._graph.in_edges(node_id, data=True):
                if relationship_type and edge_data.get("relationship_type") != relationship_type.value:
                    continue

                source_data = self._graph.nodes[source]
                if source_data.get("placeholder") or "data" not in source_data:
                    continue

                entity = Entity.model_validate(source_data["data"])
                rel = Relationship.model_validate(edge_data["data"])
                results.append((GraphNode(entity=entity), GraphEdge(relationship=rel)))

        return results

    def node_count(self) -> int:
        """Get total number of nodes."""
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        """Get total number of edges."""
        return self._graph.number_of_edges()

    def save(self, path: Path | None = None) -> None:
        """
        Save the graph to disk.

        Args:
            path: Path to save to (uses persist_path if not specified)
        """
        save_path = path or self.persist_path
        if not save_path:
            raise GraphStoreError("No save path specified")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        data = nx.node_link_data(self._graph)

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved graph to {save_path} ({self.node_count()} nodes, {self.edge_count()} edges)")

    def _load(self) -> None:
        """Load the graph from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path) as f:
                data = json.load(f)

            self._graph = nx.node_link_graph(data, directed=True)
            logger.info(
                f"Loaded graph from {self.persist_path} "
                f"({self.node_count()} nodes, {self.edge_count()} edges)"
            )
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            self._graph = nx.DiGraph()

    def visualize(
        self,
        output_path: Path,
        height: str = "800px",
        width: str = "100%",
        show_labels: bool = True,
    ) -> None:
        """
        Generate an interactive HTML visualization of the graph.

        Args:
            output_path: Path to save the HTML file
            height: Height of the visualization
            width: Width of the visualization
            show_labels: Whether to show node labels
        """
        net = Network(
            height=height,
            width=width,
            directed=True,
            notebook=False,
            select_menu=True,
            filter_menu=True,
        )

        # Color scheme for entity types
        colors = {
            EntityType.PERSON.value: "#4CAF50",
            EntityType.ORGANIZATION.value: "#2196F3",
            EntityType.MONETARY_AMOUNT.value: "#FFC107",
            EntityType.PERCENTAGE.value: "#FF9800",
            EntityType.DATE.value: "#9C27B0",
            EntityType.CLAUSE.value: "#607D8B",
            EntityType.DOCUMENT.value: "#795548",
        }
        default_color = "#9E9E9E"

        # Add nodes
        for node_id in self._graph.nodes():
            node_data = self._graph.nodes[node_id]
            entity_type = node_data.get("entity_type", "unknown")
            value = node_data.get("value", node_id)

            net.add_node(
                node_id,
                label=value if show_labels else "",
                title=f"{entity_type}: {value}",
                color=colors.get(entity_type, default_color),
            )

        # Add edges
        for source, target, edge_data in self._graph.edges(data=True):
            rel_type = edge_data.get("relationship_type", "related")
            net.add_edge(source, target, title=rel_type, label=rel_type)

        # Configure physics
        net.set_options("""
        {
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -30000,
                    "springLength": 200
                }
            }
        }
        """)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(output_path))
        logger.info(f"Saved graph visualization to {output_path}")

    def merge_entities(
        self,
        keep_id: UUID,
        merge_id: UUID,
    ) -> None:
        """
        Merge two entities, keeping one and redirecting relationships.

        All relationships pointing to/from merge_id are redirected to keep_id.
        The merge_id node is then removed.

        Args:
            keep_id: Entity to keep
            merge_id: Entity to merge into keep_id
        """
        keep_str = str(keep_id)
        merge_str = str(merge_id)

        if keep_str not in self._graph:
            logger.warning(f"Keep entity {keep_id} not in graph")
            return

        if merge_str not in self._graph:
            logger.warning(f"Merge entity {merge_id} not in graph")
            return

        # Get all edges involving merge_id
        in_edges = list(self._graph.in_edges(merge_str, data=True))
        out_edges = list(self._graph.out_edges(merge_str, data=True))

        # Redirect incoming edges
        for source, _, data in in_edges:
            if source != keep_str:  # Avoid self-loops
                self._graph.add_edge(source, keep_str, **data)

        # Redirect outgoing edges
        for _, target, data in out_edges:
            if target != keep_str:  # Avoid self-loops
                self._graph.add_edge(keep_str, target, **data)

        # Merge attributes (keep canonical but add aliases)
        keep_data = self._graph.nodes[keep_str]
        merge_data = self._graph.nodes[merge_str]

        # Add merged value as alias
        if "aliases" not in keep_data:
            keep_data["aliases"] = []
        keep_data["aliases"].append(merge_data.get("value", ""))

        # Remove merged node
        self._graph.remove_node(merge_str)

        logger.debug(f"Merged entity {merge_id} into {keep_id}")

    def get_all_entities(
        self,
        document_id: UUID | None = None,
    ) -> list["GraphNode"]:
        """
        Get all entities in the graph.

        Args:
            document_id: Optional filter by source document

        Returns:
            List of all GraphNodes
        """
        nodes: list[GraphNode] = []

        for node_id, node_data in self._graph.nodes(data=True):
            if "entity" not in node_data:
                continue

            entity = node_data["entity"]

            if document_id is not None:
                if entity.source_document_id != document_id:
                    continue

            nodes.append(GraphNode(entity=entity))

        return nodes

    def get_entities_by_document(
        self,
        document_ids: list[UUID],
    ) -> dict[UUID, list["GraphNode"]]:
        """
        Get entities grouped by document.

        Args:
            document_ids: List of document IDs

        Returns:
            Dict mapping document_id to list of entities
        """
        result: dict[UUID, list[GraphNode]] = {
            doc_id: [] for doc_id in document_ids
        }

        for node_id, node_data in self._graph.nodes(data=True):
            if "entity" not in node_data:
                continue

            entity = node_data["entity"]

            if entity.source_document_id in result:
                result[entity.source_document_id].append(
                    GraphNode(entity=entity)
                )

        return result

    def clear(self) -> None:
        """Clear all nodes and edges."""
        self._graph.clear()
        logger.info("Cleared graph store")

