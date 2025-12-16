"""
Knowledge Layer - GraphRAG Components.

Vector store for semantic search + Graph store for entity relationships.
"""

from src.knowledge.vector_store import VectorStore, VectorStoreConfig
from src.knowledge.graph_store import GraphStore, GraphNode, GraphEdge
from src.knowledge.entity_extractor import EntityExtractor, ExtractedEntity
from src.knowledge.entity_resolver import EntityResolver, ResolvedEntity
from src.knowledge.cross_reference import CrossReferenceEngine, CrossReference, EntityProfile
from src.knowledge.timeline_builder import TimelineBuilder, Timeline, TimelineEvent
from src.knowledge.graph_visualizer import GraphVisualizer, visualize_knowledge_graph
from src.knowledge.schemas import (
    Entity,
    Relationship,
    EntityType,
    RelationshipType,
    KnowledgeGraphNode,
)

__all__ = [
    # Stores
    "VectorStore",
    "VectorStoreConfig",
    "GraphStore",
    "GraphNode",
    "GraphEdge",
    # Extraction
    "EntityExtractor",
    "ExtractedEntity",
    # Resolution
    "EntityResolver",
    "ResolvedEntity",
    # Cross-reference
    "CrossReferenceEngine",
    "CrossReference",
    "EntityProfile",
    # Timeline
    "TimelineBuilder",
    "Timeline",
    "TimelineEvent",
    # Visualization
    "GraphVisualizer",
    "visualize_knowledge_graph",
    # Schemas
    "Entity",
    "Relationship",
    "EntityType",
    "RelationshipType",
    "KnowledgeGraphNode",
]
