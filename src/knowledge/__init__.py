"""
Knowledge Layer - GraphRAG components.

Provides vector search, graph storage, and entity extraction
for building the Knowledge Graph from document chunks.
"""

from src.knowledge.schemas import (
    Entity,
    EntityType,
    Relationship,
    RelationType,
    VectorSearchResult,
    GraphSearchResult,
    SubGraph,
    KnowledgeGraphStats,
)
from src.knowledge.vector_store import VectorStore, VectorStoreError
from src.knowledge.graph_store import GraphStore, GraphStoreError
from src.knowledge.entity_extractor import (
    EntityExtractor,
    ExtractionError,
    extract_entities_from_document,
    normalize_monetary_amount,
    normalize_percentage,
    normalize_duration,
)

__all__ = [
    # Schemas
    "Entity",
    "EntityType",
    "Relationship",
    "RelationType",
    "VectorSearchResult",
    "GraphSearchResult",
    "SubGraph",
    "KnowledgeGraphStats",
    # Stores
    "VectorStore",
    "VectorStoreError",
    "GraphStore",
    "GraphStoreError",
    # Extractor
    "EntityExtractor",
    "ExtractionError",
    "extract_entities_from_document",
    # Utilities
    "normalize_monetary_amount",
    "normalize_percentage",
    "normalize_duration",
]
