"""
Pydantic Schemas for Knowledge Layer.

Type-safe models for entities, relationships, and graph structures.
These are the building blocks of our Knowledge Graph.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Types of entities we extract from documents."""

    PERSON = "person"
    ORGANIZATION = "organization"
    MONETARY_AMOUNT = "monetary_amount"
    PERCENTAGE = "percentage"
    DATE = "date"
    DURATION = "duration"
    CLAUSE = "clause"
    DOCUMENT = "document"
    ROLE = "role"
    LOCATION = "location"
    UNKNOWN = "unknown"


class RelationType(str, Enum):
    """Types of relationships between entities."""

    # Document relationships
    HAS_CLAUSE = "has_clause"
    REFERENCES = "references"
    SUPERCEDES = "supercedes"
    AMENDS = "amends"
    
    # Person relationships
    HAS_ROLE = "has_role"
    EMPLOYED_BY = "employed_by"
    REPORTS_TO = "reports_to"
    
    # Financial relationships
    HAS_COMPENSATION = "has_compensation"
    HAS_EQUITY = "has_equity"
    HAS_BONUS = "has_bonus"
    OWNS = "owns"
    
    # Temporal relationships
    EFFECTIVE_DATE = "effective_date"
    TERMINATION_DATE = "termination_date"
    DURATION = "duration"
    
    # Generic
    RELATED_TO = "related_to"
    CONTAINS = "contains"
    DEFINED_IN = "defined_in"


class Entity(BaseModel):
    """
    An entity extracted from a document.
    
    Entities are nodes in our Knowledge Graph.
    """

    entity_id: UUID = Field(default_factory=uuid4, description="Unique entity ID")
    entity_type: EntityType = Field(..., description="Type of entity")
    name: str = Field(..., description="Entity name/label")
    value: Any = Field(default=None, description="Entity value (for amounts, dates, etc)")
    normalized_value: Any = Field(
        default=None,
        description="Normalized value for comparison (e.g., all amounts in USD)",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional attributes",
    )
    
    # Source tracking (mandatory for citations)
    source_document_id: UUID = Field(..., description="Source document ID")
    source_chunk_id: UUID = Field(..., description="Source chunk ID")
    source_page: int = Field(..., ge=1, description="Page number")
    source_text: str = Field(..., description="Original text snippet")
    
    # Extraction metadata
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
    extraction_method: str = Field(default="llm", description="How entity was extracted")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def __hash__(self) -> int:
        """Make entity hashable for graph operations."""
        return hash(self.entity_id)
    
    def __eq__(self, other: object) -> bool:
        """Equality based on entity_id."""
        if isinstance(other, Entity):
            return self.entity_id == other.entity_id
        return False


class Relationship(BaseModel):
    """
    A relationship between two entities.
    
    Relationships are edges in our Knowledge Graph.
    """

    relationship_id: UUID = Field(default_factory=uuid4, description="Unique relationship ID")
    relationship_type: RelationType = Field(..., description="Type of relationship")
    source_entity_id: UUID = Field(..., description="Source entity ID (from)")
    target_entity_id: UUID = Field(..., description="Target entity ID (to)")
    
    # Relationship properties
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional properties (e.g., effective_date for HAS_ROLE)",
    )
    weight: float = Field(default=1.0, ge=0.0, description="Relationship weight/strength")
    
    # Source tracking
    source_document_id: UUID = Field(..., description="Document where relationship found")
    source_chunk_id: UUID = Field(..., description="Chunk where relationship found")
    source_page: int = Field(..., ge=1, description="Page number")
    source_text: str = Field(..., description="Original text showing relationship")
    
    # Metadata
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class KnowledgeGraphStats(BaseModel):
    """Statistics about the knowledge graph."""

    total_entities: int = Field(default=0)
    total_relationships: int = Field(default=0)
    entities_by_type: dict[str, int] = Field(default_factory=dict)
    relationships_by_type: dict[str, int] = Field(default_factory=dict)
    documents_indexed: int = Field(default=0)


class EntityExtractionPrompt(BaseModel):
    """
    Structured output format for LLM entity extraction.
    
    The LLM must output in this exact format.
    """

    entities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of extracted entities",
    )
    relationships: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of relationships between entities",
    )


class VectorSearchResult(BaseModel):
    """Result from vector similarity search."""

    chunk_id: UUID = Field(..., description="Matched chunk ID")
    document_id: UUID = Field(..., description="Source document ID")
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    page_number: int = Field(..., ge=1, description="Page number")
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphSearchResult(BaseModel):
    """Result from graph traversal/search."""

    entity: Entity = Field(..., description="Found entity")
    path_length: int = Field(default=0, ge=0, description="Hops from query entity")
    connected_entities: list[UUID] = Field(
        default_factory=list,
        description="IDs of directly connected entities",
    )
    relationships: list[Relationship] = Field(
        default_factory=list,
        description="Relationships involving this entity",
    )


class SubGraph(BaseModel):
    """A subgraph extracted for a specific query."""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    query: str = Field(..., description="Query that generated this subgraph")
    
    @property
    def entity_ids(self) -> set[UUID]:
        """Get all entity IDs in subgraph."""
        return {e.entity_id for e in self.entities}
