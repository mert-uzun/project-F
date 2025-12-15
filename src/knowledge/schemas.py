"""
Pydantic Schemas for Knowledge Layer.

Defines entities, relationships, and graph structures for the Knowledge Graph.
These models form the foundation of our GraphRAG approach.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """
    Types of entities we extract from documents.
    
    These are the "nodes" in our Knowledge Graph.
    """
    
    # People & Organizations
    PERSON = "person"
    ORGANIZATION = "organization"
    ROLE = "role"  # CEO, CFO, Director, etc.
    
    # Financial
    MONETARY_AMOUNT = "monetary_amount"
    PERCENTAGE = "percentage"
    EQUITY = "equity"
    SALARY = "salary"
    
    # Legal / Contractual
    CLAUSE = "clause"
    TERM = "term"
    OBLIGATION = "obligation"
    RIGHT = "right"
    
    # Temporal
    DATE = "date"
    DURATION = "duration"
    DEADLINE = "deadline"
    
    # Documents
    DOCUMENT = "document"
    SECTION = "section"
    TABLE = "table"
    
    # Generic
    LOCATION = "location"
    EVENT = "event"
    OTHER = "other"


class RelationshipType(str, Enum):
    """
    Types of relationships between entities.
    
    These are the "edges" in our Knowledge Graph.
    """
    
    # Document structure
    HAS_SECTION = "has_section"
    HAS_CLAUSE = "has_clause"
    HAS_TABLE = "has_table"
    CONTAINS = "contains"
    REFERENCES = "references"
    
    # People/Org relationships
    EMPLOYS = "employs"
    EMPLOYED_BY = "employed_by"
    HAS_ROLE = "has_role"
    REPORTS_TO = "reports_to"
    OWNS = "owns"
    OWNED_BY = "owned_by"
    
    # Financial relationships
    HAS_SALARY = "has_salary"
    HAS_EQUITY = "has_equity"
    HAS_AMOUNT = "has_amount"
    PAYS = "pays"
    RECEIVES = "receives"
    
    # Temporal relationships
    EFFECTIVE_DATE = "effective_date"
    TERMINATION_DATE = "termination_date"
    HAS_DURATION = "has_duration"
    
    # Contractual relationships
    GRANTS = "grants"
    RESTRICTS = "restricts"
    OBLIGATES = "obligates"
    ENTITLES = "entitles"
    
    # Comparison (for conflict detection)
    SAME_AS = "same_as"
    CONFLICTS_WITH = "conflicts_with"
    SUPERSEDES = "supersedes"


class Entity(BaseModel):
    """
    An entity extracted from a document.
    
    Entities are the nodes in our Knowledge Graph.
    Each entity has a type, value, and source citation.
    """
    
    entity_id: UUID = Field(default_factory=uuid4, description="Unique entity ID")
    entity_type: EntityType = Field(..., description="Type of entity")
    value: str = Field(..., description="The entity value as extracted")
    normalized_value: str | None = Field(
        default=None, 
        description="Normalized form for comparison (e.g., 'CEO' -> 'chief executive officer')"
    )
    
    # Source citation (critical for trust)
    source_document_id: UUID = Field(..., description="Document this came from")
    source_chunk_id: UUID = Field(..., description="Chunk this was extracted from")
    source_page: int = Field(..., ge=1, description="Page number for citation")
    source_text: str = Field(..., description="Original text context")
    
    # Extraction metadata
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
    extraction_method: str = Field(default="llm", description="How it was extracted")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Optional structured data
    attributes: dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional structured attributes"
    )

    class Config:
        json_encoders = {UUID: str}


class Relationship(BaseModel):
    """
    A relationship between two entities.
    
    Relationships are the edges in our Knowledge Graph.
    They connect entities and carry semantic meaning.
    """
    
    relationship_id: UUID = Field(default_factory=uuid4, description="Unique relationship ID")
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    
    # Source and target entities
    source_entity_id: UUID = Field(..., description="Source entity ID")
    target_entity_id: UUID = Field(..., description="Target entity ID")
    
    # Relationship properties
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional relationship properties"
    )
    
    # Source citation
    source_document_id: UUID = Field(..., description="Document this came from")
    source_chunk_id: UUID = Field(..., description="Chunk this was extracted from")
    source_page: int = Field(..., ge=1, description="Page number")
    source_text: str = Field(..., description="Text context for this relationship")
    
    # Extraction metadata
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {UUID: str}


class KnowledgeGraphNode(BaseModel):
    """
    A node in the Knowledge Graph with its relationships.
    
    Used for graph queries and traversal.
    """
    
    entity: Entity = Field(..., description="The entity at this node")
    
    # Outgoing relationships
    outgoing_relationships: list[Relationship] = Field(
        default_factory=list,
        description="Relationships where this entity is the source"
    )
    
    # Incoming relationships
    incoming_relationships: list[Relationship] = Field(
        default_factory=list,
        description="Relationships where this entity is the target"
    )
    
    @property
    def degree(self) -> int:
        """Total number of relationships."""
        return len(self.outgoing_relationships) + len(self.incoming_relationships)


class ConflictCandidate(BaseModel):
    """
    A potential conflict detected between two entities.
    
    This is used by the Comparator Agent to flag inconsistencies.
    """
    
    conflict_id: UUID = Field(default_factory=uuid4)
    conflict_type: str = Field(..., description="Type of conflict (value_mismatch, date_conflict, etc.)")
    
    # The conflicting entities
    entity_a: Entity = Field(..., description="First entity")
    entity_b: Entity = Field(..., description="Second entity (conflicting)")
    
    # What's the conflict?
    description: str = Field(..., description="Human-readable conflict description")
    severity: str = Field(default="medium", description="low, medium, high, critical")
    
    # Context for verification
    context_a: str = Field(..., description="Surrounding context for entity A")
    context_b: str = Field(..., description="Surrounding context for entity B")
    
    # Verification status
    verified: bool = Field(default=False, description="Has the Judge verified this?")
    is_true_positive: bool | None = Field(default=None, description="Is this a real conflict?")
    verification_notes: str | None = Field(default=None, description="Judge's notes")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExtractionResult(BaseModel):
    """
    Result of entity extraction from a document chunk.
    """
    
    chunk_id: UUID = Field(..., description="Source chunk ID")
    document_id: UUID = Field(..., description="Source document ID")
    
    entities: list[Entity] = Field(default_factory=list, description="Extracted entities")
    relationships: list[Relationship] = Field(default_factory=list, description="Extracted relationships")
    
    # Metadata
    extraction_time_seconds: float = Field(default=0.0)
    model_used: str = Field(default="unknown")
    errors: list[str] = Field(default_factory=list)
    
    @property
    def entity_count(self) -> int:
        return len(self.entities)
    
    @property
    def relationship_count(self) -> int:
        return len(self.relationships)
