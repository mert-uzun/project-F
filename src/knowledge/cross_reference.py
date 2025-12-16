"""
Cross-Reference Engine - Find Entity Mentions Across Documents.

Answers: "Where else is [entity] mentioned?"
Combines graph queries and semantic search for comprehensive results.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.knowledge.schemas import Entity, EntityType, Relationship, RelationshipType
from src.knowledge.graph_store import GraphStore, GraphNode
from src.knowledge.vector_store import VectorStore, SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Schemas
# ============================================================================

class CrossReference(BaseModel):
    """A cross-reference result showing where an entity is mentioned."""
    
    entity: Entity
    document_id: UUID
    document_name: str
    page_number: int
    context: str = Field(description="Surrounding text for context")
    relationship_to_query: str = Field(
        default="exact_match",
        description="How this relates to the search query"
    )
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)


class EntityProfile(BaseModel):
    """Complete profile of an entity across all documents."""
    
    profile_id: UUID = Field(default_factory=uuid4)
    entity_value: str
    entity_type: EntityType
    
    # Occurrences
    total_mentions: int = 0
    documents: list[UUID] = Field(default_factory=list)
    document_names: list[str] = Field(default_factory=list)
    
    # Relationships
    roles: list[str] = Field(
        default_factory=list,
        description="Roles this entity plays (from relationships)"
    )
    related_amounts: list[str] = Field(
        default_factory=list,
        description="Monetary amounts associated with this entity"
    )
    related_organizations: list[str] = Field(
        default_factory=list,
        description="Organizations this entity is connected to"
    )
    related_people: list[str] = Field(
        default_factory=list,
        description="People this entity is connected to"
    )
    
    # Timeline
    dates: list[str] = Field(
        default_factory=list,
        description="Dates associated with this entity"
    )
    
    # All mentions with context
    mentions: list[CrossReference] = Field(default_factory=list)
    
    @property
    def document_count(self) -> int:
        return len(set(self.documents))
    
    def to_summary(self) -> str:
        """Generate text summary of profile."""
        parts = [
            f"**{self.entity_value}** ({self.entity_type.value})",
            f"Mentioned {self.total_mentions} times across {self.document_count} documents.",
        ]
        
        if self.roles:
            parts.append(f"Roles: {', '.join(self.roles)}")
        
        if self.related_amounts:
            parts.append(f"Amounts: {', '.join(self.related_amounts[:5])}")
        
        if self.related_organizations:
            parts.append(f"Organizations: {', '.join(self.related_organizations[:3])}")
        
        return "\n".join(parts)


class SearchQuery(BaseModel):
    """A cross-reference search query."""
    
    query: str = Field(..., description="Search query text")
    entity_types: list[EntityType] | None = Field(
        default=None,
        description="Filter by entity types"
    )
    document_ids: list[UUID] | None = Field(
        default=None,
        description="Filter by documents"
    )
    min_relevance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score"
    )
    max_results: int = Field(default=50, ge=1, le=200)
    include_semantic: bool = Field(
        default=True,
        description="Include semantic search results"
    )


# ============================================================================
# Cross-Reference Engine
# ============================================================================

class CrossReferenceEngine:
    """
    Find and analyze cross-references across documents.
    
    Combines:
    1. Graph queries - Find exact entity matches and related entities
    2. Semantic search - Find similar mentions via embeddings
    
    Usage:
        engine = CrossReferenceEngine(vector_store, graph_store)
        results = engine.search("John Doe")
        profile = engine.get_entity_profile("John Doe")
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
    ) -> None:
        """
        Initialize the Cross-Reference Engine.
        
        Args:
            vector_store: Vector store for semantic search
            graph_store: Graph store for entity queries
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
    
    def search(
        self,
        query: str | SearchQuery,
        entity_types: list[EntityType] | None = None,
        document_ids: list[UUID] | None = None,
        max_results: int = 50,
    ) -> list[CrossReference]:
        """
        Search for entity mentions.
        
        Args:
            query: Search query (string or SearchQuery)
            entity_types: Optional filter by entity types
            document_ids: Optional filter by documents
            max_results: Maximum results to return
            
        Returns:
            List of CrossReference results sorted by relevance
        """
        if isinstance(query, str):
            query = SearchQuery(
                query=query,
                entity_types=entity_types,
                document_ids=document_ids,
                max_results=max_results,
            )
        
        results: list[CrossReference] = []
        seen_entities: set[UUID] = set()
        
        # 1. Exact graph search (fuzzy matching)
        graph_results = self.graph_store.find_entities_by_value(
            query.query,
            fuzzy=True,
        )
        
        for node in graph_results:
            entity = node.entity
            
            # Apply filters
            if query.entity_types and entity.entity_type not in query.entity_types:
                continue
            if query.document_ids and entity.source_document_id not in query.document_ids:
                continue
            
            if entity.entity_id in seen_entities:
                continue
            seen_entities.add(entity.entity_id)
            
            results.append(CrossReference(
                entity=entity,
                document_id=entity.source_document_id,
                document_name=f"Document {str(entity.source_document_id)[:8]}",
                page_number=entity.source_page,
                context=entity.source_text[:200] if entity.source_text else "",
                relationship_to_query="exact_match" if query.query.lower() in entity.value.lower() else "partial_match",
                relevance_score=1.0 if query.query.lower() == entity.value.lower() else 0.9,
            ))
        
        # 2. Semantic search (if enabled)
        if query.include_semantic and len(results) < query.max_results:
            try:
                # Note: VectorStore.search doesn't support multiple document_ids,
                # so we search without filter and apply document filter manually
                semantic_results = self.vector_store.search(
                    query=query.query,
                    top_k=query.max_results - len(results),
                )
                
                for search_result in semantic_results:
                    # Skip results not in requested documents
                    if query.document_ids:
                        result_doc_id = UUID(search_result.document_id)
                        if result_doc_id not in query.document_ids:
                            continue
                    
                    # Extract entities from the chunk
                    chunk_entities = self._get_entities_from_chunk(
                        search_result.chunk_id
                    )
                    
                    for entity in chunk_entities:
                        if entity.entity_id in seen_entities:
                            continue
                        
                        if query.entity_types and entity.entity_type not in query.entity_types:
                            continue
                        
                        seen_entities.add(entity.entity_id)
                        
                        results.append(CrossReference(
                            entity=entity,
                            document_id=search_result.document_id,
                            document_name=f"Document {str(search_result.document_id)[:8]}",
                            page_number=search_result.metadata.get("page_number", 1),
                            context=search_result.content[:200],
                            relationship_to_query="semantic_match",
                            relevance_score=search_result.similarity,
                        ))
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")
        
        # Sort by relevance
        results.sort(key=lambda r: -r.relevance_score)
        
        return results[:query.max_results]
    
    def get_entity_profile(
        self,
        entity_value: str,
        entity_type: EntityType | None = None,
    ) -> EntityProfile:
        """
        Build a comprehensive profile of an entity.
        
        Args:
            entity_value: Entity value to profile
            entity_type: Optional type filter
            
        Returns:
            EntityProfile with all information about the entity
        """
        # Find all matching entities
        matching_nodes = self.graph_store.find_entities_by_value(
            entity_value,
            fuzzy=True,
        )
        
        if entity_type:
            matching_nodes = [n for n in matching_nodes if n.entity.entity_type == entity_type]
        
        if not matching_nodes:
            # Return empty profile
            return EntityProfile(
                entity_value=entity_value,
                entity_type=entity_type or EntityType.OTHER,
            )
        
        # Use the first entity's type if not specified
        primary_entity = matching_nodes[0].entity
        inferred_type = entity_type or primary_entity.entity_type
        
        # Collect all mentions
        mentions: list[CrossReference] = []
        documents: list[UUID] = []
        
        for node in matching_nodes:
            entity = node.entity
            documents.append(entity.source_document_id)
            
            mentions.append(CrossReference(
                entity=entity,
                document_id=entity.source_document_id,
                document_name=f"Document {str(entity.source_document_id)[:8]}",
                page_number=entity.source_page,
                context=entity.source_text[:200] if entity.source_text else "",
                relationship_to_query="exact_match",
                relevance_score=1.0,
            ))
        
        # Get relationships to understand roles
        roles: list[str] = []
        related_amounts: list[str] = []
        related_orgs: list[str] = []
        related_people: list[str] = []
        dates: list[str] = []
        
        for node in matching_nodes:
            entity_id = node.entity.entity_id
            
            # Get neighborhood
            neighborhood = self.graph_store.get_entity_neighborhood(
                entity_id,
                hops=1,
                max_nodes=20,
            )
            
            for edge in neighborhood.edges:
                rel = edge.relationship
                
                # Get target entity
                target_node = self.graph_store.get_entity(rel.target_entity_id)
                if not target_node:
                    continue
                
                target = target_node.entity
                
                # Categorize by relationship type
                if rel.relationship_type in (RelationshipType.HAS_ROLE, RelationshipType.EMPLOYED_BY):
                    if rel.relationship_type == RelationshipType.HAS_ROLE:
                        roles.append(target.value)
                    else:
                        related_orgs.append(target.value)
                
                elif rel.relationship_type == RelationshipType.HAS_SALARY:
                    related_amounts.append(target.value)
                
                elif rel.relationship_type == RelationshipType.HAS_EQUITY:
                    related_amounts.append(f"{target.value} equity")
                
                elif target.entity_type == EntityType.ORGANIZATION:
                    related_orgs.append(target.value)
                
                elif target.entity_type == EntityType.PERSON:
                    related_people.append(target.value)
                
                elif target.entity_type == EntityType.DATE:
                    dates.append(target.value)
        
        # Deduplicate
        profile = EntityProfile(
            entity_value=entity_value,
            entity_type=inferred_type,
            total_mentions=len(mentions),
            documents=list(set(documents)),
            document_names=[f"Document {str(d)[:8]}" for d in set(documents)],
            roles=list(set(roles)),
            related_amounts=list(set(related_amounts)),
            related_organizations=list(set(related_orgs)),
            related_people=list(set(related_people)),
            dates=list(set(dates)),
            mentions=mentions,
        )
        
        logger.info(
            f"Built profile for '{entity_value}': "
            f"{profile.total_mentions} mentions in {profile.document_count} docs"
        )
        
        return profile
    
    def find_co_occurring_entities(
        self,
        entity_value: str,
        max_results: int = 20,
    ) -> list[tuple[Entity, int]]:
        """
        Find entities that frequently appear with a given entity.
        
        Args:
            entity_value: Entity to find co-occurrences for
            max_results: Maximum results
            
        Returns:
            List of (entity, count) tuples sorted by count
        """
        # Find the entity
        matching_nodes = self.graph_store.find_entities_by_value(
            entity_value,
            fuzzy=True,
        )
        
        if not matching_nodes:
            return []
        
        # Collect documents where this entity appears
        entity_docs: set[UUID] = set()
        for node in matching_nodes:
            entity_docs.add(node.entity.source_document_id)
        
        # Find all entities in those documents
        co_occur_counts: dict[str, tuple[Entity, int]] = {}
        
        for doc_id in entity_docs:
            doc_entities = self.graph_store.get_all_entities(document_id=doc_id)
            
            for node in doc_entities:
                entity = node.entity
                
                # Skip the query entity itself
                if entity_value.lower() in entity.value.lower():
                    continue
                
                # Use normalized value as key
                key = f"{entity.entity_type.value}:{entity.value.lower()}"
                
                if key not in co_occur_counts:
                    co_occur_counts[key] = (entity, 0)
                
                current = co_occur_counts[key]
                co_occur_counts[key] = (current[0], current[1] + 1)
        
        # Sort by count
        sorted_results = sorted(
            co_occur_counts.values(),
            key=lambda x: -x[1]
        )
        
        return sorted_results[:max_results]
    
    def _get_entities_from_chunk(self, chunk_id: UUID) -> list[Entity]:
        """Get all entities extracted from a specific chunk."""
        results: list[Entity] = []
        
        for node in self.graph_store.get_all_entities():
            entity = node.entity
            if entity.source_chunk_id == chunk_id:
                results.append(entity)
        
        return results
