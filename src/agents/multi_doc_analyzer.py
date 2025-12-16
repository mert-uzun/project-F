"""
Multi-Document Analyzer - Cross-Document Intelligence.

Analyzes conflicts and patterns across N documents (not just pairs).
Builds unified entity tracking and finds multi-way conflicts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.agents.schemas import (
    Conflict,
    ConflictEvidence,
    ConflictSeverity,
    ConflictStatus,
    ConflictType,
    SourceCitation,
)
from src.knowledge.schemas import Entity, EntityType
from src.knowledge.graph_store import GraphStore
from src.knowledge.vector_store import VectorStore
from src.knowledge.entity_resolver import EntityResolver, ResolvedEntity
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Schemas
# ============================================================================

class DocumentSet(BaseModel):
    """A set of documents for multi-document analysis."""
    
    document_ids: list[UUID] = Field(..., description="List of document UUIDs")
    document_names: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of document ID (as string) to name"
    )
    
    def get_name(self, doc_id: UUID) -> str:
        """Get document name by ID."""
        return self.document_names.get(str(doc_id), f"Document {str(doc_id)[:8]}")
    
    @property
    def count(self) -> int:
        return len(self.document_ids)


class EntityOccurrence(BaseModel):
    """An occurrence of an entity in a document."""
    
    document_id: UUID
    document_name: str
    value: str
    normalized_value: str | None = None
    page_number: int
    chunk_id: UUID
    confidence: float = 1.0


class EntityVariation(BaseModel):
    """Same entity type with different values across documents."""
    
    variation_id: UUID = Field(default_factory=uuid4)
    entity_type: EntityType
    canonical_value: str = Field(description="Most common or first seen value")
    occurrences: list[EntityOccurrence] = Field(default_factory=list)
    
    @property
    def is_conflict(self) -> bool:
        """True if values differ across documents."""
        unique_values = set(self._normalize(o.value) for o in self.occurrences)
        return len(unique_values) > 1
    
    @property
    def unique_values(self) -> list[str]:
        """Get unique values sorted by frequency."""
        value_counts: dict[str, int] = {}
        for o in self.occurrences:
            value_counts[o.value] = value_counts.get(o.value, 0) + 1
        return sorted(value_counts.keys(), key=lambda v: -value_counts[v])
    
    @property
    def document_count(self) -> int:
        """Number of documents this entity appears in."""
        return len(set(o.document_id for o in self.occurrences))
    
    @property
    def conflict_severity(self) -> ConflictSeverity | None:
        """Compute severity based on variation magnitude."""
        if not self.is_conflict:
            return None
        
        # Simple heuristic: more documents disagreeing = higher severity
        docs_per_value: dict[str, set[UUID]] = {}
        for o in self.occurrences:
            norm_val = self._normalize(o.value)
            if norm_val not in docs_per_value:
                docs_per_value[norm_val] = set()
            docs_per_value[norm_val].add(o.document_id)
        
        # If all docs disagree, it's critical
        if len(docs_per_value) == self.document_count:
            return ConflictSeverity.CRITICAL
        
        # If majority agrees but some don't
        max_agreement = max(len(docs) for docs in docs_per_value.values())
        agreement_ratio = max_agreement / self.document_count
        
        if agreement_ratio < 0.5:
            return ConflictSeverity.HIGH
        elif agreement_ratio < 0.75:
            return ConflictSeverity.MEDIUM
        else:
            return ConflictSeverity.LOW
    
    def _normalize(self, value: str) -> str:
        """Normalize value for comparison."""
        return value.lower().strip()


class MultiDocConflict(BaseModel):
    """A conflict involving multiple documents."""
    
    conflict_id: UUID = Field(default_factory=uuid4)
    entity_type: EntityType
    conflict_type: ConflictType
    severity: ConflictSeverity
    status: ConflictStatus = ConflictStatus.DETECTED
    
    title: str
    description: str
    
    # Evidence from each document
    variations: list[EntityVariation] = Field(default_factory=list)
    
    # Summary
    unique_values: list[str] = Field(default_factory=list)
    document_count: int = 0
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MultiDocReport(BaseModel):
    """Complete report from multi-document analysis."""
    
    report_id: UUID = Field(default_factory=uuid4)
    document_set: DocumentSet
    
    # Statistics
    total_entities: int = 0
    total_relationships: int = 0
    total_resolved_entities: int = 0
    
    # Findings
    conflicts: list[MultiDocConflict] = Field(default_factory=list)
    variations: list[EntityVariation] = Field(default_factory=list)
    unanimous_entities: list[str] = Field(
        default_factory=list,
        description="Entity values that are consistent across all docs"
    )
    
    # Resolved entities
    resolved_entities: list[ResolvedEntity] = Field(default_factory=list)
    
    # Metadata
    analysis_time_seconds: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def critical_conflicts(self) -> list[MultiDocConflict]:
        return [c for c in self.conflicts if c.severity == ConflictSeverity.CRITICAL]
    
    @property
    def high_conflicts(self) -> list[MultiDocConflict]:
        return [c for c in self.conflicts if c.severity == ConflictSeverity.HIGH]
    
    @property
    def conflict_count(self) -> int:
        return len(self.conflicts)
    
    def to_summary(self) -> str:
        """Generate text summary."""
        return (
            f"Analyzed {self.document_set.count} documents. "
            f"Found {self.conflict_count} conflicts "
            f"({len(self.critical_conflicts)} critical, {len(self.high_conflicts)} high). "
            f"Resolved {self.total_resolved_entities} entity groups."
        )


# ============================================================================
# Multi-Document Analyzer
# ============================================================================

class MultiDocAnalyzer:
    """
    Analyze conflicts and patterns across multiple documents.
    
    Unlike pairwise comparison, this builds a unified view across all
    documents and finds N-way conflicts.
    
    Usage:
        analyzer = MultiDocAnalyzer(vector_store, graph_store)
        report = await analyzer.analyze(document_set)
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        entity_resolver: EntityResolver | None = None,
    ) -> None:
        """
        Initialize the Multi-Document Analyzer.
        
        Args:
            vector_store: Vector store for semantic search
            graph_store: Graph store with entities/relationships
            entity_resolver: Optional pre-configured resolver
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.entity_resolver = entity_resolver or EntityResolver(graph_store)
    
    async def analyze(
        self,
        document_set: DocumentSet,
        focus_areas: list[str] | None = None,
        resolve_entities: bool = True,
    ) -> MultiDocReport:
        """
        Run full multi-document analysis.
        
        Args:
            document_set: Set of documents to analyze
            focus_areas: Optional focus areas (e.g., ["salary", "equity"])
            resolve_entities: Whether to resolve entity aliases
            
        Returns:
            MultiDocReport with all findings
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting multi-document analysis of {document_set.count} documents")
        
        # Get all entities from graph, grouped by document
        entities_by_doc = self.graph_store.get_entities_by_document(
            document_set.document_ids
        )
        
        # Flatten entities
        all_entities: list[Entity] = []
        for doc_entities in entities_by_doc.values():
            all_entities.extend([node.entity for node in doc_entities])
        
        # Resolve entity aliases
        resolved: list[ResolvedEntity] = []
        if resolve_entities and all_entities:
            resolved = self.entity_resolver.resolve_entities(all_entities)
        
        # Find variations by entity type
        variations = self._find_variations(
            entities_by_doc,
            document_set,
            focus_areas,
        )
        
        # Convert significant variations to conflicts
        conflicts = self._variations_to_conflicts(variations)
        
        # Find unanimous entities (consistent across all docs)
        unanimous = self._find_unanimous_entities(
            entities_by_doc,
            document_set,
        )
        
        # Build report
        analysis_time = time.time() - start_time
        
        report = MultiDocReport(
            document_set=document_set,
            total_entities=len(all_entities),
            total_relationships=self.graph_store.graph.number_of_edges(),
            total_resolved_entities=len(resolved),
            conflicts=conflicts,
            variations=[v for v in variations if v.is_conflict],
            unanimous_entities=unanimous,
            resolved_entities=resolved,
            analysis_time_seconds=analysis_time,
        )
        
        logger.info(
            f"Multi-doc analysis complete: {report.conflict_count} conflicts, "
            f"{len(resolved)} resolved entities in {analysis_time:.2f}s"
        )
        
        return report
    
    def _find_variations(
        self,
        entities_by_doc: dict[UUID, list],
        document_set: DocumentSet,
        focus_areas: list[str] | None,
    ) -> list[EntityVariation]:
        """Find entity variations across documents."""
        # Group entities by type
        by_type: dict[EntityType, list[EntityOccurrence]] = {}
        
        for doc_id, nodes in entities_by_doc.items():
            doc_name = document_set.get_name(doc_id)
            
            for node in nodes:
                entity = node.entity
                
                # Filter by focus areas if specified
                if focus_areas:
                    type_matches = any(
                        area.lower() in entity.entity_type.value.lower()
                        for area in focus_areas
                    )
                    if not type_matches:
                        continue
                
                occurrence = EntityOccurrence(
                    document_id=doc_id,
                    document_name=doc_name,
                    value=entity.value,
                    normalized_value=entity.normalized_value,
                    page_number=entity.source_page,
                    chunk_id=entity.source_chunk_id,
                    confidence=entity.confidence,
                )
                
                by_type.setdefault(entity.entity_type, []).append(occurrence)
        
        # Build variations for each type
        variations: list[EntityVariation] = []
        
        for entity_type, occurrences in by_type.items():
            if not occurrences:
                continue
            
            # Use most common value as canonical
            value_counts: dict[str, int] = {}
            for o in occurrences:
                value_counts[o.value] = value_counts.get(o.value, 0) + 1
            
            canonical = max(value_counts.keys(), key=lambda v: value_counts[v])
            
            variation = EntityVariation(
                entity_type=entity_type,
                canonical_value=canonical,
                occurrences=occurrences,
            )
            
            variations.append(variation)
        
        return variations
    
    def _variations_to_conflicts(
        self,
        variations: list[EntityVariation],
    ) -> list[MultiDocConflict]:
        """Convert entity variations to conflicts."""
        conflicts: list[MultiDocConflict] = []
        
        for variation in variations:
            if not variation.is_conflict:
                continue
            
            severity = variation.conflict_severity or ConflictSeverity.MEDIUM
            
            # Determine conflict type based on entity type
            conflict_type = self._entity_type_to_conflict_type(variation.entity_type)
            
            conflict = MultiDocConflict(
                entity_type=variation.entity_type,
                conflict_type=conflict_type,
                severity=severity,
                title=f"{variation.entity_type.value} Mismatch",
                description=(
                    f"Found {len(variation.unique_values)} different values for "
                    f"{variation.entity_type.value} across {variation.document_count} documents: "
                    f"{', '.join(variation.unique_values[:5])}"
                ),
                unique_values=variation.unique_values,
                document_count=variation.document_count,
            )
            
            conflicts.append(conflict)
        
        # Sort by severity
        conflicts.sort(key=lambda c: {
            ConflictSeverity.CRITICAL: 0,
            ConflictSeverity.HIGH: 1,
            ConflictSeverity.MEDIUM: 2,
            ConflictSeverity.LOW: 3,
        }.get(c.severity, 4))
        
        return conflicts
    
    def _entity_type_to_conflict_type(self, entity_type: EntityType) -> ConflictType:
        """Map entity type to conflict type."""
        mapping = {
            EntityType.MONETARY_AMOUNT: ConflictType.AMOUNT_MISMATCH,
            EntityType.SALARY: ConflictType.SALARY_MISMATCH,
            EntityType.EQUITY: ConflictType.EQUITY_MISMATCH,
            EntityType.PERCENTAGE: ConflictType.PERCENTAGE_MISMATCH,
            EntityType.DATE: ConflictType.DATE_CONFLICT,
            EntityType.DURATION: ConflictType.TERM_MISMATCH,
            EntityType.TERM: ConflictType.TERM_MISMATCH,
            EntityType.CLAUSE: ConflictType.CLAUSE_CONFLICT,
        }
        return mapping.get(entity_type, ConflictType.DEFINITION_MISMATCH)
    
    def _find_unanimous_entities(
        self,
        entities_by_doc: dict[UUID, list],
        document_set: DocumentSet,
    ) -> list[str]:
        """Find entity values that are consistent across all documents."""
        unanimous: list[str] = []
        
        # Group by (entity_type, normalized_value)
        type_values: dict[tuple[EntityType, str], set[UUID]] = {}
        
        for doc_id, nodes in entities_by_doc.items():
            for node in nodes:
                entity = node.entity
                key = (entity.entity_type, entity.value.lower().strip())
                
                if key not in type_values:
                    type_values[key] = set()
                type_values[key].add(doc_id)
        
        # Find values present in all documents
        all_doc_ids = set(document_set.document_ids)
        
        for (entity_type, value), doc_ids in type_values.items():
            if doc_ids == all_doc_ids:
                # Capitalize nicely
                display_value = value.title() if value.islower() else value
                unanimous.append(f"{entity_type.value}: {display_value}")
        
        return sorted(unanimous)
    
    def get_entity_across_documents(
        self,
        entity_value: str,
        document_set: DocumentSet,
    ) -> list[EntityOccurrence]:
        """Get all occurrences of an entity across documents."""
        occurrences: list[EntityOccurrence] = []
        
        entities_by_doc = self.graph_store.get_entities_by_document(
            document_set.document_ids
        )
        
        search_value = entity_value.lower().strip()
        
        for doc_id, nodes in entities_by_doc.items():
            for node in nodes:
                entity = node.entity
                if search_value in entity.value.lower():
                    occurrences.append(EntityOccurrence(
                        document_id=doc_id,
                        document_name=document_set.get_name(doc_id),
                        value=entity.value,
                        normalized_value=entity.normalized_value,
                        page_number=entity.source_page,
                        chunk_id=entity.source_chunk_id,
                        confidence=entity.confidence,
                    ))
        
        return occurrences
