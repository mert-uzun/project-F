"""
Comparator Agent - Conflict Detection.

The Comparator Agent finds potential conflicts between documents by:
1. Querying the vector store for related chunks
2. Querying the graph store for entity relationships
3. Comparing values and detecting mismatches
4. Generating conflict candidates for the Judge

This is the "detection" half of the conflict detection pipeline.
"""

import asyncio
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from src.agents.schemas import (
    Conflict,
    ConflictEvidence,
    ConflictSeverity,
    ConflictStatus,
    ConflictType,
    ComparisonQuery,
    SourceCitation,
)
from src.knowledge.graph_store import GraphStore
from src.knowledge.vector_store import VectorStore, SearchResult
from src.knowledge.schemas import Entity, EntityType, RelationshipType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ComparatorError(Exception):
    """Raised when comparison fails."""
    pass


# ============================================================================
# Comparison Prompts
# ============================================================================

CONFLICT_DETECTION_PROMPT = """You are an expert at detecting inconsistencies and conflicts in legal/financial documents.

## Context from Document A:
{context_a}

## Context from Document B:
{context_b}

## Task:
Compare these two document excerpts and identify any conflicts or inconsistencies.

Focus on:
- Numerical differences (salaries, percentages, amounts)
- Date/timeline conflicts
- Contradictory statements
- Different terms for the same thing

For each conflict found, provide:
1. Type of conflict
2. Severity (low/medium/high/critical)
3. What Document A says
4. What Document B says
5. Why this is a conflict

Respond in JSON format with a "conflicts" array.
"""


class DetectedConflict(BaseModel):
    """LLM output schema for detected conflict."""
    
    conflict_type: str
    severity: str
    value_a: str
    value_b: str
    description: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ConflictDetectionOutput(BaseModel):
    """LLM output schema."""
    
    conflicts: list[DetectedConflict] = Field(default_factory=list)
    summary: str = ""


# ============================================================================
# Comparator Agent
# ============================================================================

class ComparatorAgent:
    """
    Agent that detects conflicts between documents.
    
    The Comparator uses both semantic search (vector store) and 
    relationship analysis (graph store) to find potential conflicts.
    
    Usage:
        comparator = ComparatorAgent(vector_store, graph_store)
        conflicts = await comparator.compare(query)
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        llm: Any = None,
    ) -> None:
        """
        Initialize the Comparator Agent.
        
        Args:
            vector_store: Vector store for semantic search
            graph_store: Graph store for relationship queries
            llm: Optional LLM instance
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self._llm = llm
    
    @property
    def llm(self) -> Any:
        """Get LLM instance."""
        if self._llm is None:
            from src.utils.llm_factory import get_llm
            self._llm = get_llm()
        return self._llm
    
    async def compare(self, query: ComparisonQuery) -> list[Conflict]:
        """
        Compare documents and detect conflicts.
        
        Args:
            query: Comparison query with document IDs and focus areas
            
        Returns:
            List of detected conflicts
        """
        logger.info(f"Starting comparison for {len(query.document_ids)} documents")
        
        all_conflicts: list[Conflict] = []
        
        # 1. Find entity-based conflicts
        entity_conflicts = await self._find_entity_conflicts(query)
        all_conflicts.extend(entity_conflicts)
        
        # 2. Find semantic conflicts (via LLM comparison)
        semantic_conflicts = await self._find_semantic_conflicts(query)
        all_conflicts.extend(semantic_conflicts)
        
        # 3. Filter by confidence
        filtered = [c for c in all_conflicts if c.confidence >= query.min_confidence]
        
        # 4. Filter by severity if requested
        if not query.include_low_severity:
            filtered = [c for c in filtered if c.severity != ConflictSeverity.LOW]
        
        logger.info(f"Found {len(filtered)} conflicts (from {len(all_conflicts)} candidates)")
        
        return filtered
    
    async def _find_entity_conflicts(self, query: ComparisonQuery) -> list[Conflict]:
        """
        Find conflicts by comparing extracted entities.
        
        Looks for:
        - Same entity type with different values across documents
        - Salary/equity/percentage mismatches
        """
        conflicts: list[Conflict] = []
        
        # Get entities grouped by document
        doc_entities: dict[UUID, list[Entity]] = {}
        
        for doc_id in query.document_ids:
            entities = []
            # Find all entities from this document
            for entity_type in EntityType:
                found = self.graph_store.find_entities_by_type(entity_type, doc_id)
                entities.extend([node.entity for node in found])
            doc_entities[doc_id] = entities
        
        # Compare entities across documents
        doc_ids = list(query.document_ids)
        
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                doc_a, doc_b = doc_ids[i], doc_ids[j]
                entities_a = doc_entities.get(doc_a, [])
                entities_b = doc_entities.get(doc_b, [])
                
                # Find conflicts between these two documents
                pair_conflicts = self._compare_entity_sets(
                    entities_a, entities_b, query.focus_areas
                )
                conflicts.extend(pair_conflicts)
        
        return conflicts
    
    def _compare_entity_sets(
        self,
        entities_a: list[Entity],
        entities_b: list[Entity],
        focus_areas: list[str],
    ) -> list[Conflict]:
        """Compare two sets of entities for conflicts."""
        conflicts: list[Conflict] = []
        
        # Group entities by type for easier comparison
        by_type_a: dict[EntityType, list[Entity]] = {}
        by_type_b: dict[EntityType, list[Entity]] = {}
        
        for e in entities_a:
            by_type_a.setdefault(e.entity_type, []).append(e)
        for e in entities_b:
            by_type_b.setdefault(e.entity_type, []).append(e)
        
        # Check monetary amounts
        if "salary" in focus_areas or "amount" in focus_areas:
            conflicts.extend(self._compare_monetary_values(
                by_type_a.get(EntityType.MONETARY_AMOUNT, []),
                by_type_b.get(EntityType.MONETARY_AMOUNT, []),
            ))
            conflicts.extend(self._compare_monetary_values(
                by_type_a.get(EntityType.SALARY, []),
                by_type_b.get(EntityType.SALARY, []),
            ))
        
        # Check percentages/equity
        if "equity" in focus_areas or "percentage" in focus_areas:
            conflicts.extend(self._compare_percentage_values(
                by_type_a.get(EntityType.PERCENTAGE, []) + by_type_a.get(EntityType.EQUITY, []),
                by_type_b.get(EntityType.PERCENTAGE, []) + by_type_b.get(EntityType.EQUITY, []),
            ))
        
        # Check dates
        if "dates" in focus_areas:
            conflicts.extend(self._compare_date_values(
                by_type_a.get(EntityType.DATE, []),
                by_type_b.get(EntityType.DATE, []),
            ))
        
        return conflicts
    
    def _compare_monetary_values(
        self,
        amounts_a: list[Entity],
        amounts_b: list[Entity],
    ) -> list[Conflict]:
        """Compare monetary amounts for mismatches."""
        conflicts: list[Conflict] = []
        
        for a in amounts_a:
            for b in amounts_b:
                # Check if these might be referring to the same thing
                # by comparing normalized values
                norm_a = self._normalize_amount(a.normalized_value or a.value)
                norm_b = self._normalize_amount(b.normalized_value or b.value)
                
                if norm_a is None or norm_b is None:
                    continue
                
                # If values are different, it's a potential conflict
                if norm_a != norm_b:
                    # Calculate difference
                    diff_pct = abs(norm_a - norm_b) / max(norm_a, norm_b) * 100
                    
                    # Determine severity based on difference
                    if diff_pct >= 50:
                        severity = ConflictSeverity.CRITICAL
                    elif diff_pct >= 20:
                        severity = ConflictSeverity.HIGH
                    elif diff_pct >= 5:
                        severity = ConflictSeverity.MEDIUM
                    else:
                        severity = ConflictSeverity.LOW
                    
                    conflicts.append(Conflict(
                        conflict_type=ConflictType.AMOUNT_MISMATCH,
                        severity=severity,
                        status=ConflictStatus.DETECTED,
                        title=f"Amount Mismatch: {a.value} vs {b.value}",
                        description=(
                            f"Document A states '{a.value}' while Document B states '{b.value}'. "
                            f"Difference: {diff_pct:.1f}%"
                        ),
                        evidence_a=ConflictEvidence(
                            entity=a,
                            citation=SourceCitation(
                                document_id=a.source_document_id,
                                document_name="Document A",
                                page_number=a.source_page,
                                chunk_id=a.source_chunk_id,
                                excerpt=a.source_text[:200],
                            ),
                            extracted_value=a.value,
                            normalized_value=str(norm_a),
                        ),
                        evidence_b=ConflictEvidence(
                            entity=b,
                            citation=SourceCitation(
                                document_id=b.source_document_id,
                                document_name="Document B",
                                page_number=b.source_page,
                                chunk_id=b.source_chunk_id,
                                excerpt=b.source_text[:200],
                            ),
                            extracted_value=b.value,
                            normalized_value=str(norm_b),
                        ),
                        value_a=a.value,
                        value_b=b.value,
                        difference=f"{diff_pct:.1f}%",
                        confidence=min(a.confidence, b.confidence) * 0.9,
                    ))
        
        return conflicts
    
    def _compare_percentage_values(
        self,
        pcts_a: list[Entity],
        pcts_b: list[Entity],
    ) -> list[Conflict]:
        """Compare percentage/equity values."""
        conflicts: list[Conflict] = []
        
        for a in pcts_a:
            for b in pcts_b:
                pct_a = self._normalize_percentage(a.normalized_value or a.value)
                pct_b = self._normalize_percentage(b.normalized_value or b.value)
                
                if pct_a is None or pct_b is None:
                    continue
                
                if pct_a != pct_b:
                    diff = abs(pct_a - pct_b)
                    
                    if diff > 10:
                        severity = ConflictSeverity.CRITICAL
                    elif diff > 5:
                        severity = ConflictSeverity.HIGH
                    elif diff > 1:
                        severity = ConflictSeverity.MEDIUM
                    else:
                        severity = ConflictSeverity.LOW
                    
                    conflicts.append(Conflict(
                        conflict_type=ConflictType.PERCENTAGE_MISMATCH,
                        severity=severity,
                        status=ConflictStatus.DETECTED,
                        title=f"Percentage Mismatch: {a.value} vs {b.value}",
                        description=(
                            f"Document A states '{a.value}' while Document B states '{b.value}'. "
                            f"Difference: {diff:.1f} percentage points"
                        ),
                        evidence_a=ConflictEvidence(
                            entity=a,
                            citation=SourceCitation(
                                document_id=a.source_document_id,
                                document_name="Document A",
                                page_number=a.source_page,
                                chunk_id=a.source_chunk_id,
                                excerpt=a.source_text[:200],
                            ),
                            extracted_value=a.value,
                            normalized_value=str(pct_a),
                        ),
                        evidence_b=ConflictEvidence(
                            entity=b,
                            citation=SourceCitation(
                                document_id=b.source_document_id,
                                document_name="Document B",
                                page_number=b.source_page,
                                chunk_id=b.source_chunk_id,
                                excerpt=b.source_text[:200],
                            ),
                            extracted_value=b.value,
                            normalized_value=str(pct_b),
                        ),
                        value_a=a.value,
                        value_b=b.value,
                        difference=f"{diff:.1f}pp",
                        confidence=min(a.confidence, b.confidence) * 0.9,
                    ))
        
        return conflicts
    
    def _compare_date_values(
        self,
        dates_a: list[Entity],
        dates_b: list[Entity],
    ) -> list[Conflict]:
        """Compare date values for timeline conflicts."""
        conflicts: list[Conflict] = []
        
        for a in dates_a:
            for b in dates_b:
                date_a = self._parse_date(a.normalized_value or a.value)
                date_b = self._parse_date(b.normalized_value or b.value)
                
                if date_a is None or date_b is None:
                    continue
                
                # Calculate difference in days
                diff_days = abs((date_a - date_b).days)
                
                if diff_days == 0:
                    continue  # Same date, no conflict
                
                # Determine severity based on difference
                if diff_days > 365:
                    severity = ConflictSeverity.CRITICAL
                elif diff_days > 90:
                    severity = ConflictSeverity.HIGH
                elif diff_days > 30:
                    severity = ConflictSeverity.MEDIUM
                else:
                    severity = ConflictSeverity.LOW
                
                conflicts.append(Conflict(
                    conflict_type=ConflictType.DATE_CONFLICT,
                    severity=severity,
                    status=ConflictStatus.DETECTED,
                    title=f"Date Mismatch: {a.value} vs {b.value}",
                    description=(
                        f"Document A states '{a.value}' while Document B states '{b.value}'. "
                        f"Difference: {diff_days} days"
                    ),
                    evidence_a=ConflictEvidence(
                        entity=a,
                        citation=SourceCitation(
                            document_id=a.source_document_id,
                            document_name="Document A",
                            page_number=a.source_page,
                            chunk_id=a.source_chunk_id,
                            excerpt=a.source_text[:200],
                        ),
                        extracted_value=a.value,
                        normalized_value=date_a.isoformat() if date_a else None,
                    ),
                    evidence_b=ConflictEvidence(
                        entity=b,
                        citation=SourceCitation(
                            document_id=b.source_document_id,
                            document_name="Document B",
                            page_number=b.source_page,
                            chunk_id=b.source_chunk_id,
                            excerpt=b.source_text[:200],
                        ),
                        extracted_value=b.value,
                        normalized_value=date_b.isoformat() if date_b else None,
                    ),
                    value_a=a.value,
                    value_b=b.value,
                    difference=f"{diff_days} days",
                    confidence=min(a.confidence, b.confidence) * 0.85,
                ))
        
        return conflicts
    
    def _parse_date(self, value: str) -> "date | None":
        """Parse a date string into a date object."""
        from datetime import date, datetime
        import re
        
        # Try common date formats
        formats = [
            "%Y-%m-%d",          # 2024-01-15
            "%m/%d/%Y",          # 01/15/2024
            "%d/%m/%Y",          # 15/01/2024
            "%B %d, %Y",         # January 15, 2024
            "%b %d, %Y",         # Jan 15, 2024
            "%d %B %Y",          # 15 January 2024
            "%d %b %Y",          # 15 Jan 2024
            "%Y/%m/%d",          # 2024/01/15
        ]
        
        cleaned = value.strip()
        
        for fmt in formats:
            try:
                return datetime.strptime(cleaned, fmt).date()
            except ValueError:
                continue
        
        # Try to extract year at minimum
        year_match = re.search(r'\b(19|20)\d{2}\b', cleaned)
        if year_match:
            year = int(year_match.group())
            # Default to January 1 if only year found
            return date(year, 1, 1)
        
        return None
    
    async def _find_semantic_conflicts(self, query: ComparisonQuery) -> list[Conflict]:
        """
        Find conflicts using LLM semantic comparison.
        
        Queries the vector store for related content and uses LLM
        to identify subtle conflicts.
        """
        conflicts: list[Conflict] = []
        
        # Build search queries from focus areas
        search_queries = []
        if query.specific_query:
            search_queries.append(query.specific_query)
        
        for area in query.focus_areas:
            if area == "salary":
                search_queries.append("compensation salary base pay annual")
            elif area == "equity":
                search_queries.append("equity shares stock options vesting")
            elif area == "dates":
                search_queries.append("effective date start termination expiry")
            elif area == "parties":
                search_queries.append("parties agreement between company employee")
        
        # For each document pair, find related chunks and compare
        doc_ids = list(query.document_ids)
        
        for search_query in search_queries[:3]:  # Limit queries
            for i in range(len(doc_ids)):
                for j in range(i + 1, len(doc_ids)):
                    # Get relevant chunks from each document
                    chunks_a = self.vector_store.search(
                        search_query, 
                        top_k=3, 
                        filter_document_id=doc_ids[i]
                    )
                    chunks_b = self.vector_store.search(
                        search_query, 
                        top_k=3, 
                        filter_document_id=doc_ids[j]
                    )
                    
                    if chunks_a and chunks_b:
                        # Use LLM to compare
                        llm_conflicts = await self._llm_compare(
                            chunks_a, chunks_b, doc_ids[i], doc_ids[j]
                        )
                        conflicts.extend(llm_conflicts)
        
        return conflicts
    
    async def _llm_compare(
        self,
        chunks_a: list[SearchResult],
        chunks_b: list[SearchResult],
        doc_id_a: UUID,
        doc_id_b: UUID,
    ) -> list[Conflict]:
        """Use LLM to compare chunks and find conflicts."""
        import json
        
        # Build context
        context_a = "\n\n".join([c.content for c in chunks_a])
        context_b = "\n\n".join([c.content for c in chunks_b])
        
        prompt = CONFLICT_DETECTION_PROMPT.format(
            context_a=context_a[:2000],
            context_b=context_b[:2000],
        )
        
        try:
            response = await self.llm.acomplete(prompt)
            text = response.text.strip()
            
            # Parse JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(text)
            output = ConflictDetectionOutput.model_validate(data)
            
            # Convert to Conflict objects
            conflicts: list[Conflict] = []
            
            for detected in output.conflicts:
                # Map conflict type
                try:
                    conflict_type = ConflictType(detected.conflict_type.lower().replace(" ", "_"))
                except ValueError:
                    conflict_type = ConflictType.LOGICAL_INCONSISTENCY
                
                # Map severity
                try:
                    severity = ConflictSeverity(detected.severity.lower())
                except ValueError:
                    severity = ConflictSeverity.MEDIUM
                
                # Create placeholder entities for LLM-detected conflicts
                entity_a = Entity(
                    entity_type=EntityType.OTHER,
                    value=detected.value_a[:100],
                    source_document_id=doc_id_a,
                    source_chunk_id=UUID(chunks_a[0].chunk_id) if chunks_a else uuid4(),
                    source_page=chunks_a[0].metadata.get("page_number", 1) if chunks_a else 1,
                    source_text=context_a[:200],
                    confidence=detected.confidence,
                )
                
                entity_b = Entity(
                    entity_type=EntityType.OTHER,
                    value=detected.value_b[:100],
                    source_document_id=doc_id_b,
                    source_chunk_id=UUID(chunks_b[0].chunk_id) if chunks_b else uuid4(),
                    source_page=chunks_b[0].metadata.get("page_number", 1) if chunks_b else 1,
                    source_text=context_b[:200],
                    confidence=detected.confidence,
                )
                
                conflicts.append(Conflict(
                    conflict_type=conflict_type,
                    severity=severity,
                    status=ConflictStatus.DETECTED,
                    title=detected.description[:80],
                    description=detected.description,
                    evidence_a=ConflictEvidence(
                        entity=entity_a,
                        citation=SourceCitation(
                            document_id=doc_id_a,
                            document_name="Document A",
                            page_number=entity_a.source_page,
                            chunk_id=entity_a.source_chunk_id,
                            excerpt=context_a[:200],
                        ),
                        extracted_value=detected.value_a,
                    ),
                    evidence_b=ConflictEvidence(
                        entity=entity_b,
                        citation=SourceCitation(
                            document_id=doc_id_b,
                            document_name="Document B",
                            page_number=entity_b.source_page,
                            chunk_id=entity_b.source_chunk_id,
                            excerpt=context_b[:200],
                        ),
                        extracted_value=detected.value_b,
                    ),
                    value_a=detected.value_a,
                    value_b=detected.value_b,
                    confidence=detected.confidence,
                ))
            
            return conflicts
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM conflict output: {e}")
            return []
        except Exception as e:
            logger.error(f"LLM comparison failed: {e}")
            return []
    
    # ========================================================================
    # Utility methods
    # ========================================================================
    
    def _normalize_amount(self, value: str) -> float | None:
        """Normalize monetary amount to float."""
        import re
        
        # Remove currency symbols and commas
        cleaned = re.sub(r"[,$£€¥]", "", value)
        
        # Handle K/M/B suffixes
        multiplier = 1
        if cleaned.lower().endswith("k"):
            multiplier = 1000
            cleaned = cleaned[:-1]
        elif cleaned.lower().endswith("m"):
            multiplier = 1_000_000
            cleaned = cleaned[:-1]
        elif cleaned.lower().endswith("b"):
            multiplier = 1_000_000_000
            cleaned = cleaned[:-1]
        
        try:
            return float(cleaned) * multiplier
        except ValueError:
            return None
    
    def _normalize_percentage(self, value: str) -> float | None:
        """Normalize percentage to float."""
        import re
        
        # Remove % sign
        cleaned = re.sub(r"[%]", "", value).strip()
        
        try:
            return float(cleaned)
        except ValueError:
            return None


# Required import for uuid4
from uuid import uuid4
