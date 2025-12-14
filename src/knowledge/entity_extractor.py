"""
Entity Extractor - LLM-Based Entity and Relationship Extraction.

Uses structured output to extract entities and relationships from document chunks.
This is the bridge between raw text and the Knowledge Graph.

Key principles:
1. Every extraction MUST have a source citation
2. Output MUST be structured (Pydantic models)
3. Normalize values for comparison (e.g., "3 months" → 90 days)
"""

import re
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from src.ingestion.schemas import DocumentChunk
from src.knowledge.schemas import (
    Entity,
    EntityType,
    Relationship,
    RelationType,
)
from src.utils.llm_factory import get_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExtractionError(Exception):
    """Raised when entity extraction fails."""

    pass


# === Structured Output Models for LLM ===

class ExtractedEntity(BaseModel):
    """LLM output format for a single entity."""

    entity_type: str = Field(..., description="Type: person, organization, monetary_amount, percentage, date, duration, role, clause")
    name: str = Field(..., description="Entity name or label")
    value: str | None = Field(default=None, description="Value if applicable (amount, percentage, date)")
    context: str = Field(..., description="Surrounding text context (1-2 sentences)")


class ExtractedRelationship(BaseModel):
    """LLM output format for a relationship."""

    source_entity_name: str = Field(..., description="Name of source entity")
    relationship_type: str = Field(..., description="Type: has_role, employed_by, has_compensation, has_equity, effective_date, etc")
    target_entity_name: str = Field(..., description="Name of target entity")
    context: str = Field(..., description="Text showing this relationship")


class ExtractionResult(BaseModel):
    """Complete LLM extraction output."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


# === Normalization Utilities ===

def normalize_monetary_amount(value: str) -> float | None:
    """
    Normalize monetary amounts to USD float.
    
    Examples:
        "$100,000" → 100000.0
        "100k" → 100000.0
        "$1.5M" → 1500000.0
    """
    if not value:
        return None
    
    # Remove currency symbols and whitespace
    cleaned = re.sub(r"[$€£¥,\s]", "", value.lower())
    
    # Handle K/M/B suffixes
    multiplier = 1.0
    if cleaned.endswith("k"):
        multiplier = 1_000
        cleaned = cleaned[:-1]
    elif cleaned.endswith("m"):
        multiplier = 1_000_000
        cleaned = cleaned[:-1]
    elif cleaned.endswith("b"):
        multiplier = 1_000_000_000
        cleaned = cleaned[:-1]
    
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


def normalize_percentage(value: str) -> float | None:
    """
    Normalize percentage to float.
    
    Examples:
        "5%" → 5.0
        "0.05" → 5.0 (if appears to be decimal)
    """
    if not value:
        return None
    
    cleaned = re.sub(r"[%\s]", "", value)
    
    try:
        num = float(cleaned)
        # If less than 1, assume it's a decimal representation
        if num < 1 and "%" not in value:
            return num * 100
        return num
    except ValueError:
        return None


def normalize_duration(value: str) -> int | None:
    """
    Normalize duration to days.
    
    Examples:
        "3 months" → 90
        "90 days" → 90
        "1 year" → 365
    """
    if not value:
        return None
    
    value_lower = value.lower()
    
    # Extract number
    match = re.search(r"(\d+(?:\.\d+)?)", value_lower)
    if not match:
        return None
    
    num = float(match.group(1))
    
    # Determine unit
    if "year" in value_lower:
        return int(num * 365)
    elif "month" in value_lower:
        return int(num * 30)
    elif "week" in value_lower:
        return int(num * 7)
    elif "day" in value_lower:
        return int(num)
    
    return None


def get_entity_type(type_str: str) -> EntityType:
    """Map string to EntityType enum."""
    type_map = {
        "person": EntityType.PERSON,
        "organization": EntityType.ORGANIZATION,
        "company": EntityType.ORGANIZATION,
        "monetary_amount": EntityType.MONETARY_AMOUNT,
        "money": EntityType.MONETARY_AMOUNT,
        "salary": EntityType.MONETARY_AMOUNT,
        "percentage": EntityType.PERCENTAGE,
        "equity": EntityType.PERCENTAGE,
        "date": EntityType.DATE,
        "duration": EntityType.DURATION,
        "period": EntityType.DURATION,
        "role": EntityType.ROLE,
        "position": EntityType.ROLE,
        "title": EntityType.ROLE,
        "clause": EntityType.CLAUSE,
        "location": EntityType.LOCATION,
        "document": EntityType.DOCUMENT,
    }
    return type_map.get(type_str.lower(), EntityType.UNKNOWN)


def get_relationship_type(type_str: str) -> RelationType:
    """Map string to RelationType enum."""
    type_map = {
        "has_role": RelationType.HAS_ROLE,
        "employed_by": RelationType.EMPLOYED_BY,
        "works_for": RelationType.EMPLOYED_BY,
        "reports_to": RelationType.REPORTS_TO,
        "has_compensation": RelationType.HAS_COMPENSATION,
        "has_salary": RelationType.HAS_COMPENSATION,
        "earns": RelationType.HAS_COMPENSATION,
        "has_equity": RelationType.HAS_EQUITY,
        "owns": RelationType.OWNS,
        "has_bonus": RelationType.HAS_BONUS,
        "effective_date": RelationType.EFFECTIVE_DATE,
        "starts": RelationType.EFFECTIVE_DATE,
        "termination_date": RelationType.TERMINATION_DATE,
        "ends": RelationType.TERMINATION_DATE,
        "duration": RelationType.DURATION,
        "has_clause": RelationType.HAS_CLAUSE,
        "contains": RelationType.CONTAINS,
        "references": RelationType.REFERENCES,
        "defined_in": RelationType.DEFINED_IN,
    }
    return type_map.get(type_str.lower(), RelationType.RELATED_TO)


# === Entity Extractor ===

EXTRACTION_PROMPT = """You are an expert at extracting structured information from legal and financial documents.

Given the following text chunk from a document, extract:
1. ENTITIES: People, organizations, monetary amounts, percentages, dates, durations, roles
2. RELATIONSHIPS: How entities are connected (who has what role, who earns what, etc.)

CRITICAL RULES:
- Only extract information EXPLICITLY stated in the text
- Do NOT infer or assume information not present
- Include the exact text context for each extraction
- For monetary amounts, include the full value (e.g., "$500,000" not just "salary")
- For percentages, include the % symbol

TEXT CHUNK:
---
{text}
---

Extract entities and relationships in the following JSON format:
{{
    "entities": [
        {{
            "entity_type": "person|organization|monetary_amount|percentage|date|duration|role|clause",
            "name": "entity name or label",
            "value": "value if applicable (amount, percentage, date)",
            "context": "1-2 sentences showing where this entity appears"
        }}
    ],
    "relationships": [
        {{
            "source_entity_name": "name of source entity",
            "relationship_type": "has_role|employed_by|has_compensation|has_equity|effective_date|etc",
            "target_entity_name": "name of target entity",
            "context": "text showing this relationship"
        }}
    ]
}}

Return ONLY valid JSON, no other text."""


class EntityExtractor:
    """
    LLM-based entity and relationship extractor.
    
    Extracts structured entities from document chunks and creates
    normalized Entity and Relationship objects for the Knowledge Graph.
    
    Usage:
        extractor = EntityExtractor()
        entities, relationships = await extractor.extract(chunk)
    """

    def __init__(self, llm: Any = None) -> None:
        """
        Initialize extractor.
        
        Args:
            llm: LLM instance (uses default from factory if None)
        """
        self._llm = llm
    
    def _get_llm(self) -> Any:
        """Get LLM instance."""
        if self._llm is None:
            self._llm = get_llm()
        return self._llm
    
    async def extract(
        self,
        chunk: DocumentChunk,
    ) -> tuple[list[Entity], list[Relationship]]:
        """
        Extract entities and relationships from a chunk.
        
        Args:
            chunk: Document chunk to process
            
        Returns:
            Tuple of (entities, relationships)
        """
        llm = self._get_llm()
        
        # Build prompt
        prompt = EXTRACTION_PROMPT.format(text=chunk.content)
        
        try:
            # Call LLM
            response = await llm.acomplete(prompt)
            response_text = response.text.strip()
            
            # Parse JSON response
            extraction = self._parse_response(response_text)
            
            # Convert to Entity and Relationship objects
            entities = self._build_entities(extraction.entities, chunk)
            relationships = self._build_relationships(
                extraction.relationships,
                entities,
                chunk,
            )
            
            logger.debug(
                f"Extracted {len(entities)} entities, {len(relationships)} relationships "
                f"from chunk {chunk.metadata.chunk_id}"
            )
            
            return entities, relationships
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            raise ExtractionError(f"Extraction failed: {e}") from e
    
    async def extract_from_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> tuple[list[Entity], list[Relationship]]:
        """
        Extract from multiple chunks.
        
        Args:
            chunks: List of chunks to process
            
        Returns:
            Tuple of (all_entities, all_relationships)
        """
        all_entities: list[Entity] = []
        all_relationships: list[Relationship] = []
        
        for i, chunk in enumerate(chunks):
            try:
                entities, relationships = await self.extract(chunk)
                all_entities.extend(entities)
                all_relationships.extend(relationships)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
                    
            except ExtractionError as e:
                logger.warning(f"Skipping chunk {chunk.metadata.chunk_id}: {e}")
                continue
        
        logger.info(
            f"Extraction complete: {len(all_entities)} entities, "
            f"{len(all_relationships)} relationships from {len(chunks)} chunks"
        )
        
        return all_entities, all_relationships
    
    def _parse_response(self, response_text: str) -> ExtractionResult:
        """Parse LLM response into ExtractionResult."""
        import json
        
        # Handle markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        try:
            data = json.loads(response_text)
            return ExtractionResult.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return ExtractionResult(entities=[], relationships=[])
    
    def _build_entities(
        self,
        extracted: list[ExtractedEntity],
        chunk: DocumentChunk,
    ) -> list[Entity]:
        """Convert extracted entities to Entity objects."""
        entities: list[Entity] = []
        
        for ext in extracted:
            entity_type = get_entity_type(ext.entity_type)
            
            # Normalize value based on type
            normalized_value = None
            if entity_type == EntityType.MONETARY_AMOUNT and ext.value:
                normalized_value = normalize_monetary_amount(ext.value)
            elif entity_type == EntityType.PERCENTAGE and ext.value:
                normalized_value = normalize_percentage(ext.value)
            elif entity_type == EntityType.DURATION and ext.value:
                normalized_value = normalize_duration(ext.value)
            
            entity = Entity(
                entity_type=entity_type,
                name=ext.name,
                value=ext.value,
                normalized_value=normalized_value,
                source_document_id=chunk.metadata.document_id,
                source_chunk_id=chunk.metadata.chunk_id,
                source_page=chunk.metadata.page_number,
                source_text=ext.context,
            )
            
            entities.append(entity)
        
        return entities
    
    def _build_relationships(
        self,
        extracted: list[ExtractedRelationship],
        entities: list[Entity],
        chunk: DocumentChunk,
    ) -> list[Relationship]:
        """Convert extracted relationships to Relationship objects."""
        relationships: list[Relationship] = []
        
        # Build name to entity map
        entity_map: dict[str, Entity] = {e.name.lower(): e for e in entities}
        
        for ext in extracted:
            source_entity = entity_map.get(ext.source_entity_name.lower())
            target_entity = entity_map.get(ext.target_entity_name.lower())
            
            if not source_entity or not target_entity:
                logger.debug(
                    f"Skipping relationship: entity not found "
                    f"({ext.source_entity_name} -> {ext.target_entity_name})"
                )
                continue
            
            rel_type = get_relationship_type(ext.relationship_type)
            
            relationship = Relationship(
                relationship_type=rel_type,
                source_entity_id=source_entity.entity_id,
                target_entity_id=target_entity.entity_id,
                source_document_id=chunk.metadata.document_id,
                source_chunk_id=chunk.metadata.chunk_id,
                source_page=chunk.metadata.page_number,
                source_text=ext.context,
            )
            
            relationships.append(relationship)
        
        return relationships


# === Convenience Functions ===

async def extract_entities_from_document(
    chunks: list[DocumentChunk],
) -> tuple[list[Entity], list[Relationship]]:
    """
    Extract all entities and relationships from document chunks.
    
    Args:
        chunks: Document chunks
        
    Returns:
        Tuple of (entities, relationships)
    """
    extractor = EntityExtractor()
    return await extractor.extract_from_chunks(chunks)
