"""
Entity Extractor - LLM-based Entity and Relationship Extraction.

Uses structured LLM output to extract entities and relationships from document chunks.
This is the bridge between raw text and the Knowledge Graph.
"""

import time
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from src.ingestion.schemas import DocumentChunk
from src.knowledge.schemas import (
    Entity,
    EntityType,
    ExtractionResult,
    Relationship,
    RelationshipType,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EntityExtractionError(Exception):
    """Raised when entity extraction fails."""

    pass


# ============================================================================
# Pydantic Models for Structured LLM Output
# ============================================================================


class ExtractedEntity(BaseModel):
    """Schema for LLM-extracted entity."""

    entity_type: str = Field(
        ...,
        description="Type: person, organization, monetary_amount, percentage, date, clause, etc.",
    )
    value: str = Field(..., description="The entity value as it appears in the text")
    normalized_value: str | None = Field(
        None, description="Normalized form (e.g., '$100,000' -> '100000')"
    )
    confidence: float = Field(0.9, ge=0.0, le=1.0, description="Extraction confidence")
    context: str = Field(..., description="Surrounding text context (1-2 sentences)")


class ExtractedRelationship(BaseModel):
    """Schema for LLM-extracted relationship."""

    source_entity: str = Field(..., description="The source entity value")
    relationship_type: str = Field(..., description="Type: has_salary, has_equity, employs, etc.")
    target_entity: str = Field(..., description="The target entity value")
    confidence: float = Field(0.9, ge=0.0, le=1.0)
    context: str = Field(..., description="Text supporting this relationship")


class ExtractionOutput(BaseModel):
    """Complete extraction output from LLM."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


# ============================================================================
# Extraction Prompts
# ============================================================================

ENTITY_EXTRACTION_PROMPT = """You are an expert at extracting structured information from legal and financial documents.

Analyze the following text and extract ALL entities and relationships.

## Entity Types to Extract:
- **person**: Names of individuals (e.g., "John Doe", "CEO")
- **organization**: Company names (e.g., "ABC Corp", "Acme Inc")
- **role**: Job titles (e.g., "Chief Executive Officer", "Director")
- **monetary_amount**: Money values (e.g., "$500,000", "100k USD")
- **percentage**: Percentages (e.g., "5%", "three percent")
- **equity**: Equity stakes (e.g., "5% equity", "1000 shares")
- **date**: Dates (e.g., "January 1, 2024", "Q1 2024")
- **duration**: Time periods (e.g., "90 days", "2 years")
- **clause**: Contract clauses (e.g., "Termination Clause", "Non-compete")

## Relationship Types to Extract:
- **has_salary**: Person has a salary amount
- **has_equity**: Person/Org owns equity
- **employs**: Organization employs person
- **has_role**: Person has a role/title
- **effective_date**: Something has an effective date
- **grants**: Document/Clause grants something
- **obligates**: Clause obligates someone to something

## Text to Analyze:
{text}

## Instructions:
1. Extract ALL entities you can find
2. For each entity, provide the normalized form if applicable
3. Extract relationships between entities
4. Include surrounding context for verification
5. Be thorough but accurate - only extract what's clearly stated

Respond with a JSON object containing "entities" and "relationships" arrays.
"""

FOCUSED_EXTRACTION_PROMPT = """Extract specific information from this document chunk.

Focus on extracting: {focus_areas}

Text:
{text}

Return entities and relationships as JSON.
"""


# ============================================================================
# Entity Extractor
# ============================================================================


class EntityExtractor:
    """
    LLM-based entity and relationship extractor.

    Converts document chunks into structured entities and relationships
    for the Knowledge Graph.

    Usage:
        extractor = EntityExtractor(llm)
        result = await extractor.extract(chunk)

        # Add to graph
        graph_store.add_entities(result.entities)
        graph_store.add_relationships(result.relationships)
    """

    def __init__(
        self,
        llm: Any = None,
        use_structured_output: bool = True,
    ) -> None:
        """
        Initialize the entity extractor.

        Args:
            llm: LLM instance (if None, will use default from llm_factory)
            use_structured_output: Whether to use Pydantic structured output
        """
        self._llm = llm
        self._use_structured_output = use_structured_output

    @property
    def llm(self) -> Any:
        """Get the LLM instance."""
        if self._llm is None:
            from src.utils.llm_factory import get_llm

            self._llm = get_llm()
        return self._llm

    async def extract(
        self,
        chunk: DocumentChunk,
        focus_areas: list[str] | None = None,
    ) -> ExtractionResult:
        """
        Extract entities and relationships from a document chunk.

        Args:
            chunk: The document chunk to process
            focus_areas: Optional specific areas to focus on

        Returns:
            ExtractionResult with entities and relationships
        """
        start_time = time.time()
        errors: list[str] = []

        logger.debug(f"Extracting entities from chunk {chunk.metadata.chunk_id}")

        try:
            # Build prompt
            if focus_areas:
                prompt = FOCUSED_EXTRACTION_PROMPT.format(
                    focus_areas=", ".join(focus_areas),
                    text=chunk.content,
                )
            else:
                prompt = ENTITY_EXTRACTION_PROMPT.format(text=chunk.content)

            # Call LLM
            if self._use_structured_output:
                extraction = await self._extract_structured(prompt)
            else:
                extraction = await self._extract_unstructured(prompt)

            # Convert to our schema
            entities = self._convert_entities(
                extraction.entities,
                chunk.metadata.document_id,
                chunk.metadata.chunk_id,
                chunk.metadata.page_number,
            )

            relationships = self._convert_relationships(
                extraction.relationships,
                entities,
                chunk.metadata.document_id,
                chunk.metadata.chunk_id,
                chunk.metadata.page_number,
            )

            extraction_time = time.time() - start_time

            logger.info(
                f"Extracted {len(entities)} entities, {len(relationships)} relationships "
                f"from chunk {chunk.metadata.chunk_id} in {extraction_time:.2f}s"
            )

            return ExtractionResult(
                chunk_id=chunk.metadata.chunk_id,
                document_id=chunk.metadata.document_id,
                entities=entities,
                relationships=relationships,
                extraction_time_seconds=extraction_time,
                model_used=str(type(self.llm).__name__),
                errors=errors,
            )

        except Exception as e:
            logger.error(f"Extraction failed for chunk {chunk.metadata.chunk_id}: {e}")
            errors.append(str(e))

            return ExtractionResult(
                chunk_id=chunk.metadata.chunk_id,
                document_id=chunk.metadata.document_id,
                entities=[],
                relationships=[],
                extraction_time_seconds=time.time() - start_time,
                model_used="unknown",
                errors=errors,
            )

    async def extract_batch(
        self,
        chunks: list[DocumentChunk],
        focus_areas: list[str] | None = None,
    ) -> list[ExtractionResult]:
        """
        Extract entities from multiple chunks.

        Args:
            chunks: List of document chunks
            focus_areas: Optional focus areas

        Returns:
            List of ExtractionResult objects
        """
        results: list[ExtractionResult] = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
            result = await self.extract(chunk, focus_areas)
            results.append(result)

        total_entities = sum(r.entity_count for r in results)
        total_relationships = sum(r.relationship_count for r in results)

        logger.info(
            f"Batch extraction complete: {total_entities} entities, "
            f"{total_relationships} relationships from {len(chunks)} chunks"
        )

        return results

    async def _extract_structured(self, prompt: str) -> ExtractionOutput:
        """Extract using structured Pydantic output."""
        try:
            from llama_index.core.program import LLMTextCompletionProgram

            program = LLMTextCompletionProgram.from_defaults(
                output_cls=ExtractionOutput,
                llm=self.llm,
                prompt_template_str=prompt + "\n\nJSON Output:",
            )

            result = await program.acall()
            return result

        except ImportError:
            logger.warning("LLMTextCompletionProgram not available, falling back to unstructured")
            return await self._extract_unstructured(prompt)
        except Exception as e:
            logger.warning(f"Structured extraction failed: {e}, falling back to unstructured")
            return await self._extract_unstructured(prompt)

    async def _extract_unstructured(self, prompt: str) -> ExtractionOutput:
        """Extract using raw LLM output and parse JSON."""
        import json

        full_prompt = prompt + "\n\nRespond with valid JSON only:"

        try:
            response = await self.llm.acomplete(full_prompt)
            text = response.text.strip()

            # Try to extract JSON from response
            # Handle cases where LLM might wrap in markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)
            return ExtractionOutput.model_validate(data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM output as JSON: {e}")
            return ExtractionOutput(entities=[], relationships=[])
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise EntityExtractionError(f"LLM extraction failed: {e}") from e

    def _convert_entities(
        self,
        extracted: list[ExtractedEntity],
        document_id: UUID,
        chunk_id: UUID,
        page_number: int,
    ) -> list[Entity]:
        """Convert extracted entities to our schema."""
        entities: list[Entity] = []

        for ext in extracted:
            # Map string type to enum
            try:
                entity_type = EntityType(ext.entity_type.lower())
            except ValueError:
                entity_type = EntityType.OTHER

            entities.append(
                Entity(
                    entity_type=entity_type,
                    value=ext.value,
                    normalized_value=ext.normalized_value,
                    source_document_id=document_id,
                    source_chunk_id=chunk_id,
                    source_page=page_number,
                    source_text=ext.context,
                    confidence=ext.confidence,
                    extraction_method="llm",
                )
            )

        return entities

    def _convert_relationships(
        self,
        extracted: list[ExtractedRelationship],
        entities: list[Entity],
        document_id: UUID,
        chunk_id: UUID,
        page_number: int,
    ) -> list[Relationship]:
        """Convert extracted relationships to our schema."""
        relationships: list[Relationship] = []

        # Build entity lookup by value
        entity_map: dict[str, Entity] = {e.value.lower(): e for e in entities}

        for ext in extracted:
            # Find source and target entities
            source = entity_map.get(ext.source_entity.lower())
            target = entity_map.get(ext.target_entity.lower())

            if not source or not target:
                logger.debug(
                    f"Relationship entities not found: {ext.source_entity} -> {ext.target_entity}"
                )
                continue

            # Map relationship type
            try:
                rel_type = RelationshipType(ext.relationship_type.lower())
            except ValueError:
                rel_type = RelationshipType.REFERENCES

            relationships.append(
                Relationship(
                    relationship_type=rel_type,
                    source_entity_id=source.entity_id,
                    target_entity_id=target.entity_id,
                    source_document_id=document_id,
                    source_chunk_id=chunk_id,
                    source_page=page_number,
                    source_text=ext.context,
                    confidence=ext.confidence,
                )
            )

        return relationships


# ============================================================================
# Convenience Functions
# ============================================================================


async def extract_from_chunks(
    chunks: list[DocumentChunk],
    llm: Any = None,
) -> tuple[list[Entity], list[Relationship]]:
    """
    Convenience function to extract entities and relationships from chunks.

    Args:
        chunks: Document chunks to process
        llm: Optional LLM instance

    Returns:
        Tuple of (all_entities, all_relationships)
    """
    extractor = EntityExtractor(llm=llm)
    results = await extractor.extract_batch(chunks)

    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []

    for result in results:
        all_entities.extend(result.entities)
        all_relationships.extend(result.relationships)

    return all_entities, all_relationships
