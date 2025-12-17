"""
Entity Resolver - Deduplicate and Merge Entity Aliases.

Recognizes that "John Doe", "J. Doe", "Mr. Doe" are the same person.
Uses rule-based matching, embedding similarity, and context analysis.
"""

import re
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.knowledge.graph_store import GraphStore
from src.knowledge.schemas import Entity, EntityType
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_SIMILARITY_THRESHOLD = 0.85

# Common titles to strip for matching
TITLES = {
    "mr",
    "mrs",
    "ms",
    "miss",
    "dr",
    "prof",
    "professor",
    "sir",
    "madam",
    "lord",
    "lady",
    "hon",
    "honorable",
    "ceo",
    "cfo",
    "coo",
    "cto",
    "cio",
    "president",
    "chairman",
    "director",
    "manager",
    "partner",
    "associate",
    "vp",
    "vice president",
    "executive",
    "chief",
    "senior",
    "junior",
}

# Common organizational suffixes
ORG_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "llc",
    "llp",
    "ltd",
    "limited",
    "plc",
    "co",
    "company",
    "group",
    "holdings",
    "partners",
    "associates",
    "international",
    "global",
}


# ============================================================================
# Schemas
# ============================================================================


class ResolvedEntity(BaseModel):
    """An entity with resolved aliases."""

    canonical_id: UUID = Field(default_factory=uuid4)
    canonical_value: str = Field(..., description="Primary/canonical name")
    aliases: list[str] = Field(default_factory=list, description="All name variations")
    entity_type: EntityType = Field(...)
    source_entity_ids: list[UUID] = Field(default_factory=list, description="Original entity IDs")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    merge_reason: str = Field(default="", description="Why entities were merged")

    @property
    def total_occurrences(self) -> int:
        """Total number of source entities merged."""
        return len(self.source_entity_ids)


class EntityMatch(BaseModel):
    """A potential match between two entities."""

    entity_a: Entity
    entity_b: Entity
    similarity_score: float = Field(ge=0.0, le=1.0)
    match_type: str = Field(description="Type of match: exact, case, initial, title, semantic")
    reasoning: str = Field(default="")


# ============================================================================
# Entity Resolver
# ============================================================================


class EntityResolver:
    """
    Resolve entity aliases and duplicates.

    Uses multiple matching strategies:
    1. Exact match (case-insensitive)
    2. Initial matching ("John Doe" ≈ "J. Doe")
    3. Title stripping ("Mr. John Doe" ≈ "John Doe")
    4. Semantic similarity (embedding-based)
    5. Context matching (same role/org = likely same person)

    Usage:
        resolver = EntityResolver(graph_store)
        resolved = resolver.resolve_entities(entities, threshold=0.85)
    """

    def __init__(
        self,
        graph_store: GraphStore,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        """
        Initialize the Entity Resolver.

        Args:
            graph_store: Graph store for querying relationships
            similarity_threshold: Minimum score for auto-merge (0-1)
        """
        self.graph_store = graph_store
        self.similarity_threshold = similarity_threshold
        self._embedding_model = None

    @property
    def embedding_model(self):
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            try:
                from src.utils.llm_factory import get_embedding_model

                self._embedding_model = get_embedding_model()
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
        return self._embedding_model

    def resolve_entities(
        self,
        entities: list[Entity],
        threshold: float | None = None,
    ) -> list[ResolvedEntity]:
        """
        Group entities that refer to the same thing.

        Args:
            entities: List of entities to resolve
            threshold: Override similarity threshold

        Returns:
            List of resolved entities with merged aliases
        """
        if not entities:
            return []

        threshold = threshold or self.similarity_threshold

        # Group by entity type first
        by_type: dict[EntityType, list[Entity]] = {}
        for entity in entities:
            by_type.setdefault(entity.entity_type, []).append(entity)

        resolved: list[ResolvedEntity] = []

        for entity_type, type_entities in by_type.items():
            type_resolved = self._resolve_type_group(type_entities, threshold)
            resolved.extend(type_resolved)

        logger.info(f"Resolved {len(entities)} entities into {len(resolved)} canonical entities")

        return resolved

    def _resolve_type_group(
        self,
        entities: list[Entity],
        threshold: float,
    ) -> list[ResolvedEntity]:
        """Resolve entities of the same type."""
        if not entities:
            return []

        # Build similarity matrix
        n = len(entities)
        merged: set[int] = set()
        groups: list[list[int]] = []

        for i in range(n):
            if i in merged:
                continue

            group = [i]
            merged.add(i)

            for j in range(i + 1, n):
                if j in merged:
                    continue

                score, match_type, reason = self._compute_similarity(entities[i], entities[j])

                if score >= threshold:
                    group.append(j)
                    merged.add(j)
                    logger.debug(
                        f"Merged '{entities[i].value}' with '{entities[j].value}' "
                        f"(score={score:.2f}, type={match_type})"
                    )

            groups.append(group)

        # Convert groups to ResolvedEntity
        resolved: list[ResolvedEntity] = []
        for group in groups:
            group_entities = [entities[i] for i in group]
            resolved.append(self._create_resolved_entity(group_entities))

        return resolved

    def _compute_similarity(
        self,
        entity_a: Entity,
        entity_b: Entity,
    ) -> tuple[float, str, str]:
        """
        Compute similarity between two entities.

        Returns:
            Tuple of (score, match_type, reasoning)
        """
        value_a = entity_a.value.strip()
        value_b = entity_b.value.strip()

        # 1. Exact match (case-insensitive)
        if value_a.lower() == value_b.lower():
            return (0.98, "exact", "Case-insensitive exact match")

        # 2. Normalized match (strip titles, punctuation)
        norm_a = self._normalize_name(value_a)
        norm_b = self._normalize_name(value_b)

        if norm_a == norm_b:
            return (0.95, "normalized", "Match after normalization")

        # 3. Initial matching for PERSON entities
        if entity_a.entity_type == EntityType.PERSON:
            initial_score = self._match_initials(value_a, value_b)
            if initial_score >= 0.85:
                return (initial_score, "initial", "Initial/abbreviation match")

        # 4. Organization suffix matching
        if entity_a.entity_type == EntityType.ORGANIZATION:
            org_score = self._match_organizations(value_a, value_b)
            if org_score >= 0.85:
                return (org_score, "org_suffix", "Organization name match")

        # 5. Monetary amount matching
        if entity_a.entity_type in (EntityType.MONETARY_AMOUNT, EntityType.SALARY):
            amount_score = self._match_amounts(entity_a, entity_b)
            if amount_score >= 0.99:
                return (amount_score, "amount", "Same monetary value")

        # 6. Token overlap (Jaccard similarity)
        token_score = self._token_similarity(norm_a, norm_b)
        if token_score >= 0.70:
            return (token_score, "token", f"Token overlap: {token_score:.2f}")

        # 7. Semantic similarity (if embeddings available)
        if self.embedding_model and len(value_a) > 3 and len(value_b) > 3:
            semantic_score = self._semantic_similarity(value_a, value_b)
            if semantic_score >= 0.85:
                return (semantic_score, "semantic", "Embedding similarity")

        return (0.0, "none", "No match found")

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for comparison."""
        # Lowercase
        name = name.lower()

        # Remove punctuation
        name = re.sub(r"[^\w\s]", "", name)

        # Strip titles
        words = name.split()
        words = [w for w in words if w not in TITLES]

        # Remove organization suffixes
        words = [w for w in words if w not in ORG_SUFFIXES]

        return " ".join(words).strip()

    def _match_initials(self, name_a: str, name_b: str) -> float:
        """
        Match names with initials.

        "John Doe" ≈ "J. Doe" (0.88)
        "John Smith Doe" ≈ "J. S. Doe" (0.88)
        "J. Doe" ≈ "John D." - doesn't match (different initial position)
        """
        # Normalize
        a_parts = self._normalize_name(name_a).split()
        b_parts = self._normalize_name(name_b).split()

        if not a_parts or not b_parts:
            return 0.0

        # Check if one is abbreviated version of other
        if len(a_parts) != len(b_parts):
            return 0.0

        matches = 0
        for a_part, b_part in zip(a_parts, b_parts):
            if a_part == b_part:
                matches += 1
            elif len(a_part) == 1 and b_part.startswith(a_part):
                matches += 0.9  # Initial match
            elif len(b_part) == 1 and a_part.startswith(b_part):
                matches += 0.9  # Initial match
            else:
                return 0.0  # Mismatch

        return matches / len(a_parts)

    def _match_organizations(self, org_a: str, org_b: str) -> float:
        """Match organization names ignoring suffixes."""
        norm_a = self._normalize_name(org_a)
        norm_b = self._normalize_name(org_b)

        if norm_a == norm_b:
            return 0.95

        # Check if one is prefix of other
        if norm_a.startswith(norm_b) or norm_b.startswith(norm_a):
            return 0.88

        return self._token_similarity(norm_a, norm_b)

    def _match_amounts(self, entity_a: Entity, entity_b: Entity) -> float:
        """Match monetary amounts by normalized value."""
        # Use normalized_value if available
        val_a = entity_a.normalized_value or entity_a.value
        val_b = entity_b.normalized_value or entity_b.value

        try:
            # Parse to float
            num_a = self._parse_amount(val_a)
            num_b = self._parse_amount(val_b)

            if num_a is not None and num_b is not None:
                if num_a == num_b:
                    return 1.0
                # Allow small rounding differences
                if abs(num_a - num_b) / max(num_a, num_b) < 0.001:
                    return 0.99
        except Exception:
            pass

        return 0.0

    def _parse_amount(self, value: str) -> float | None:
        """Parse monetary amount string to float."""
        if isinstance(value, (int, float)):
            return float(value)

        # Remove currency symbols and commas
        clean = re.sub(r"[$€£¥,]", "", str(value))

        # Handle K/M/B suffixes
        multipliers = {"k": 1000, "m": 1000000, "b": 1000000000}

        match = re.match(r"^([\d.]+)\s*([kmb])?$", clean.lower())
        if match:
            num = float(match.group(1))
            suffix = match.group(2)
            if suffix:
                num *= multipliers.get(suffix, 1)
            return num

        try:
            return float(clean)
        except ValueError:
            return None

    def _token_similarity(self, text_a: str, text_b: str) -> float:
        """Compute Jaccard similarity of word tokens."""
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)

        return intersection / union if union > 0 else 0.0

    def _semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Compute embedding-based semantic similarity."""
        if not self.embedding_model:
            return 0.0

        try:
            emb_a = self.embedding_model.get_text_embedding(text_a)
            emb_b = self.embedding_model.get_text_embedding(text_b)

            # Cosine similarity
            import numpy as np

            dot = np.dot(emb_a, emb_b)
            norm_a = np.linalg.norm(emb_a)
            norm_b = np.linalg.norm(emb_b)

            if norm_a > 0 and norm_b > 0:
                return float(dot / (norm_a * norm_b))
        except Exception as e:
            logger.warning(f"Semantic similarity failed: {e}")

        return 0.0

    def _create_resolved_entity(
        self,
        entities: list[Entity],
    ) -> ResolvedEntity:
        """Create a ResolvedEntity from a group of matching entities."""
        # Use most common value as canonical, or longest if tie
        value_counts: dict[str, int] = {}
        for e in entities:
            value_counts[e.value] = value_counts.get(e.value, 0) + 1

        max_count = max(value_counts.values())
        candidates = [v for v, c in value_counts.items() if c == max_count]

        # Prefer longest (most complete) name
        canonical = max(candidates, key=len)

        # Collect all unique aliases
        aliases = list(set(e.value for e in entities if e.value != canonical))

        # Average confidence
        avg_confidence = sum(e.confidence for e in entities) / len(entities)

        return ResolvedEntity(
            canonical_value=canonical,
            aliases=sorted(aliases),
            entity_type=entities[0].entity_type,
            source_entity_ids=[e.entity_id for e in entities],
            confidence=avg_confidence,
            merge_reason=f"Merged {len(entities)} occurrences",
        )

    def find_matches(
        self,
        entity: Entity,
        candidates: list[Entity],
        threshold: float | None = None,
    ) -> list[EntityMatch]:
        """
        Find all entities that match a given entity.

        Args:
            entity: Entity to match against
            candidates: List of candidate entities
            threshold: Minimum similarity score

        Returns:
            List of matches sorted by score (descending)
        """
        threshold = threshold or self.similarity_threshold
        matches: list[EntityMatch] = []

        for candidate in candidates:
            if candidate.entity_id == entity.entity_id:
                continue

            score, match_type, reason = self._compute_similarity(entity, candidate)

            if score >= threshold:
                matches.append(
                    EntityMatch(
                        entity_a=entity,
                        entity_b=candidate,
                        similarity_score=score,
                        match_type=match_type,
                        reasoning=reason,
                    )
                )

        return sorted(matches, key=lambda m: m.similarity_score, reverse=True)

    def merge_in_graph(
        self,
        resolved_entities: list[ResolvedEntity],
    ) -> None:
        """
        Merge resolved entities in the graph store.

        Updates relationships to point to canonical entity.
        """
        for resolved in resolved_entities:
            if len(resolved.source_entity_ids) <= 1:
                continue  # No merge needed

            # Keep the first entity as canonical
            canonical_id = resolved.source_entity_ids[0]

            for alias_id in resolved.source_entity_ids[1:]:
                self.graph_store.merge_entities(
                    keep_id=canonical_id,
                    merge_id=alias_id,
                )

        logger.info(f"Merged {len(resolved_entities)} entity groups in graph")
