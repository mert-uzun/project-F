"""
Reference Detector - Find Missing Document References.

Detects when documents reference other documents that weren't uploaded.
E.g., "pursuant to Exhibit A" when Exhibit A is not in the document set.
"""

import re
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.ingestion.schemas import DocumentChunk, DocumentMetadata
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Reference Patterns
# ============================================================================

# Patterns to detect document references
REFERENCE_PATTERNS = [
    # "pursuant to the Employment Agreement"
    (
        r"(?:pursuant to|as defined in|per|under|in accordance with|"
        r"subject to|referenced in|described in|set forth in)\s+"
        r"(?:the\s+)?([A-Z][A-Za-z\s]+(?:Agreement|Contract|Letter|Plan|Policy))",
        "agreement"
    ),
    # "Exhibit A", "Schedule 1", "Appendix B-1"
    (
        r"(?:Exhibit|Schedule|Appendix|Annex|Attachment)\s+([A-Z0-9][-A-Z0-9]*)",
        "exhibit"
    ),
    # "Section 4.2 of the Operating Agreement"
    (
        r"(?:Section|Article|Clause)\s+(\d+(?:\.\d+)*)\s+of\s+"
        r"(?:the\s+)?([A-Z][A-Za-z\s]+(?:Agreement|Contract))",
        "section"
    ),
    # "Side Letter dated January 1, 2024"
    (
        r"(Side Letter|Amendment|Addendum|Supplement)\s+"
        r"(?:dated\s+)?([A-Za-z]+\s+\d{1,2},?\s+\d{4})?",
        "amendment"
    ),
    # "the Stock Option Plan"
    (
        r"the\s+([A-Z][A-Za-z\s]+(?:Plan|Agreement|Schedule|Policy))",
        "plan"
    ),
]


class ReferenceType(str, Enum):
    """Types of document references."""

    EXHIBIT = "exhibit"
    SCHEDULE = "schedule"
    AGREEMENT = "agreement"
    AMENDMENT = "amendment"
    SECTION = "section"
    PLAN = "plan"
    OTHER = "other"


# ============================================================================
# Schemas
# ============================================================================

class DocumentReference(BaseModel):
    """A reference to another document found in text."""

    reference_id: UUID = Field(default_factory=uuid4)
    reference_text: str = Field(..., description="The reference as found in text")
    reference_type: ReferenceType
    normalized_name: str = Field(..., description="Normalized reference name for matching")

    # Source location
    source_document_id: UUID
    source_document_name: str
    source_page: int
    source_chunk_id: UUID | None = None
    context: str = Field(default="", description="Surrounding text")

    # Match status
    is_uploaded: bool = Field(default=False, description="Whether this doc was uploaded")
    matched_document_id: UUID | None = Field(
        default=None,
        description="ID of matched uploaded document"
    )
    matched_document_name: str | None = None

    # Confidence
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    @property
    def is_missing(self) -> bool:
        """True if referenced but not uploaded."""
        return not self.is_uploaded


class MissingDocumentReport(BaseModel):
    """Report of missing document references."""

    report_id: UUID = Field(default_factory=uuid4)

    # All references found
    total_references: int = 0
    references: list[DocumentReference] = Field(default_factory=list)

    # Missing documents
    missing_documents: list[DocumentReference] = Field(default_factory=list)
    matched_references: list[DocumentReference] = Field(default_factory=list)

    # By type
    missing_by_type: dict[str, int] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def missing_count(self) -> int:
        return len(self.missing_documents)

    @property
    def has_missing(self) -> bool:
        return self.missing_count > 0

    def to_summary(self) -> str:
        """Generate text summary."""
        if not self.has_missing:
            return f"All {self.total_references} document references were resolved."

        summary = f"Found {self.missing_count} missing document references:\n"
        for ref in self.missing_documents[:10]:
            summary += f"  - {ref.reference_text} (referenced in {ref.source_document_name}, p.{ref.source_page})\n"

        if self.missing_count > 10:
            summary += f"  ... and {self.missing_count - 10} more\n"

        return summary


# ============================================================================
# Reference Detector
# ============================================================================

class ReferenceDetector:
    """
    Detect references to external documents.

    Scans document text for patterns like:
    - "Exhibit A"
    - "pursuant to the Employment Agreement"
    - "Side Letter dated January 1, 2024"

    Then matches against uploaded documents to find missing ones.

    Usage:
        detector = ReferenceDetector()
        refs = detector.extract_references(chunk, doc_id, doc_name)
        missing = detector.find_missing_documents(refs, uploaded_docs)
    """

    def __init__(
        self,
        custom_patterns: list[tuple[str, str]] | None = None,
    ) -> None:
        """
        Initialize the Reference Detector.

        Args:
            custom_patterns: Optional additional regex patterns
        """
        self.patterns = list(REFERENCE_PATTERNS)
        if custom_patterns:
            self.patterns.extend(custom_patterns)

        # Compile patterns
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), ref_type)
            for pattern, ref_type in self.patterns
        ]

    def extract_references(
        self,
        chunk: DocumentChunk,
        document_id: UUID,
        document_name: str,
    ) -> list[DocumentReference]:
        """
        Extract document references from a chunk.

        Args:
            chunk: Document chunk to scan
            document_id: Source document ID
            document_name: Source document name

        Returns:
            List of DocumentReference objects
        """
        text = chunk.content
        references: list[DocumentReference] = []
        seen: set[str] = set()

        for pattern, ref_type in self._compiled_patterns:
            for match in pattern.finditer(text):
                # Get the reference text
                ref_text = match.group(0).strip()

                # Get captured groups for normalized name
                groups = match.groups()
                if groups:
                    normalized = " ".join(g for g in groups if g).strip()
                else:
                    normalized = ref_text

                # Skip duplicates
                norm_key = normalized.lower()
                if norm_key in seen:
                    continue
                seen.add(norm_key)

                # Skip self-references (the document referencing itself)
                if self._is_self_reference(normalized, document_name):
                    continue

                # Get context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                references.append(DocumentReference(
                    reference_text=ref_text,
                    reference_type=ReferenceType(ref_type),
                    normalized_name=normalized,
                    source_document_id=document_id,
                    source_document_name=document_name,
                    source_page=chunk.metadata.page_number,
                    source_chunk_id=chunk.metadata.chunk_id,
                    context=f"...{context}...",
                ))

        return references

    def extract_references_from_text(
        self,
        text: str,
        document_id: UUID,
        document_name: str,
        page_number: int = 1,
    ) -> list[DocumentReference]:
        """
        Extract references from raw text (without chunk).

        Args:
            text: Text to scan
            document_id: Source document ID
            document_name: Source document name
            page_number: Page number

        Returns:
            List of DocumentReference objects
        """
        references: list[DocumentReference] = []
        seen: set[str] = set()

        for pattern, ref_type in self._compiled_patterns:
            for match in pattern.finditer(text):
                ref_text = match.group(0).strip()

                groups = match.groups()
                if groups:
                    normalized = " ".join(g for g in groups if g).strip()
                else:
                    normalized = ref_text

                norm_key = normalized.lower()
                if norm_key in seen:
                    continue
                seen.add(norm_key)

                if self._is_self_reference(normalized, document_name):
                    continue

                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                references.append(DocumentReference(
                    reference_text=ref_text,
                    reference_type=ReferenceType(ref_type),
                    normalized_name=normalized,
                    source_document_id=document_id,
                    source_document_name=document_name,
                    source_page=page_number,
                    context=f"...{context}...",
                ))

        return references

    def find_missing_documents(
        self,
        references: list[DocumentReference],
        uploaded_documents: list[DocumentMetadata] | list[dict[str, Any]],
    ) -> MissingDocumentReport:
        """
        Find references that don't match any uploaded document.

        Args:
            references: All extracted references
            uploaded_documents: List of uploaded document metadata

        Returns:
            MissingDocumentReport with missing and matched references
        """
        # Build index of uploaded documents
        uploaded_names: dict[str, tuple[UUID, str]] = {}

        for doc in uploaded_documents:
            if isinstance(doc, dict):
                doc_id = doc.get("document_id") or doc.get("id")
                doc_name = doc.get("filename") or doc.get("name", "")
            else:
                doc_id = doc.document_id
                doc_name = doc.filename

            # Index by various normalizations
            for norm in self._get_normalizations(doc_name):
                uploaded_names[norm] = (doc_id, doc_name)

        # Match references
        missing: list[DocumentReference] = []
        matched: list[DocumentReference] = []

        for ref in references:
            # Try to match
            match_id, match_name = self._find_match(ref, uploaded_names)

            if match_id:
                ref.is_uploaded = True
                ref.matched_document_id = match_id
                ref.matched_document_name = match_name
                matched.append(ref)
            else:
                ref.is_uploaded = False
                missing.append(ref)

        # Count by type
        missing_by_type: dict[str, int] = {}
        for ref in missing:
            type_key = ref.reference_type.value
            missing_by_type[type_key] = missing_by_type.get(type_key, 0) + 1

        report = MissingDocumentReport(
            total_references=len(references),
            references=references,
            missing_documents=missing,
            matched_references=matched,
            missing_by_type=missing_by_type,
        )

        logger.info(
            f"Reference analysis: {len(references)} total, "
            f"{len(matched)} matched, {len(missing)} missing"
        )

        return report

    def _is_self_reference(self, normalized: str, document_name: str) -> bool:
        """Check if reference is to the source document itself."""
        norm_ref = normalized.lower().strip()
        norm_doc = document_name.lower().replace(".pdf", "").replace("_", " ").strip()

        # Exact match
        if norm_ref == norm_doc:
            return True

        # One contains the other
        if norm_ref in norm_doc or norm_doc in norm_ref:
            return True

        return False

    def _get_normalizations(self, name: str) -> list[str]:
        """Get various normalizations of a document name for matching."""
        normalizations: list[str] = []

        # Base normalization
        base = name.lower().replace(".pdf", "").replace("_", " ").replace("-", " ").strip()
        normalizations.append(base)

        # Without common words
        without_common = re.sub(
            r"\b(the|a|an|of|for|and|or)\b",
            "",
            base,
            flags=re.IGNORECASE
        ).strip()
        normalizations.append(without_common)

        # Extract exhibit/schedule identifier
        exhibit_match = re.search(r"exhibit\s*([a-z0-9]+)", base, re.IGNORECASE)
        if exhibit_match:
            normalizations.append(f"exhibit {exhibit_match.group(1)}")
            normalizations.append(exhibit_match.group(1))

        schedule_match = re.search(r"schedule\s*([a-z0-9]+)", base, re.IGNORECASE)
        if schedule_match:
            normalizations.append(f"schedule {schedule_match.group(1)}")

        return normalizations

    def _find_match(
        self,
        ref: DocumentReference,
        uploaded_names: dict[str, tuple[UUID, str]],
    ) -> tuple[UUID | None, str | None]:
        """Try to match a reference to an uploaded document."""
        normalized = ref.normalized_name.lower().strip()

        # Direct match
        if normalized in uploaded_names:
            return uploaded_names[normalized]

        # Try variations
        # "Exhibit A" -> "exhibit a"
        # "the Employment Agreement" -> "employment agreement"
        variations = [
            normalized,
            normalized.replace("the ", ""),
            re.sub(r"\s+", " ", normalized),
        ]

        for var in variations:
            if var in uploaded_names:
                return uploaded_names[var]

        # Fuzzy match - check if any uploaded name contains the reference
        for uploaded_norm, (doc_id, doc_name) in uploaded_names.items():
            if normalized in uploaded_norm or uploaded_norm in normalized:
                return (doc_id, doc_name)

        return (None, None)


def detect_missing_documents(
    chunks: list[DocumentChunk],
    document_id: UUID,
    document_name: str,
    uploaded_documents: list[DocumentMetadata],
) -> MissingDocumentReport:
    """
    Convenience function to detect missing documents from chunks.

    Args:
        chunks: Document chunks to scan
        document_id: Source document ID
        document_name: Source document name
        uploaded_documents: List of all uploaded documents

    Returns:
        MissingDocumentReport
    """
    detector = ReferenceDetector()

    all_references: list[DocumentReference] = []

    for chunk in chunks:
        refs = detector.extract_references(chunk, document_id, document_name)
        all_references.extend(refs)

    return detector.find_missing_documents(all_references, uploaded_documents)
