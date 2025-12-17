"""
Pydantic Schemas for Agents Layer.

Models for conflict detection, red flag reports, and verification results.
"""

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.knowledge.schemas import Entity


class ConflictType(str, Enum):
    """Types of conflicts we can detect."""

    # Value mismatches
    SALARY_MISMATCH = "salary_mismatch"
    EQUITY_MISMATCH = "equity_mismatch"
    PERCENTAGE_MISMATCH = "percentage_mismatch"
    AMOUNT_MISMATCH = "amount_mismatch"

    # Date conflicts
    DATE_CONFLICT = "date_conflict"
    DURATION_CONFLICT = "duration_conflict"
    TIMELINE_INCONSISTENCY = "timeline_inconsistency"

    # Entity conflicts
    ROLE_CONFLICT = "role_conflict"
    PARTY_MISMATCH = "party_mismatch"
    ENTITY_AMBIGUITY = "entity_ambiguity"

    # Clause conflicts
    CONTRADICTORY_CLAUSES = "contradictory_clauses"
    MISSING_CLAUSE = "missing_clause"
    OVERLAPPING_OBLIGATIONS = "overlapping_obligations"

    # General
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    OTHER = "other"


class ConflictSeverity(str, Enum):
    """Severity levels for conflicts."""

    LOW = "low"           # Minor discrepancy, likely formatting
    MEDIUM = "medium"     # Notable difference, needs review
    HIGH = "high"         # Significant conflict, potential risk
    CRITICAL = "critical" # Major contradiction, deal-breaker


class ConflictStatus(str, Enum):
    """Status of a conflict in the verification pipeline."""

    DETECTED = "detected"      # Just found by Comparator
    VERIFIED = "verified"      # Confirmed by Judge as real
    REJECTED = "rejected"      # Rejected by Judge as false positive
    NEEDS_REVIEW = "needs_review"  # Judge uncertain, needs human


class SourceCitation(BaseModel):
    """Citation for where information was found."""

    document_id: UUID = Field(..., description="Source document")
    document_name: str = Field(..., description="Document filename")
    page_number: int = Field(..., ge=1, description="Page number")
    chunk_id: UUID = Field(..., description="Chunk containing this info")
    excerpt: str = Field(..., description="Relevant text excerpt")


class ConflictEvidence(BaseModel):
    """Evidence supporting a conflict detection."""

    entity: Entity = Field(..., description="The entity involved")
    citation: SourceCitation = Field(..., description="Where it was found")
    extracted_value: str = Field(..., description="The value we extracted")
    normalized_value: str | None = Field(None, description="Normalized for comparison")


class Conflict(BaseModel):
    """
    A detected conflict between documents.

    This is the core output of the Comparator Agent.
    """

    conflict_id: UUID = Field(default_factory=uuid4)
    conflict_type: ConflictType = Field(..., description="Type of conflict")
    severity: ConflictSeverity = Field(default=ConflictSeverity.MEDIUM)
    status: ConflictStatus = Field(default=ConflictStatus.DETECTED)

    # What's the conflict?
    title: str = Field(..., description="Brief conflict title")
    description: str = Field(..., description="Detailed conflict description")

    # Evidence from each side
    evidence_a: ConflictEvidence = Field(..., description="First source")
    evidence_b: ConflictEvidence = Field(..., description="Conflicting source")

    # Comparison details
    value_a: str = Field(..., description="Value from source A")
    value_b: str = Field(..., description="Value from source B")
    difference: str | None = Field(None, description="Quantified difference if applicable")

    # Metadata
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    # Verification (filled by Judge)
    verified_at: datetime | None = Field(default=None)
    verification_notes: str | None = Field(default=None)
    is_false_positive: bool = Field(default=False)


class VerificationResult(BaseModel):
    """Result of Judge verification for a single conflict."""

    conflict_id: UUID = Field(..., description="ID of conflict being verified")
    is_valid: bool = Field(..., description="Is this a real conflict?")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Judge's confidence")
    reasoning: str = Field(..., description="Why the Judge made this decision")

    # Updated severity/status
    updated_severity: ConflictSeverity | None = Field(None)
    updated_status: ConflictStatus = Field(...)

    # Additional context
    additional_context: str | None = Field(None, description="Extra info found")
    recommendations: list[str] = Field(default_factory=list)


class RedFlag(BaseModel):
    """
    A verified red flag for the final report.

    Red flags are conflicts that passed Judge verification.
    """

    flag_id: UUID = Field(default_factory=uuid4)
    conflict: Conflict = Field(..., description="The underlying conflict")
    verification: VerificationResult = Field(..., description="Judge's verification")

    # Report-ready fields
    summary: str = Field(..., description="One-line summary for executives")
    impact: str = Field(..., description="Potential business impact")
    recommended_action: str = Field(..., description="What to do about it")

    # Priority for report ordering
    priority: int = Field(default=1, ge=1, le=5, description="1=highest, 5=lowest")


class ConflictReport(BaseModel):
    """
    Final conflict detection report.

    Contains all red flags, organized by severity.
    """

    report_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Documents analyzed
    document_ids: list[UUID] = Field(..., description="Documents compared")
    document_names: list[str] = Field(..., description="Document filenames")

    # Summary stats
    total_conflicts_detected: int = Field(default=0)
    total_verified: int = Field(default=0)
    total_rejected: int = Field(default=0)

    # The findings
    red_flags: list[RedFlag] = Field(default_factory=list)

    # Grouped by severity
    @property
    def critical_flags(self) -> list[RedFlag]:
        return [f for f in self.red_flags if f.conflict.severity == ConflictSeverity.CRITICAL]

    @property
    def high_flags(self) -> list[RedFlag]:
        return [f for f in self.red_flags if f.conflict.severity == ConflictSeverity.HIGH]

    @property
    def medium_flags(self) -> list[RedFlag]:
        return [f for f in self.red_flags if f.conflict.severity == ConflictSeverity.MEDIUM]

    @property
    def low_flags(self) -> list[RedFlag]:
        return [f for f in self.red_flags if f.conflict.severity == ConflictSeverity.LOW]

    def to_summary(self) -> str:
        """Generate executive summary."""
        return (
            f"Conflict Report: {len(self.red_flags)} issues found across "
            f"{len(self.document_names)} documents.\n"
            f"Critical: {len(self.critical_flags)} | High: {len(self.high_flags)} | "
            f"Medium: {len(self.medium_flags)} | Low: {len(self.low_flags)}"
        )


class ComparisonQuery(BaseModel):
    """Query for what to compare between documents."""

    query_id: UUID = Field(default_factory=uuid4)
    document_ids: list[UUID] = Field(..., min_length=2, description="Documents to compare")

    # What to focus on
    focus_areas: list[str] = Field(
        default_factory=lambda: ["salary", "equity", "dates", "parties"],
        description="Areas to focus comparison on"
    )

    # Specific query
    specific_query: str | None = Field(
        None,
        description="Specific thing to look for, e.g., 'Compare CEO compensation'"
    )

    # Thresholds
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    include_low_severity: bool = Field(default=False)
