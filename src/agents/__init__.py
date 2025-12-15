"""
Agents Layer - Conflict Detection & Verification.

Two-agent architecture:
1. Comparator Agent - Finds potential conflicts between documents
2. Judge Agent - Verifies conflicts and generates the final report
"""

from src.agents.comparator import ComparatorAgent
from src.agents.judge import JudgeAgent, detect_and_verify
from src.agents.schemas import (
    Conflict,
    ConflictEvidence,
    ConflictReport,
    ConflictSeverity,
    ConflictStatus,
    ConflictType,
    ComparisonQuery,
    RedFlag,
    SourceCitation,
    VerificationResult,
)

__all__ = [
    # Agents
    "ComparatorAgent",
    "JudgeAgent",
    "detect_and_verify",
    # Schemas
    "Conflict",
    "ConflictEvidence",
    "ConflictReport",
    "ConflictSeverity",
    "ConflictStatus",
    "ConflictType",
    "ComparisonQuery",
    "RedFlag",
    "SourceCitation",
    "VerificationResult",
]
