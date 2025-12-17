"""
Agents Layer - Conflict Detection & Verification.

Two-agent architecture:
1. Comparator Agent - Finds potential conflicts between documents
2. Judge Agent - Verifies conflicts and generates the final report

Plus multi-document analysis, reference detection, and report generation.
"""

from src.agents.comparator import ComparatorAgent
from src.agents.judge import JudgeAgent, detect_and_verify
from src.agents.multi_doc_analyzer import (
    DocumentSet,
    EntityVariation,
    MultiDocAnalyzer,
    MultiDocConflict,
    MultiDocReport,
)
from src.agents.reference_detector import (
    DocumentReference,
    MissingDocumentReport,
    ReferenceDetector,
)
from src.agents.report_generator import (
    ExecutiveSummary,
    ReportConfig,
    ReportGenerator,
)
from src.agents.schemas import (
    ComparisonQuery,
    Conflict,
    ConflictEvidence,
    ConflictReport,
    ConflictSeverity,
    ConflictStatus,
    ConflictType,
    RedFlag,
    SourceCitation,
    VerificationResult,
)

__all__ = [
    # Core Agents
    "ComparatorAgent",
    "JudgeAgent",
    "detect_and_verify",
    # Multi-Doc Analysis
    "MultiDocAnalyzer",
    "MultiDocReport",
    "MultiDocConflict",
    "DocumentSet",
    "EntityVariation",
    # Reference Detection
    "ReferenceDetector",
    "DocumentReference",
    "MissingDocumentReport",
    # Report Generation
    "ReportGenerator",
    "ExecutiveSummary",
    "ReportConfig",
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
