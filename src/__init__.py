"""
Cross-Document Conflict Detector - Source Package.

This package contains the core functionality for:
- Document ingestion and parsing
- Knowledge graph and vector store management
- Agent-based conflict detection
- Utility functions
"""

from src.ingestion import PDFParser, SemanticChunker
from src.knowledge import (
    VectorStore,
    GraphStore,
    EntityExtractor,
    EntityResolver,
    CrossReferenceEngine,
    TimelineBuilder,
    GraphVisualizer,
)
from src.agents import (
    ComparatorAgent,
    JudgeAgent,
    MultiDocAnalyzer,
    ReferenceDetector,
    ReportGenerator,
)

__all__ = [
    # Ingestion
    "PDFParser",
    "SemanticChunker",
    # Knowledge
    "VectorStore",
    "GraphStore",
    "EntityExtractor",
    "EntityResolver",
    "CrossReferenceEngine",
    "TimelineBuilder",
    "GraphVisualizer",
    # Agents
    "ComparatorAgent",
    "JudgeAgent",
    "MultiDocAnalyzer",
    "ReferenceDetector",
    "ReportGenerator",
]
