"""
Cross-Document Conflict Detector - Source Package.

This package contains the core functionality for:
- Document ingestion and parsing
- Knowledge graph and vector store management
- Agent-based conflict detection
- Utility functions
"""

from src.agents import (
    ComparatorAgent,
    JudgeAgent,
    MultiDocAnalyzer,
    ReferenceDetector,
    ReportGenerator,
)
from src.ingestion import PDFParser, SemanticChunker
from src.knowledge import (
    CrossReferenceEngine,
    EntityExtractor,
    EntityResolver,
    GraphStore,
    GraphVisualizer,
    TimelineBuilder,
    VectorStore,
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
