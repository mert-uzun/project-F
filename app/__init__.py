"""
Cross-Document Conflict Detector - FastAPI Application.

Provides REST API endpoints for document ingestion,
conflict detection, analysis, and reporting.
"""

from app.main import app
from app.config import get_settings, Settings

__all__ = [
    "app",
    "get_settings",
    "Settings",
]
