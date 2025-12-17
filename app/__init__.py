"""
Cross-Document Conflict Detector - FastAPI Application.

Provides REST API endpoints for document ingestion,
conflict detection, analysis, and reporting.
"""

from app.config import Settings, get_settings
from app.main import app

__all__ = [
    "app",
    "get_settings",
    "Settings",
]
