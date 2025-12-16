"""
UI Components Package.

All Streamlit UI components for the conflict detector.
"""

from ui.components.upload import render_upload
from ui.components.inspector import render_inspector
from ui.components.analysis import render_analysis
from ui.components.conflicts import render_conflicts
from ui.components.graph import render_graph
from ui.components.timeline import render_timeline
from ui.components.report import render_report
from ui.components.sidebar import (
    render_document_summary,
    render_analysis_status,
    render_quick_actions,
)

__all__ = [
    # Main render functions
    "render_upload",
    "render_inspector",
    "render_analysis",
    "render_conflicts",
    "render_graph",
    "render_timeline",
    "render_report",
    # Sidebar utilities
    "render_document_summary",
    "render_analysis_status",
    "render_quick_actions",
]
