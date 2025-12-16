"""
Display Formatting Helpers.

Utility functions for formatting data in the Streamlit UI.
"""

from datetime import datetime
from typing import Any


def format_severity_badge(severity: str) -> str:
    """Generate HTML for a severity badge."""
    colors = {
        "critical": ("#DC3545", "white"),
        "high": ("#FF6B35", "white"),
        "medium": ("#FFC107", "black"),
        "low": ("#28A745", "white"),
    }
    bg_color, text_color = colors.get(severity.lower(), ("#888", "white"))
    
    return f"""
    <span style="
        background-color: {bg_color};
        color: {text_color};
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
        text-transform: uppercase;
    ">{severity.upper()}</span>
    """


def format_timestamp(dt: datetime | str | None) -> str:
    """Format a datetime for display."""
    if dt is None:
        return "N/A"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return dt
    return dt.strftime("%Y-%m-%d %H:%M")


def format_date(dt: datetime | str | None) -> str:
    """Format a date for display."""
    if dt is None:
        return "N/A"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return dt
    return dt.strftime("%B %d, %Y")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_entity_type(entity_type: str) -> str:
    """Format entity type for display."""
    return entity_type.replace("_", " ").title()


def format_conflict_card(conflict: dict[str, Any]) -> str:
    """Generate HTML for a conflict card."""
    severity = conflict.get("severity", "medium").lower()
    severity_colors = {
        "critical": "#DC3545",
        "high": "#FF6B35",
        "medium": "#FFC107",
        "low": "#28A745",
    }
    border_color = severity_colors.get(severity, "#FFC107")
    
    title = truncate_text(conflict.get("title", "Conflict"), 40)
    description = truncate_text(conflict.get("description", ""), 80)
    doc_count = conflict.get("document_count", 2)
    conflict_type = conflict.get("type", "value_conflict")
    
    return f"""
    <div class="conflict-card" style="border-left-color: {border_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: bold; color: #FAFAFA;">{title}</span>
            {format_severity_badge(severity)}
        </div>
        <div style="color: #B0B0B0; font-size: 13px; margin-top: 6px;">
            {conflict_type} | {doc_count} docs
        </div>
        <div style="color: #888; font-size: 12px; margin-top: 4px;">
            {description}
        </div>
    </div>
    """


def format_metric_card(value: Any, label: str, color: str = "#D4AF37") -> str:
    """Generate HTML for a metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {color};">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def format_log_line(log_type: str, message: str, timestamp: datetime | None = None) -> str:
    """Format a reasoning log line."""
    if timestamp is None:
        timestamp = datetime.now()
    
    ts_str = timestamp.strftime("%H:%M:%S.%f")[:-3]
    
    return f"""
    <div class="log-line">
        <span class="log-timestamp">[{ts_str}]</span>
        <span class="log-type-{log_type}">[{log_type}]</span>
        {message}
    </div>
    """
