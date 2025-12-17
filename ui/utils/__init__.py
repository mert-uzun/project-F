"""
UI Utilities Package.

Helper functions and API client for the Streamlit UI.
"""

from ui.utils.api_client import APIClient, get_client
from ui.utils.formatters import (
    format_conflict_card,
    format_date,
    format_entity_type,
    format_file_size,
    format_log_line,
    format_metric_card,
    format_severity_badge,
    format_timestamp,
    truncate_text,
)

__all__ = [
    # API Client
    "APIClient",
    "get_client",
    # Formatters
    "format_severity_badge",
    "format_timestamp",
    "format_date",
    "format_file_size",
    "truncate_text",
    "format_entity_type",
    "format_conflict_card",
    "format_metric_card",
    "format_log_line",
]
