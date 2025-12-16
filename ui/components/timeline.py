"""
Timeline Component.

Chronological event display with conflict highlighting.
"""

import streamlit as st
from datetime import datetime

from ui.utils.api_client import get_client


def render_timeline() -> None:
    """Render the timeline view."""
    st.header("📅 Timeline")
    st.caption("Chronological view of events across documents")
    
    # Check for selected documents
    if not st.session_state.selected_document_ids:
        st.info("Select documents first. Go to Upload page.")
        if st.button("📁 Go to Upload"):
            st.session_state.current_page = "upload"
            st.rerun()
        return
    
    # Fetch timeline
    with st.spinner("Building timeline..."):
        try:
            client = get_client()
            timeline_data = client.get_timeline(st.session_state.selected_document_ids)
            
            events = timeline_data.get("events", [])
            conflicts = timeline_data.get("conflicts", [])
            
        except Exception as e:
            st.error(f"Failed to load timeline: {e}")
            return
    
    st.markdown("---")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Events", len(events))
    with col2:
        st.metric("Timeline Conflicts", len(conflicts))
    with col3:
        if events:
            date_range = f"{events[0].get('date', 'N/A')} to {events[-1].get('date', 'N/A')}"
            st.caption(f"Range: {date_range}")
    
    st.markdown("---")
    
    # Controls
    col_filter, col_sort = st.columns(2)
    
    with col_filter:
        event_types = list(set(e.get("event_type", "unknown") for e in events))
        selected_types = st.multiselect(
            "Filter by Event Type",
            options=event_types,
            default=event_types,
        )
    
    with col_sort:
        sort_order = st.radio("Sort", ["Chronological", "Reverse"], horizontal=True)
    
    st.markdown("---")
    
    # Timeline conflicts (if any)
    if conflicts:
        st.subheader("⚠️ Timeline Conflicts")
        for conflict in conflicts:
            with st.container():
                st.markdown(f"""
                <div style="
                    border-left: 4px solid #DC3545;
                    background-color: #262730;
                    padding: 12px;
                    margin-bottom: 8px;
                    border-radius: 4px;
                ">
                    <strong style="color: #DC3545;">{conflict.get('conflict_type', 'Temporal Conflict')}</strong>
                    <p style="color: #B0B0B0; margin: 4px 0;">{conflict.get('description', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Event list
    st.subheader("📋 Events")
    
    # Filter and sort
    filtered_events = [e for e in events if e.get("event_type") in selected_types]
    
    if sort_order == "Reverse":
        filtered_events = list(reversed(filtered_events))
    
    if not filtered_events:
        st.info("No events match the selected filters")
        return
    
    # Render timeline
    for idx, event in enumerate(filtered_events):
        _render_event(event, idx)


def _render_event(event: dict, idx: int) -> None:
    """Render a single timeline event."""
    
    # Event type colors
    type_colors = {
        "employment_start": "#28A745",
        "employment_end": "#DC3545",
        "vesting": "#D4AF37",
        "signing": "#2196F3",
        "amendment": "#9C27B0",
    }
    
    event_type = event.get("event_type", "unknown")
    color = type_colors.get(event_type, "#B0B0B0")
    
    # Format date
    date_str = event.get("date", "Unknown Date")
    
    col_marker, col_content = st.columns([0.1, 0.9])
    
    with col_marker:
        st.markdown(f"""
        <div style="
            width: 12px;
            height: 12px;
            background-color: {color};
            border-radius: 50%;
            margin-top: 8px;
        "></div>
        """, unsafe_allow_html=True)
    
    with col_content:
        st.markdown(f"""
        <div style="
            background-color: #262730;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 8px;
        ">
            <div style="display: flex; justify-content: space-between;">
                <span style="color: {color}; font-weight: bold;">{event_type.replace('_', ' ').title()}</span>
                <span style="color: #888;">{date_str}</span>
            </div>
            <p style="color: #FAFAFA; margin: 8px 0 4px 0;">{event.get('description', 'No description')}</p>
            <span style="color: #666; font-size: 12px;">
                📄 {event.get('source_document', 'Unknown')} | Page {event.get('source_page', '?')}
            </span>
        </div>
        """, unsafe_allow_html=True)
