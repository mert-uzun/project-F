"""
Sidebar Component.

Global navigation and document context.
Already integrated into app.py, this module provides
additional sidebar utilities if needed.
"""

import streamlit as st


def render_document_summary() -> None:
    """Render a compact document summary for the sidebar."""
    docs = st.session_state.get("uploaded_documents", [])
    
    if not docs:
        st.caption("No documents")
        return
    
    processed = len([d for d in docs if d.get("status") == "processed"])
    total = len(docs)
    
    st.caption(f"📂 {processed}/{total} processed")


def render_analysis_status() -> None:
    """Render analysis status indicator."""
    results = st.session_state.get("analysis_results")
    
    if results:
        conflict_count = results.get("conflict_count", 0)
        critical = results.get("critical_conflicts", 0)
        
        if critical > 0:
            st.error(f"⚠️ {critical} critical conflicts")
        elif conflict_count > 0:
            st.warning(f"📊 {conflict_count} conflicts found")
        else:
            st.success("✅ No conflicts")
    else:
        st.caption("Analysis not run")


def render_quick_actions() -> None:
    """Render quick action buttons."""
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⚡ Analyze", use_container_width=True):
            st.session_state.current_page = "analysis"
            st.rerun()
    
    with col2:
        if st.button("📄 Report", use_container_width=True):
            st.session_state.current_page = "report"
            st.rerun()
