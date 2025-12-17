"""
Conflict Workbench Component.

Master-detail split view with conflict cards and PDF viewer.
"""

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer


def render_conflicts() -> None:
    """Render the conflict workbench with master-detail split."""
    st.header("⚠️ Conflict Workbench")

    # Check for conflicts
    if not st.session_state.conflicts:
        st.info("No conflicts detected yet. Run analysis first.")
        if st.button("⚡ Go to Analysis"):
            st.session_state.current_page = "analysis"
            st.rerun()
        return

    conflicts = st.session_state.conflicts
    st.caption(f"{len(conflicts)} conflicts detected")

    st.markdown("---")

    # Master-detail split (40/60)
    col_list, col_detail = st.columns([0.4, 0.6])

    with col_list:
        st.subheader("📋 Conflict Feed")
        _render_conflict_list(conflicts)

    with col_detail:
        st.subheader("📄 Document View")
        _render_conflict_detail()


def _render_conflict_list(conflicts: list[dict]) -> None:
    """Render the scrollable list of conflict cards."""

    for idx, conflict in enumerate(conflicts):
        is_selected = idx == st.session_state.current_conflict_idx

        # Severity styling
        severity = conflict.get("severity", "medium").lower()
        severity_colors = {
            "critical": "#DC3545",
            "high": "#FF6B35",
            "medium": "#FFC107",
            "low": "#28A745",
        }
        border_color = severity_colors.get(severity, "#FFC107")
        bg_color = "#363740" if is_selected else "#262730"

        # Render conflict card
        card_html = f"""
        <div style="
            border-left: 4px solid {border_color};
            background-color: {bg_color};
            padding: 12px;
            margin-bottom: 8px;
            border-radius: 4px;
            cursor: pointer;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold; color: #FAFAFA;">{conflict.get("title", "Conflict")[:40]}</span>
                <span style="
                    background-color: {border_color};
                    color: {"black" if severity == "medium" else "white"};
                    padding: 2px 8px;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: bold;
                ">{severity.upper()}</span>
            </div>
            <div style="color: #B0B0B0; font-size: 13px; margin-top: 6px;">
                {conflict.get("type", "value_conflict")} | {conflict.get("document_count", 2)} docs
            </div>
            <div style="color: #888; font-size: 12px; margin-top: 4px;">
                {conflict.get("description", "")[:80]}...
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # Button to select this conflict
        if st.button("View Details", key=f"conflict_{idx}", use_container_width=True):
            st.session_state.current_conflict_idx = idx
            # Try to navigate to the source page
            if conflict.get("source_page"):
                st.session_state.pdf_page = conflict["source_page"]
            st.rerun()


def _render_conflict_detail() -> None:
    """Render the detail view for the selected conflict."""
    conflicts = st.session_state.conflicts
    idx = st.session_state.current_conflict_idx

    if idx >= len(conflicts):
        st.info("Select a conflict from the list")
        return

    conflict = conflicts[idx]

    # Conflict detail card
    st.markdown(f"""
    ### {conflict.get("title", "Conflict")}

    **Type:** {conflict.get("type", "N/A")}
    **Severity:** {conflict.get("severity", "N/A").upper()}
    **Documents:** {conflict.get("document_count", 0)}
    """)

    st.markdown("---")

    # Values comparison
    st.markdown("#### 📊 Value Comparison")

    unique_values = conflict.get("unique_values", [])
    if unique_values:
        for i, value in enumerate(unique_values[:5]):
            col_val, col_doc = st.columns([2, 1])
            with col_val:
                st.code(value)
            with col_doc:
                st.caption(f"Document {i + 1}")
    else:
        st.caption("No value details available")

    st.markdown("---")

    # Description
    st.markdown("#### 📝 Description")
    st.write(conflict.get("description", "No description available"))

    st.markdown("---")

    # PDF Viewer (if we have the document)
    _render_source_pdf()


def _render_source_pdf() -> None:
    """Render PDF viewer for the source document."""
    st.markdown("#### 📄 Source Document")

    # Get documents with file content
    docs_with_content = [
        doc
        for doc in st.session_state.uploaded_documents
        if doc.get("file_content") and doc.get("id") in st.session_state.selected_document_ids
    ]

    if not docs_with_content:
        st.caption("PDF preview not available")
        return

    # Document selector
    selected_doc_name = st.selectbox(
        "Select Document",
        options=[doc["name"] for doc in docs_with_content],
        key="conflict_pdf_selector",
    )

    selected_doc = next(
        (doc for doc in docs_with_content if doc["name"] == selected_doc_name), None
    )

    if not selected_doc:
        return

    # Page selector
    page = st.number_input(
        "Page",
        min_value=1,
        value=st.session_state.get("pdf_page", 1),
        key="conflict_pdf_page",
    )

    # Render PDF
    st.markdown('<div class="pdf-container">', unsafe_allow_html=True)
    try:
        pdf_viewer(
            selected_doc["file_content"],
            width=650,
            height=600,
            pages_to_render=[page],
        )
    except Exception as e:
        st.error(f"Could not render PDF: {e}")
    st.markdown("</div>", unsafe_allow_html=True)
