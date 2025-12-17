"""
Data Inspector Component.

Split view showing raw PDF vs parsed output.
Proves parsing quality (Table Supremacy).
"""

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer


def render_inspector() -> None:
    """Render the data inspector split view."""
    st.header("🔍 Data Inspector")
    st.caption("Compare raw PDF with parsed output to verify extraction quality")

    # Check for documents
    if not st.session_state.uploaded_documents:
        st.warning("No documents uploaded. Go to Upload to add documents.")
        return

    processed_docs = [
        doc for doc in st.session_state.uploaded_documents
        if doc["status"] == "processed" and doc.get("file_content")
    ]

    if not processed_docs:
        st.warning("No processed documents available.")
        return

    # Document selector
    selected_doc_name = st.selectbox(
        "Select Document",
        options=[doc["name"] for doc in processed_docs],
    )

    selected_doc = next(
        (doc for doc in processed_docs if doc["name"] == selected_doc_name),
        None
    )

    if not selected_doc:
        return

    st.markdown("---")

    # Controls row
    col_page, col_view_mode = st.columns([1, 1])

    with col_page:
        page_num = st.number_input(
            "Page Number",
            min_value=1,
            value=1,
            step=1,
            key="inspector_page",
        )

    with col_view_mode:
        view_mode = st.radio(
            "View Mode",
            options=["Markdown Tables", "Raw JSON", "Text Only"],
            horizontal=True,
        )

    st.markdown("---")

    # Split view
    col_pdf, col_parsed = st.columns([1, 1])

    with col_pdf:
        st.subheader("📄 Raw PDF")
        with st.container():
            st.markdown('<div class="pdf-container">', unsafe_allow_html=True)
            try:
                pdf_viewer(
                    selected_doc["file_content"],
                    width=600,
                    height=700,
                    pages_to_render=[page_num],
                )
            except Exception as e:
                st.error(f"Could not render PDF: {e}")
                st.info("PDF preview not available. The document was still processed successfully.")
            st.markdown('</div>', unsafe_allow_html=True)

    with col_parsed:
        st.subheader("📊 Parsed Output")

        # Fetch parsed data for this page (mock for now, would call API)
        _render_parsed_output(selected_doc, page_num, view_mode)


def _render_parsed_output(doc: dict, page_num: int, view_mode: str) -> None:
    """Render the parsed output for a document page."""

    # This would normally fetch from an API endpoint
    # For now, show document metadata and placeholder

    with st.container():
        st.markdown('<div class="inspector-panel">', unsafe_allow_html=True)

        # Document stats
        st.metric("Chunks Extracted", doc.get("chunks", 0))
        st.metric("Entities Found", doc.get("entities", 0))

        st.markdown("---")

        if view_mode == "Markdown Tables":
            st.markdown("""
            ### Extracted Tables

            | Field | Value |
            |-------|-------|
            | Document ID | `{doc_id}` |
            | Page | {page} |
            | Status | ✅ Processed |

            > **Note**: Full table extraction data would appear here
            > after API integration is complete.
            """.format(doc_id=doc.get("id", "N/A")[:8], page=page_num))

        elif view_mode == "Raw JSON":
            st.json({
                "document_id": doc.get("id"),
                "filename": doc.get("name"),
                "page": page_num,
                "chunk_count": doc.get("chunks", 0),
                "entity_count": doc.get("entities", 0),
                "status": doc.get("status"),
            })

        else:  # Text Only
            st.text_area(
                "Extracted Text",
                value=f"[Page {page_num} text would appear here after full API integration]",
                height=400,
                disabled=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)
