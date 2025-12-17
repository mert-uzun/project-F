"""
Document Upload Component.

Drag-and-drop PDF upload with status indicators.
"""

import tempfile
from pathlib import Path

import streamlit as st

from ui.utils.api_client import get_client


def render_upload() -> None:
    """Render the document upload interface."""
    st.header("📁 Document Upload")
    st.caption("Upload PDF documents for conflict analysis")

    # API status indicator
    client = get_client()
    col_status, col_spacer = st.columns([1, 3])
    with col_status:
        if client.is_healthy():
            st.success("🟢 API Connected")
        else:
            st.error("🔴 API Offline")
            st.caption("Start the API: `python -m uvicorn app.main:app`")

    st.markdown("---")

    # File uploader
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        help="Upload employment agreements, offer letters, or other due diligence documents",
    )

    if uploaded_files:
        st.subheader("📋 Files to Process")

        # Show files ready for upload
        for idx, file in enumerate(uploaded_files):
            col_name, col_size, col_action = st.columns([3, 1, 1])

            with col_name:
                st.text(f"📄 {file.name}")
            with col_size:
                size_mb = file.size / (1024 * 1024)
                st.caption(f"{size_mb:.2f} MB")
            with col_action:
                # Check if already uploaded
                already_uploaded = any(
                    doc["name"] == file.name for doc in st.session_state.uploaded_documents
                )
                if already_uploaded:
                    st.caption("✅ Done")

        # Upload button
        st.markdown("---")

        if st.button("⬆️ Upload All Documents", type="primary", use_container_width=True):
            _process_uploads(uploaded_files)

    # Show uploaded documents
    st.markdown("---")
    st.subheader("📂 Uploaded Documents")

    if st.session_state.uploaded_documents:
        _render_document_list()
    else:
        st.info("No documents uploaded yet. Use the uploader above to get started.")


def _process_uploads(files: list) -> None:
    """Process and upload files to the API."""
    client = get_client()

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, file in enumerate(files):
        # Skip if already processed
        if any(doc["name"] == file.name for doc in st.session_state.uploaded_documents):
            continue

        status_text.text(f"Processing: {file.name}...")

        try:
            # Save to temp file and upload
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file.getvalue())
                tmp_path = Path(tmp.name)

            result = client.upload_document(tmp_path, file.name)

            # Store in session state
            st.session_state.uploaded_documents.append(
                {
                    "id": result.get("document_id"),
                    "name": file.name,
                    "status": "processed",
                    "chunks": result.get("chunk_count", 0),
                    "entities": result.get("entity_count", 0),
                    "file_content": file.getvalue(),  # Keep for PDF viewer
                }
            )

            # Clean up temp file
            tmp_path.unlink(missing_ok=True)

        except Exception as e:
            st.error(f"Failed to upload {file.name}: {e}")
            st.session_state.uploaded_documents.append(
                {
                    "id": None,
                    "name": file.name,
                    "status": "error",
                    "error": str(e),
                }
            )

        progress_bar.progress((idx + 1) / len(files))

    status_text.text("✅ Upload complete!")
    st.rerun()


def _render_document_list() -> None:
    """Render the list of uploaded documents."""
    for idx, doc in enumerate(st.session_state.uploaded_documents):
        with st.container():
            col_icon, col_name, col_stats, col_actions = st.columns([0.5, 3, 2, 1])

            with col_icon:
                if doc["status"] == "processed":
                    st.markdown("✅")
                elif doc["status"] == "error":
                    st.markdown("❌")
                else:
                    st.markdown("⏳")

            with col_name:
                st.markdown(f"**{doc['name']}**")
                if doc.get("id"):
                    st.caption(f"ID: {doc['id'][:8]}...")

            with col_stats:
                if doc["status"] == "processed":
                    st.caption(
                        f"📦 {doc.get('chunks', 0)} chunks | 🏷️ {doc.get('entities', 0)} entities"
                    )
                elif doc["status"] == "error":
                    st.caption(f"Error: {doc.get('error', 'Unknown')[:30]}")

            with col_actions:
                if st.button("🗑️", key=f"delete_{idx}", help="Remove document"):
                    st.session_state.uploaded_documents.pop(idx)
                    st.rerun()

        st.divider()

    # Select documents for analysis
    if len(st.session_state.uploaded_documents) >= 2:
        processed_docs = [
            doc for doc in st.session_state.uploaded_documents if doc["status"] == "processed"
        ]

        if len(processed_docs) >= 2:
            st.markdown("### 🎯 Select Documents for Analysis")

            selected = st.multiselect(
                "Choose documents to analyze",
                options=[doc["name"] for doc in processed_docs],
                default=[doc["name"] for doc in processed_docs[:2]],
            )

            st.session_state.selected_document_ids = [
                doc["id"] for doc in processed_docs if doc["name"] in selected
            ]

            if len(selected) >= 2:
                if st.button("⚡ Go to Analysis", type="primary"):
                    st.session_state.current_page = "analysis"
                    st.rerun()
