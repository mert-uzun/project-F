"""
Cross-Document Conflict Detector - Streamlit UI.

Investment banking due diligence workbench.
"""

import streamlit as st

# Must be first Streamlit call
st.set_page_config(
    page_title="Conflict Detector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css() -> None:
    """Inject custom CSS for dark mode IB styling."""
    css_path = "ui/static/style.css"
    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback inline styles
        st.markdown("""
        <style>
        .stApp { background-color: #0E1117; }
        .conflict-card {
            border-left: 4px solid #ff4b4b;
            background-color: #262730;
            padding: 12px;
            margin-bottom: 8px;
            border-radius: 4px;
        }
        .reasoning-log {
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 12px;
            color: #00cc96;
            background-color: #0e1117;
            padding: 10px;
            border: 1px solid #333;
        }
        </style>
        """, unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize session state variables."""
    defaults = {
        "uploaded_documents": [],  # List of {id, name, status}
        "current_page": "upload",
        "selected_document_ids": [],
        "analysis_results": None,
        "conflicts": [],
        "current_conflict_idx": 0,
        "pdf_page": 1,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> None:
    """Render sidebar with navigation and document list."""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x50/1E3A5F/D4AF37?text=Conflict+Detector", width=200)
        st.markdown("---")

        # Navigation
        st.subheader("📍 Navigation")
        pages = {
            "upload": "📁 Upload Documents",
            "inspector": "🔍 Data Inspector",
            "analysis": "⚡ Run Analysis",
            "conflicts": "⚠️ Conflict Workbench",
            "graph": "🕸️ Knowledge Graph",
            "timeline": "📅 Timeline",
            "report": "📄 Executive Summary",
        }

        for page_id, page_name in pages.items():
            if st.button(page_name, key=f"nav_{page_id}", use_container_width=True):
                st.session_state.current_page = page_id
                st.rerun()

        st.markdown("---")

        # Document list
        st.subheader("📂 Documents")
        if st.session_state.uploaded_documents:
            for doc in st.session_state.uploaded_documents:
                status_icon = "✅" if doc.get("status") == "processed" else "⏳"
                st.text(f"{status_icon} {doc['name'][:25]}")
        else:
            st.caption("No documents uploaded")


def render_main_content() -> None:
    """Render main content based on current page."""
    page = st.session_state.current_page

    if page == "upload":
        from ui.components.upload import render_upload
        render_upload()
    elif page == "inspector":
        from ui.components.inspector import render_inspector
        render_inspector()
    elif page == "analysis":
        from ui.components.analysis import render_analysis
        render_analysis()
    elif page == "conflicts":
        from ui.components.conflicts import render_conflicts
        render_conflicts()
    elif page == "graph":
        from ui.components.graph import render_graph
        render_graph()
    elif page == "timeline":
        from ui.components.timeline import render_timeline
        render_timeline()
    elif page == "report":
        from ui.components.report import render_report
        render_report()
    else:
        st.error(f"Unknown page: {page}")


def main() -> None:
    """Main application entry point."""
    inject_custom_css()
    init_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
