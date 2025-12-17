"""
Graph Visualization Component.

Read-only PyVis graph with zoom/pan.
"""

import streamlit as st
import streamlit.components.v1 as components

from ui.utils.api_client import get_client


def render_graph() -> None:
    """Render the knowledge graph visualization."""
    st.header("🕸️ Knowledge Graph")
    st.caption("Interactive visualization of entities and relationships")

    # Check for selected documents
    if not st.session_state.selected_document_ids:
        st.info("Select documents first. Go to Upload page.")
        if st.button("📁 Go to Upload"):
            st.session_state.current_page = "upload"
            st.rerun()
        return

    # Controls
    col_controls, col_legend = st.columns([2, 1])

    with col_controls:
        max_nodes = st.slider("Max Nodes", min_value=20, max_value=200, value=100)
        st.checkbox("Highlight Conflicts", value=True)

    with col_legend:
        st.markdown("""
        **Legend**
        - 🟢 Person
        - 🔵 Organization
        - 🟠 Money
        - 🟣 Date
        - 🔴 Conflict
        """)

    st.markdown("---")

    # Fetch and render graph
    with st.spinner("Loading graph..."):
        try:
            client = get_client()

            # Get graph HTML from API
            graph_html = client.get_graph_html(
                document_ids=st.session_state.selected_document_ids,
                max_nodes=max_nodes,
            )

            # Embed the HTML
            components.html(graph_html, height=700, scrolling=True)

        except Exception as e:
            st.error(f"Failed to load graph: {e}")

            # Show fallback message
            st.info("""
            **Graph could not be loaded.**

            Make sure:
            1. The FastAPI server is running
            2. Documents have been analyzed
            3. Entities have been extracted
            """)

            # Show graph data as JSON fallback
            with st.expander("📊 View Graph Data (JSON)"):
                try:
                    graph_data = client.get_graph_data(
                        document_ids=st.session_state.selected_document_ids,
                        max_nodes=max_nodes,
                    )
                    st.json(graph_data)
                except Exception:
                    st.caption("Graph data not available")

    st.markdown("---")

    # Graph stats
    st.subheader("📈 Graph Statistics")

    try:
        client = get_client()
        graph_data = client.get_graph_data(
            document_ids=st.session_state.selected_document_ids,
            max_nodes=max_nodes,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            node_count = len(graph_data.get("nodes", []))
            st.metric("Nodes", node_count)

        with col2:
            edge_count = len(graph_data.get("edges", []))
            st.metric("Edges", edge_count)

        with col3:
            conflict_nodes = [n for n in graph_data.get("nodes", []) if n.get("is_conflict")]
            st.metric("Conflict Nodes", len(conflict_nodes))

    except Exception:
        st.caption("Statistics not available")
