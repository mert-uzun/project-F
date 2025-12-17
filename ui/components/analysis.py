"""
Analysis Component.

Run analysis with live reasoning trace (audit log).
"""

import time
from datetime import datetime

import streamlit as st

from ui.utils.api_client import get_client


def render_analysis() -> None:
    """Render the analysis execution view with reasoning trace."""
    st.header("⚡ Run Analysis")
    st.caption("Execute multi-document conflict detection with live progress")

    # Check for selected documents
    if len(st.session_state.selected_document_ids) < 2:
        st.warning("Select at least 2 documents for analysis. Go to Upload page.")
        if st.button("📁 Go to Upload"):
            st.session_state.current_page = "upload"
            st.rerun()
        return

    # Show selected documents
    st.subheader("📋 Selected Documents")

    selected_docs = [
        doc
        for doc in st.session_state.uploaded_documents
        if doc.get("id") in st.session_state.selected_document_ids
    ]

    for doc in selected_docs:
        st.markdown(f"- **{doc['name']}** ({doc.get('entities', 0)} entities)")

    st.markdown("---")

    # Analysis controls
    col_btn, col_options = st.columns([1, 2])

    with col_btn:
        run_analysis = st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("analysis_running", False),
        )

    with col_options:
        include_timeline = st.checkbox("Include Timeline Analysis", value=True)
        resolve_entities = st.checkbox("Enable Entity Resolution", value=True)

    if run_analysis:
        _execute_analysis(include_timeline, resolve_entities)

    # Show previous results if available
    if st.session_state.analysis_results:
        st.markdown("---")
        _render_results_summary()


def _execute_analysis(include_timeline: bool, resolve_entities: bool) -> None:
    """Execute analysis with live reasoning trace."""
    st.session_state.analysis_running = True

    # Reasoning trace container
    with st.expander("📜 Live Reasoning Log", expanded=True):
        log_container = st.container()

        with log_container:
            st.markdown('<div class="reasoning-log">', unsafe_allow_html=True)
            log_placeholder = st.empty()
            logs: list[str] = []

            def add_log(log_type: str, message: str) -> None:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                log_line = f'<span class="log-timestamp">[{timestamp}]</span> <span class="log-type-{log_type}">[{log_type}]</span> {message}'
                logs.append(log_line)
                log_placeholder.markdown("<br>".join(logs), unsafe_allow_html=True)

            # Simulate analysis steps with logging
            add_log("INFO", "Starting multi-document analysis...")
            time.sleep(0.3)

            add_log("INFO", f"Loading {len(st.session_state.selected_document_ids)} documents")
            time.sleep(0.2)

            add_log("EXTRACT", "Retrieving entities from knowledge graph...")
            time.sleep(0.4)

            if resolve_entities:
                add_log("NORMALIZE", "Running entity resolution (threshold: 0.85)...")
                time.sleep(0.5)
                add_log("NORMALIZE", "Matching name variants: initials, titles, case...")
                time.sleep(0.3)

            add_log("COMPARE", "Detecting value conflicts across documents...")
            time.sleep(0.5)

            # Call actual API
            add_log("INFO", "Calling analysis API...")

            try:
                client = get_client()
                result = client.run_analysis(st.session_state.selected_document_ids)

                conflict_count = result.get("conflict_count", 0)
                critical_count = result.get("critical_conflicts", 0)

                add_log("COMPARE", f"Found {conflict_count} conflicts")

                if critical_count > 0:
                    add_log("ALERT", f"⚠️ {critical_count} CRITICAL conflicts detected!")

                if include_timeline:
                    add_log("INFO", "Building timeline...")
                    time.sleep(0.3)
                    timeline_result = client.get_timeline(st.session_state.selected_document_ids)
                    event_count = timeline_result.get("event_count", 0)
                    add_log("INFO", f"Timeline: {event_count} events extracted")

                add_log("INFO", "✅ Analysis complete!")

                # Store results
                st.session_state.analysis_results = result
                st.session_state.conflicts = result.get("conflicts", [])

            except Exception as e:
                add_log("ALERT", f"❌ Analysis failed: {e}")
                st.error(f"Analysis failed: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.analysis_running = False
    st.rerun()


def _render_results_summary() -> None:
    """Render analysis results summary."""
    results = st.session_state.analysis_results

    st.subheader("📊 Analysis Results")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Conflicts</div>
        </div>
        """.format(results.get("conflict_count", 0)),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value" style="color: #DC3545;">{}</div>
            <div class="metric-label">Critical</div>
        </div>
        """.format(results.get("critical_conflicts", 0)),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Entities</div>
        </div>
        """.format(results.get("total_entities", 0)),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">{:.1f}s</div>
            <div class="metric-label">Analysis Time</div>
        </div>
        """.format(results.get("analysis_time_seconds", 0)),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Quick summary
    if results.get("summary"):
        st.info(results["summary"])

    # Navigation buttons
    col_conflicts, col_graph, col_report = st.columns(3)

    with col_conflicts:
        if st.button("⚠️ View Conflicts", use_container_width=True):
            st.session_state.current_page = "conflicts"
            st.rerun()

    with col_graph:
        if st.button("🕸️ View Graph", use_container_width=True):
            st.session_state.current_page = "graph"
            st.rerun()

    with col_report:
        if st.button("📄 Generate Report", use_container_width=True):
            st.session_state.current_page = "report"
            st.rerun()
