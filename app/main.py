"""
FastAPI Application Entry Point.

Cross-Document Conflict Detector API.
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events."""
    # Startup
    logger.info("Starting Conflict Detector API...")
    logger.info(f"LLM Backend: {settings.llm_backend.value}")
    logger.info(f"Embedding Backend: {settings.embedding_backend.value}")
    settings.ensure_directories()
    yield
    # Shutdown
    logger.info("Shutting down Conflict Detector API...")


app = FastAPI(
    title="Cross-Document Conflict Detector",
    description="GraphRAG-based Due Diligence tool for identifying contradictions in financial documents",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/config")
async def get_config() -> dict[str, str]:
    """Get current configuration (non-sensitive values only)."""
    return {
        "llm_backend": settings.llm_backend.value,
        "embedding_backend": settings.embedding_backend.value,
        "embedding_model": settings.embedding_model,
        "ollama_model": settings.ollama_model if settings.llm_backend.value == "ollama" else "N/A",
    }


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)) -> dict[str, str | int]:
    """
    Ingest a PDF document.
    
    Parses the document, extracts tables, and stores in vector/graph databases.
    """
    from pathlib import Path
    from uuid import uuid4
    
    from src.ingestion.pdf_parser import PDFParser
    from src.ingestion.chunker import chunk_document
    from src.knowledge.vector_store import VectorStore, VectorStoreConfig
    from src.knowledge.graph_store import GraphStore
    from src.knowledge.entity_extractor import EntityExtractor
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    logger.info(f"Received document for ingestion: {file.filename}")
    
    try:
        # 1. Save uploaded file
        upload_path = settings.data_uploads_dir / f"{uuid4()}_{file.filename}"
        content = await file.read()
        upload_path.write_bytes(content)
        logger.info(f"Saved upload to {upload_path}")
        
        # 2. Parse with LlamaParse (or fallback)
        parser = PDFParser(
            llama_api_key=settings.llama_cloud_api_key or None,
            prefer_llamaparse=bool(settings.llama_cloud_api_key),
        )
        parse_result = await parser.parse(upload_path)
        
        # 3. Chunk semantically
        parse_result = chunk_document(parse_result)
        logger.info(f"Created {len(parse_result.chunks)} chunks")
        
        # 4. Store in vector DB
        vector_store = VectorStore(VectorStoreConfig(
            persist_directory=settings.chroma_persist_dir,
        ))
        chunks_added = vector_store.add_chunks(parse_result.chunks)
        
        # 5. Extract entities to graph
        graph_store = GraphStore(
            persist_path=settings.data_graphs_dir / "knowledge_graph.json"
        )
        
        extractor = EntityExtractor()
        entity_count = 0
        relationship_count = 0
        
        for chunk in parse_result.chunks:
            result = await extractor.extract(chunk)
            graph_store.add_entities(result.entities)
            graph_store.add_relationships(result.relationships)
            entity_count += len(result.entities)
            relationship_count += len(result.relationships)
        
        # Save graph
        graph_store.save()
        
        logger.info(
            f"Ingestion complete: {chunks_added} chunks, "
            f"{entity_count} entities, {relationship_count} relationships"
        )
        
        return {
            "status": "success",
            "message": f"Document '{file.filename}' ingested successfully",
            "filename": file.filename,
            "document_id": str(parse_result.metadata.document_id),
            "pages": parse_result.metadata.num_pages,
            "tables": parse_result.metadata.num_tables,
            "chunks": len(parse_result.chunks),
            "entities": entity_count,
            "relationships": relationship_count,
        }
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/detect-conflicts")
async def detect_conflicts(document_ids: list[str]) -> dict[str, Any]:
    """
    Detect conflicts across specified documents.
    
    Uses the Comparator Agent to find contradictions,
    then the Judge Agent to verify and generate the report.
    """
    from uuid import UUID
    
    from src.agents.comparator import ComparatorAgent
    from src.agents.judge import JudgeAgent
    from src.agents.schemas import ComparisonQuery
    from src.knowledge.vector_store import VectorStore, VectorStoreConfig
    from src.knowledge.graph_store import GraphStore
    
    if len(document_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 documents required for conflict detection",
        )
    
    logger.info(f"Conflict detection requested for documents: {document_ids}")
    
    try:
        # Parse document IDs
        uuids = [UUID(doc_id) for doc_id in document_ids]
        
        # Initialize stores
        vector_store = VectorStore(VectorStoreConfig(
            persist_directory=settings.chroma_persist_dir,
        ))
        graph_store = GraphStore(
            persist_path=settings.data_graphs_dir / "knowledge_graph.json"
        )
        
        # Create comparison query
        query = ComparisonQuery(
            document_ids=uuids,
            focus_areas=["salary", "equity", "dates", "parties"],
        )
        
        # 1. Run Comparator Agent
        comparator = ComparatorAgent(vector_store, graph_store)
        conflicts = await comparator.compare(query)
        
        logger.info(f"Comparator found {len(conflicts)} potential conflicts")
        
        # 2. Run Judge Agent for verification
        judge = JudgeAgent(vector_store, graph_store)
        report = await judge.verify_and_report(
            conflicts,
            uuids,
            document_ids,  # Use IDs as names for now
        )
        
        logger.info(f"Judge verified: {report.total_verified} of {report.total_conflicts_detected}")
        
        # 3. Return report as JSON
        return {
            "status": "success",
            "summary": report.to_summary(),
            "total_detected": report.total_conflicts_detected,
            "total_verified": report.total_verified,
            "total_rejected": report.total_rejected,
            "red_flags": [
                {
                    "flag_id": str(flag.flag_id),
                    "summary": flag.summary,
                    "severity": flag.conflict.severity.value,
                    "type": flag.conflict.conflict_type.value,
                    "value_a": flag.conflict.value_a,
                    "value_b": flag.conflict.value_b,
                    "impact": flag.impact,
                    "recommended_action": flag.recommended_action,
                    "page_a": flag.conflict.evidence_a.citation.page_number,
                    "page_b": flag.conflict.evidence_b.citation.page_number,
                }
                for flag in report.red_flags
            ],
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid document ID: {e}")
    except Exception as e:
        logger.error(f"Conflict detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Conflict detection failed: {str(e)}")


# ============================================================================
# New Investment Banking Feature Endpoints
# ============================================================================

@app.post("/analyze")
async def analyze_documents(document_ids: list[str]) -> dict[str, Any]:
    """
    Run multi-document analysis across specified documents.
    
    Detects entity variations, conflicts, and unanimous entities.
    Uses Entity Resolution with 0.85 threshold.
    """
    from uuid import UUID
    
    from src.agents.multi_doc_analyzer import MultiDocAnalyzer, DocumentSet
    from src.knowledge.vector_store import VectorStore, VectorStoreConfig
    from src.knowledge.graph_store import GraphStore
    
    if not document_ids:
        raise HTTPException(status_code=400, detail="No document IDs provided")
    
    logger.info(f"Multi-document analysis requested for {len(document_ids)} documents")
    
    try:
        # Parse document IDs
        uuids = [UUID(doc_id) for doc_id in document_ids]
        
        # Initialize stores
        vector_store = VectorStore(VectorStoreConfig(
            persist_directory=settings.chroma_persist_dir,
        ))
        graph_store = GraphStore(
            persist_path=settings.data_graphs_dir / "knowledge_graph.json"
        )
        
        # Create document set
        doc_set = DocumentSet(
            document_ids=uuids,
            document_names={str(uid): f"Document_{str(uid)[:8]}" for uid in uuids},
        )
        
        # Run analysis
        analyzer = MultiDocAnalyzer(vector_store, graph_store)
        report = await analyzer.analyze(doc_set, resolve_entities=True)
        
        logger.info(f"Analysis complete: {report.conflict_count} conflicts found")
        
        return {
            "status": "success",
            "summary": report.to_summary(),
            "document_count": report.document_set.count,
            "total_entities": report.total_entities,
            "total_resolved_entities": report.total_resolved_entities,
            "conflict_count": report.conflict_count,
            "critical_conflicts": len(report.critical_conflicts),
            "high_conflicts": len(report.high_conflicts),
            "unanimous_entities": report.unanimous_entities[:20],  # Top 20
            "conflicts": [
                {
                    "conflict_id": str(c.conflict_id),
                    "title": c.title,
                    "description": c.description,
                    "severity": c.severity.value,
                    "type": c.conflict_type.value,
                    "unique_values": c.unique_values,
                    "document_count": c.document_count,
                }
                for c in report.conflicts[:50]  # Top 50 conflicts
            ],
            "analysis_time_seconds": report.analysis_time_seconds,
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid document ID: {e}")
    except Exception as e:
        logger.error(f"Multi-doc analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/timeline")
async def get_timeline(document_ids: list[str]) -> dict[str, Any]:
    """
    Build a chronological timeline from documents.
    
    Extracts DATE entities, detects event types, and finds temporal conflicts.
    """
    from uuid import UUID
    
    from src.knowledge.timeline_builder import TimelineBuilder
    from src.knowledge.graph_store import GraphStore
    
    if not document_ids:
        raise HTTPException(status_code=400, detail="No document IDs provided")
    
    logger.info(f"Timeline requested for {len(document_ids)} documents")
    
    try:
        uuids = [UUID(doc_id) for doc_id in document_ids]
        
        graph_store = GraphStore(
            persist_path=settings.data_graphs_dir / "knowledge_graph.json"
        )
        
        builder = TimelineBuilder(graph_store)
        timeline = builder.build_timeline(uuids)
        
        logger.info(f"Timeline built: {timeline.event_count} events, {timeline.conflict_count} conflicts")
        
        return {
            "status": "success",
            "event_count": timeline.event_count,
            "conflict_count": timeline.conflict_count,
            "span_days": timeline.span_days,
            "earliest_date": str(timeline.earliest_date) if timeline.earliest_date else None,
            "latest_date": str(timeline.latest_date) if timeline.latest_date else None,
            "events": [
                {
                    "event_id": str(e.event_id),
                    "date": str(e.event_date),
                    "type": e.event_type.value,
                    "description": e.description,
                    "source_document": e.source_document_name,
                    "page": e.source_page,
                    "related_entities": e.related_entity_values[:5],
                }
                for e in timeline.events[:100]  # First 100 events
            ],
            "conflicts": [
                {
                    "conflict_id": str(c.conflict_id),
                    "type": c.conflict_type,
                    "description": c.description,
                    "severity": c.severity,
                    "days_difference": c.days_difference,
                    "event_a_date": str(c.event_a.event_date),
                    "event_b_date": str(c.event_b.event_date),
                }
                for c in timeline.conflicts
            ],
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid document ID: {e}")
    except Exception as e:
        logger.error(f"Timeline building failed: {e}")
        raise HTTPException(status_code=500, detail=f"Timeline failed: {str(e)}")


@app.get("/search")
async def search_entities(
    query: str,
    entity_type: str | None = None,
    document_id: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Search for entity mentions across documents.
    
    Combines graph queries and semantic search.
    """
    from uuid import UUID
    
    from src.knowledge.cross_reference import CrossReferenceEngine
    from src.knowledge.vector_store import VectorStore, VectorStoreConfig
    from src.knowledge.graph_store import GraphStore
    from src.knowledge.schemas import EntityType
    
    if not query or len(query) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
    
    logger.info(f"Entity search: '{query}', type={entity_type}, doc={document_id}")
    
    try:
        vector_store = VectorStore(VectorStoreConfig(
            persist_directory=settings.chroma_persist_dir,
        ))
        graph_store = GraphStore(
            persist_path=settings.data_graphs_dir / "knowledge_graph.json"
        )
        
        engine = CrossReferenceEngine(vector_store, graph_store)
        
        # Parse filters
        parsed_entity_type = None
        if entity_type:
            try:
                parsed_entity_type = EntityType(entity_type.lower())
            except ValueError:
                pass  # Ignore invalid type
        
        parsed_doc_id = UUID(document_id) if document_id else None
        
        # Run search (synchronous method)
        results = engine.search(
            query=query,
            entity_types=[parsed_entity_type] if parsed_entity_type else None,
            document_ids=[parsed_doc_id] if parsed_doc_id else None,
            max_results=limit,
        )
        
        return {
            "status": "success",
            "query": query,
            "result_count": len(results),
            "results": [
                {
                    "entity_id": str(r.entity.entity_id),
                    "entity_value": r.entity.value,
                    "entity_type": r.entity.entity_type.value,
                    "document_id": str(r.document_id),
                    "document_name": r.document_name,
                    "page": r.page_number,
                    "context": r.context[:200] if r.context else None,
                    "match_type": r.relationship_to_query,
                    "relevance_score": r.relevance_score,
                }
                for r in results
            ],
        }
        
    except Exception as e:
        logger.error(f"Entity search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/report")
async def generate_report(
    document_ids: list[str],
    include_timeline: bool = True,
    include_missing_docs: bool = True,
) -> dict[str, Any]:
    """
    Generate an executive summary report.
    
    Formal investment banking memo style with key metrics,
    critical issues, and recommendations.
    """
    from uuid import UUID
    
    from src.agents.report_generator import ReportGenerator, ReportConfig
    from src.agents.multi_doc_analyzer import MultiDocAnalyzer, DocumentSet
    from src.agents.reference_detector import ReferenceDetector
    from src.knowledge.timeline_builder import TimelineBuilder
    from src.knowledge.vector_store import VectorStore, VectorStoreConfig
    from src.knowledge.graph_store import GraphStore
    
    if not document_ids:
        raise HTTPException(status_code=400, detail="No document IDs provided")
    
    logger.info(f"Executive report requested for {len(document_ids)} documents")
    
    try:
        uuids = [UUID(doc_id) for doc_id in document_ids]
        
        vector_store = VectorStore(VectorStoreConfig(
            persist_directory=settings.chroma_persist_dir,
        ))
        graph_store = GraphStore(
            persist_path=settings.data_graphs_dir / "knowledge_graph.json"
        )
        
        # 1. Run multi-doc analysis
        doc_set = DocumentSet(
            document_ids=uuids,
            document_names={str(uid): f"Document_{str(uid)[:8]}" for uid in uuids},
        )
        analyzer = MultiDocAnalyzer(vector_store, graph_store)
        analysis_report = await analyzer.analyze(doc_set)
        
        # 2. Build timeline if requested
        timeline = None
        if include_timeline:
            builder = TimelineBuilder(graph_store)
            timeline = builder.build_timeline(uuids)
        
        # 3. Generate executive summary
        config = ReportConfig(
            include_timeline=include_timeline,
            include_missing_docs=include_missing_docs,
            formal_style=True,
        )
        
        generator = ReportGenerator()
        summary = await generator.generate_executive_summary(
            analysis_report,
            timeline=timeline,
            config=config,
        )
        
        logger.info(f"Report generated, model used: {summary.model_used}")
        
        return {
            "status": "success",
            "document_count": summary.document_count,
            "conflict_count": summary.conflict_count,
            "has_critical_issues": summary.has_critical_issues,
            "model_used": summary.model_used,
            "summary_markdown": summary.summary_markdown,
            "critical_issues": summary.critical_issues,
            "key_findings": summary.key_findings,
            "action_items": summary.action_items,
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid document ID: {e}")
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report failed: {str(e)}")


@app.get("/graph")
async def get_graph_data(
    document_ids: str | None = None,
    max_nodes: int = 100,
) -> dict[str, Any]:
    """
    Get knowledge graph data for visualization.
    
    Returns nodes and edges in a format suitable for frontend rendering.
    """
    from uuid import UUID
    
    from src.knowledge.graph_store import GraphStore
    from src.knowledge.graph_visualizer import ENTITY_COLORS, _Severity, SEVERITY_COLORS
    
    logger.info(f"Graph data requested, max_nodes={max_nodes}")
    
    try:
        graph_store = GraphStore(
            persist_path=settings.data_graphs_dir / "knowledge_graph.json"
        )
        
        # Parse document IDs if provided
        doc_filter = None
        if document_ids:
            doc_filter = [UUID(d.strip()) for d in document_ids.split(",")]
        
        # Get all entities
        if doc_filter:
            entities_by_doc = graph_store.get_entities_by_document(doc_filter)
            all_nodes = []
            for nodes in entities_by_doc.values():
                all_nodes.extend(nodes)
        else:
            all_nodes = graph_store.get_all_entities()
        
        # Limit nodes
        all_nodes = all_nodes[:max_nodes]
        
        # Build node list
        nodes = []
        node_ids = set()
        
        for graph_node in all_nodes:
            entity = graph_node.entity
            node_id = str(entity.entity_id)
            
            if node_id in node_ids:
                continue
            node_ids.add(node_id)
            
            color = ENTITY_COLORS.get(entity.entity_type, "#94A3B8")
            
            nodes.append({
                "id": node_id,
                "label": entity.value[:30],
                "type": entity.entity_type.value,
                "color": color,
                "document_id": str(entity.source_document_id),
                "page": entity.source_page,
            })
        
        # Build edge list
        edges = []
        for source, target, edge_data in graph_store.graph.edges(data=True):
            if source in node_ids and target in node_ids:
                rel_type = edge_data.get("relationship_type", "related")
                if hasattr(rel_type, "value"):
                    rel_type = rel_type.value
                
                edges.append({
                    "source": source,
                    "target": target,
                    "type": rel_type,
                })
        
        return {
            "status": "success",
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }
        
    except Exception as e:
        logger.error(f"Graph data fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Graph failed: {str(e)}")


@app.get("/graph/html")
async def get_graph_html(
    document_ids: str | None = None,
    max_nodes: int = 200,
) -> dict[str, str]:
    """
    Generate interactive PyVis HTML visualization.
    
    Returns the HTML content as a string.
    """
    from uuid import UUID
    from pathlib import Path
    import tempfile
    
    from src.knowledge.graph_store import GraphStore
    from src.knowledge.graph_visualizer import GraphVisualizer
    
    logger.info(f"Graph HTML requested, max_nodes={max_nodes}")
    
    try:
        graph_store = GraphStore(
            persist_path=settings.data_graphs_dir / "knowledge_graph.json"
        )
        
        # Parse document IDs if provided
        doc_filter = None
        if document_ids:
            doc_filter = [UUID(d.strip()) for d in document_ids.split(",")]
        
        # Generate visualization
        visualizer = GraphVisualizer(graph_store)
        visualizer.generate_network(
            document_ids=doc_filter,
            max_nodes=max_nodes,
        )
        
        # Save to temp file and read content
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            temp_path = Path(f.name)
        
        visualizer.save_html(temp_path, include_legend=True)
        html_content = temp_path.read_text()
        temp_path.unlink()  # Clean up
        
        return {
            "status": "success",
            "html": html_content,
        }
        
    except Exception as e:
        logger.error(f"Graph HTML generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Graph HTML failed: {str(e)}")


@app.post("/missing-documents")
async def detect_missing_documents(
    document_ids: list[str],
) -> dict[str, Any]:
    """
    Detect referenced documents that weren't uploaded.
    
    Scans for patterns like "Exhibit A", "pursuant to the Agreement", etc.
    """
    from uuid import UUID
    
    from src.agents.reference_detector import ReferenceDetector
    from src.knowledge.vector_store import VectorStore, VectorStoreConfig
    from src.knowledge.graph_store import GraphStore
    
    if not document_ids:
        raise HTTPException(status_code=400, detail="No document IDs provided")
    
    logger.info(f"Missing document detection for {len(document_ids)} documents")
    
    try:
        uuids = [UUID(doc_id) for doc_id in document_ids]
        
        vector_store = VectorStore(VectorStoreConfig(
            persist_directory=settings.chroma_persist_dir,
        ))
        
        detector = ReferenceDetector()
        
        # Get chunks from vector store via search and extract references
        all_references = []
        for uid in uuids:
            # Search for all chunks in this document
            search_results = vector_store.search(
                query="",
                top_k=1000,
                filter_document_id=uid,
            )
            for result in search_results:
                # Use text-based extraction instead of chunk-based
                refs = detector.extract_references_from_text(
                    result.content,
                    uid,
                    f"Document_{str(uid)[:8]}",
                    result.metadata.get("page_number", 1),
                )
                all_references.extend(refs)
        
        # Build uploaded document list
        uploaded_docs = [
            {"document_id": uid, "filename": f"Document_{str(uid)[:8]}.pdf"}
            for uid in uuids
        ]
        
        report = detector.find_missing_documents(all_references, uploaded_docs)
        
        return {
            "status": "success",
            "total_references": report.total_references,
            "missing_count": report.missing_count,
            "matched_count": len(report.matched_references),
            "has_missing": report.has_missing,
            "summary": report.to_summary(),
            "missing_documents": [
                {
                    "reference_id": str(ref.reference_id),
                    "reference_text": ref.reference_text,
                    "type": ref.reference_type.value,
                    "source_document": ref.source_document_name,
                    "source_page": ref.source_page,
                    "context": ref.context[:100] if ref.context else None,
                }
                for ref in report.missing_documents[:50]
            ],
        }
        
    except Exception as e:
        logger.error(f"Missing document detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
