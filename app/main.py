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


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
