"""
FastAPI Application Entry Point.

Cross-Document Conflict Detector API.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

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
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    logger.info(f"Received document for ingestion: {file.filename}")
    
    # TODO: Implement full ingestion pipeline
    # 1. Save uploaded file
    # 2. Parse with LlamaParse
    # 3. Chunk semantically
    # 4. Store in vector DB
    # 5. Extract entities to graph
    
    return {
        "status": "pending",
        "message": f"Document '{file.filename}' queued for processing",
        "filename": file.filename,
    }


@app.post("/detect-conflicts")
async def detect_conflicts(document_ids: list[str]) -> dict[str, str]:
    """
    Detect conflicts across specified documents.
    
    Uses the Comparator Agent to find contradictions.
    """
    if len(document_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 documents required for conflict detection",
        )
    
    logger.info(f"Conflict detection requested for documents: {document_ids}")
    
    # TODO: Implement conflict detection
    # 1. Retrieve relevant subgraphs
    # 2. Run Comparator Agent
    # 3. Verify with Judge Agent
    # 4. Return Red Flag Report
    
    return {
        "status": "pending",
        "message": "Conflict detection pipeline not yet implemented",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
