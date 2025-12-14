"""
Vector Store - ChromaDB Integration.

Provides semantic search over document chunks using embeddings.
Part of the GraphRAG architecture - works alongside the Graph Store.
"""

from pathlib import Path
from typing import Any
from uuid import UUID

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.ingestion.schemas import DocumentChunk
from src.knowledge.schemas import VectorSearchResult
from src.utils.llm_factory import get_embedding_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""

    pass


class VectorStore:
    """
    ChromaDB-backed vector store for semantic search.
    
    Stores document chunks with embeddings for similarity search.
    Each chunk maintains metadata for citation tracking.
    
    Usage:
        store = VectorStore(persist_dir=Path("./data/chroma"))
        await store.add_chunks(chunks)
        results = await store.search("salary information", top_k=5)
    """

    COLLECTION_NAME = "document_chunks"

    def __init__(
        self,
        persist_dir: Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        """
        Initialize vector store.
        
        Args:
            persist_dir: Directory for persistence (None for in-memory)
            collection_name: Name of the collection
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name or self.COLLECTION_NAME
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._embed_model = None
    
    def _get_client(self) -> chromadb.ClientAPI:
        """Get or create ChromaDB client."""
        if self._client is None:
            if self.persist_dir:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_dir),
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    ),
                )
                logger.info(f"Initialized persistent ChromaDB at {self.persist_dir}")
            else:
                self._client = chromadb.Client(
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
                logger.info("Initialized in-memory ChromaDB")
        
        return self._client
    
    def _get_collection(self) -> chromadb.Collection:
        """Get or create collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )
            logger.info(f"Using collection: {self.collection_name}")
        
        return self._collection
    
    def _get_embed_model(self) -> Any:
        """Get embedding model (lazy load)."""
        if self._embed_model is None:
            self._embed_model = get_embedding_model()
        return self._embed_model
    
    async def add_chunks(
        self,
        chunks: list[DocumentChunk],
        batch_size: int = 100,
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk to add
            batch_size: Batch size for embedding generation
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        collection = self._get_collection()
        embed_model = self._get_embed_model()
        
        added_count = 0
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            
            # Prepare data
            ids = [str(chunk.metadata.chunk_id) for chunk in batch]
            documents = [chunk.content for chunk in batch]
            metadatas = [
                {
                    "document_id": str(chunk.metadata.document_id),
                    "chunk_id": str(chunk.metadata.chunk_id),
                    "page_number": chunk.metadata.page_number,
                    "chunk_index": chunk.metadata.chunk_index,
                    "contains_table": chunk.metadata.contains_table,
                    "section_title": chunk.metadata.section_title or "",
                    "char_start": chunk.metadata.char_start,
                    "char_end": chunk.metadata.char_end,
                }
                for chunk in batch
            ]
            
            # Generate embeddings
            try:
                embeddings = [
                    embed_model.get_text_embedding(doc)
                    for doc in documents
                ]
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                raise VectorStoreError(f"Failed to generate embeddings: {e}") from e
            
            # Add to collection
            try:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                added_count += len(batch)
                logger.debug(f"Added batch of {len(batch)} chunks")
            except Exception as e:
                logger.error(f"Failed to add chunks to ChromaDB: {e}")
                raise VectorStoreError(f"Failed to add chunks: {e}") from e
        
        logger.info(f"Added {added_count} chunks to vector store")
        return added_count
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_document_id: UUID | None = None,
        filter_contains_table: bool | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_document_id: Filter by document ID
            filter_contains_table: Filter by table content
            
        Returns:
            List of VectorSearchResult
        """
        collection = self._get_collection()
        embed_model = self._get_embed_model()
        
        # Generate query embedding
        try:
            query_embedding = embed_model.get_text_embedding(query)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise VectorStoreError(f"Failed to embed query: {e}") from e
        
        # Build filter
        where_filter: dict[str, Any] | None = None
        if filter_document_id or filter_contains_table is not None:
            conditions = []
            if filter_document_id:
                conditions.append({"document_id": {"$eq": str(filter_document_id)}})
            if filter_contains_table is not None:
                conditions.append({"contains_table": {"$eq": filter_contains_table}})
            
            if len(conditions) == 1:
                where_filter = conditions[0]
            else:
                where_filter = {"$and": conditions}
        
        # Query
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise VectorStoreError(f"Search failed: {e}") from e
        
        # Convert to VectorSearchResult
        search_results: list[VectorSearchResult] = []
        
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                document = results["documents"][0][i] if results["documents"] else ""
                distance = results["distances"][0][i] if results["distances"] else 0.0
                
                # Convert distance to similarity score (ChromaDB uses L2 by default, we use cosine)
                # For cosine, distance is 1 - similarity
                score = max(0.0, 1.0 - distance)
                
                search_results.append(
                    VectorSearchResult(
                        chunk_id=UUID(chunk_id),
                        document_id=UUID(metadata.get("document_id", chunk_id)),
                        content=document,
                        score=score,
                        page_number=metadata.get("page_number", 1),
                        metadata=metadata,
                    )
                )
        
        logger.debug(f"Found {len(search_results)} results for query: {query[:50]}...")
        return search_results
    
    async def delete_document(self, document_id: UUID) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Number of chunks deleted
        """
        collection = self._get_collection()
        
        # Get chunks for this document
        try:
            results = collection.get(
                where={"document_id": {"$eq": str(document_id)}},
                include=[],
            )
            
            if results["ids"]:
                collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
                return len(results["ids"])
            
            return 0
            
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise VectorStoreError(f"Failed to delete document: {e}") from e
    
    async def get_stats(self) -> dict[str, Any]:
        """Get vector store statistics."""
        collection = self._get_collection()
        
        return {
            "collection_name": self.collection_name,
            "total_chunks": collection.count(),
            "persist_dir": str(self.persist_dir) if self.persist_dir else "in-memory",
        }
    
    def reset(self) -> None:
        """Reset the collection (delete all data)."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            self._collection = None
            logger.warning(f"Reset collection: {self.collection_name}")
        except Exception:
            pass  # Collection might not exist
