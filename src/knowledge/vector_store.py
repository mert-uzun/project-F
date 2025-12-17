"""
Vector Store - ChromaDB Integration.

Provides semantic search over document chunks using embeddings.
This is the "retrieval" part of RAG.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.ingestion.schemas import DocumentChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""
    pass


@dataclass
class VectorStoreConfig:
    """Configuration for the vector store."""

    persist_directory: Path = Path("./data/chroma")
    collection_name: str = "document_chunks"
    embedding_function: Any = None  # Will use default if None
    distance_metric: str = "cosine"  # cosine, l2, ip


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any]
    distance: float

    @property
    def similarity(self) -> float:
        """Convert distance to similarity score (0-1)."""
        # For cosine distance, similarity = 1 - distance
        return max(0.0, 1.0 - self.distance)


class VectorStore:
    """
    ChromaDB-based vector store for semantic search.

    Stores document chunks with embeddings for similarity search.
    Used to retrieve relevant context for LLM queries.

    Usage:
        store = VectorStore(config)
        store.add_chunks(chunks)
        results = store.search("salary information", top_k=5)
    """

    def __init__(self, config: VectorStoreConfig | None = None) -> None:
        """
        Initialize the vector store.

        Args:
            config: Vector store configuration
        """
        self.config = config or VectorStoreConfig()
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    def _ensure_initialized(self) -> None:
        """Lazy initialization of ChromaDB client and collection."""
        if self._client is None:
            logger.info(f"Initializing ChromaDB at {self.config.persist_directory}")

            # Ensure directory exists
            self.config.persist_directory.mkdir(parents=True, exist_ok=True)

            # Create persistent client
            self._client = chromadb.PersistentClient(
                path=str(self.config.persist_directory),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.distance_metric},
                embedding_function=self.config.embedding_function,
            )

            logger.info(
                f"ChromaDB initialized. Collection '{self.config.collection_name}' "
                f"has {self._collection.count()} documents"
            )

    @property
    def collection(self) -> chromadb.Collection:
        """Get the ChromaDB collection."""
        self._ensure_initialized()
        assert self._collection is not None
        return self._collection

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]] | None = None,
    ) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of document chunks to add
            embeddings: Optional pre-computed embeddings (if None, ChromaDB will compute)

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        logger.info(f"Adding {len(chunks)} chunks to vector store")

        # Prepare data for ChromaDB
        ids = [str(chunk.metadata.chunk_id) for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "document_id": str(chunk.metadata.document_id),
                "page_number": chunk.metadata.page_number,
                "chunk_index": chunk.metadata.chunk_index,
                "contains_table": chunk.metadata.contains_table,
                "section_title": chunk.metadata.section_title or "",
                "char_start": chunk.metadata.char_start,
                "char_end": chunk.metadata.char_end,
            }
            for chunk in chunks
        ]

        try:
            if embeddings:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
            else:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )

            logger.info(f"Successfully added {len(chunks)} chunks")
            return len(chunks)

        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise VectorStoreError(f"Failed to add chunks: {e}") from e

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_document_id: UUID | None = None,
        filter_contains_table: bool | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar chunks.

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_document_id: Optional filter to specific document
            filter_contains_table: Optional filter for table chunks

        Returns:
            List of SearchResult objects
        """
        logger.debug(f"Searching for: '{query[:50]}...' (top_k={top_k})")

        # Build where clause
        where: dict[str, Any] | None = None
        if filter_document_id or filter_contains_table is not None:
            conditions = []
            if filter_document_id:
                conditions.append({"document_id": {"$eq": str(filter_document_id)}})
            if filter_contains_table is not None:
                conditions.append({"contains_table": {"$eq": filter_contains_table}})

            if len(conditions) == 1:
                where = conditions[0]
            else:
                where = {"$and": conditions}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            # Parse results
            search_results: list[SearchResult] = []

            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    search_results.append(SearchResult(
                        chunk_id=chunk_id,
                        document_id=results["metadatas"][0][i].get("document_id", ""),
                        content=results["documents"][0][i] if results["documents"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        distance=results["distances"][0][i] if results["distances"] else 0.0,
                    ))

            logger.debug(f"Found {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorStoreError(f"Search failed: {e}") from e

    def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Search using a pre-computed embedding vector.

        Args:
            embedding: Embedding vector
            top_k: Number of results

        Returns:
            List of SearchResult objects
        """
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            search_results: list[SearchResult] = []

            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    search_results.append(SearchResult(
                        chunk_id=chunk_id,
                        document_id=results["metadatas"][0][i].get("document_id", ""),
                        content=results["documents"][0][i] if results["documents"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        distance=results["distances"][0][i] if results["distances"] else 0.0,
                    ))

            return search_results

        except Exception as e:
            raise VectorStoreError(f"Embedding search failed: {e}") from e

    def get_chunk(self, chunk_id: UUID) -> SearchResult | None:
        """
        Retrieve a specific chunk by ID.

        Args:
            chunk_id: UUID of the chunk

        Returns:
            SearchResult or None if not found
        """
        try:
            results = self.collection.get(
                ids=[str(chunk_id)],
                include=["documents", "metadatas"],
            )

            if results["ids"]:
                return SearchResult(
                    chunk_id=results["ids"][0],
                    document_id=results["metadatas"][0].get("document_id", ""),
                    content=results["documents"][0] if results["documents"] else "",
                    metadata=results["metadatas"][0] if results["metadatas"] else {},
                    distance=0.0,
                )

            return None

        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None

    def delete_document(self, document_id: UUID) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: UUID of the document

        Returns:
            Number of chunks deleted
        """
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"document_id": {"$eq": str(document_id)}},
                include=[],
            )

            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
                return len(results["ids"])

            return 0

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            raise VectorStoreError(f"Delete failed: {e}") from e

    def count(self) -> int:
        """Get total number of chunks in the store."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete all data in the collection. Use with caution!"""
        logger.warning("Resetting vector store - all data will be deleted")
        self._ensure_initialized()
        assert self._client is not None
        self._client.delete_collection(self.config.collection_name)
        self._collection = None
        self._ensure_initialized()
