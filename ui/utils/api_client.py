"""
FastAPI Client for Streamlit UI.

Wraps all backend API calls.
"""

import httpx
from typing import Any
from pathlib import Path


class APIClient:
    """Client for the FastAPI backend."""
    
    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """Initialize client with base URL."""
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(60.0, connect=10.0)
    
    def _url(self, path: str) -> str:
        """Build full URL."""
        return f"{self.base_url}{path}"
    
    # === Health ===
    
    def health_check(self) -> dict[str, Any]:
        """Check API health."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self._url("/health"))
            response.raise_for_status()
            return response.json()
    
    def is_healthy(self) -> bool:
        """Check if API is reachable."""
        try:
            result = self.health_check()
            return result.get("status") == "healthy"
        except Exception:
            return False
    
    # === Document Upload ===
    
    def upload_document(self, file_path: Path | str, filename: str | None = None) -> dict[str, Any]:
        """Upload a PDF document."""
        file_path = Path(file_path)
        filename = filename or file_path.name
        
        with httpx.Client(timeout=self.timeout) as client:
            with open(file_path, "rb") as f:
                files = {"file": (filename, f, "application/pdf")}
                response = client.post(self._url("/ingest"), files=files)
                response.raise_for_status()
                return response.json()
    
    def upload_document_bytes(self, content: bytes, filename: str) -> dict[str, Any]:
        """Upload PDF from bytes."""
        with httpx.Client(timeout=self.timeout) as client:
            files = {"file": (filename, content, "application/pdf")}
            response = client.post(self._url("/ingest"), files=files)
            response.raise_for_status()
            return response.json()
    
    # === Analysis ===
    
    def detect_conflicts(self, document_ids: list[str]) -> dict[str, Any]:
        """Run pairwise conflict detection."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self._url("/detect-conflicts"),
                json=document_ids,
            )
            response.raise_for_status()
            return response.json()
    
    def run_analysis(self, document_ids: list[str]) -> dict[str, Any]:
        """Run multi-document analysis."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self._url("/analyze"),
                json=document_ids,
            )
            response.raise_for_status()
            return response.json()
    
    # === Search ===
    
    def search_entities(
        self,
        query: str,
        entity_type: str | None = None,
        document_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search for entity mentions."""
        params = {"query": query, "limit": limit}
        if entity_type:
            params["entity_type"] = entity_type
        if document_id:
            params["document_id"] = document_id
        
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self._url("/search"), params=params)
            response.raise_for_status()
            return response.json()
    
    # === Timeline ===
    
    def get_timeline(self, document_ids: list[str]) -> dict[str, Any]:
        """Get timeline for documents."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self._url("/timeline"),
                json=document_ids,
            )
            response.raise_for_status()
            return response.json()
    
    # === Report ===
    
    def generate_report(
        self,
        document_ids: list[str],
        include_timeline: bool = True,
        include_missing_docs: bool = True,
    ) -> dict[str, Any]:
        """Generate executive summary report."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self._url("/report"),
                params={
                    "include_timeline": include_timeline,
                    "include_missing_docs": include_missing_docs,
                },
                json=document_ids,
            )
            response.raise_for_status()
            return response.json()
    
    # === Graph ===
    
    def get_graph_data(
        self,
        document_ids: list[str] | None = None,
        max_nodes: int = 100,
    ) -> dict[str, Any]:
        """Get graph data as JSON."""
        params = {"max_nodes": max_nodes}
        if document_ids:
            params["document_ids"] = ",".join(document_ids)
        
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self._url("/graph"), params=params)
            response.raise_for_status()
            return response.json()
    
    def get_graph_html(
        self,
        document_ids: list[str] | None = None,
        max_nodes: int = 100,
    ) -> str:
        """Get graph as HTML string."""
        params = {"max_nodes": max_nodes}
        if document_ids:
            params["document_ids"] = ",".join(document_ids)
        
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self._url("/graph/html"), params=params)
            response.raise_for_status()
            return response.text
    
    # === Missing Documents ===
    
    def detect_missing_documents(self, document_ids: list[str]) -> dict[str, Any]:
        """Detect referenced but not uploaded documents."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self._url("/missing-documents"),
                json=document_ids,
            )
            response.raise_for_status()
            return response.json()


# Singleton instance
_client: APIClient | None = None


def get_client(base_url: str = "http://localhost:8000") -> APIClient:
    """Get or create API client singleton."""
    global _client
    if _client is None:
        _client = APIClient(base_url)
    return _client
