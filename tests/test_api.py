"""
Tests for FastAPI Endpoints - REAL Integration Tests.

Uses FastAPI TestClient for actual HTTP requests.
No mocking - tests real endpoint behavior.
"""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

# ============================================================================
# Client Fixture
# ============================================================================


@pytest.fixture
def client():
    """Create FastAPI TestClient."""
    from app.main import app

    return TestClient(app)


# ============================================================================
# Health & Config Endpoints
# ============================================================================


class TestHealthEndpoints:
    """Tests for health and config endpoints."""

    def test_health_check(self, client) -> None:
        """Test /health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_config_endpoint(self, client) -> None:
        """Test /config endpoint returns configuration."""
        response = client.get("/config")

        assert response.status_code == 200
        data = response.json()
        assert "llm_backend" in data
        assert "embedding_backend" in data
        assert "embedding_model" in data


# ============================================================================
# Ingest Endpoint Tests
# ============================================================================


class TestIngestEndpoint:
    """Tests for /ingest endpoint."""

    def test_ingest_rejects_non_pdf(self, client) -> None:
        """Test that non-PDF files are rejected."""
        # Create a fake text file
        response = client.post(
            "/ingest",
            files={"file": ("test.txt", b"This is not a PDF", "text/plain")},
        )

        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_ingest_requires_file(self, client) -> None:
        """Test that file is required."""
        response = client.post("/ingest")

        # FastAPI returns 422 for missing required fields
        assert response.status_code == 422

    def test_ingest_validates_filename(self, client) -> None:
        """Test that PDF extension is validated."""
        response = client.post(
            "/ingest",
            files={"file": ("document.doc", b"fake content", "application/msword")},
        )

        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]


# ============================================================================
# Detect Conflicts Endpoint Tests
# ============================================================================


class TestDetectConflictsEndpoint:
    """Tests for /detect-conflicts endpoint."""

    def test_detect_conflicts_requires_two_docs(self, client) -> None:
        """Test that at least 2 documents are required."""
        response = client.post(
            "/detect-conflicts",
            json=[str(uuid4())],  # Only 1 document
        )

        assert response.status_code == 400
        assert "2 documents" in response.json()["detail"]

    def test_detect_conflicts_validates_uuid(self, client) -> None:
        """Test that invalid UUIDs are rejected."""
        response = client.post(
            "/detect-conflicts",
            json=["not-a-uuid", "also-not-uuid"],
        )

        assert response.status_code == 400
        assert "Invalid" in response.json()["detail"]

    def test_detect_conflicts_accepts_valid_input(self, client) -> None:
        """
        Test that valid input is accepted.
        Note: Will fail on actual detection since docs don't exist,
        but validates input parsing.
        """
        doc1 = str(uuid4())
        doc2 = str(uuid4())

        response = client.post(
            "/detect-conflicts",
            json=[doc1, doc2],
        )

        # May fail because documents don't exist in store,
        # but should not be a 400 (validation) error
        # It will be 500 if docs not found, which is expected
        assert response.status_code in [200, 500]


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling across endpoints."""

    def test_404_for_unknown_endpoint(self, client) -> None:
        """Test that unknown endpoints return 404."""
        response = client.get("/nonexistent")

        assert response.status_code == 404

    def test_method_not_allowed(self, client) -> None:
        """Test that wrong HTTP methods are rejected."""
        # GET on POST-only endpoint
        response = client.get("/ingest")

        assert response.status_code == 405

    def test_invalid_json_body(self, client) -> None:
        """Test that invalid JSON is rejected."""
        response = client.post(
            "/detect-conflicts",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422


# ============================================================================
# Response Format Tests
# ============================================================================


class TestResponseFormats:
    """Tests for API response formats."""

    def test_health_response_format(self, client) -> None:
        """Test health response has expected structure."""
        response = client.get("/health")
        data = response.json()

        assert isinstance(data, dict)
        assert "status" in data
        assert "version" in data
        assert isinstance(data["status"], str)
        assert isinstance(data["version"], str)

    def test_config_response_format(self, client) -> None:
        """Test config response has expected structure."""
        response = client.get("/config")
        data = response.json()

        assert isinstance(data, dict)
        required_keys = ["llm_backend", "embedding_backend", "embedding_model"]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_error_response_format(self, client) -> None:
        """Test error responses have expected structure."""
        response = client.post(
            "/detect-conflicts",
            json=["invalid-uuid"],
        )

        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], str)
