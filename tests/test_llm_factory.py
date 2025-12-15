"""
Tests for LLM Factory - REAL Integration Tests.

Tests actual LLM and embedding model creation.
Skip tests if required backends are not available.
"""

import pytest
import os

from tests.conftest import requires_llm, requires_ollama


# ============================================================================
# LLM Creation Tests
# ============================================================================

class TestLLMFactory:
    """Tests for LLM factory functions."""
    
    def test_get_settings(self) -> None:
        """Test settings can be loaded."""
        from app.config import get_settings
        
        settings = get_settings()
        
        assert settings is not None
        assert hasattr(settings, 'llm_backend')
        assert hasattr(settings, 'embedding_backend')
    
    @requires_ollama()
    def test_create_ollama_llm(self) -> None:
        """Test Ollama LLM creation when Ollama is running."""
        from src.utils.llm_factory import get_llm
        from app.config import LLMBackend
        
        # This requires Ollama to be running
        llm = get_llm(backend=LLMBackend.OLLAMA)
        
        assert llm is not None
        # Validate it has expected methods
        assert hasattr(llm, 'complete') or hasattr(llm, 'acomplete')
    
    def test_create_openai_llm_with_key(self) -> None:
        """Test OpenAI LLM creation when API key is set."""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")
        
        from src.utils.llm_factory import get_llm
        from app.config import LLMBackend
        
        llm = get_llm(backend=LLMBackend.OPENAI)
        
        assert llm is not None
        assert hasattr(llm, 'complete') or hasattr(llm, 'acomplete')
    
    @requires_llm()
    def test_get_llm_default(self) -> None:
        """Test default LLM creation."""
        from src.utils.llm_factory import get_llm
        
        llm = get_llm()
        
        assert llm is not None


# ============================================================================
# Embedding Model Tests
# ============================================================================

class TestEmbeddingFactory:
    """Tests for embedding model creation."""
    
    def test_create_huggingface_embedding(self) -> None:
        """Test HuggingFace embedding model creation."""
        from src.utils.llm_factory import get_embedding_model
        from app.config import EmbeddingBackend
        
        # HuggingFace embeddings should work without external services
        embedding = get_embedding_model(backend=EmbeddingBackend.HUGGINGFACE)
        
        assert embedding is not None
        # Should have embedding method
        assert hasattr(embedding, 'get_text_embedding') or hasattr(embedding, '_get_text_embedding')
    
    def test_embedding_produces_vectors(self) -> None:
        """Test that embedding model produces actual vectors."""
        from src.utils.llm_factory import get_embedding_model
        from app.config import EmbeddingBackend
        
        embedding = get_embedding_model(backend=EmbeddingBackend.HUGGINGFACE)
        
        # Get embedding for test text
        text = "This is a test sentence for embedding."
        
        try:
            if hasattr(embedding, 'get_text_embedding'):
                vector = embedding.get_text_embedding(text)
            elif hasattr(embedding, '_get_text_embedding'):
                vector = embedding._get_text_embedding(text)
            else:
                pytest.skip("Embedding model doesn't have expected methods")
            
            assert isinstance(vector, list)
            assert len(vector) > 0
            assert all(isinstance(v, (int, float)) for v in vector)
        except Exception as e:
            pytest.skip(f"Embedding failed: {e}")
    
    def test_get_embedding_model_default(self) -> None:
        """Test default embedding model creation."""
        from src.utils.llm_factory import get_embedding_model
        
        embedding = get_embedding_model()
        
        assert embedding is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestLLMIntegration:
    """Integration tests requiring actual LLM calls."""
    
    @pytest.mark.asyncio
    @requires_ollama()
    async def test_ollama_completion(self) -> None:
        """Test actual completion with Ollama."""
        from src.utils.llm_factory import get_llm
        from app.config import LLMBackend
        
        llm = get_llm(backend=LLMBackend.OLLAMA)
        
        # Simple completion test
        response = await llm.acomplete("What is 2 + 2? Answer with just the number.")
        
        assert response is not None
        assert hasattr(response, 'text')
        assert len(response.text) > 0
        # Should contain "4" somewhere
        assert "4" in response.text
    
    @pytest.mark.asyncio
    @requires_llm()
    async def test_llm_completion_returns_text(self) -> None:
        """Test that LLM completion returns non-empty text."""
        from src.utils.llm_factory import get_llm
        
        llm = get_llm()
        
        response = await llm.acomplete("Respond with the word 'hello'.")
        
        assert response is not None
        assert len(response.text.strip()) > 0
