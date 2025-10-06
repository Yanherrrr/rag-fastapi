"""Tests for query API endpoint"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import Intent, SearchResult, Chunk, ResponseMetadata


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing"""
    return [
        Chunk(
            chunk_id="chunk_1",
            text="Machine learning is a subset of artificial intelligence.",
            source_file="ai.pdf",
            page_number=1,
            chunk_index=0
        ),
        Chunk(
            chunk_id="chunk_2",
            text="Deep learning uses neural networks with many layers.",
            source_file="ai.pdf",
            page_number=2,
            chunk_index=1
        ),
        Chunk(
            chunk_id="chunk_3",
            text="Natural language processing enables computers to understand text.",
            source_file="nlp.pdf",
            page_number=1,
            chunk_index=0
        )
    ]


@pytest.fixture
def sample_search_results(sample_chunks):
    """Sample search results"""
    return [
        SearchResult(chunk=sample_chunks[0], score=0.95, rank=1),
        SearchResult(chunk=sample_chunks[1], score=0.85, rank=2),
        SearchResult(chunk=sample_chunks[2], score=0.75, rank=3)
    ]


class TestQueryEndpoint:
    """Tests for the /api/query endpoint"""
    
    def test_greeting_intent(self, client):
        """Test greeting intent handling"""
        with patch('app.api.query.detect_intent') as mock_intent, \
             patch('app.api.query.is_conversational') as mock_conv, \
             patch('app.api.query.get_simple_response') as mock_response:
            
            mock_intent.return_value = Intent.GREETING
            mock_conv.return_value = True
            mock_response.return_value = "Hello! How can I help you today?"
            
            response = client.post(
                "/api/query",
                json={"query": "hello", "top_k": 5}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["intent"] == "greeting"
            assert "Hello" in data["answer"]
            assert len(data["sources"]) == 0
            assert data["has_sufficient_evidence"] is True
    
    def test_chitchat_intent(self, client):
        """Test chitchat intent handling"""
        with patch('app.api.query.detect_intent') as mock_intent, \
             patch('app.api.query.is_conversational') as mock_conv, \
             patch('app.api.query.get_simple_response') as mock_response:
            
            mock_intent.return_value = Intent.CHITCHAT
            mock_conv.return_value = True
            mock_response.return_value = "I'm here to help answer questions based on your documents."
            
            response = client.post(
                "/api/query",
                json={"query": "how are you?", "top_k": 5}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["intent"] == "chitchat"
            assert "documents" in data["answer"]
    
    def test_empty_knowledge_base(self, client):
        """Test query when knowledge base is empty"""
        with patch('app.api.query.detect_intent') as mock_intent, \
             patch('app.api.query.is_conversational') as mock_conv, \
             patch('app.api.query.get_vector_store') as mock_store:
            
            mock_intent.return_value = Intent.SEARCH_KNOWLEDGE_BASE
            mock_conv.return_value = False
            mock_store.return_value.__len__.return_value = 0
            
            response = client.post(
                "/api/query",
                json={"query": "What is machine learning?", "top_k": 5}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "documents" in data["answer"].lower()
            assert "knowledge base" in data["answer"].lower()
            assert data["has_sufficient_evidence"] is False
    
    def test_successful_search_query(self, client, sample_search_results):
        """Test successful search query with results"""
        with patch('app.api.query.detect_intent') as mock_intent, \
             patch('app.api.query.is_conversational') as mock_conv, \
             patch('app.api.query.get_vector_store') as mock_store, \
             patch('app.api.query.get_bm25_index') as mock_bm25, \
             patch('app.api.query.hybrid_search_with_fallback') as mock_search, \
             patch('app.api.query.rerank_results') as mock_rerank, \
             patch('app.api.query.has_sufficient_evidence') as mock_evidence, \
             patch('app.api.query.generate_answer') as mock_llm:
            
            mock_intent.return_value = Intent.SEARCH_KNOWLEDGE_BASE
            mock_conv.return_value = False
            mock_store.return_value.__len__.return_value = 100
            mock_search.return_value = sample_search_results
            mock_rerank.return_value = sample_search_results
            mock_evidence.return_value = True
            mock_llm.return_value = {
                "status": "success",
                "answer": "Machine learning is a subset of AI that enables systems to learn from data."
            }
            
            response = client.post(
                "/api/query",
                json={
                    "query": "What is machine learning?",
                    "top_k": 3,
                    "include_sources": True
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["intent"] == "search"
            assert "Machine learning" in data["answer"]
            assert len(data["sources"]) == 3
            assert data["has_sufficient_evidence"] is True
            assert "search_time_ms" in data["metadata"]
            assert "llm_time_ms" in data["metadata"]
            
            # Verify sources structure
            source = data["sources"][0]
            assert "chunk_id" in source
            assert "text" in source
            assert "source_file" in source
            assert "page_number" in source
            assert "similarity_score" in source
    
    def test_no_search_results(self, client):
        """Test when search returns no results"""
        with patch('app.api.query.detect_intent') as mock_intent, \
             patch('app.api.query.is_conversational') as mock_conv, \
             patch('app.api.query.get_vector_store') as mock_store, \
             patch('app.api.query.get_bm25_index') as mock_bm25, \
             patch('app.api.query.hybrid_search_with_fallback') as mock_search:
            
            mock_intent.return_value = Intent.SEARCH_KNOWLEDGE_BASE
            mock_conv.return_value = False
            mock_store.return_value.__len__.return_value = 100
            mock_search.return_value = []
            
            response = client.post(
                "/api/query",
                json={"query": "quantum physics", "top_k": 5}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "couldn't find" in data["answer"].lower()
            assert len(data["sources"]) == 0
            assert data["has_sufficient_evidence"] is False
    
    def test_llm_error_handling(self, client, sample_search_results):
        """Test handling of LLM errors"""
        with patch('app.api.query.detect_intent') as mock_intent, \
             patch('app.api.query.is_conversational') as mock_conv, \
             patch('app.api.query.get_vector_store') as mock_store, \
             patch('app.api.query.get_bm25_index') as mock_bm25, \
             patch('app.api.query.hybrid_search_with_fallback') as mock_search, \
             patch('app.api.query.rerank_results') as mock_rerank, \
             patch('app.api.query.has_sufficient_evidence') as mock_evidence, \
             patch('app.api.query.generate_answer') as mock_llm:
            
            mock_intent.return_value = Intent.SEARCH_KNOWLEDGE_BASE
            mock_conv.return_value = False
            mock_store.return_value.__len__.return_value = 100
            mock_search.return_value = sample_search_results
            mock_rerank.return_value = sample_search_results
            mock_evidence.return_value = True
            mock_llm.return_value = {
                "status": "error",
                "answer": "API key not configured"
            }
            
            response = client.post(
                "/api/query",
                json={"query": "test query", "top_k": 3}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "error" in data["answer"].lower()
            assert len(data["sources"]) == 3  # Still returns sources
    
    def test_query_without_sources(self, client, sample_search_results):
        """Test query with include_sources=False"""
        with patch('app.api.query.detect_intent') as mock_intent, \
             patch('app.api.query.is_conversational') as mock_conv, \
             patch('app.api.query.get_vector_store') as mock_store, \
             patch('app.api.query.get_bm25_index') as mock_bm25, \
             patch('app.api.query.hybrid_search_with_fallback') as mock_search, \
             patch('app.api.query.rerank_results') as mock_rerank, \
             patch('app.api.query.has_sufficient_evidence') as mock_evidence, \
             patch('app.api.query.generate_answer') as mock_llm:
            
            mock_intent.return_value = Intent.SEARCH_KNOWLEDGE_BASE
            mock_conv.return_value = False
            mock_store.return_value.__len__.return_value = 100
            mock_search.return_value = sample_search_results
            mock_rerank.return_value = sample_search_results
            mock_evidence.return_value = True
            mock_llm.return_value = {
                "status": "success",
                "answer": "Test answer"
            }
            
            response = client.post(
                "/api/query",
                json={
                    "query": "test query",
                    "top_k": 3,
                    "include_sources": False
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["sources"]) == 0
    
    def test_query_validation_empty(self, client):
        """Test query validation for empty query"""
        response = client.post(
            "/api/query",
            json={"query": "", "top_k": 5}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_query_validation_too_long(self, client):
        """Test query validation for very long query"""
        long_query = "a" * 1001  # Max is 1000
        response = client.post(
            "/api/query",
            json={"query": long_query, "top_k": 5}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_top_k_validation(self, client):
        """Test top_k parameter validation"""
        # Too small
        response = client.post(
            "/api/query",
            json={"query": "test", "top_k": 0}
        )
        assert response.status_code == 422
        
        # Too large
        response = client.post(
            "/api/query",
            json={"query": "test", "top_k": 21}
        )
        assert response.status_code == 422
    
    def test_exception_handling(self, client):
        """Test generic exception handling"""
        with patch('app.api.query.detect_intent') as mock_intent:
            mock_intent.side_effect = Exception("Unexpected error")
            
            response = client.post(
                "/api/query",
                json={"query": "test query", "top_k": 5}
            )
            
            assert response.status_code == 500
            assert "error" in response.json()["detail"].lower()


class TestSourceInfoConversion:
    """Tests for convert_to_source_info helper"""
    
    def test_conversion(self, sample_search_results):
        """Test conversion from SearchResult to SourceInfo"""
        from app.api.query import convert_to_source_info
        
        sources = convert_to_source_info(sample_search_results)
        
        assert len(sources) == 3
        assert sources[0].chunk_id == "chunk_1"
        assert sources[0].text == "Machine learning is a subset of artificial intelligence."
        assert sources[0].source_file == "ai.pdf"
        assert sources[0].page_number == 1
        assert sources[0].similarity_score == 0.95
    
    def test_score_rounding(self, sample_chunks):
        """Test that similarity scores are rounded to 4 decimal places"""
        from app.api.query import convert_to_source_info
        
        results = [
            SearchResult(chunk=sample_chunks[0], score=0.123456789, rank=1)
        ]
        
        sources = convert_to_source_info(results)
        assert sources[0].similarity_score == 0.1235
    
    def test_empty_list(self):
        """Test conversion with empty list"""
        from app.api.query import convert_to_source_info
        
        sources = convert_to_source_info([])
        assert sources == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

