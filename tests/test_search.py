"""Tests for search functionality"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from app.core.search import (
    cosine_similarity_search,
    semantic_search,
    compute_similarity_scores,
    get_top_k_indices,
    multi_vector_search,
    euclidean_distance,
    convert_distance_to_similarity,
    calculate_search_quality_metrics,
    has_sufficient_evidence
)
from app.storage.vector_store import VectorStore
from app.models.schemas import Chunk, SearchResult


@pytest.fixture
def sample_embeddings():
    """Create sample normalized embeddings"""
    embeddings = np.random.rand(10, 384).astype(np.float32)
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


@pytest.fixture
def sample_chunks():
    """Create sample chunks"""
    chunks = []
    for i in range(10):
        chunk = Chunk(
            chunk_id=f"chunk_{i}",
            text=f"This is sample text for chunk {i}",
            source_file="test.pdf",
            page_number=1,
            chunk_index=i,
            metadata={}
        )
        chunks.append(chunk)
    return chunks


@pytest.fixture
def populated_vector_store(sample_chunks, sample_embeddings):
    """Create a vector store with sample data"""
    temp_path = Path(tempfile.mktemp(suffix=".pkl"))
    store = VectorStore(store_path=str(temp_path))
    store.add_documents(sample_chunks, sample_embeddings)
    
    yield store
    
    # Cleanup
    temp_path.unlink(missing_ok=True)


class TestCosineSimilaritySearch:
    """Test cosine similarity search"""
    
    def test_basic_search(self, sample_embeddings):
        """Test basic cosine similarity search"""
        query = sample_embeddings[0]
        
        indices, scores = cosine_similarity_search(
            query,
            sample_embeddings,
            top_k=3
        )
        
        assert len(indices) == 3
        assert len(scores) == 3
        assert scores[0] >= scores[1] >= scores[2]  # Sorted
        assert indices[0] == 0  # Query should match itself best
        assert abs(scores[0] - 1.0) < 0.01  # Should be ~1.0
    
    def test_with_threshold(self, sample_embeddings):
        """Test search with similarity threshold"""
        query = sample_embeddings[0]
        
        indices, scores = cosine_similarity_search(
            query,
            sample_embeddings,
            top_k=10,
            threshold=0.9
        )
        
        # Should only return results above threshold
        assert all(score >= 0.9 for score in scores)
    
    def test_empty_documents(self):
        """Test search with no documents"""
        query = np.random.rand(384)
        empty_docs = np.array([]).reshape(0, 384)
        
        indices, scores = cosine_similarity_search(query, empty_docs, top_k=5)
        
        assert len(indices) == 0
        assert len(scores) == 0
    
    def test_normalization(self, sample_embeddings):
        """Test that unnormalized queries are handled"""
        # Create unnormalized query
        query = np.random.rand(384) * 10  # Not normalized
        
        indices, scores = cosine_similarity_search(
            query,
            sample_embeddings,
            top_k=3
        )
        
        # Should still work and return valid scores
        assert len(indices) == 3
        assert all(0 <= score <= 1 for score in scores)


class TestSemanticSearch:
    """Test semantic search wrapper"""
    
    def test_semantic_search(self, populated_vector_store, sample_embeddings):
        """Test semantic search function"""
        query = sample_embeddings[0]
        
        results = semantic_search(
            query,
            populated_vector_store,
            top_k=3
        )
        
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].score >= results[1].score >= results[2].score
    
    def test_with_threshold(self, populated_vector_store, sample_embeddings):
        """Test semantic search with threshold"""
        query = sample_embeddings[0]
        
        results = semantic_search(
            query,
            populated_vector_store,
            top_k=10,
            threshold=0.8
        )
        
        # All results should be above threshold
        assert all(r.score >= 0.8 for r in results)


class TestComputeSimilarityScores:
    """Test similarity score computation"""
    
    def test_compute_scores(self, sample_embeddings):
        """Test computing similarity scores"""
        query = sample_embeddings[0]
        
        scores = compute_similarity_scores(query, sample_embeddings)
        
        assert len(scores) == len(sample_embeddings)
        assert all(0 <= score <= 1 for score in scores)
        assert abs(scores[0] - 1.0) < 0.01  # Self-similarity


class TestGetTopKIndices:
    """Test top-k selection"""
    
    def test_top_k(self):
        """Test getting top-k indices"""
        scores = np.array([0.9, 0.5, 0.8, 0.3, 0.7])
        
        indices, top_scores = get_top_k_indices(scores, k=3)
        
        assert len(indices) == 3
        assert list(indices) == [0, 2, 4]  # Indices of top-3 scores
        assert list(top_scores) == [0.9, 0.8, 0.7]
    
    def test_with_threshold(self):
        """Test top-k with threshold"""
        scores = np.array([0.9, 0.5, 0.8, 0.3, 0.7])
        
        indices, top_scores = get_top_k_indices(scores, k=5, threshold=0.7)
        
        # Should only return scores >= 0.7
        assert all(score >= 0.7 for score in top_scores)
        assert len(indices) == 3  # 0.9, 0.8, 0.7


class TestMultiVectorSearch:
    """Test multi-vector search"""
    
    def test_multi_vector_max(self, populated_vector_store, sample_embeddings):
        """Test multi-vector search with max aggregation"""
        queries = [sample_embeddings[0], sample_embeddings[1]]
        
        results = multi_vector_search(
            queries,
            populated_vector_store,
            top_k=3,
            aggregation="max"
        )
        
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)
    
    def test_multi_vector_mean(self, populated_vector_store, sample_embeddings):
        """Test multi-vector search with mean aggregation"""
        queries = [sample_embeddings[0], sample_embeddings[1]]
        
        results = multi_vector_search(
            queries,
            populated_vector_store,
            top_k=3,
            aggregation="mean"
        )
        
        assert len(results) <= 3
    
    def test_empty_queries(self, populated_vector_store):
        """Test with no queries"""
        results = multi_vector_search([], populated_vector_store, top_k=3)
        assert len(results) == 0


class TestEuclideanDistance:
    """Test Euclidean distance computation"""
    
    def test_euclidean(self, sample_embeddings):
        """Test Euclidean distance calculation"""
        query = sample_embeddings[0]
        
        distances = euclidean_distance(query, sample_embeddings)
        
        assert len(distances) == len(sample_embeddings)
        assert abs(distances[0]) < 0.01  # Distance to itself should be ~0
        assert all(distances >= 0)  # Distances are non-negative
    
    def test_distance_to_similarity(self):
        """Test converting distances to similarities"""
        distances = np.array([0.0, 1.0, 2.0, 10.0])
        
        similarities = convert_distance_to_similarity(distances)
        
        assert similarities[0] == 1.0  # Distance 0 → similarity 1
        assert all(0 <= s <= 1 for s in similarities)
        assert similarities[0] > similarities[1] > similarities[2]  # Decreasing


class TestSearchQualityMetrics:
    """Test search quality metrics"""
    
    def test_quality_metrics(self, populated_vector_store, sample_embeddings):
        """Test calculating quality metrics"""
        query = sample_embeddings[0]
        results = semantic_search(query, populated_vector_store, top_k=5)
        
        metrics = calculate_search_quality_metrics(results, threshold=0.5)
        
        assert "total_results" in metrics
        assert "avg_score" in metrics
        assert "max_score" in metrics
        assert "min_score" in metrics
        assert "above_threshold" in metrics
        
        assert metrics["total_results"] == len(results)
        assert 0 <= metrics["avg_score"] <= 1
    
    def test_empty_results(self):
        """Test metrics with no results"""
        metrics = calculate_search_quality_metrics([])
        
        assert metrics["total_results"] == 0
        assert metrics["avg_score"] == 0.0
    
    def test_sufficient_evidence(self, populated_vector_store, sample_embeddings):
        """Test checking for sufficient evidence"""
        query = sample_embeddings[0]
        results = semantic_search(query, populated_vector_store, top_k=5)
        
        # Should have sufficient evidence with high similarity to itself
        assert has_sufficient_evidence(results, threshold=0.8, min_results=1)
        
        # Should not have evidence with impossible threshold
        assert not has_sufficient_evidence(results, threshold=1.1, min_results=1)
    
    def test_insufficient_evidence_empty(self):
        """Test insufficient evidence with no results"""
        assert not has_sufficient_evidence([], threshold=0.5, min_results=1)

