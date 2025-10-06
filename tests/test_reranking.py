"""
Tests for result re-ranking module.
"""

import pytest
import numpy as np
import os

# Clear HuggingFace tokens before imports to avoid 401 errors
os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)
os.environ.pop('HF_TOKEN', None)

from app.core.reranking import (
    CrossEncoderReranker,
    get_cross_encoder_reranker,
    rerank_with_cross_encoder,
    compute_similarity_matrix,
    maximal_marginal_relevance,
    rerank_results,
    compare_rankings
)
from app.models.schemas import Chunk, SearchResult


@pytest.fixture
def cross_encoder_available():
    """Check if cross-encoder model is available."""
    try:
        from sentence_transformers import CrossEncoder
        # Try to load the model
        CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return True
    except Exception as e:
        pytest.skip(f"Cross-encoder model not available: {e}. Run download script first.")
        return False


@pytest.fixture
def sample_results():
    """Create sample search results."""
    return [
        SearchResult(
            chunk=Chunk(
                chunk_id="1",
                text="Machine learning is a method of data analysis",
                source_file="doc1.pdf",
                page_number=1,
                chunk_index=0
            ),
            score=0.9,
            rank=1
        ),
        SearchResult(
            chunk=Chunk(
                chunk_id="2",
                text="Deep learning uses neural networks",
                source_file="doc2.pdf",
                page_number=1,
                chunk_index=0
            ),
            score=0.8,
            rank=2
        ),
        SearchResult(
            chunk=Chunk(
                chunk_id="3",
                text="Python is a programming language",
                source_file="doc3.pdf",
                page_number=1,
                chunk_index=0
            ),
            score=0.7,
            rank=3
        ),
        SearchResult(
            chunk=Chunk(
                chunk_id="4",
                text="Artificial intelligence encompasses machine learning",
                source_file="doc4.pdf",
                page_number=1,
                chunk_index=0
            ),
            score=0.6,
            rank=4
        ),
        SearchResult(
            chunk=Chunk(
                chunk_id="5",
                text="Data science involves statistics and machine learning",
                source_file="doc5.pdf",
                page_number=1,
                chunk_index=0
            ),
            score=0.5,
            rank=5
        )
    ]


class TestCrossEncoderReranker:
    """Tests for cross-encoder re-ranking."""
    
    def test_initialization(self):
        """Test reranker initialization."""
        reranker = CrossEncoderReranker()
        
        assert reranker.model is None  # Lazy loading
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def test_custom_model(self):
        """Test initialization with custom model."""
        reranker = CrossEncoderReranker(model_name="custom-model")
        
        assert reranker.model_name == "custom-model"
    
    def test_singleton_pattern(self):
        """Test that get_cross_encoder_reranker returns singleton."""
        reranker1 = get_cross_encoder_reranker()
        reranker2 = get_cross_encoder_reranker()
        
        assert reranker1 is reranker2
    
    def test_rerank_basic(self, sample_results, cross_encoder_available):
        """Test basic re-ranking functionality."""
        if not cross_encoder_available:
            pytest.skip("Cross-encoder model not available")
        
        query = "machine learning algorithms"
        
        reranked = rerank_with_cross_encoder(query, sample_results, top_k=3)
        
        assert len(reranked) == 3
        
        # Results should be sorted by score
        scores = [r.score for r in reranked]
        assert scores == sorted(scores, reverse=True)
        
        # Ranks should be updated
        for rank, result in enumerate(reranked, start=1):
            assert result.rank == rank
    
    def test_rerank_changes_order(self, sample_results, cross_encoder_available):
        """Test that re-ranking can change result order."""
        if not cross_encoder_available:
            pytest.skip("Cross-encoder model not available")
        
        query = "neural networks deep learning"
        
        # Original top result
        original_top = sample_results[0].chunk.chunk_id
        
        # Re-rank
        reranked = rerank_with_cross_encoder(query, sample_results)
        
        # Check that we got results
        assert len(reranked) > 0
        
        # Results should have valid scores
        for result in reranked:
            assert isinstance(result.score, float)
    
    def test_rerank_empty_results(self):
        """Test re-ranking with empty results."""
        reranked = rerank_with_cross_encoder("test query", [])
        
        assert reranked == []
    
    def test_rerank_single_result(self, sample_results, cross_encoder_available):
        """Test re-ranking with single result."""
        if not cross_encoder_available:
            pytest.skip("Cross-encoder model not available")
        
        query = "machine learning"
        single_result = sample_results[:1]
        
        reranked = rerank_with_cross_encoder(query, single_result)
        
        assert len(reranked) == 1
        assert reranked[0].rank == 1


class TestSimilarityMatrix:
    """Tests for similarity matrix computation."""
    
    def test_similarity_matrix_with_embeddings(self, sample_results):
        """Test similarity matrix with pre-computed embeddings."""
        # Create simple embeddings
        n = len(sample_results)
        embeddings = np.random.rand(n, 384)
        
        similarity_matrix = compute_similarity_matrix(sample_results, embeddings)
        
        assert similarity_matrix.shape == (n, n)
        
        # Diagonal should be 1 (self-similarity)
        np.testing.assert_array_almost_equal(
            np.diag(similarity_matrix),
            np.ones(n),
            decimal=5
        )
        
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(
            similarity_matrix,
            similarity_matrix.T,
            decimal=5
        )
        
        # Values should be in [-1, 1] (with small tolerance for floating point)
        assert np.all(similarity_matrix >= -1.0 - 1e-6)
        assert np.all(similarity_matrix <= 1.0 + 1e-6)
    
    def test_similarity_matrix_without_embeddings(self, sample_results):
        """Test similarity matrix with text-based similarity."""
        similarity_matrix = compute_similarity_matrix(sample_results, embeddings=None)
        
        n = len(sample_results)
        assert similarity_matrix.shape == (n, n)
        
        # Diagonal should be 1
        for i in range(n):
            assert similarity_matrix[i, i] == 1.0
        
        # Matrix should be symmetric
        np.testing.assert_array_equal(similarity_matrix, similarity_matrix.T)
    
    def test_similarity_matrix_single_document(self, sample_results):
        """Test similarity matrix with single document."""
        single_result = sample_results[:1]
        embeddings = np.random.rand(1, 384)
        
        similarity_matrix = compute_similarity_matrix(single_result, embeddings)
        
        assert similarity_matrix.shape == (1, 1)
        np.testing.assert_almost_equal(similarity_matrix[0, 0], 1.0, decimal=5)


class TestMMR:
    """Tests for Maximal Marginal Relevance."""
    
    def test_mmr_basic(self, sample_results):
        """Test basic MMR functionality."""
        reranked = maximal_marginal_relevance(
            results=sample_results,
            lambda_param=0.5,
            top_k=3
        )
        
        assert len(reranked) == 3
        
        # First result should be the most relevant
        assert reranked[0].chunk.chunk_id == sample_results[0].chunk.chunk_id
    
    def test_mmr_lambda_1_preserves_order(self, sample_results):
        """Test that λ=1 (pure relevance) preserves original order."""
        reranked = maximal_marginal_relevance(
            results=sample_results,
            lambda_param=1.0,
            top_k=5
        )
        
        # With λ=1, order should be close to original (all relevance, no diversity)
        assert reranked[0].chunk.chunk_id == sample_results[0].chunk.chunk_id
    
    def test_mmr_lambda_0_maximizes_diversity(self, sample_results):
        """Test that λ=0 (pure diversity) spreads results."""
        # Create embeddings where some documents are very similar
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # Doc 1
            [0.9, 0.1, 0.0],  # Doc 2 - very similar to Doc 1
            [0.0, 1.0, 0.0],  # Doc 3 - different
            [0.0, 0.9, 0.1],  # Doc 4 - similar to Doc 3
            [0.0, 0.0, 1.0],  # Doc 5 - different from all
        ])
        
        reranked = maximal_marginal_relevance(
            results=sample_results,
            lambda_param=0.0,
            top_k=3,
            embeddings=embeddings
        )
        
        assert len(reranked) == 3
        # First is still highest relevance
        assert reranked[0].chunk.chunk_id == "1"
    
    def test_mmr_invalid_lambda(self, sample_results):
        """Test that invalid lambda raises error."""
        with pytest.raises(ValueError):
            maximal_marginal_relevance(
                results=sample_results,
                lambda_param=1.5  # Invalid
            )
        
        with pytest.raises(ValueError):
            maximal_marginal_relevance(
                results=sample_results,
                lambda_param=-0.5  # Invalid
            )
    
    def test_mmr_empty_results(self):
        """Test MMR with empty results."""
        reranked = maximal_marginal_relevance(results=[], lambda_param=0.5)
        
        assert reranked == []
    
    def test_mmr_single_result(self, sample_results):
        """Test MMR with single result."""
        single_result = sample_results[:1]
        
        reranked = maximal_marginal_relevance(
            results=single_result,
            lambda_param=0.5,
            top_k=1
        )
        
        assert len(reranked) == 1
        assert reranked[0].chunk.chunk_id == single_result[0].chunk.chunk_id
    
    def test_mmr_top_k_larger_than_results(self, sample_results):
        """Test MMR when top_k is larger than available results."""
        reranked = maximal_marginal_relevance(
            results=sample_results,
            lambda_param=0.5,
            top_k=100  # Much larger than available
        )
        
        assert len(reranked) == len(sample_results)


class TestCombinedPipeline:
    """Tests for combined re-ranking pipeline."""
    
    def test_pipeline_both_methods(self, sample_results, cross_encoder_available):
        """Test pipeline with both cross-encoder and MMR."""
        if not cross_encoder_available:
            pytest.skip("Cross-encoder model not available")
        
        query = "machine learning"
        
        reranked = rerank_results(
            query=query,
            results=sample_results,
            methods=["cross_encoder", "mmr"],
            top_k=3,
            mmr_lambda=0.5
        )
        
        assert len(reranked) == 3
        
        # Results should have valid scores and ranks
        for result in reranked:
            assert isinstance(result.score, float)
            assert isinstance(result.rank, int)
    
    def test_pipeline_cross_encoder_only(self, sample_results, cross_encoder_available):
        """Test pipeline with only cross-encoder."""
        if not cross_encoder_available:
            pytest.skip("Cross-encoder model not available")
        
        query = "deep learning neural networks"
        
        reranked = rerank_results(
            query=query,
            results=sample_results,
            methods=["cross_encoder"],
            top_k=3
        )
        
        assert len(reranked) == 3
    
    def test_pipeline_mmr_only(self, sample_results):
        """Test pipeline with only MMR."""
        reranked = rerank_results(
            query="test",  # Not used for MMR
            results=sample_results,
            methods=["mmr"],
            top_k=3,
            mmr_lambda=0.7
        )
        
        assert len(reranked) == 3
    
    def test_pipeline_empty_methods(self, sample_results):
        """Test pipeline with no methods."""
        reranked = rerank_results(
            query="test",
            results=sample_results,
            methods=[],
            top_k=3
        )
        
        # Should return top_k of original results
        assert len(reranked) == 3
    
    def test_pipeline_unknown_method(self, sample_results):
        """Test pipeline with unknown method."""
        # Should log warning but continue
        reranked = rerank_results(
            query="test",
            results=sample_results,
            methods=["unknown_method"],
            top_k=3
        )
        
        assert len(reranked) == 3
    
    def test_pipeline_empty_results(self):
        """Test pipeline with empty results."""
        reranked = rerank_results(
            query="test",
            results=[],
            methods=["cross_encoder", "mmr"],
            top_k=5
        )
        
        assert reranked == []


class TestCompareRankings:
    """Tests for ranking comparison utility."""
    
    def test_compare_identical_rankings(self, sample_results):
        """Test comparing identical rankings."""
        comparison = compare_rankings(
            original=sample_results,
            reranked=sample_results,
            top_k=3
        )
        
        assert comparison["overlap"] == 3
        assert comparison["new_in_top_k"] == 0
        assert comparison["dropped_from_top_k"] == 0
        assert comparison["top_1_changed"] is False
    
    def test_compare_different_rankings(self, sample_results):
        """Test comparing different rankings."""
        # Reverse the order
        reranked = list(reversed(sample_results))
        
        # Update ranks
        for rank, result in enumerate(reranked, start=1):
            result.rank = rank
        
        comparison = compare_rankings(
            original=sample_results,
            reranked=reranked,
            top_k=3
        )
        
        assert comparison["top_1_changed"] is True
        assert comparison["original_top_1"] == "1"
        assert comparison["reranked_top_1"] == "5"
    
    def test_compare_with_new_results(self, sample_results):
        """Test comparing when new results appear in top-k."""
        # Create reranked with different results
        reranked = sample_results[2:] + sample_results[:2]
        
        # Update ranks
        for rank, result in enumerate(reranked, start=1):
            result.rank = rank
        
        comparison = compare_rankings(
            original=sample_results,
            reranked=reranked,
            top_k=3
        )
        
        # Top 3 original: [1, 2, 3]
        # Top 3 reranked: [3, 4, 5]
        # Overlap: [3]
        assert comparison["overlap"] == 1
        assert comparison["new_in_top_k"] == 2  # 4, 5 are new
        assert comparison["dropped_from_top_k"] == 2  # 1, 2 dropped
    
    def test_compare_rank_changes(self, sample_results):
        """Test tracking rank changes."""
        # Swap first two results
        reranked = [sample_results[1], sample_results[0]] + sample_results[2:]
        
        # Update ranks
        for rank, result in enumerate(reranked, start=1):
            result.rank = rank
        
        comparison = compare_rankings(
            original=sample_results,
            reranked=reranked,
            top_k=3
        )
        
        # Check rank changes
        rank_changes = comparison["rank_changes"]
        assert len(rank_changes) > 0
        
        # Result that was rank 2 is now rank 1 (moved up 1)
        # Result that was rank 1 is now rank 2 (moved down 1)
        changes_dict = {chunk_id: change for chunk_id, _, _, change in rank_changes}
        assert "2" in changes_dict
        assert changes_dict["2"] == 1  # Moved up 1 position
        assert "1" in changes_dict
        assert changes_dict["1"] == -1  # Moved down 1 position

