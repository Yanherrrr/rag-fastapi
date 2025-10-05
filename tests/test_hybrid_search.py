"""
Tests for hybrid search implementation.
"""

import pytest
import numpy as np
from app.core.hybrid_search import (
    normalize_scores_minmax,
    normalize_scores_zscore,
    normalize_scores_softmax,
    weighted_score_fusion,
    reciprocal_rank_fusion,
    max_score_fusion,
    hybrid_search,
    hybrid_search_with_fallback,
    compare_search_methods
)
from app.models.schemas import Chunk, SearchResult
from app.storage.vector_store import VectorStore
from app.core.keyword_search import BM25Index
from app.core.embeddings import generate_embeddings


class TestScoreNormalization:
    """Tests for score normalization functions."""
    
    def test_minmax_normalization(self):
        """Test min-max normalization."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_scores_minmax(scores)
        
        assert normalized[0] == 0.0  # min
        assert normalized[-1] == 1.0  # max
        assert 0.0 <= np.all(normalized) <= 1.0
    
    def test_minmax_same_scores(self):
        """Test min-max with identical scores."""
        scores = np.array([5.0, 5.0, 5.0])
        normalized = normalize_scores_minmax(scores)
        
        assert np.all(normalized == 1.0)
    
    def test_minmax_empty(self):
        """Test min-max with empty array."""
        scores = np.array([])
        normalized = normalize_scores_minmax(scores)
        
        assert len(normalized) == 0
    
    def test_zscore_normalization(self):
        """Test z-score normalization."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_scores_zscore(scores)
        
        assert np.abs(np.mean(normalized)) < 1e-10  # mean ≈ 0
        assert np.abs(np.std(normalized) - 1.0) < 1e-10  # std ≈ 1
    
    def test_zscore_same_scores(self):
        """Test z-score with identical scores."""
        scores = np.array([5.0, 5.0, 5.0])
        normalized = normalize_scores_zscore(scores)
        
        assert np.all(normalized == 0.0)
    
    def test_softmax_normalization(self):
        """Test softmax normalization."""
        scores = np.array([1.0, 2.0, 3.0])
        normalized = normalize_scores_softmax(scores)
        
        assert np.abs(np.sum(normalized) - 1.0) < 1e-10  # sum = 1
        assert np.all(normalized >= 0.0)  # all positive
        assert np.all(normalized <= 1.0)  # all <= 1
    
    def test_softmax_temperature(self):
        """Test softmax temperature effect."""
        scores = np.array([1.0, 2.0, 3.0])
        
        # Low temperature = more peaked distribution
        norm_low_temp = normalize_scores_softmax(scores, temperature=0.5)
        
        # High temperature = more uniform distribution
        norm_high_temp = normalize_scores_softmax(scores, temperature=2.0)
        
        # Low temp should have higher max probability
        assert np.max(norm_low_temp) > np.max(norm_high_temp)


class TestFusionStrategies:
    """Tests for score fusion strategies."""
    
    def test_weighted_fusion_equal_weights(self):
        """Test weighted fusion with equal weights."""
        semantic_scores = {"doc1": 0.8, "doc2": 0.6}
        keyword_scores = {"doc1": 0.4, "doc2": 0.9}
        
        combined = weighted_score_fusion(semantic_scores, keyword_scores, alpha=0.5)
        
        # Use approximate comparison for floating point
        assert np.abs(combined["doc1"] - 0.6) < 1e-10  # (0.8 + 0.4) / 2
        assert np.abs(combined["doc2"] - 0.75) < 1e-10  # (0.6 + 0.9) / 2
    
    def test_weighted_fusion_semantic_only(self):
        """Test weighted fusion favoring semantic."""
        semantic_scores = {"doc1": 0.8}
        keyword_scores = {"doc1": 0.4}
        
        combined = weighted_score_fusion(semantic_scores, keyword_scores, alpha=1.0)
        
        assert combined["doc1"] == 0.8  # 100% semantic
    
    def test_weighted_fusion_keyword_only(self):
        """Test weighted fusion favoring keyword."""
        semantic_scores = {"doc1": 0.8}
        keyword_scores = {"doc1": 0.4}
        
        combined = weighted_score_fusion(semantic_scores, keyword_scores, alpha=0.0)
        
        assert combined["doc1"] == 0.4  # 100% keyword
    
    def test_weighted_fusion_missing_scores(self):
        """Test weighted fusion with missing scores."""
        semantic_scores = {"doc1": 0.8, "doc2": 0.6}
        keyword_scores = {"doc2": 0.9, "doc3": 0.7}
        
        combined = weighted_score_fusion(semantic_scores, keyword_scores, alpha=0.5)
        
        assert "doc1" in combined  # Has semantic only
        assert "doc2" in combined  # Has both
        assert "doc3" in combined  # Has keyword only
        assert combined["doc1"] == 0.4  # 0.5 * 0.8 + 0.5 * 0
        assert combined["doc3"] == 0.35  # 0.5 * 0 + 0.5 * 0.7
    
    def test_rrf_basic(self):
        """Test reciprocal rank fusion."""
        sem_results = [
            SearchResult(
                chunk=Chunk(chunk_id="doc1", text="text1", source_file="f", page_number=1, chunk_index=0),
                score=0.9,
                rank=1
            ),
            SearchResult(
                chunk=Chunk(chunk_id="doc2", text="text2", source_file="f", page_number=1, chunk_index=1),
                score=0.8,
                rank=2
            )
        ]
        
        kw_results = [
            SearchResult(
                chunk=Chunk(chunk_id="doc2", text="text2", source_file="f", page_number=1, chunk_index=1),
                score=5.0,
                rank=1
            ),
            SearchResult(
                chunk=Chunk(chunk_id="doc3", text="text3", source_file="f", page_number=1, chunk_index=2),
                score=4.0,
                rank=2
            )
        ]
        
        rrf_scores = reciprocal_rank_fusion(sem_results, kw_results, k=60)
        
        # doc2 appears in both, should have highest score
        assert "doc1" in rrf_scores
        assert "doc2" in rrf_scores
        assert "doc3" in rrf_scores
        assert rrf_scores["doc2"] > rrf_scores["doc1"]
        assert rrf_scores["doc2"] > rrf_scores["doc3"]
    
    def test_rrf_formula(self):
        """Test RRF formula calculation."""
        # Single result at rank 1
        result = SearchResult(
            chunk=Chunk(chunk_id="doc1", text="text", source_file="f", page_number=1, chunk_index=0),
            score=1.0,
            rank=1
        )
        
        rrf_scores = reciprocal_rank_fusion([result], [], k=60)
        
        # RRF score = 1 / (60 + 1) = 1/61
        expected = 1.0 / 61.0
        assert np.abs(rrf_scores["doc1"] - expected) < 1e-10
    
    def test_max_fusion(self):
        """Test maximum score fusion."""
        semantic_scores = {"doc1": 0.8, "doc2": 0.6}
        keyword_scores = {"doc1": 0.4, "doc2": 0.9}
        
        combined = max_score_fusion(semantic_scores, keyword_scores)
        
        assert combined["doc1"] == 0.8  # max(0.8, 0.4)
        assert combined["doc2"] == 0.9  # max(0.6, 0.9)
    
    def test_max_fusion_missing_scores(self):
        """Test max fusion with missing scores."""
        semantic_scores = {"doc1": 0.8}
        keyword_scores = {"doc2": 0.9}
        
        combined = max_score_fusion(semantic_scores, keyword_scores)
        
        assert combined["doc1"] == 0.8  # max(0.8, 0)
        assert combined["doc2"] == 0.9  # max(0, 0.9)


class TestHybridSearch:
    """Tests for hybrid search."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks."""
        return [
            Chunk(
                chunk_id="1",
                text="Machine learning uses algorithms to learn from data",
                source_file="doc1.pdf",
                page_number=1,
                chunk_index=0
            ),
            Chunk(
                chunk_id="2",
                text="Deep learning is a subset of machine learning using neural networks",
                source_file="doc1.pdf",
                page_number=2,
                chunk_index=1
            ),
            Chunk(
                chunk_id="3",
                text="Python is popular for data science and machine learning",
                source_file="doc2.pdf",
                page_number=1,
                chunk_index=0
            )
        ]
    
    @pytest.fixture
    def vector_store(self, sample_chunks, tmp_path):
        """Create vector store with sample chunks."""
        store = VectorStore(store_path=tmp_path / "test_store.pkl")
        
        # Generate embeddings
        embeddings = generate_embeddings([c.text for c in sample_chunks])
        
        # Add to store
        store.add_documents(sample_chunks, embeddings)
        
        return store
    
    @pytest.fixture
    def bm25_index(self, sample_chunks):
        """Create BM25 index with sample chunks."""
        index = BM25Index()
        index.add_documents(sample_chunks)
        return index
    
    def test_hybrid_search_weighted(self, vector_store, bm25_index):
        """Test hybrid search with weighted fusion."""
        results = hybrid_search(
            query="machine learning algorithms",
            vector_store=vector_store,
            bm25_index=bm25_index,
            top_k=3,
            fusion_method="weighted",
            semantic_weight=0.5
        )
        
        assert len(results) > 0
        assert len(results) <= 3
        
        # Results should be sorted by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_hybrid_search_rrf(self, vector_store, bm25_index):
        """Test hybrid search with RRF fusion."""
        results = hybrid_search(
            query="machine learning",
            vector_store=vector_store,
            bm25_index=bm25_index,
            top_k=3,
            fusion_method="rrf"
        )
        
        assert len(results) > 0
        
        # Results should have valid scores
        for result in results:
            assert result.score > 0
    
    def test_hybrid_search_max(self, vector_store, bm25_index):
        """Test hybrid search with max fusion."""
        results = hybrid_search(
            query="Python data science",
            vector_store=vector_store,
            bm25_index=bm25_index,
            top_k=3,
            fusion_method="max"
        )
        
        assert len(results) > 0
    
    def test_hybrid_search_semantic_weight(self, vector_store, bm25_index):
        """Test that semantic weight affects results."""
        # High semantic weight
        results_semantic = hybrid_search(
            query="machine learning",
            vector_store=vector_store,
            bm25_index=bm25_index,
            top_k=3,
            semantic_weight=0.9
        )
        
        # High keyword weight
        results_keyword = hybrid_search(
            query="machine learning",
            vector_store=vector_store,
            bm25_index=bm25_index,
            top_k=3,
            semantic_weight=0.1
        )
        
        # Both should return results
        assert len(results_semantic) > 0
        assert len(results_keyword) > 0
    
    def test_hybrid_search_no_results(self, tmp_path):
        """Test hybrid search with no documents."""
        # Use a fresh temporary path to ensure no existing data
        empty_store = VectorStore(store_path=tmp_path / "empty_store.pkl")
        empty_index = BM25Index()
        
        results = hybrid_search(
            query="test",
            vector_store=empty_store,
            bm25_index=empty_index,
            top_k=5
        )
        
        assert results == []


class TestHybridSearchWithFallback:
    """Tests for hybrid search with fallback."""
    
    @pytest.fixture
    def setup_stores(self, tmp_path):
        """Setup vector store and BM25 index."""
        chunks = [
            Chunk(
                chunk_id="1",
                text="Artificial intelligence and machine learning",
                source_file="doc.pdf",
                page_number=1,
                chunk_index=0
            ),
            Chunk(
                chunk_id="2",
                text="Natural language processing",
                source_file="doc.pdf",
                page_number=2,
                chunk_index=1
            )
        ]
        
        # Vector store
        store = VectorStore(store_path=tmp_path / "store.pkl")
        embeddings = generate_embeddings([c.text for c in chunks])
        store.add_documents(chunks, embeddings)
        
        # BM25 index
        index = BM25Index()
        index.add_documents(chunks)
        
        return store, index
    
    def test_fallback_hybrid(self, setup_stores):
        """Test fallback uses hybrid when both are good."""
        store, index = setup_stores
        
        results, method = hybrid_search_with_fallback(
            query="artificial intelligence",
            vector_store=store,
            bm25_index=index,
            top_k=2,
            semantic_threshold=0.3  # Low threshold
        )
        
        assert len(results) > 0
        assert method in ["hybrid", "semantic", "keyword"]
    
    def test_fallback_no_results(self, tmp_path):
        """Test fallback with no results."""
        # Use a fresh temporary path to ensure no existing data
        empty_store = VectorStore(store_path=tmp_path / "empty_store.pkl")
        empty_index = BM25Index()
        
        results, method = hybrid_search_with_fallback(
            query="test",
            vector_store=empty_store,
            bm25_index=empty_index,
            top_k=5
        )
        
        assert results == []
        assert method == "none"


class TestCompareSearchMethods:
    """Tests for search method comparison."""
    
    @pytest.fixture
    def setup(self, tmp_path):
        """Setup for comparison tests."""
        chunks = [
            Chunk(
                chunk_id="1",
                text="Machine learning algorithms",
                source_file="doc.pdf",
                page_number=1,
                chunk_index=0
            ),
            Chunk(
                chunk_id="2",
                text="Deep learning neural networks",
                source_file="doc.pdf",
                page_number=2,
                chunk_index=1
            )
        ]
        
        store = VectorStore(store_path=tmp_path / "store.pkl")
        embeddings = generate_embeddings([c.text for c in chunks])
        store.add_documents(chunks, embeddings)
        
        index = BM25Index()
        index.add_documents(chunks)
        
        return store, index
    
    def test_compare_methods(self, setup):
        """Test comparing different search methods."""
        store, index = setup
        
        comparison = compare_search_methods(
            query="machine learning",
            vector_store=store,
            bm25_index=index,
            top_k=2
        )
        
        assert "query" in comparison
        assert "results" in comparison
        assert "overlap" in comparison
        assert "counts" in comparison
        
        assert "semantic" in comparison["results"]
        assert "keyword" in comparison["results"]
        assert "hybrid_weighted" in comparison["results"]
        assert "hybrid_rrf" in comparison["results"]

