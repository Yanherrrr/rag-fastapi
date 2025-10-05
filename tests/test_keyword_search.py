"""
Tests for BM25 keyword search implementation.
"""

import pytest
import numpy as np
from app.core.keyword_search import (
    tokenize,
    remove_stopwords,
    preprocess_text,
    BM25Index,
    keyword_search,
    build_bm25_index
)
from app.models.schemas import Chunk


class TestTokenization:
    """Tests for text tokenization."""
    
    def test_basic_tokenization(self):
        """Test basic tokenization."""
        text = "Hello world! This is a test."
        tokens = tokenize(text)
        
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
    
    def test_lowercase(self):
        """Test lowercase conversion."""
        text = "Python and PYTHON"
        tokens = tokenize(text, lowercase=True)
        
        assert tokens.count("python") == 2
    
    def test_no_lowercase(self):
        """Test without lowercase conversion."""
        text = "Python and PYTHON"
        tokens = tokenize(text, lowercase=False)
        
        assert "Python" in tokens
        assert "PYTHON" in tokens
    
    def test_numbers_in_words(self):
        """Test that words with numbers are kept."""
        text = "Python3 ML2023 version1.5"
        tokens = tokenize(text)
        
        assert "python3" in tokens
        assert "ml2023" in tokens
        assert "version1" in tokens
        assert "5" in tokens
    
    def test_special_characters(self):
        """Test that special characters are removed."""
        text = "hello@world #test $money"
        tokens = tokenize(text)
        
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "money" in tokens
        assert "@" not in tokens
        assert "#" not in tokens


class TestStopwords:
    """Tests for stopword removal."""
    
    def test_remove_common_stopwords(self):
        """Test removing common stopwords."""
        tokens = ["the", "quick", "brown", "fox", "is", "running"]
        filtered = remove_stopwords(tokens)
        
        assert "quick" in filtered
        assert "brown" in filtered
        assert "fox" in filtered
        assert "running" in filtered
        assert "the" not in filtered
        assert "is" not in filtered
    
    def test_custom_stopwords(self):
        """Test with custom stopwords."""
        tokens = ["hello", "world", "test"]
        custom_stops = {"hello", "test"}
        filtered = remove_stopwords(tokens, stopwords=custom_stops)
        
        assert filtered == ["world"]
    
    def test_no_stopwords_removed(self):
        """Test when no stopwords present."""
        tokens = ["machine", "learning", "algorithm"]
        filtered = remove_stopwords(tokens)
        
        assert len(filtered) == 3


class TestPreprocessing:
    """Tests for text preprocessing."""
    
    def test_preprocess_with_stopwords(self):
        """Test full preprocessing with stopword removal."""
        text = "The quick brown fox is running"
        tokens = preprocess_text(text, remove_stops=True)
        
        assert "quick" in tokens
        assert "brown" in tokens
        assert "the" not in tokens
        assert "is" not in tokens
    
    def test_preprocess_without_stopwords(self):
        """Test preprocessing without stopword removal."""
        text = "The quick brown fox"
        tokens = preprocess_text(text, remove_stops=False)
        
        assert "the" in tokens
        assert "quick" in tokens


class TestBM25Index:
    """Tests for BM25 index."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            Chunk(
                chunk_id="1",
                text="Machine learning is a subset of artificial intelligence",
                source_file="doc1.pdf",
                page_number=1,
                chunk_index=0
            ),
            Chunk(
                chunk_id="2",
                text="Deep learning is a type of machine learning using neural networks",
                source_file="doc1.pdf",
                page_number=2,
                chunk_index=1
            ),
            Chunk(
                chunk_id="3",
                text="Python is a popular programming language for machine learning",
                source_file="doc2.pdf",
                page_number=1,
                chunk_index=0
            ),
            Chunk(
                chunk_id="4",
                text="Natural language processing is an application of artificial intelligence",
                source_file="doc2.pdf",
                page_number=2,
                chunk_index=1
            )
        ]
    
    def test_initialization(self):
        """Test index initialization."""
        index = BM25Index(k1=1.5, b=0.75)
        
        assert index.k1 == 1.5
        assert index.b == 0.75
        assert index.num_docs == 0
        assert len(index) == 0
    
    def test_add_documents(self, sample_chunks):
        """Test adding documents to index."""
        index = BM25Index()
        index.add_documents(sample_chunks)
        
        assert index.num_docs == 4
        assert len(index.chunks) == 4
        assert len(index.term_freqs) == 4
        assert len(index.doc_lengths) == 4
        assert index.avg_doc_length > 0
    
    def test_inverted_index_structure(self, sample_chunks):
        """Test that inverted index is built correctly."""
        index = BM25Index()
        index.add_documents(sample_chunks)
        
        # "machine" appears in docs 0, 1, 2
        assert "machine" in index.inverted_index
        assert len(index.inverted_index["machine"]) == 3
        
        # "python" appears only in doc 2
        assert "python" in index.inverted_index
        assert len(index.inverted_index["python"]) == 1
    
    def test_document_frequency(self, sample_chunks):
        """Test document frequency calculation."""
        index = BM25Index()
        index.add_documents(sample_chunks)
        
        # "learning" appears in docs 0, 1, 2
        assert index.doc_freqs["learning"] == 3
        
        # "python" appears in doc 2 only
        assert index.doc_freqs["python"] == 1
    
    def test_idf_computation(self, sample_chunks):
        """Test IDF computation."""
        index = BM25Index()
        index.add_documents(sample_chunks)
        
        # Term appearing in 1 doc should have higher IDF
        idf_python = index.compute_idf("python")
        
        # Term appearing in 3 docs should have lower IDF
        idf_learning = index.compute_idf("learning")
        
        assert idf_python > idf_learning
        
        # Non-existent term should have maximum IDF
        idf_nonexistent = index.compute_idf("nonexistent")
        assert idf_nonexistent > idf_python
    
    def test_bm25_score_computation(self, sample_chunks):
        """Test BM25 score computation."""
        index = BM25Index()
        index.add_documents(sample_chunks)
        
        query_terms = ["machine", "learning"]
        
        # Doc 0 contains both terms
        score_0 = index.compute_bm25_score(query_terms, 0)
        
        # Doc 3 contains neither term
        score_3 = index.compute_bm25_score(query_terms, 3)
        
        assert score_0 > 0
        assert score_3 == 0
    
    def test_search_basic(self, sample_chunks):
        """Test basic search functionality."""
        index = BM25Index()
        index.add_documents(sample_chunks)
        
        results = index.search("machine learning", top_k=3)
        
        assert len(results) > 0
        assert len(results) <= 3
        
        # Results should be sorted by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_search_with_threshold(self, sample_chunks):
        """Test search with minimum score threshold."""
        index = BM25Index()
        index.add_documents(sample_chunks)
        
        # High threshold should return fewer results
        results_high = index.search("machine learning", top_k=10, min_score=5.0)
        results_low = index.search("machine learning", top_k=10, min_score=0.0)
        
        assert len(results_high) <= len(results_low)
    
    def test_search_no_results(self, sample_chunks):
        """Test search with no matching documents."""
        index = BM25Index()
        index.add_documents(sample_chunks)
        
        results = index.search("quantum computing blockchain", top_k=5)
        
        # Should return empty list or very low scores
        assert len(results) == 0 or results[0].score < 1.0
    
    def test_search_empty_index(self):
        """Test search on empty index."""
        index = BM25Index()
        results = index.search("test query", top_k=5)
        
        assert results == []
    
    def test_search_ranking(self, sample_chunks):
        """Test that search ranking makes sense."""
        index = BM25Index()
        index.add_documents(sample_chunks)
        
        # Query for "machine learning" should rank docs with both terms higher
        results = index.search("machine learning", top_k=4)
        
        # The top result should contain both "machine" and "learning"
        if results:
            top_text = results[0].chunk.text.lower()
            assert "machine" in top_text
            assert "learning" in top_text
    
    def test_get_stats(self, sample_chunks):
        """Test getting index statistics."""
        index = BM25Index()
        index.add_documents(sample_chunks)
        
        stats = index.get_stats()
        
        assert stats["num_documents"] == 4
        assert stats["num_unique_terms"] > 0
        assert stats["avg_doc_length"] > 0
        assert stats["k1"] == 1.5
        assert stats["b"] == 0.75
    
    def test_clear(self, sample_chunks):
        """Test clearing the index."""
        index = BM25Index()
        index.add_documents(sample_chunks)
        
        assert len(index) > 0
        
        index.clear()
        
        assert len(index) == 0
        assert index.num_docs == 0
        assert len(index.chunks) == 0
    
    def test_repr(self, sample_chunks):
        """Test string representation."""
        index = BM25Index()
        index.add_documents(sample_chunks)
        
        repr_str = repr(index)
        
        assert "BM25Index" in repr_str
        assert "docs=4" in repr_str


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks."""
        return [
            Chunk(
                chunk_id="1",
                text="Python is great for data science",
                source_file="doc.pdf",
                page_number=1,
                chunk_index=0
            ),
            Chunk(
                chunk_id="2",
                text="Data science involves statistics and machine learning",
                source_file="doc.pdf",
                page_number=2,
                chunk_index=1
            )
        ]
    
    def test_build_bm25_index(self, sample_chunks):
        """Test building a new index."""
        index = build_bm25_index(sample_chunks, k1=2.0, b=0.8)
        
        assert index.num_docs == 2
        assert index.k1 == 2.0
        assert index.b == 0.8
    
    def test_keyword_search_with_custom_index(self, sample_chunks):
        """Test keyword search with custom index."""
        index = build_bm25_index(sample_chunks)
        results = keyword_search("data science", top_k=2, index=index)
        
        assert len(results) > 0
        assert results[0].chunk.text is not None


class TestBM25Parameters:
    """Tests for BM25 parameter effects."""
    
    @pytest.fixture
    def chunks(self):
        """Create test chunks."""
        return [
            Chunk(
                chunk_id="1",
                text="machine learning machine learning machine learning",
                source_file="doc.pdf",
                page_number=1,
                chunk_index=0
            ),
            Chunk(
                chunk_id="2",
                text="machine learning",
                source_file="doc.pdf",
                page_number=2,
                chunk_index=1
            )
        ]
    
    def test_k1_parameter_effect(self, chunks):
        """Test that k1 affects term frequency saturation."""
        # Lower k1 = more saturation (diminishing returns for repeated terms)
        index_low_k1 = BM25Index(k1=0.5, b=0.0)
        index_low_k1.add_documents(chunks)
        
        # Higher k1 = less saturation
        index_high_k1 = BM25Index(k1=3.0, b=0.0)
        index_high_k1.add_documents(chunks)
        
        query_terms = ["machine", "learning"]
        
        score_low_0 = index_low_k1.compute_bm25_score(query_terms, 0)
        score_high_0 = index_high_k1.compute_bm25_score(query_terms, 0)
        
        # With higher k1, the document with more term occurrences
        # should get a relatively higher boost
        # (This is a qualitative test)
        assert score_low_0 > 0
        assert score_high_0 > 0
    
    def test_b_parameter_effect(self, chunks):
        """Test that b affects length normalization."""
        # b=0: no length normalization
        index_no_norm = BM25Index(k1=1.5, b=0.0)
        index_no_norm.add_documents(chunks)
        
        # b=1: full length normalization
        index_full_norm = BM25Index(k1=1.5, b=1.0)
        index_full_norm.add_documents(chunks)
        
        # Both should still produce valid scores
        query_terms = ["machine"]
        
        score_no_norm = index_no_norm.compute_bm25_score(query_terms, 0)
        score_full_norm = index_full_norm.compute_bm25_score(query_terms, 0)
        
        assert score_no_norm > 0
        assert score_full_norm > 0

