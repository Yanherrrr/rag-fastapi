"""Tests for embedding generation"""

import pytest
import numpy as np

from app.core.embeddings import (
    EmbeddingGenerator,
    get_embedding_generator,
    generate_embeddings,
    generate_query_embedding
)


# Fixture to check if embedding model is available
@pytest.fixture(scope="module")
def embedding_available():
    """Check if embedding model can be loaded"""
    try:
        generator = get_embedding_generator()
        return True
    except (OSError, RuntimeError) as e:
        pytest.skip(f"Embedding model not available: {e}. Run 'python download_models.py' first.")
        return False


class TestEmbeddingGenerator:
    """Test EmbeddingGenerator class"""
    
    def test_singleton_pattern(self, embedding_available):
        """Test that EmbeddingGenerator follows singleton pattern"""
        gen1 = EmbeddingGenerator()
        gen2 = EmbeddingGenerator()
        assert gen1 is gen2
    
    def test_generate_single_embedding(self, embedding_available):
        """Test generating single embedding"""
        generator = get_embedding_generator()
        text = "This is a test sentence."
        
        embedding = generator.generate_single_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert len(embedding) == generator.get_embedding_dimension()
        
        # Check that embedding is normalized
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01  # Should be close to 1
    
    def test_generate_embeddings_batch(self, embedding_available):
        """Test generating embeddings for multiple texts"""
        generator = get_embedding_generator()
        texts = [
            "First sentence here.",
            "Second sentence here.",
            "Third sentence here."
        ]
        
        embeddings = generator.generate_embeddings(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (len(texts), generator.get_embedding_dimension())
        
        # Check that all embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=0.01)
    
    def test_empty_text_handling(self, embedding_available):
        """Test handling of empty text"""
        generator = get_embedding_generator()
        
        # Single empty text
        embedding = generator.generate_single_embedding("")
        assert isinstance(embedding, np.ndarray)
        
        # Batch with empty texts
        texts = ["Valid text", "", "Another valid text"]
        embeddings = generator.generate_embeddings(texts)
        assert embeddings.shape[0] == len(texts)
    
    def test_encode_query(self, embedding_available):
        """Test query encoding"""
        generator = get_embedding_generator()
        query = "What is artificial intelligence?"
        
        embedding = generator.encode_query(query)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
    
    def test_convenience_functions(self, embedding_available):
        """Test convenience functions"""
        texts = ["Test sentence one.", "Test sentence two."]
        
        # Test generate_embeddings
        embeddings = generate_embeddings(texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        
        # Test generate_query_embedding
        query_embedding = generate_query_embedding("test query")
        assert isinstance(query_embedding, np.ndarray)
        assert query_embedding.ndim == 1


class TestEmbeddingSimilarity:
    """Test embedding similarity properties"""
    
    def test_similar_texts_have_high_similarity(self, embedding_available):
        """Test that similar texts have high cosine similarity"""
        texts = [
            "The cat sat on the mat.",
            "A cat was sitting on a mat.",
            "Dogs are playing in the park."
        ]
        
        embeddings = generate_embeddings(texts)
        
        # Calculate cosine similarities
        sim_01 = np.dot(embeddings[0], embeddings[1])
        sim_02 = np.dot(embeddings[0], embeddings[2])
        
        # Similar sentences (0 and 1) should be more similar than dissimilar ones (0 and 2)
        assert sim_01 > sim_02
        assert sim_01 > 0.7  # High similarity threshold
    
    def test_embedding_consistency(self, embedding_available):
        """Test that same text always produces same embedding"""
        text = "Consistency test sentence."
        
        emb1 = generate_query_embedding(text)
        emb2 = generate_query_embedding(text)
        
        # Should be identical or very close
        assert np.allclose(emb1, emb2, atol=1e-6)

