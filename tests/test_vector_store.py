"""Tests for vector store"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from app.storage.vector_store import VectorStore
from app.models.schemas import Chunk


@pytest.fixture
def temp_store_path():
    """Create temporary store path"""
    # Create a temporary file path but don't create the file
    # This ensures VectorStore starts fresh without trying to load empty file
    temp_dir = tempfile.gettempdir()
    path = Path(temp_dir) / f"test_store_{tempfile.mkstemp()[1]}.pkl"
    
    yield str(path)
    
    # Cleanup
    path.unlink(missing_ok=True)


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing"""
    return [
        Chunk(
            chunk_id="doc1_1_0",
            text="This is the first chunk about artificial intelligence.",
            source_file="doc1.pdf",
            page_number=1,
            chunk_index=0,
            metadata={}
        ),
        Chunk(
            chunk_id="doc1_1_1",
            text="This is the second chunk about machine learning.",
            source_file="doc1.pdf",
            page_number=1,
            chunk_index=1,
            metadata={}
        ),
        Chunk(
            chunk_id="doc2_1_0",
            text="This chunk is from a different document about neural networks.",
            source_file="doc2.pdf",
            page_number=1,
            chunk_index=0,
            metadata={}
        )
    ]


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings"""
    # Create 3 embeddings with dimension 384 (matching all-MiniLM-L6-v2)
    embeddings = np.random.rand(3, 384).astype(np.float32)
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


class TestVectorStore:
    """Test VectorStore class"""
    
    def test_initialization(self, temp_store_path):
        """Test vector store initialization"""
        store = VectorStore(store_path=temp_store_path)
        
        assert store.embeddings is None
        assert len(store.chunks) == 0
        assert store.metadata["version"] == "1.0"
    
    def test_add_documents(self, temp_store_path, sample_chunks, sample_embeddings):
        """Test adding documents to store"""
        store = VectorStore(store_path=temp_store_path)
        
        store.add_documents(sample_chunks, sample_embeddings)
        
        assert store.embeddings is not None
        assert store.embeddings.shape == (3, 384)
        assert len(store.chunks) == 3
        assert store.metadata["total_documents"] == 2  # 2 unique files
    
    def test_add_documents_validation(self, temp_store_path, sample_chunks):
        """Test validation when adding documents"""
        store = VectorStore(store_path=temp_store_path)
        
        # Mismatched lengths should raise error
        wrong_embeddings = np.random.rand(2, 384)
        
        with pytest.raises(ValueError):
            store.add_documents(sample_chunks, wrong_embeddings)
    
    def test_search(self, temp_store_path, sample_chunks, sample_embeddings):
        """Test searching for similar chunks"""
        store = VectorStore(store_path=temp_store_path)
        store.add_documents(sample_chunks, sample_embeddings)
        
        # Use one of the embeddings as query
        query_embedding = sample_embeddings[0]
        
        results = store.search(query_embedding, top_k=2)
        
        assert len(results) <= 2
        assert all(hasattr(r, 'chunk') for r in results)
        assert all(hasattr(r, 'score') for r in results)
        assert all(hasattr(r, 'rank') for r in results)
        
        # Results should be sorted by score (descending)
        if len(results) > 1:
            assert results[0].score >= results[1].score
    
    def test_search_with_threshold(self, temp_store_path, sample_chunks, sample_embeddings):
        """Test search with similarity threshold"""
        store = VectorStore(store_path=temp_store_path)
        store.add_documents(sample_chunks, sample_embeddings)
        
        query_embedding = sample_embeddings[0]
        
        # High threshold should return fewer results
        results_high = store.search(query_embedding, top_k=10, threshold=0.95)
        results_low = store.search(query_embedding, top_k=10, threshold=0.5)
        
        assert len(results_high) <= len(results_low)
    
    def test_get_by_ids(self, temp_store_path, sample_chunks, sample_embeddings):
        """Test retrieving chunks by IDs"""
        store = VectorStore(store_path=temp_store_path)
        store.add_documents(sample_chunks, sample_embeddings)
        
        chunk_ids = ["doc1_1_0", "doc2_1_0"]
        chunks = store.get_by_ids(chunk_ids)
        
        assert len(chunks) == 2
        assert all(c.chunk_id in chunk_ids for c in chunks)
    
    def test_get_by_source_file(self, temp_store_path, sample_chunks, sample_embeddings):
        """Test retrieving chunks by source file"""
        store = VectorStore(store_path=temp_store_path)
        store.add_documents(sample_chunks, sample_embeddings)
        
        chunks = store.get_by_source_file("doc1.pdf")
        
        assert len(chunks) == 2
        assert all(c.source_file == "doc1.pdf" for c in chunks)
    
    def test_delete_by_source_file(self, temp_store_path, sample_chunks, sample_embeddings):
        """Test deleting chunks by source file"""
        store = VectorStore(store_path=temp_store_path)
        store.add_documents(sample_chunks, sample_embeddings)
        
        initial_count = len(store.chunks)
        store.delete_by_source_file("doc1.pdf")
        
        assert len(store.chunks) < initial_count
        assert len(store.chunks) == 1
        assert all(c.source_file != "doc1.pdf" for c in store.chunks)
    
    def test_save_and_load(self, temp_store_path, sample_chunks, sample_embeddings):
        """Test saving and loading vector store"""
        # Create and populate store
        store1 = VectorStore(store_path=temp_store_path)
        store1.add_documents(sample_chunks, sample_embeddings)
        store1.save()
        
        # Load into new store
        store2 = VectorStore(store_path=temp_store_path)
        
        assert len(store2.chunks) == len(store1.chunks)
        assert store2.embeddings.shape == store1.embeddings.shape
        assert np.allclose(store2.embeddings, store1.embeddings)
    
    def test_clear(self, temp_store_path, sample_chunks, sample_embeddings):
        """Test clearing vector store"""
        store = VectorStore(store_path=temp_store_path)
        store.add_documents(sample_chunks, sample_embeddings)
        
        store.clear()
        
        assert store.embeddings is None
        assert len(store.chunks) == 0
        assert store.metadata["total_documents"] == 0
    
    def test_get_stats(self, temp_store_path, sample_chunks, sample_embeddings):
        """Test getting store statistics"""
        store = VectorStore(store_path=temp_store_path)
        store.add_documents(sample_chunks, sample_embeddings)
        
        stats = store.get_stats()
        
        assert "total_chunks" in stats
        assert "total_documents" in stats
        assert "embedding_dimension" in stats
        assert "memory_usage_mb" in stats
        
        assert stats["total_chunks"] == 3
        assert stats["total_documents"] == 2
    
    def test_len(self, temp_store_path, sample_chunks, sample_embeddings):
        """Test __len__ method"""
        store = VectorStore(store_path=temp_store_path)
        assert len(store) == 0
        
        store.add_documents(sample_chunks, sample_embeddings)
        assert len(store) == 3
    
    def test_empty_search(self, temp_store_path):
        """Test searching in empty store"""
        store = VectorStore(store_path=temp_store_path)
        query_embedding = np.random.rand(384)
        
        results = store.search(query_embedding, top_k=5)
        
        assert len(results) == 0

