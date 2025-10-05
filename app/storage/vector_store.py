"""Custom vector store implementation using numpy"""

import pickle
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from datetime import datetime

from app.models.schemas import Chunk, SearchResult
from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Custom vector store for storing and searching document embeddings.
    
    Uses numpy arrays for efficient similarity search and pickle for persistence.
    Suitable for small to medium-sized datasets (up to 100k chunks).
    """
    
    def __init__(self, store_path: Optional[str] = None):
        """
        Initialize vector store.
        
        Args:
            store_path: Path to store file (default from settings)
        """
        self.store_path = Path(store_path or settings.vector_store_path)
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[Chunk] = []
        self.metadata: Dict[str, Any] = {
            "version": "1.0",
            "created_at": None,
            "updated_at": None,
            "total_documents": 0,
            "embedding_dimension": settings.embedding_dimension
        }
        
        # Try to load existing store
        if self.store_path.exists():
            self.load()
        else:
            logger.info("No existing vector store found. Starting fresh.")
    
    def add_documents(self, chunks: List[Chunk], embeddings: np.ndarray):
        """
        Add documents (chunks) with their embeddings to the store.
        
        Args:
            chunks: List of Chunk objects
            embeddings: numpy array of embeddings (shape: [n_chunks, embedding_dim])
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match number of embeddings ({len(embeddings)})"
            )
        
        logger.info(f"Adding {len(chunks)} documents to vector store")
        
        # Initialize or concatenate embeddings
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        # Add chunks
        self.chunks.extend(chunks)
        
        # Update metadata
        self.metadata["updated_at"] = datetime.now().isoformat()
        if self.metadata["created_at"] is None:
            self.metadata["created_at"] = self.metadata["updated_at"]
        
        # Count unique documents
        unique_files = set(chunk.source_file for chunk in self.chunks)
        self.metadata["total_documents"] = len(unique_files)
        
        logger.info(f"Vector store now contains {len(self.chunks)} chunks from {len(unique_files)} documents")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold (optional)
            
        Returns:
            List of SearchResult objects sorted by similarity score
        """
        if self.embeddings is None or len(self.chunks) == 0:
            logger.warning("Vector store is empty")
            return []
        
        try:
            # Ensure query embedding is 1D
            if query_embedding.ndim > 1:
                query_embedding = query_embedding.flatten()
            
            # Normalize query embedding (embeddings should already be normalized)
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
            
            # Calculate cosine similarity (dot product since vectors are normalized)
            similarities = np.dot(self.embeddings, query_embedding)
            
            # Get top-k indices
            top_k = min(top_k, len(self.chunks))
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Create SearchResult objects
            results = []
            for rank, idx in enumerate(top_indices, start=1):
                score = float(similarities[idx])
                
                # Apply threshold if specified
                if threshold is not None and score < threshold:
                    continue
                
                results.append(
                    SearchResult(
                        chunk=self.chunks[idx],
                        score=score,
                        rank=rank
                    )
                )
            
            logger.info(f"Found {len(results)} results (top_k={top_k}, threshold={threshold})")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise
    
    def get_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """
        Get chunks by their IDs.
        
        Args:
            chunk_ids: List of chunk IDs
            
        Returns:
            List of matching Chunk objects
        """
        id_set = set(chunk_ids)
        return [chunk for chunk in self.chunks if chunk.chunk_id in id_set]
    
    def get_by_source_file(self, source_file: str) -> List[Chunk]:
        """
        Get all chunks from a specific source file.
        
        Args:
            source_file: Source filename
            
        Returns:
            List of Chunk objects from that file
        """
        return [chunk for chunk in self.chunks if chunk.source_file == source_file]
    
    def delete_by_source_file(self, source_file: str):
        """
        Delete all chunks from a specific source file.
        
        Args:
            source_file: Source filename to delete
        """
        # Find indices to keep
        indices_to_keep = [
            i for i, chunk in enumerate(self.chunks)
            if chunk.source_file != source_file
        ]
        
        if len(indices_to_keep) == len(self.chunks):
            logger.warning(f"No chunks found for source file: {source_file}")
            return
        
        # Update chunks and embeddings
        self.chunks = [self.chunks[i] for i in indices_to_keep]
        
        if self.embeddings is not None:
            self.embeddings = self.embeddings[indices_to_keep]
        
        # Update metadata
        unique_files = set(chunk.source_file for chunk in self.chunks)
        self.metadata["total_documents"] = len(unique_files)
        self.metadata["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"Deleted chunks from {source_file}. Remaining: {len(self.chunks)} chunks")
    
    def save(self):
        """Save the vector store to disk."""
        try:
            # Ensure directory exists
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data
            data = {
                "embeddings": self.embeddings,
                "chunks": self.chunks,
                "metadata": self.metadata
            }
            
            # Save with pickle
            with open(self.store_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Get file size
            size_mb = self.store_path.stat().st_size / (1024 * 1024)
            logger.info(f"Vector store saved to {self.store_path} ({size_mb:.2f} MB)")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def load(self):
        """Load the vector store from disk."""
        try:
            if not self.store_path.exists():
                logger.warning(f"Vector store file not found: {self.store_path}")
                return
            
            # Check if file is empty
            if self.store_path.stat().st_size == 0:
                logger.warning(f"Vector store file is empty: {self.store_path}")
                return
            
            logger.info(f"Loading vector store from {self.store_path}")
            
            with open(self.store_path, 'rb') as f:
                data = pickle.load(f)
            
            self.embeddings = data.get("embeddings")
            self.chunks = data.get("chunks", [])
            self.metadata = data.get("metadata", self.metadata)
            
            logger.info(
                f"Loaded {len(self.chunks)} chunks from {self.metadata.get('total_documents', 0)} documents"
            )
            
        except EOFError:
            logger.warning(f"Vector store file is corrupted or empty: {self.store_path}")
            # Continue with empty store
            return
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def clear(self):
        """Clear all data from the vector store."""
        logger.warning("Clearing vector store")
        self.embeddings = None
        self.chunks = []
        self.metadata = {
            "version": "1.0",
            "created_at": None,
            "updated_at": None,
            "total_documents": 0,
            "embedding_dimension": settings.embedding_dimension
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_chunks": len(self.chunks),
            "total_documents": self.metadata.get("total_documents", 0),
            "embedding_dimension": self.metadata.get("embedding_dimension", 0),
            "created_at": self.metadata.get("created_at"),
            "updated_at": self.metadata.get("updated_at"),
        }
        
        # Calculate store size if file exists
        if self.store_path.exists():
            size_mb = self.store_path.stat().st_size / (1024 * 1024)
            stats["store_size_mb"] = round(size_mb, 2)
        else:
            stats["store_size_mb"] = 0.0
        
        # Calculate memory usage
        if self.embeddings is not None:
            memory_mb = self.embeddings.nbytes / (1024 * 1024)
            stats["memory_usage_mb"] = round(memory_mb, 2)
        else:
            stats["memory_usage_mb"] = 0.0
        
        # Document breakdown
        if self.chunks:
            file_counts = {}
            for chunk in self.chunks:
                file_counts[chunk.source_file] = file_counts.get(chunk.source_file, 0) + 1
            stats["documents"] = file_counts
        
        return stats
    
    def __len__(self) -> int:
        """Return number of chunks in store"""
        return len(self.chunks)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"VectorStore(chunks={len(self.chunks)}, documents={self.metadata.get('total_documents', 0)})"


# Global instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """
    Get the global vector store instance.
    
    Returns:
        VectorStore instance
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

