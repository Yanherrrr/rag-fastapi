"""Embedding generation using sentence-transformers"""

import logging
from typing import List, Union
import numpy as np

from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings for text using sentence-transformers.
    
    This class handles the initialization and usage of the embedding model,
    with support for batch processing and GPU acceleration if available.
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times"""
        if cls._instance is None:
            cls._instance = super(EmbeddingGenerator, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the embedding model (only once due to singleton)"""
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            try:
                # Clear any invalid HuggingFace tokens that might cause 401 errors
                import os
                os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)
                os.environ.pop('HF_TOKEN', None)
                
                # Try to load model with cache first, then download if needed
                self._model = SentenceTransformer(
                    settings.embedding_model,
                    cache_folder=None,  # Use default cache
                    use_auth_token=False  # Don't use auth for public models
                )
                self.embedding_dimension = self._model.get_sentence_embedding_dimension()
                
                # Log device information
                device = self._model.device
                logger.info(f"Embedding model loaded successfully on device: {device}")
                logger.info(f"Embedding dimension: {self.embedding_dimension}")
                
            except OSError as e:
                logger.error(f"Network/connection error loading model: {e}")
                logger.error("Please ensure you have internet connection for first-time model download")
                logger.error("Or manually download the model first")
                raise RuntimeError(
                    f"Failed to load embedding model '{settings.embedding_model}'. "
                    "This usually happens on first run when the model needs to be downloaded. "
                    "Please ensure you have internet connection or manually download the model."
                ) from e
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing (default: 32)
            
        Returns:
            numpy array of shape (len(texts), embedding_dimension)
        """
        if not texts:
            return np.array([])
        
        try:
            # Filter out empty texts
            non_empty_texts = [text if text.strip() else " " for text in texts]
            
            logger.info(f"Generating embeddings for {len(non_empty_texts)} texts...")
            
            # Generate embeddings in batches
            embeddings = self._model.encode(
                non_empty_texts,
                batch_size=batch_size,
                show_progress_bar=len(non_empty_texts) > 100,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            numpy array of shape (embedding_dimension,)
        """
        if not text.strip():
            text = " "
        
        try:
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dimension
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query string for semantic search.
        Alias for generate_single_embedding for clarity.
        
        Args:
            query: Query string
            
        Returns:
            Query embedding vector
        """
        return self.generate_single_embedding(query)


# Global instance
_embedding_generator = None


def get_embedding_generator() -> EmbeddingGenerator:
    """
    Get the global embedding generator instance.
    
    Returns:
        EmbeddingGenerator instance
    """
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator


def generate_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Convenience function to generate embeddings.
    
    Args:
        texts: List of text strings
        batch_size: Batch size for processing
        
    Returns:
        numpy array of embeddings
    """
    generator = get_embedding_generator()
    return generator.generate_embeddings(texts, batch_size)


def generate_query_embedding(query: str) -> np.ndarray:
    """
    Convenience function to generate query embedding.
    
    Args:
        query: Query string
        
    Returns:
        Query embedding vector
    """
    generator = get_embedding_generator()
    return generator.encode_query(query)

