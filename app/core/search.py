"""Search functionality including semantic and keyword search"""

import logging
from typing import List, Tuple, Optional
import numpy as np

from app.models.schemas import SearchResult, Chunk
from app.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ============= Low-Level Utilities =============

def compute_similarity_scores(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity scores between query and all documents.
    
    This is a low-level utility that just computes raw similarity scores
    without any filtering or selection.
    
    Args:
        query_embedding: Query embedding vector
        doc_embeddings: Document embedding matrix
        
    Returns:
        Array of similarity scores
    """
    # Ensure query is 1D and normalized
    if query_embedding.ndim > 1:
        query_embedding = query_embedding.flatten()
    
    query_norm = np.linalg.norm(query_embedding)
    if query_norm > 0:
        query_embedding = query_embedding / query_norm
    
    # Compute similarities (dot product for normalized vectors)
    similarities = np.dot(doc_embeddings, query_embedding)
    
    return similarities


def get_top_k_indices(
    scores: np.ndarray,
    k: int,
    threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get indices and scores of top-k elements.
    
    Args:
        scores: Array of scores
        k: Number of top elements to return
        threshold: Optional minimum score threshold
        
    Returns:
        Tuple of (indices, scores) for top-k elements
    """
    # Apply threshold if specified
    if threshold is not None:
        valid_mask = scores >= threshold
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return np.array([]), np.array([])
        
        scores = scores[valid_indices]
    else:
        valid_indices = np.arange(len(scores))
    
    # Get top-k
    k = min(k, len(scores))
    top_indices_in_valid = np.argsort(scores)[::-1][:k]
    top_indices = valid_indices[top_indices_in_valid]
    top_scores = scores[top_indices_in_valid]
    
    return top_indices, top_scores


# ============= Semantic Search =============

def cosine_similarity_search(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    top_k: int = 5,
    threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform cosine similarity search with top-k selection.
    
    This is a convenience function that combines similarity computation
    with threshold filtering and top-k selection. It delegates to lower-level
    functions to avoid code duplication.
    
    Args:
        query_embedding: Query embedding vector (normalized)
        doc_embeddings: Document embedding matrix (normalized)
        top_k: Number of top results to return
        threshold: Optional minimum similarity threshold
        
    Returns:
        Tuple of (indices, scores) for top-k results
    """
    if doc_embeddings.shape[0] == 0:
        return np.array([]), np.array([])
    
    # Compute all similarity scores (reuses common logic)
    similarities = compute_similarity_scores(query_embedding, doc_embeddings)
    
    # Use get_top_k_indices for selection (reuses common logic)
    top_indices, top_scores = get_top_k_indices(similarities, top_k, threshold)
    
    return top_indices, top_scores


def semantic_search(
    query_embedding: np.ndarray,
    vector_store: VectorStore,
    top_k: int = 5,
    threshold: Optional[float] = None
) -> List[SearchResult]:
    """
    Perform semantic search using cosine similarity.
    
    This is a convenience wrapper around the vector store's search method
    with additional logging and error handling.
    
    Args:
        query_embedding: Query embedding vector
        vector_store: Vector store containing documents
        top_k: Number of results to return
        threshold: Optional minimum similarity threshold
        
    Returns:
        List of SearchResult objects sorted by similarity score
    """
    try:
        logger.info(f"Performing semantic search (top_k={top_k}, threshold={threshold})")
        
        # Use vector store's built-in search
        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold
        )
        
        logger.info(f"Semantic search returned {len(results)} results")
        
        if results:
            logger.debug(f"Top result score: {results[0].score:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise


# ============= Advanced Semantic Search =============

def multi_vector_search(
    query_embeddings: List[np.ndarray],
    vector_store: VectorStore,
    top_k: int = 5,
    aggregation: str = "max"
) -> List[SearchResult]:
    """
    Perform search with multiple query embeddings.
    
    Useful for queries that can be broken into multiple sub-queries
    or for ensemble search.
    
    Args:
        query_embeddings: List of query embedding vectors
        vector_store: Vector store containing documents
        top_k: Number of results to return
        aggregation: How to combine scores ("max", "mean", "sum")
        
    Returns:
        List of SearchResult objects
    """
    if not query_embeddings:
        return []
    
    logger.info(f"Multi-vector search with {len(query_embeddings)} queries")
    
    # Get embeddings from store
    if vector_store.embeddings is None or len(vector_store.chunks) == 0:
        return []
    
    # Compute similarity for each query
    all_similarities = []
    for query_emb in query_embeddings:
        similarities = compute_similarity_scores(query_emb, vector_store.embeddings)
        all_similarities.append(similarities)
    
    # Aggregate scores
    all_similarities = np.array(all_similarities)
    
    if aggregation == "max":
        final_scores = np.max(all_similarities, axis=0)
    elif aggregation == "mean":
        final_scores = np.mean(all_similarities, axis=0)
    elif aggregation == "sum":
        final_scores = np.sum(all_similarities, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    # Get top-k
    top_indices, top_scores = get_top_k_indices(final_scores, top_k)
    
    # Build results
    results = []
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores), start=1):
        results.append(
            SearchResult(
                chunk=vector_store.chunks[idx],
                score=float(score),
                rank=rank
            )
        )
    
    logger.info(f"Multi-vector search returned {len(results)} results")
    return results


def search_with_score_breakdown(
    query_embedding: np.ndarray,
    vector_store: VectorStore,
    top_k: int = 5
) -> List[dict]:
    """
    Perform search and return detailed score breakdown.
    
    Useful for debugging and understanding search results.
    
    Args:
        query_embedding: Query embedding vector
        vector_store: Vector store containing documents
        top_k: Number of results to return
        
    Returns:
        List of dicts with chunk, score, and breakdown info
    """
    results = semantic_search(query_embedding, vector_store, top_k)
    
    detailed_results = []
    for result in results:
        detailed_results.append({
            "chunk": result.chunk,
            "score": result.score,
            "rank": result.rank,
            "breakdown": {
                "cosine_similarity": result.score,
                "normalized": True,
                "method": "dot_product"
            }
        })
    
    return detailed_results


# ============= Similarity Metrics =============

def euclidean_distance(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute Euclidean distance between query and documents.
    
    Note: Smaller distances = more similar.
    For normalized vectors, this is related to cosine similarity.
    
    Args:
        query_embedding: Query embedding vector
        doc_embeddings: Document embedding matrix
        
    Returns:
        Array of distances
    """
    # Ensure query is 1D
    if query_embedding.ndim > 1:
        query_embedding = query_embedding.flatten()
    
    # Compute L2 distances
    distances = np.linalg.norm(doc_embeddings - query_embedding, axis=1)
    
    return distances


def convert_distance_to_similarity(distances: np.ndarray) -> np.ndarray:
    """
    Convert distances to similarity scores.
    
    Uses formula: similarity = 1 / (1 + distance)
    
    Args:
        distances: Array of distances
        
    Returns:
        Array of similarity scores [0, 1]
    """
    return 1.0 / (1.0 + distances)


# ============= Search Quality Metrics =============

def calculate_search_quality_metrics(
    results: List[SearchResult],
    threshold: float = 0.6
) -> dict:
    """
    Calculate quality metrics for search results.
    
    Args:
        results: List of search results
        threshold: Similarity threshold for "good" results
        
    Returns:
        Dict with quality metrics
    """
    if not results:
        return {
            "total_results": 0,
            "avg_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "above_threshold": 0,
            "score_std": 0.0
        }
    
    scores = [r.score for r in results]
    
    return {
        "total_results": len(results),
        "avg_score": float(np.mean(scores)),
        "max_score": float(np.max(scores)),
        "min_score": float(np.min(scores)),
        "above_threshold": sum(1 for s in scores if s >= threshold),
        "score_std": float(np.std(scores)),
        "score_range": float(np.max(scores) - np.min(scores))
    }


def has_sufficient_evidence(
    results: List[SearchResult],
    threshold: float = 0.6,
    min_results: int = 1
) -> bool:
    """
    Check if search results have sufficient evidence.
    
    Args:
        results: List of search results
        threshold: Minimum similarity threshold
        min_results: Minimum number of results above threshold
        
    Returns:
        True if sufficient evidence exists
    """
    if not results:
        return False
    
    above_threshold = sum(1 for r in results if r.score >= threshold)
    
    return above_threshold >= min_results

