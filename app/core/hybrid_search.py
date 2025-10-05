"""
Hybrid Search Implementation

This module combines semantic (vector) search and keyword (BM25) search
to leverage the strengths of both approaches:
- Semantic search: understands meaning and context
- Keyword search: finds exact term matches

Multiple fusion strategies are supported:
- Weighted combination: α×semantic + (1-α)×keyword
- Reciprocal Rank Fusion (RRF): rank-based combination
- Maximum score: take the best score from either method
"""

import logging
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import numpy as np

from app.models.schemas import SearchResult, Chunk
from app.core.search import semantic_search, compute_similarity_scores
from app.core.keyword_search import BM25Index, keyword_search
from app.storage.vector_store import VectorStore
from app.core.embeddings import generate_query_embedding

logger = logging.getLogger(__name__)


# ============= Score Normalization =============

def normalize_scores_minmax(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores to [0, 1] range using min-max normalization.
    
    Args:
        scores: Array of scores
        
    Returns:
        Normalized scores
    """
    if len(scores) == 0:
        return scores
    
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        # All scores are the same
        return np.ones_like(scores)
    
    return (scores - min_score) / (max_score - min_score)


def normalize_scores_zscore(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores using z-score normalization.
    
    Args:
        scores: Array of scores
        
    Returns:
        Normalized scores (mean=0, std=1)
    """
    if len(scores) == 0:
        return scores
    
    mean = np.mean(scores)
    std = np.std(scores)
    
    if std == 0:
        return np.zeros_like(scores)
    
    return (scores - mean) / std


def normalize_scores_softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Normalize scores using softmax (converts to probability distribution).
    
    Args:
        scores: Array of scores
        temperature: Temperature parameter (lower = more peaked)
        
    Returns:
        Normalized scores (sum to 1)
    """
    if len(scores) == 0:
        return scores
    
    # Apply temperature scaling
    scores = scores / temperature
    
    # Subtract max for numerical stability
    scores = scores - np.max(scores)
    
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)


# ============= Score Fusion Strategies =============

def weighted_score_fusion(
    semantic_scores: Dict[str, float],
    keyword_scores: Dict[str, float],
    alpha: float = 0.5
) -> Dict[str, float]:
    """
    Combine scores using weighted average.
    
    Formula: final_score = α × semantic_score + (1-α) × keyword_score
    
    Args:
        semantic_scores: Dict mapping chunk_id to semantic score
        keyword_scores: Dict mapping chunk_id to keyword score
        alpha: Weight for semantic scores (0-1)
        
    Returns:
        Dict mapping chunk_id to combined score
    """
    # Get all unique chunk IDs
    all_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
    
    combined_scores = {}
    for chunk_id in all_ids:
        sem_score = semantic_scores.get(chunk_id, 0.0)
        kw_score = keyword_scores.get(chunk_id, 0.0)
        
        combined_scores[chunk_id] = alpha * sem_score + (1 - alpha) * kw_score
    
    return combined_scores


def reciprocal_rank_fusion(
    semantic_results: List[SearchResult],
    keyword_results: List[SearchResult],
    k: int = 60
) -> Dict[str, float]:
    """
    Combine results using Reciprocal Rank Fusion (RRF).
    
    RRF formula: score(d) = Σ 1/(k + rank_i(d))
    
    This is a rank-based method that doesn't require score normalization.
    It's particularly robust when scores are on different scales.
    
    Args:
        semantic_results: Results from semantic search
        keyword_results: Results from keyword search
        k: Constant to prevent high ranks from dominating (typically 60)
        
    Returns:
        Dict mapping chunk_id to RRF score
    """
    rrf_scores = defaultdict(float)
    
    # Add semantic search ranks
    for rank, result in enumerate(semantic_results, start=1):
        chunk_id = result.chunk.chunk_id
        rrf_scores[chunk_id] += 1.0 / (k + rank)
    
    # Add keyword search ranks
    for rank, result in enumerate(keyword_results, start=1):
        chunk_id = result.chunk.chunk_id
        rrf_scores[chunk_id] += 1.0 / (k + rank)
    
    return dict(rrf_scores)


def max_score_fusion(
    semantic_scores: Dict[str, float],
    keyword_scores: Dict[str, float]
) -> Dict[str, float]:
    """
    Combine scores by taking the maximum of each.
    
    This is useful when you want to surface documents that are highly
    relevant by either metric.
    
    Args:
        semantic_scores: Dict mapping chunk_id to semantic score
        keyword_scores: Dict mapping chunk_id to keyword score
        
    Returns:
        Dict mapping chunk_id to max score
    """
    all_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
    
    combined_scores = {}
    for chunk_id in all_ids:
        sem_score = semantic_scores.get(chunk_id, 0.0)
        kw_score = keyword_scores.get(chunk_id, 0.0)
        
        combined_scores[chunk_id] = max(sem_score, kw_score)
    
    return combined_scores


# ============= Hybrid Search =============

def hybrid_search(
    query: str,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    top_k: int = 5,
    semantic_weight: float = 0.5,
    fusion_method: str = "weighted",
    normalize_method: str = "minmax",
    rrf_k: int = 60
) -> List[SearchResult]:
    """
    Perform hybrid search combining semantic and keyword search.
    
    Args:
        query: Search query
        vector_store: Vector store for semantic search
        bm25_index: BM25 index for keyword search
        top_k: Number of top results to return
        semantic_weight: Weight for semantic scores (0-1) [for weighted fusion]
        fusion_method: How to combine scores ("weighted", "rrf", "max")
        normalize_method: How to normalize scores ("minmax", "zscore", "softmax")
        rrf_k: RRF constant [for rrf fusion]
        
    Returns:
        List of SearchResult objects, sorted by combined score
    """
    logger.info(f"Hybrid search: query='{query}', method={fusion_method}")
    
    # Generate query embedding for semantic search
    query_embedding = generate_query_embedding(query)
    
    # Perform both searches (get more results than needed for fusion)
    retrieval_k = top_k * 3  # Retrieve 3x more for better fusion
    
    semantic_results = semantic_search(
        query_embedding,
        vector_store,
        top_k=retrieval_k
    )
    
    keyword_results = bm25_index.search(
        query,
        top_k=retrieval_k
    )
    
    logger.debug(
        f"Retrieved {len(semantic_results)} semantic, "
        f"{len(keyword_results)} keyword results"
    )
    
    # Handle edge cases
    if not semantic_results and not keyword_results:
        logger.warning("No results from either search method")
        return []
    
    if not semantic_results:
        logger.info("No semantic results, using keyword only")
        return keyword_results[:top_k]
    
    if not keyword_results:
        logger.info("No keyword results, using semantic only")
        return semantic_results[:top_k]
    
    # Combine results based on fusion method
    if fusion_method == "rrf":
        # RRF doesn't need score normalization
        combined_scores = reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            k=rrf_k
        )
    else:
        # For score-based fusion, normalize scores first
        
        # Extract scores into arrays
        sem_scores_dict = {r.chunk.chunk_id: r.score for r in semantic_results}
        kw_scores_dict = {r.chunk.chunk_id: r.score for r in keyword_results}
        
        # Normalize scores
        if normalize_method == "minmax":
            sem_scores_array = np.array(list(sem_scores_dict.values()))
            kw_scores_array = np.array(list(kw_scores_dict.values()))
            
            sem_scores_norm = normalize_scores_minmax(sem_scores_array)
            kw_scores_norm = normalize_scores_minmax(kw_scores_array)
            
            sem_scores_dict = dict(zip(sem_scores_dict.keys(), sem_scores_norm))
            kw_scores_dict = dict(zip(kw_scores_dict.keys(), kw_scores_norm))
        
        elif normalize_method == "zscore":
            sem_scores_array = np.array(list(sem_scores_dict.values()))
            kw_scores_array = np.array(list(kw_scores_dict.values()))
            
            sem_scores_norm = normalize_scores_zscore(sem_scores_array)
            kw_scores_norm = normalize_scores_zscore(kw_scores_array)
            
            # Convert z-scores to positive range (shift and scale)
            sem_scores_norm = sem_scores_norm - np.min(sem_scores_norm)
            kw_scores_norm = kw_scores_norm - np.min(kw_scores_norm)
            
            sem_scores_dict = dict(zip(sem_scores_dict.keys(), sem_scores_norm))
            kw_scores_dict = dict(zip(kw_scores_dict.keys(), kw_scores_norm))
        
        # Apply fusion strategy
        if fusion_method == "weighted":
            combined_scores = weighted_score_fusion(
                sem_scores_dict,
                kw_scores_dict,
                alpha=semantic_weight
            )
        elif fusion_method == "max":
            combined_scores = max_score_fusion(
                sem_scores_dict,
                kw_scores_dict
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    # Build chunk lookup (from both result sets)
    chunk_lookup: Dict[str, Chunk] = {}
    for result in semantic_results + keyword_results:
        chunk_lookup[result.chunk.chunk_id] = result.chunk
    
    # Sort by combined score and take top-k
    sorted_items = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    # Build final results
    results = []
    for rank, (chunk_id, score) in enumerate(sorted_items, start=1):
        if chunk_id in chunk_lookup:
            results.append(
                SearchResult(
                    chunk=chunk_lookup[chunk_id],
                    score=float(score),
                    rank=rank
                )
            )
    
    logger.info(f"Hybrid search returned {len(results)} results")
    if results:
        logger.debug(f"Top score: {results[0].score:.4f}")
    
    return results


def hybrid_search_with_fallback(
    query: str,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    top_k: int = 5,
    semantic_threshold: float = 0.6,
    **kwargs
) -> Tuple[List[SearchResult], str]:
    """
    Perform hybrid search with intelligent fallback.
    
    Strategy:
    1. Try hybrid search
    2. If semantic results are poor, use keyword only
    3. If keyword results are poor, use semantic only
    4. Return results with method used
    
    Args:
        query: Search query
        vector_store: Vector store for semantic search
        bm25_index: BM25 index for keyword search
        top_k: Number of results
        semantic_threshold: Minimum semantic score for good results
        **kwargs: Additional arguments for hybrid_search
        
    Returns:
        Tuple of (results, method_used)
    """
    # Try semantic search first to check quality
    query_embedding = generate_query_embedding(query)
    semantic_results = semantic_search(query_embedding, vector_store, top_k=top_k)
    
    # Check if semantic results are good
    has_good_semantic = (
        len(semantic_results) > 0 and
        semantic_results[0].score >= semantic_threshold
    )
    
    # Try keyword search
    keyword_results = bm25_index.search(query, top_k=top_k)
    has_keyword = len(keyword_results) > 0
    
    if has_good_semantic and has_keyword:
        # Both are good, use hybrid
        results = hybrid_search(
            query,
            vector_store,
            bm25_index,
            top_k=top_k,
            **kwargs
        )
        return results, "hybrid"
    
    elif has_good_semantic and not has_keyword:
        # Only semantic is good
        logger.info("Using semantic search (no keyword matches)")
        return semantic_results, "semantic"
    
    elif not has_good_semantic and has_keyword:
        # Only keyword is good
        logger.info("Using keyword search (low semantic confidence)")
        return keyword_results, "keyword"
    
    else:
        # Neither is great, but try hybrid anyway
        if semantic_results or keyword_results:
            logger.warning("Low confidence results, using hybrid as fallback")
            results = hybrid_search(
                query,
                vector_store,
                bm25_index,
                top_k=top_k,
                **kwargs
            )
            return results, "hybrid_low_confidence"
        else:
            logger.warning("No results from any method")
            return [], "none"


# ============= Analysis and Debugging =============

def compare_search_methods(
    query: str,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    top_k: int = 5
) -> Dict:
    """
    Compare results from different search methods.
    
    Useful for understanding how different methods perform.
    
    Args:
        query: Search query
        vector_store: Vector store
        bm25_index: BM25 index
        top_k: Number of results
        
    Returns:
        Dict with results from each method and comparison stats
    """
    query_embedding = generate_query_embedding(query)
    
    # Get results from each method
    semantic_results = semantic_search(query_embedding, vector_store, top_k=top_k)
    keyword_results = bm25_index.search(query, top_k=top_k)
    hybrid_results = hybrid_search(
        query,
        vector_store,
        bm25_index,
        top_k=top_k,
        fusion_method="weighted",
        semantic_weight=0.5
    )
    hybrid_rrf_results = hybrid_search(
        query,
        vector_store,
        bm25_index,
        top_k=top_k,
        fusion_method="rrf"
    )
    
    # Calculate overlap
    sem_ids = {r.chunk.chunk_id for r in semantic_results}
    kw_ids = {r.chunk.chunk_id for r in keyword_results}
    hybrid_ids = {r.chunk.chunk_id for r in hybrid_results}
    
    return {
        "query": query,
        "results": {
            "semantic": semantic_results,
            "keyword": keyword_results,
            "hybrid_weighted": hybrid_results,
            "hybrid_rrf": hybrid_rrf_results
        },
        "overlap": {
            "semantic_keyword": len(sem_ids & kw_ids),
            "semantic_only": len(sem_ids - kw_ids),
            "keyword_only": len(kw_ids - sem_ids),
            "hybrid_from_semantic": len(hybrid_ids & sem_ids),
            "hybrid_from_keyword": len(hybrid_ids & kw_ids)
        },
        "counts": {
            "semantic": len(semantic_results),
            "keyword": len(keyword_results),
            "hybrid": len(hybrid_results)
        }
    }

