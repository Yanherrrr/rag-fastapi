"""
Result Re-ranking Module

This module provides advanced re-ranking capabilities to improve search result quality:
1. Cross-encoder re-ranking: Uses transformer models to score query-document relevance
2. Maximal Marginal Relevance (MMR): Diversifies results to reduce redundancy

Re-ranking is a second-stage refinement applied to initial search results.
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import CrossEncoder

from app.models.schemas import SearchResult

logger = logging.getLogger(__name__)


# ============= Cross-Encoder Re-ranking =============

class CrossEncoderReranker:
    """
    Cross-encoder based re-ranker.
    
    Unlike bi-encoders (which encode query and doc separately), cross-encoders
    process [query, document] as a single input, allowing the model to learn
    interactions between them. This is more accurate but computationally expensive.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder re-ranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model_name = model_name
        self.model: Optional[CrossEncoder] = None
        logger.info(f"Initializing CrossEncoderReranker with model: {model_name}")
    
    def load_model(self) -> None:
        """Load the cross-encoder model (lazy loading)."""
        if self.model is None:
            try:
                logger.info(f"Loading cross-encoder model: {self.model_name}")
                
                # Clear HuggingFace tokens to avoid 401 errors with public models
                import os
                os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)
                os.environ.pop('HF_TOKEN', None)
                
                self.model = CrossEncoder(self.model_name, max_length=512)
                logger.info("Cross-encoder model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model: {e}")
                logger.error("Try running: python -c \"from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')\"")
                raise
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Re-rank search results using cross-encoder.
        
        Args:
            query: Search query
            results: Initial search results
            top_k: Number of top results to return (None = all)
            
        Returns:
            Re-ranked search results
        """
        if not results:
            return []
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        logger.info(f"Re-ranking {len(results)} results with cross-encoder")
        
        # Prepare query-document pairs
        pairs = [[query, result.chunk.text] for result in results]
        
        # Get cross-encoder scores
        try:
            scores = self.model.predict(pairs)
            
            # Convert to numpy array if not already
            if not isinstance(scores, np.ndarray):
                scores = np.array(scores)
            
            logger.debug(f"Cross-encoder scores range: [{scores.min():.4f}, {scores.max():.4f}]")
        except Exception as e:
            logger.error(f"Error during cross-encoder prediction: {e}")
            return results  # Return original results on error
        
        # Create new SearchResult objects with updated scores
        reranked_results = []
        for idx, (result, score) in enumerate(zip(results, scores)):
            reranked_results.append(
                SearchResult(
                    chunk=result.chunk,
                    score=float(score),
                    rank=idx + 1  # Temporary rank, will update after sorting
                )
            )
        
        # Sort by new scores
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for rank, result in enumerate(reranked_results, start=1):
            result.rank = rank
        
        # Return top-k if specified
        if top_k is not None:
            reranked_results = reranked_results[:top_k]
        
        logger.info(f"Re-ranking complete, returning {len(reranked_results)} results")
        if reranked_results:
            logger.debug(f"Top score after re-ranking: {reranked_results[0].score:.4f}")
        
        return reranked_results


# Global reranker instance (singleton pattern)
_cross_encoder_reranker: Optional[CrossEncoderReranker] = None


def get_cross_encoder_reranker() -> CrossEncoderReranker:
    """
    Get the global cross-encoder reranker instance.
    
    Returns:
        CrossEncoderReranker instance
    """
    global _cross_encoder_reranker
    if _cross_encoder_reranker is None:
        _cross_encoder_reranker = CrossEncoderReranker()
    return _cross_encoder_reranker


def rerank_with_cross_encoder(
    query: str,
    results: List[SearchResult],
    top_k: Optional[int] = None
) -> List[SearchResult]:
    """
    Re-rank results using cross-encoder (convenience function).
    
    Args:
        query: Search query
        results: Initial search results
        top_k: Number of top results to return
        
    Returns:
        Re-ranked search results
    """
    reranker = get_cross_encoder_reranker()
    return reranker.rerank(query, results, top_k)


# ============= Maximal Marginal Relevance (MMR) =============

def compute_similarity_matrix(
    results: List[SearchResult],
    embeddings: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute similarity matrix between documents.
    
    Args:
        results: Search results
        embeddings: Pre-computed embeddings (if None, uses chunk texts)
        
    Returns:
        Similarity matrix (N x N)
    """
    n = len(results)
    
    if embeddings is not None:
        # Use provided embeddings
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings / norms
        
        # Compute cosine similarity matrix
        similarity_matrix = np.dot(normalized, normalized.T)
    else:
        # Fallback: use simple text overlap (Jaccard similarity)
        logger.warning("No embeddings provided, using text-based similarity (slower)")
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            tokens_i = set(results[i].chunk.text.lower().split())
            for j in range(i, n):
                tokens_j = set(results[j].chunk.text.lower().split())
                
                # Jaccard similarity
                intersection = len(tokens_i & tokens_j)
                union = len(tokens_i | tokens_j)
                similarity = intersection / union if union > 0 else 0.0
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
    
    return similarity_matrix


def maximal_marginal_relevance(
    results: List[SearchResult],
    lambda_param: float = 0.5,
    top_k: Optional[int] = None,
    embeddings: Optional[np.ndarray] = None
) -> List[SearchResult]:
    """
    Re-rank results using Maximal Marginal Relevance (MMR).
    
    MMR balances relevance and diversity. The formula:
        MMR = λ × Relevance(doc, query) - (1-λ) × max(Similarity(doc, selected_docs))
    
    Args:
        results: Search results (assumed sorted by relevance)
        lambda_param: Balance between relevance (1.0) and diversity (0.0)
        top_k: Number of results to return
        embeddings: Document embeddings for similarity computation
        
    Returns:
        Re-ranked results with diversity
    """
    if not results:
        return []
    
    if lambda_param < 0 or lambda_param > 1:
        raise ValueError("lambda_param must be between 0 and 1")
    
    n = len(results)
    if top_k is None:
        top_k = n
    
    top_k = min(top_k, n)
    
    logger.info(f"Applying MMR with λ={lambda_param} to {n} results, selecting {top_k}")
    
    # Compute similarity matrix between documents
    similarity_matrix = compute_similarity_matrix(results, embeddings)
    
    # Normalize relevance scores to [0, 1]
    relevance_scores = np.array([r.score for r in results])
    max_score = np.max(relevance_scores)
    min_score = np.min(relevance_scores)
    
    if max_score > min_score:
        relevance_scores = (relevance_scores - min_score) / (max_score - min_score)
    else:
        relevance_scores = np.ones_like(relevance_scores)
    
    # MMR algorithm
    selected_indices = []
    unselected_indices = list(range(n))
    
    # Select first document (highest relevance)
    first_idx = unselected_indices[0]
    selected_indices.append(first_idx)
    unselected_indices.remove(first_idx)
    
    # Iteratively select remaining documents
    while len(selected_indices) < top_k and unselected_indices:
        mmr_scores = []
        
        for idx in unselected_indices:
            # Relevance component
            relevance = relevance_scores[idx]
            
            # Diversity component (max similarity to already selected docs)
            if selected_indices:
                max_similarity = max(
                    similarity_matrix[idx, selected_idx]
                    for selected_idx in selected_indices
                )
            else:
                max_similarity = 0.0
            
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            mmr_scores.append((idx, mmr_score))
        
        # Select document with highest MMR score
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)
    
    # Build reranked results
    reranked_results = []
    for rank, idx in enumerate(selected_indices, start=1):
        result = results[idx]
        reranked_results.append(
            SearchResult(
                chunk=result.chunk,
                score=result.score,  # Keep original relevance score
                rank=rank
            )
        )
    
    logger.info(f"MMR re-ranking complete, selected {len(reranked_results)} diverse results")
    
    return reranked_results


# ============= Combined Re-ranking Pipeline =============

def rerank_results(
    query: str,
    results: List[SearchResult],
    methods: List[str] = ["cross_encoder", "mmr"],
    top_k: int = 5,
    mmr_lambda: float = 0.5,
    cross_encoder_top_k: Optional[int] = None,
    embeddings: Optional[np.ndarray] = None
) -> List[SearchResult]:
    """
    Apply multiple re-ranking methods in sequence.
    
    Recommended pipeline:
    1. Cross-encoder: Re-rank by true relevance (slow, so limit candidates)
    2. MMR: Diversify final results (fast)
    
    Args:
        query: Search query
        results: Initial search results
        methods: List of methods to apply in order ["cross_encoder", "mmr"]
        top_k: Final number of results to return
        mmr_lambda: Lambda parameter for MMR (0=diversity, 1=relevance)
        cross_encoder_top_k: How many results to keep after cross-encoder (None=all)
        embeddings: Document embeddings for MMR similarity computation
        
    Returns:
        Re-ranked results
    """
    if not results:
        return []
    
    logger.info(f"Starting re-ranking pipeline with methods: {methods}")
    
    current_results = results
    
    for method in methods:
        if method == "cross_encoder":
            logger.info("Applying cross-encoder re-ranking")
            current_results = rerank_with_cross_encoder(
                query=query,
                results=current_results,
                top_k=cross_encoder_top_k
            )
            
        elif method == "mmr":
            logger.info("Applying MMR diversification")
            current_results = maximal_marginal_relevance(
                results=current_results,
                lambda_param=mmr_lambda,
                top_k=top_k,
                embeddings=embeddings
            )
            
        else:
            logger.warning(f"Unknown re-ranking method: {method}")
    
    # Final top-k selection
    final_results = current_results[:top_k]
    
    logger.info(f"Re-ranking pipeline complete, returning {len(final_results)} results")
    
    return final_results


# ============= Utility Functions =============

def compare_rankings(
    original: List[SearchResult],
    reranked: List[SearchResult],
    top_k: int = 5
) -> dict:
    """
    Compare original and reranked results.
    
    Args:
        original: Original search results
        reranked: Re-ranked results
        top_k: Number of top results to compare
        
    Returns:
        Dict with comparison metrics
    """
    original_ids = [r.chunk.chunk_id for r in original[:top_k]]
    reranked_ids = [r.chunk.chunk_id for r in reranked[:top_k]]
    
    # Calculate metrics
    intersection = set(original_ids) & set(reranked_ids)
    
    # Rank changes
    rank_changes = []
    for reranked_rank, chunk_id in enumerate(reranked_ids, start=1):
        if chunk_id in original_ids:
            original_rank = original_ids.index(chunk_id) + 1
            rank_change = original_rank - reranked_rank
            rank_changes.append((chunk_id, original_rank, reranked_rank, rank_change))
    
    return {
        "top_k": top_k,
        "overlap": len(intersection),
        "new_in_top_k": len(set(reranked_ids) - set(original_ids)),
        "dropped_from_top_k": len(set(original_ids) - set(reranked_ids)),
        "rank_changes": rank_changes,
        "original_top_1": original_ids[0] if original_ids else None,
        "reranked_top_1": reranked_ids[0] if reranked_ids else None,
        "top_1_changed": (original_ids[0] != reranked_ids[0]) if original_ids and reranked_ids else False
    }

