"""
BM25 Keyword Search Implementation

This module implements BM25 (Best Matching 25), a probabilistic ranking function
used for keyword-based document retrieval. BM25 is more sophisticated than
simple TF-IDF as it includes document length normalization and saturation.

BM25 Formula:
    score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D| / avgdl))
    
Where:
    - D: document
    - Q: query
    - qi: query term i
    - f(qi,D): frequency of qi in D
    - |D|: length of document D
    - avgdl: average document length
    - k1: term frequency saturation parameter (typically 1.5)
    - b: length normalization parameter (typically 0.75)
    - IDF(qi): inverse document frequency of qi
"""

import re
import logging
import math
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
import numpy as np

from app.models.schemas import Chunk, SearchResult

logger = logging.getLogger(__name__)


# ============= Text Processing =============

def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Tokenize text into terms.
    
    Args:
        text: Input text
        lowercase: Whether to lowercase tokens
        
    Returns:
        List of tokens
    """
    # Convert to lowercase if needed
    if lowercase:
        text = text.lower()
    
    # Split on non-alphanumeric characters, keep words
    # This regex keeps words with numbers (e.g., "python3", "ml2023")
    tokens = re.findall(r'\b\w+\b', text)
    
    return tokens


def remove_stopwords(tokens: List[str], stopwords: Optional[Set[str]] = None) -> List[str]:
    """
    Remove common stopwords from tokens.
    
    Args:
        tokens: List of tokens
        stopwords: Set of stopwords to remove (if None, uses default)
        
    Returns:
        Filtered list of tokens
    """
    if stopwords is None:
        # Common English stopwords
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
        }
    
    return [token for token in tokens if token not in stopwords]


def preprocess_text(text: str, remove_stops: bool = True) -> List[str]:
    """
    Preprocess text: tokenize and optionally remove stopwords.
    
    Args:
        text: Input text
        remove_stops: Whether to remove stopwords
        
    Returns:
        List of processed tokens
    """
    tokens = tokenize(text)
    
    if remove_stops:
        tokens = remove_stopwords(tokens)
    
    return tokens


# ============= BM25 Index =============

class BM25Index:
    """
    BM25 inverted index for keyword search.
    
    This index stores term frequencies, document frequencies, and other
    statistics needed for BM25 scoring.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index.
        
        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Length normalization parameter (0.0-1.0)
        """
        self.k1 = k1
        self.b = b
        
        # Index structures
        self.chunks: List[Chunk] = []
        self.inverted_index: Dict[str, List[int]] = defaultdict(list)  # term -> doc_ids
        self.term_freqs: List[Dict[str, int]] = []  # doc_id -> {term: freq}
        self.doc_lengths: List[int] = []  # doc_id -> length
        self.doc_freqs: Dict[str, int] = defaultdict(int)  # term -> num_docs_containing
        self.avg_doc_length: float = 0.0
        self.num_docs: int = 0
        
        logger.info(f"Initialized BM25Index with k1={k1}, b={b}")
    
    def add_documents(self, chunks: List[Chunk]) -> None:
        """
        Add documents to the index.
        
        Args:
            chunks: List of document chunks to index
        """
        logger.info(f"Adding {len(chunks)} documents to BM25 index")
        
        start_idx = len(self.chunks)
        self.chunks.extend(chunks)
        
        for doc_id, chunk in enumerate(chunks, start=start_idx):
            # Tokenize document
            tokens = preprocess_text(chunk.text)
            
            # Calculate term frequencies
            term_freq = Counter(tokens)
            self.term_freqs.append(term_freq)
            
            # Store document length
            doc_length = len(tokens)
            self.doc_lengths.append(doc_length)
            
            # Update inverted index and document frequencies
            for term in term_freq.keys():
                if doc_id not in self.inverted_index[term]:
                    self.inverted_index[term].append(doc_id)
                    self.doc_freqs[term] += 1
        
        # Update statistics
        self.num_docs = len(self.chunks)
        self.avg_doc_length = np.mean(self.doc_lengths) if self.doc_lengths else 0.0
        
        logger.info(
            f"Index updated: {self.num_docs} docs, "
            f"{len(self.inverted_index)} unique terms, "
            f"avg_length={self.avg_doc_length:.1f}"
        )
    
    def compute_idf(self, term: str) -> float:
        """
        Compute IDF (Inverse Document Frequency) for a term.
        
        IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
        Where N = total docs, df = docs containing term
        
        Args:
            term: The term
            
        Returns:
            IDF score
        """
        df = self.doc_freqs.get(term, 0)
        
        # BM25 IDF formula (prevents negative values)
        idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
        
        return idf
    
    def compute_bm25_score(self, query_terms: List[str], doc_id: int) -> float:
        """
        Compute BM25 score for a document given query terms.
        
        Args:
            query_terms: List of query terms
            doc_id: Document ID
            
        Returns:
            BM25 score
        """
        if doc_id >= len(self.term_freqs):
            return 0.0
        
        score = 0.0
        term_freq = self.term_freqs[doc_id]
        doc_length = self.doc_lengths[doc_id]
        
        for term in query_terms:
            if term not in term_freq:
                continue
            
            # Get term frequency in document
            tf = term_freq[term]
            
            # Compute IDF
            idf = self.compute_idf(term)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            min_score: Minimum BM25 score threshold
            
        Returns:
            List of SearchResult objects, sorted by score
        """
        if self.num_docs == 0:
            logger.warning("BM25 index is empty")
            return []
        
        # Preprocess query
        query_terms = preprocess_text(query)
        
        if not query_terms:
            logger.warning("Query resulted in no terms after preprocessing")
            return []
        
        logger.info(f"BM25 search for query terms: {query_terms}")
        
        # Get candidate documents (documents containing at least one query term)
        candidate_docs: Set[int] = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term])
        
        logger.debug(f"Found {len(candidate_docs)} candidate documents")
        
        if not candidate_docs:
            return []
        
        # Compute BM25 scores for candidates
        scores = []
        for doc_id in candidate_docs:
            score = self.compute_bm25_score(query_terms, doc_id)
            if score >= min_score:
                scores.append((doc_id, score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top-k
        scores = scores[:top_k]
        
        # Build search results
        results = []
        for rank, (doc_id, score) in enumerate(scores, start=1):
            results.append(
                SearchResult(
                    chunk=self.chunks[doc_id],
                    score=float(score),
                    rank=rank
                )
            )
        
        logger.info(f"BM25 search returned {len(results)} results")
        if results:
            logger.debug(f"Top score: {results[0].score:.4f}")
        
        return results
    
    def get_stats(self) -> Dict:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            "num_documents": self.num_docs,
            "num_unique_terms": len(self.inverted_index),
            "avg_doc_length": self.avg_doc_length,
            "total_tokens": sum(self.doc_lengths),
            "k1": self.k1,
            "b": self.b
        }
    
    def clear(self) -> None:
        """Clear the index."""
        self.chunks = []
        self.inverted_index = defaultdict(list)
        self.term_freqs = []
        self.doc_lengths = []
        self.doc_freqs = defaultdict(int)
        self.avg_doc_length = 0.0
        self.num_docs = 0
        logger.info("BM25 index cleared")
    
    def __len__(self) -> int:
        """Return number of documents in index."""
        return self.num_docs
    
    def __repr__(self) -> str:
        """String representation."""
        return f"BM25Index(docs={self.num_docs}, terms={len(self.inverted_index)})"


# ============= Convenience Functions =============

# Global index instance (singleton pattern)
_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """
    Get the global BM25 index instance.
    
    Returns:
        BM25Index instance
    """
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index


def keyword_search(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    index: Optional[BM25Index] = None
) -> List[SearchResult]:
    """
    Perform keyword search using BM25.
    
    Convenience function that uses the global index by default.
    
    Args:
        query: Search query
        top_k: Number of top results to return
        min_score: Minimum BM25 score threshold
        index: Optional BM25Index instance (uses global if None)
        
    Returns:
        List of SearchResult objects
    """
    if index is None:
        index = get_bm25_index()
    
    return index.search(query, top_k, min_score)


def build_bm25_index(chunks: List[Chunk], k1: float = 1.5, b: float = 0.75) -> BM25Index:
    """
    Build a new BM25 index from chunks.
    
    Args:
        chunks: List of document chunks
        k1: BM25 k1 parameter
        b: BM25 b parameter
        
    Returns:
        Initialized BM25Index
    """
    index = BM25Index(k1=k1, b=b)
    index.add_documents(chunks)
    return index

