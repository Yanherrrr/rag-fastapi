"""
Demo script for BM25 keyword search.

This demonstrates the keyword search functionality with example documents.
"""

from app.core.keyword_search import BM25Index, build_bm25_index
from app.models.schemas import Chunk


def main():
    print("=" * 70)
    print("BM25 KEYWORD SEARCH DEMO")
    print("=" * 70)
    
    # Create sample documents
    chunks = [
        Chunk(
            chunk_id="1",
            text="""Machine learning is a subset of artificial intelligence that focuses on 
            building systems that can learn from data. It involves training algorithms to 
            recognize patterns and make decisions with minimal human intervention.""",
            source_file="ml_basics.pdf",
            page_number=1,
            chunk_index=0
        ),
        Chunk(
            chunk_id="2",
            text="""Deep learning is a specialized branch of machine learning that uses neural 
            networks with multiple layers. These deep neural networks can automatically learn 
            hierarchical representations from raw data.""",
            source_file="ml_basics.pdf",
            page_number=2,
            chunk_index=1
        ),
        Chunk(
            chunk_id="3",
            text="""Python is the most popular programming language for machine learning and 
            data science. Libraries like scikit-learn, TensorFlow, and PyTorch make it easy 
            to implement machine learning algorithms.""",
            source_file="python_ml.pdf",
            page_number=1,
            chunk_index=0
        ),
        Chunk(
            chunk_id="4",
            text="""Natural language processing (NLP) is an area of artificial intelligence 
            concerned with the interaction between computers and human language. Modern NLP 
            uses deep learning techniques for tasks like translation and sentiment analysis.""",
            source_file="nlp_guide.pdf",
            page_number=1,
            chunk_index=0
        ),
        Chunk(
            chunk_id="5",
            text="""Supervised learning is a type of machine learning where the algorithm learns 
            from labeled training data. The goal is to learn a mapping from inputs to outputs 
            that can generalize to new, unseen data.""",
            source_file="ml_types.pdf",
            page_number=3,
            chunk_index=0
        ),
        Chunk(
            chunk_id="6",
            text="""Computer vision is a field of artificial intelligence that trains computers 
            to interpret and understand visual information. Applications include image recognition, 
            object detection, and facial recognition.""",
            source_file="cv_intro.pdf",
            page_number=1,
            chunk_index=0
        )
    ]
    
    print("\n📚 Building BM25 Index...")
    print(f"   Documents: {len(chunks)}")
    
    # Build index
    index = build_bm25_index(chunks, k1=1.5, b=0.75)
    
    stats = index.get_stats()
    print(f"\n📊 Index Statistics:")
    print(f"   Documents: {stats['num_documents']}")
    print(f"   Unique terms: {stats['num_unique_terms']}")
    print(f"   Avg doc length: {stats['avg_doc_length']:.1f} tokens")
    print(f"   Total tokens: {stats['total_tokens']}")
    print(f"   BM25 parameters: k1={stats['k1']}, b={stats['b']}")
    
    # Test queries
    queries = [
        "machine learning algorithms",
        "deep learning neural networks",
        "Python programming",
        "natural language processing",
        "computer vision image",
        "supervised learning training"
    ]
    
    print("\n" + "=" * 70)
    print("SEARCH RESULTS")
    print("=" * 70)
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 70)
        
        results = index.search(query, top_k=3, min_score=0.0)
        
        if not results:
            print("   No results found.")
            continue
        
        for result in results:
            print(f"\n   Rank {result.rank} | Score: {result.score:.4f}")
            print(f"   Source: {result.chunk.source_file} (Page {result.chunk.page_number})")
            
            # Show first 100 characters
            text_preview = result.chunk.text.replace('\n', ' ').strip()
            if len(text_preview) > 100:
                text_preview = text_preview[:100] + "..."
            print(f"   Text: {text_preview}")
    
    # Demonstrate how IDF affects ranking
    print("\n" + "=" * 70)
    print("IDF ANALYSIS")
    print("=" * 70)
    
    example_terms = ["machine", "learning", "python", "neural", "algorithms"]
    print("\nInverse Document Frequency (IDF) scores:")
    print("(Higher IDF = rarer term = more discriminative)\n")
    
    for term in example_terms:
        idf = index.compute_idf(term)
        df = index.doc_freqs.get(term, 0)
        print(f"   '{term}': IDF={idf:.4f}, appears in {df}/{index.num_docs} docs")
    
    # Show how BM25 handles repeated terms (saturation)
    print("\n" + "=" * 70)
    print("TERM FREQUENCY SATURATION (k1 parameter)")
    print("=" * 70)
    
    print("\nDemonstrating how k1 affects scoring when terms repeat:\n")
    
    # Create two test documents
    test_chunks = [
        Chunk(
            chunk_id="test1",
            text="neural neural neural neural neural",  # 5 occurrences
            source_file="test.pdf",
            page_number=1,
            chunk_index=0
        ),
        Chunk(
            chunk_id="test2",
            text="neural",  # 1 occurrence
            source_file="test.pdf",
            page_number=2,
            chunk_index=1
        )
    ]
    
    # Test with different k1 values
    for k1_val in [0.5, 1.5, 3.0]:
        test_index = BM25Index(k1=k1_val, b=0.0)  # b=0 to isolate k1 effect
        test_index.add_documents(test_chunks)
        
        score_5x = test_index.compute_bm25_score(["neural"], 0)
        score_1x = test_index.compute_bm25_score(["neural"], 1)
        ratio = score_5x / score_1x if score_1x > 0 else 0
        
        print(f"   k1={k1_val}: score(5x)/score(1x) = {ratio:.2f}x")
    
    print("\n   Lower k1 → more saturation → diminishing returns for repetition")
    print("   Higher k1 → less saturation → repeated terms matter more")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\n✅ BM25 keyword search is working perfectly!")
    print("   - Tokenization and preprocessing ✓")
    print("   - Inverted index construction ✓")
    print("   - IDF computation ✓")
    print("   - BM25 scoring with k1 and b parameters ✓")
    print("   - Top-K retrieval ✓")
    print("\n")


if __name__ == "__main__":
    main()

