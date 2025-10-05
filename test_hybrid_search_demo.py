"""
Demo script for hybrid search combining semantic and keyword search.

This demonstrates how hybrid search leverages the strengths of both:
- Semantic search: understands meaning and context
- Keyword search: finds exact term matches
"""

from app.core.hybrid_search import (
    hybrid_search,
    hybrid_search_with_fallback,
    compare_search_methods
)
from app.storage.vector_store import VectorStore
from app.core.keyword_search import BM25Index
from app.models.schemas import Chunk
from app.core.embeddings import generate_embeddings


def main():
    print("=" * 80)
    print("HYBRID SEARCH DEMO")
    print("Combining Semantic (Vector) Search + Keyword (BM25) Search")
    print("=" * 80)
    
    # Create sample documents
    chunks = [
        Chunk(
            chunk_id="1",
            text="""Machine learning is a method of data analysis that automates analytical 
            model building. It is a branch of artificial intelligence based on the idea that 
            systems can learn from data, identify patterns and make decisions.""",
            source_file="ml_intro.pdf",
            page_number=1,
            chunk_index=0
        ),
        Chunk(
            chunk_id="2",
            text="""Deep learning is part of a broader family of machine learning methods based 
            on artificial neural networks with representation learning. Learning can be 
            supervised, semi-supervised or unsupervised.""",
            source_file="dl_basics.pdf",
            page_number=1,
            chunk_index=0
        ),
        Chunk(
            chunk_id="3",
            text="""Python is an interpreted, high-level programming language for general-purpose 
            programming. It has become the de facto standard for machine learning and data 
            science due to libraries like NumPy, Pandas, and Scikit-learn.""",
            source_file="python_guide.pdf",
            page_number=2,
            chunk_index=0
        ),
        Chunk(
            chunk_id="4",
            text="""Natural language processing (NLP) enables computers to understand, interpret, 
            and generate human language. Modern NLP systems use deep learning models like 
            transformers to achieve state-of-the-art results.""",
            source_file="nlp_overview.pdf",
            page_number=1,
            chunk_index=0
        ),
        Chunk(
            chunk_id="5",
            text="""Computer vision is an interdisciplinary field that deals with how computers 
            can gain high-level understanding from digital images or videos. It seeks to 
            automate tasks that the human visual system can do.""",
            source_file="cv_intro.pdf",
            page_number=1,
            chunk_index=0
        ),
        Chunk(
            chunk_id="6",
            text="""Supervised learning algorithms learn from labeled training data to make 
            predictions on new data. Examples include linear regression, logistic regression, 
            decision trees, and neural networks.""",
            source_file="supervised_learning.pdf",
            page_number=3,
            chunk_index=0
        )
    ]
    
    print(f"\n📚 Building Search Indexes...")
    print(f"   Documents: {len(chunks)}")
    
    # Build vector store
    vector_store = VectorStore(store_path=None)
    embeddings = generate_embeddings([c.text for c in chunks])
    vector_store.add_documents(chunks, embeddings)
    
    # Build BM25 index
    bm25_index = BM25Index()
    bm25_index.add_documents(chunks)
    
    print(f"\n✅ Indexes Ready:")
    print(f"   Vector Store: {len(vector_store)} documents")
    print(f"   BM25 Index: {len(bm25_index)} documents")
    
    # Test queries
    test_queries = [
        {
            "query": "artificial intelligence learning systems",
            "description": "Conceptual query (semantic should excel)"
        },
        {
            "query": "Python programming NumPy Pandas",
            "description": "Exact terms (keyword should excel)"
        },
        {
            "query": "deep learning neural networks",
            "description": "Mixed query (both should help)"
        }
    ]
    
    print("\n" + "=" * 80)
    print("COMPARING SEARCH METHODS")
    print("=" * 80)
    
    for test_case in test_queries:
        query = test_case["query"]
        description = test_case["description"]
        
        print(f"\n🔍 Query: '{query}'")
        print(f"   Type: {description}")
        print("-" * 80)
        
        comparison = compare_search_methods(
            query=query,
            vector_store=vector_store,
            bm25_index=bm25_index,
            top_k=3
        )
        
        # Show results from each method
        print("\n   📊 SEMANTIC SEARCH (Vector/Meaning-based):")
        for result in comparison["results"]["semantic"][:3]:
            print(f"      Rank {result.rank} | Score: {result.score:.4f}")
            text_preview = result.chunk.text.replace('\n', ' ').strip()[:70]
            print(f"      {text_preview}...")
        
        print("\n   📊 KEYWORD SEARCH (BM25/Term-based):")
        for result in comparison["results"]["keyword"][:3]:
            print(f"      Rank {result.rank} | Score: {result.score:.4f}")
            text_preview = result.chunk.text.replace('\n', ' ').strip()[:70]
            print(f"      {text_preview}...")
        
        print("\n   📊 HYBRID SEARCH (Combined - Weighted):")
        for result in comparison["results"]["hybrid_weighted"][:3]:
            print(f"      Rank {result.rank} | Score: {result.score:.4f}")
            text_preview = result.chunk.text.replace('\n', ' ').strip()[:70]
            print(f"      {text_preview}...")
        
        print("\n   📊 HYBRID SEARCH (Combined - RRF):")
        for result in comparison["results"]["hybrid_rrf"][:3]:
            print(f"      Rank {result.rank} | Score: {result.score:.4f}")
            text_preview = result.chunk.text.replace('\n', ' ').strip()[:70]
            print(f"      {text_preview}...")
        
        # Show overlap statistics
        overlap = comparison["overlap"]
        print(f"\n   📈 Overlap Statistics:")
        print(f"      Semantic ∩ Keyword: {overlap['semantic_keyword']} documents")
        print(f"      Semantic only: {overlap['semantic_only']} documents")
        print(f"      Keyword only: {overlap['keyword_only']} documents")
    
    # Demonstrate fusion strategies
    print("\n" + "=" * 80)
    print("FUSION STRATEGY COMPARISON")
    print("=" * 80)
    
    query = "machine learning algorithms"
    print(f"\n🔍 Query: '{query}'")
    print("-" * 80)
    
    fusion_methods = [
        ("weighted", {"semantic_weight": 0.7}, "70% semantic, 30% keyword"),
        ("weighted", {"semantic_weight": 0.3}, "30% semantic, 70% keyword"),
        ("rrf", {"rrf_k": 60}, "Reciprocal Rank Fusion"),
        ("max", {}, "Maximum score from either")
    ]
    
    for method, params, description in fusion_methods:
        print(f"\n   📊 {method.upper()} - {description}:")
        
        results = hybrid_search(
            query=query,
            vector_store=vector_store,
            bm25_index=bm25_index,
            top_k=3,
            fusion_method=method,
            **params
        )
        
        for result in results:
            print(f"      Rank {result.rank} | Score: {result.score:.4f} | {result.chunk.source_file}")
    
    # Demonstrate intelligent fallback
    print("\n" + "=" * 80)
    print("INTELLIGENT FALLBACK")
    print("=" * 80)
    
    test_cases = [
        ("machine learning", "Good semantic and keyword matches"),
        ("quantum entanglement blockchain", "No good matches (low confidence)")
    ]
    
    for query, expected in test_cases:
        print(f"\n🔍 Query: '{query}'")
        print(f"   Expected: {expected}")
        print("-" * 80)
        
        results, method_used = hybrid_search_with_fallback(
            query=query,
            vector_store=vector_store,
            bm25_index=bm25_index,
            top_k=3,
            semantic_threshold=0.5
        )
        
        print(f"   Method used: {method_used}")
        print(f"   Results: {len(results)}")
        
        if results:
            print(f"   Top result score: {results[0].score:.4f}")
        else:
            print("   ⚠️ No confident results - would return 'I don't know'")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    print("""
    ✅ SEMANTIC SEARCH (Vector):
       • Understands meaning and context
       • Finds conceptually similar documents
       • Good for natural language queries
       • Example: "AI systems" matches "artificial intelligence"
    
    ✅ KEYWORD SEARCH (BM25):
       • Finds exact term matches
       • Good for specific terminology
       • Fast and efficient
       • Example: "Python NumPy" finds exact library names
    
    ✅ HYBRID SEARCH (Combined):
       • Best of both worlds
       • More robust across query types
       • Multiple fusion strategies available
       • Intelligent fallback to best method
    
    🎯 FUSION STRATEGIES:
       • Weighted: Balance semantic/keyword with α parameter
       • RRF: Rank-based, robust to score scales
       • Max: Takes highest score from either method
    
    💡 RECOMMENDATION:
       Use hybrid search with RRF fusion as default for most applications.
       It's robust and doesn't require tuning the α parameter.
    """)
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\n✅ Hybrid search successfully combines semantic and keyword search!")
    print("   Ready for integration into the RAG pipeline.\n")


if __name__ == "__main__":
    main()

