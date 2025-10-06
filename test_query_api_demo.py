"""
Demo script for testing the Query API endpoint

This script demonstrates the full RAG pipeline with various query types.
Requires the FastAPI server to be running and documents to be ingested.

Usage:
    # Terminal 1: Start the server
    python run.py
    
    # Terminal 2: Run this demo
    python test_query_api_demo.py
"""

import requests
import json
from typing import Dict, Any


BASE_URL = "http://localhost:8000/api"


def print_response(response: Dict[Any, Any]):
    """Pretty print query response"""
    print(f"\n{'='*70}")
    print(f"Query: {response['query']}")
    print(f"Intent: {response['intent']}")
    print(f"Status: {response['status']}")
    print(f"\n{'-'*70}")
    print(f"Answer:\n{response['answer']}")
    print(f"\n{'-'*70}")
    
    if response['sources']:
        print(f"\nSources ({len(response['sources'])} found):")
        for i, source in enumerate(response['sources'], 1):
            print(f"\n  [{i}] {source['source_file']} (Page {source['page_number']})")
            print(f"      Score: {source['similarity_score']:.4f}")
            print(f"      Text: {source['text'][:150]}...")
    else:
        print("\nNo sources returned.")
    
    print(f"\n{'-'*70}")
    print(f"Evidence Quality: {'✓ Sufficient' if response['has_sufficient_evidence'] else '✗ Insufficient'}")
    print(f"Timing:")
    print(f"  - Search: {response['metadata']['search_time_ms']:.2f}ms")
    print(f"  - LLM: {response['metadata']['llm_time_ms']:.2f}ms")
    print(f"  - Total: {response['metadata']['total_time_ms']:.2f}ms")
    print(f"{'='*70}\n")


def query(question: str, top_k: int = 5, include_sources: bool = True) -> Dict[Any, Any]:
    """Send a query to the API"""
    response = requests.post(
        f"{BASE_URL}/query",
        json={
            "query": question,
            "top_k": top_k,
            "include_sources": include_sources
        }
    )
    response.raise_for_status()
    return response.json()


def check_status():
    """Check if server is running and has documents"""
    try:
        response = requests.get(f"{BASE_URL}/status")
        response.raise_for_status()
        status = response.json()
        
        print("\n" + "="*70)
        print("SERVER STATUS")
        print("="*70)
        print(f"Status: {status['status']}")
        print(f"Total Documents: {status['statistics']['total_documents']}")
        print(f"Total Chunks: {status['statistics']['total_chunks']}")
        print(f"Embedding Dimension: {status['statistics']['embedding_dimension']}")
        print(f"Vector Store Size: {status['statistics']['vector_store_size_mb']:.2f} MB")
        print("="*70 + "\n")
        
        return status['statistics']['total_chunks'] > 0
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("Make sure the server is running: python run.py")
        return False


def demo_greeting():
    """Demo: Greeting intent"""
    print("\n" + "🌟"*23)
    print("DEMO 1: GREETING INTENT")
    print("🌟"*23)
    
    greetings = ["hello", "hi there", "hey"]
    
    for greeting in greetings:
        result = query(greeting, top_k=3)
        print_response(result)


def demo_chitchat():
    """Demo: Chitchat intent"""
    print("\n" + "🌟"*23)
    print("DEMO 2: CHITCHAT INTENT")
    print("🌟"*23)
    
    chitchats = [
        "how are you?",
        "what can you do?",
        "nice weather today"
    ]
    
    for chitchat in chitchats:
        result = query(chitchat, top_k=3)
        print_response(result)


def demo_search_queries():
    """Demo: Real search queries"""
    print("\n" + "🌟"*23)
    print("DEMO 3: SEARCH QUERIES")
    print("🌟"*23)
    
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain natural language processing",
        "What are the applications of AI?",
        "Tell me about deep learning"
    ]
    
    for q in queries:
        result = query(q, top_k=5, include_sources=True)
        print_response(result)


def demo_different_top_k():
    """Demo: Different top_k values"""
    print("\n" + "🌟"*23)
    print("DEMO 4: DIFFERENT TOP_K VALUES")
    print("🌟"*23)
    
    question = "What is artificial intelligence?"
    
    for k in [1, 3, 5]:
        print(f"\n--- Testing with top_k={k} ---")
        result = query(question, top_k=k, include_sources=True)
        print_response(result)


def demo_with_without_sources():
    """Demo: With and without sources"""
    print("\n" + "🌟"*23)
    print("DEMO 5: WITH/WITHOUT SOURCES")
    print("🌟"*23)
    
    question = "What are the challenges in AI?"
    
    print("\n--- With Sources ---")
    result = query(question, top_k=3, include_sources=True)
    print_response(result)
    
    print("\n--- Without Sources ---")
    result = query(question, top_k=3, include_sources=False)
    print_response(result)


def demo_edge_cases():
    """Demo: Edge cases"""
    print("\n" + "🌟"*23)
    print("DEMO 6: EDGE CASES")
    print("🌟"*23)
    
    edge_cases = [
        "quantum computing",  # May not be in documents
        "AI?",  # Very short query
        "Can you explain in detail how reinforcement learning algorithms work?",  # Long query
    ]
    
    for q in edge_cases:
        result = query(q, top_k=5)
        print_response(result)


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("QUERY API DEMO SCRIPT")
    print("="*70)
    print("\nThis script tests the complete RAG pipeline:")
    print("  1. Intent Detection")
    print("  2. Hybrid Search (Semantic + Keyword)")
    print("  3. Re-ranking (Cross-Encoder + MMR)")
    print("  4. Answer Generation (Mistral AI)")
    print("="*70)
    
    # Check server status
    if not check_status():
        print("\n⚠️  WARNING: No documents in knowledge base!")
        print("Please ingest some documents first using:")
        print("  1. Start server: python run.py")
        print("  2. Upload PDFs via API or frontend")
        print("\nProceeding with greeting/chitchat demos only...\n")
        
        # Only run intent-based demos
        demo_greeting()
        demo_chitchat()
        return
    
    # Run all demos
    try:
        demo_greeting()
        demo_chitchat()
        demo_search_queries()
        demo_different_top_k()
        demo_with_without_sources()
        demo_edge_cases()
        
        print("\n" + "="*70)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error communicating with API: {e}")
        print("Make sure the server is running: python run.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

