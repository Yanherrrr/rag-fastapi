"""Demo script to test embedding generation and vector store"""

import sys
import numpy as np

from app.core.embeddings import (
    get_embedding_generator,
    generate_embeddings,
    generate_query_embedding
)
from app.storage.vector_store import VectorStore
from app.models.schemas import Chunk


def test_embeddings():
    """Test embedding generation"""
    print("\n" + "="*60)
    print("1. TESTING EMBEDDING GENERATION")
    print("="*60 + "\n")
    
    try:
        # Initialize embedding generator
        print("📦 Loading embedding model...")
        generator = get_embedding_generator()
        print(f"   ✅ Model loaded: {generator._model.device}")
        print(f"   ✅ Embedding dimension: {generator.get_embedding_dimension()}\n")
        
        # Test single embedding
        print("🔤 Testing single text embedding...")
        text = "Artificial intelligence is transforming the world."
        embedding = generator.generate_single_embedding(text)
        print(f"   Text: {text}")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"   Sample values: {embedding[:5]}\n")
        
        # Test batch embeddings
        print("📚 Testing batch embeddings...")
        texts = [
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks.",
            "Natural language processing enables text understanding.",
            "Computer vision helps machines see.",
            "Robotics combines AI with physical systems."
        ]
        
        embeddings = generate_embeddings(texts, batch_size=3)
        print(f"   Texts: {len(texts)}")
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   All normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=0.01)}\n")
        
        # Test similarity
        print("🔍 Testing semantic similarity...")
        sim_matrix = np.dot(embeddings, embeddings.T)
        print(f"   Similarity matrix shape: {sim_matrix.shape}")
        print(f"   Similarity between texts 0 and 1: {sim_matrix[0, 1]:.4f}")
        print(f"   Similarity between texts 0 and 4: {sim_matrix[0, 4]:.4f}\n")
        
        return True, embeddings, texts
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_vector_store(embeddings, texts):
    """Test vector store functionality"""
    print("\n" + "="*60)
    print("2. TESTING VECTOR STORE")
    print("="*60 + "\n")
    
    try:
        # Create sample chunks
        print("📝 Creating sample chunks...")
        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                chunk_id=f"demo_doc_1_{i}",
                text=text,
                source_file="demo_document.pdf",
                page_number=1,
                chunk_index=i,
                metadata={"demo": True}
            )
            chunks.append(chunk)
        print(f"   ✅ Created {len(chunks)} chunks\n")
        
        # Initialize vector store
        print("💾 Initializing vector store...")
        store = VectorStore(store_path="data/demo_vector_store.pkl")
        store.clear()  # Start fresh
        print(f"   ✅ Vector store initialized\n")
        
        # Add documents
        print("➕ Adding documents to store...")
        store.add_documents(chunks, embeddings)
        print(f"   ✅ Added {len(chunks)} chunks")
        print(f"   ✅ Store now contains: {len(store)} chunks\n")
        
        # Get statistics
        print("📊 Store statistics:")
        stats = store.get_stats()
        for key, value in stats.items():
            if key != "documents":
                print(f"   • {key}: {value}")
        print()
        
        # Test search
        print("🔎 Testing similarity search...")
        query_text = "What is artificial intelligence and machine learning?"
        print(f"   Query: {query_text}\n")
        
        # Generate query embedding
        query_embedding = generate_query_embedding(query_text)
        
        # Search
        results = store.search(query_embedding, top_k=3)
        
        print(f"   Found {len(results)} results:\n")
        for result in results:
            print(f"   Rank {result.rank}: (Score: {result.score:.4f})")
            print(f"      ID: {result.chunk.chunk_id}")
            print(f"      Text: {result.chunk.text}")
            print()
        
        # Test save and load
        print("💾 Testing save and load...")
        store.save()
        print(f"   ✅ Saved to {store.store_path}\n")
        
        # Load in new instance
        store2 = VectorStore(store_path="data/demo_vector_store.pkl")
        print(f"   ✅ Loaded from disk")
        print(f"   ✅ Contains {len(store2)} chunks\n")
        
        # Test search on loaded store
        results2 = store2.search(query_embedding, top_k=1)
        print(f"   Search on loaded store:")
        print(f"      Top result: {results2[0].chunk.text[:50]}...")
        print(f"      Score: {results2[0].score:.4f}\n")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test full integration"""
    print("\n" + "="*60)
    print("3. INTEGRATION TEST")
    print("="*60 + "\n")
    
    try:
        print("🔄 Testing end-to-end flow...")
        
        # 1. Generate embeddings
        from app.core.chunking import chunk_text
        
        sample_text = """
        Artificial intelligence (AI) has revolutionized many industries. 
        Machine learning algorithms can now process vast amounts of data.
        Deep learning networks are capable of recognizing patterns in images and text.
        Natural language processing enables computers to understand human language.
        """
        
        chunks = chunk_text(
            text=sample_text,
            chunk_size=100,
            overlap=20,
            metadata={'source_file': 'integration_test.pdf', 'page_number': 1}
        )
        print(f"   ✅ Created {len(chunks)} chunks from text\n")
        
        # 2. Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = generate_embeddings(texts)
        print(f"   ✅ Generated {len(embeddings)} embeddings\n")
        
        # 3. Store in vector store
        store = VectorStore(store_path="data/integration_test_store.pkl")
        store.clear()
        store.add_documents(chunks, embeddings)
        print(f"   ✅ Stored in vector database\n")
        
        # 4. Query
        query = "What is machine learning?"
        query_emb = generate_query_embedding(query)
        results = store.search(query_emb, top_k=2)
        
        print(f"   Query: '{query}'")
        print(f"   ✅ Found {len(results)} relevant chunks")
        print(f"   Best match (score {results[0].score:.4f}):")
        print(f"      {results[0].chunk.text}\n")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("""
    ╔════════════════════════════════════════╗
    ║   Embeddings & Vector Store Demo      ║
    ║   Steps 2.3 & 2.4 Verification        ║
    ╚════════════════════════════════════════╝
    """)
    
    # Test embeddings
    success1, embeddings, texts = test_embeddings()
    
    if not success1:
        print("\n❌ Embedding tests failed. Please install dependencies:")
        print("   pip install -r requirements.txt")
        return 1
    
    # Test vector store
    success2 = test_vector_store(embeddings, texts)
    
    if not success2:
        print("\n❌ Vector store tests failed.")
        return 1
    
    # Integration test
    success3 = test_integration()
    
    if not success3:
        print("\n❌ Integration tests failed.")
        return 1
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")
    
    print("Summary:")
    print("  ✅ Embedding generation working")
    print("  ✅ Vector store operations working")
    print("  ✅ Similarity search working")
    print("  ✅ Save/load functionality working")
    print("  ✅ End-to-end integration working")
    print("\n🚀 Ready to proceed to Step 2.5: Ingestion API Endpoint!\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

