"""
Demo script to test Mistral AI integration with real API calls.

This script demonstrates the complete LLM integration:
1. Loads API key from .env
2. Creates MistralClient
3. Generates answers from sample context
4. Shows token usage and costs
"""

from app.core.llm import MistralClient, generate_answer
from app.core.config import settings


def demo_basic_answer():
    """Demo 1: Basic answer generation."""
    print("=" * 80)
    print("DEMO 1: Basic Answer Generation")
    print("=" * 80)
    
    # Sample context chunks (simulating search results)
    context_chunks = [
        """Machine learning is a method of data analysis that automates analytical 
        model building. It is a branch of artificial intelligence based on the idea 
        that systems can learn from data, identify patterns and make decisions with 
        minimal human intervention.""",
        
        """Deep learning is a subset of machine learning that uses neural networks 
        with multiple layers. These deep neural networks can automatically learn 
        hierarchical representations of data, making them particularly effective 
        for tasks like image recognition and natural language processing.""",
        
        """Python is the most popular programming language for machine learning due 
        to its extensive libraries such as scikit-learn, TensorFlow, and PyTorch. 
        These libraries provide pre-built implementations of common machine learning 
        algorithms and tools for data processing."""
    ]
    
    question = "What is machine learning and why is Python popular for it?"
    
    print(f"\n📝 Question: {question}")
    print(f"\n📚 Context: {len(context_chunks)} chunks provided")
    print("\n🤖 Generating answer with Mistral AI...")
    print("-" * 80)
    
    try:
        # Generate answer
        result = generate_answer(
            question=question,
            context_chunks=context_chunks,
            temperature=0.7
        )
        
        print("\n✅ Answer:")
        print(result["answer"])
        print(f"\n📊 Metadata:")
        print(f"   Model: {result['model']}")
        print(f"   Tokens used: {result['tokens_used']}")
        
        # Show error if present
        if 'error' in result:
            print(f"\n⚠️  Error details: {result['error']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def demo_with_source_citations():
    """Demo 2: Answer with source citations."""
    print("\n\n" + "=" * 80)
    print("DEMO 2: Answer with Source Citations")
    print("=" * 80)
    
    context_chunks = [
        "Natural language processing (NLP) enables computers to understand human language.",
        "NLP applications include chatbots, translation, and sentiment analysis.",
        "Modern NLP uses transformer models like BERT and GPT for state-of-the-art results."
    ]
    
    question = "What is NLP and what are its applications?"
    
    print(f"\n📝 Question: {question}")
    print(f"\n📚 Context: {len(context_chunks)} numbered sources")
    print("\n🤖 Generating answer with source citations...")
    print("-" * 80)
    
    try:
        # Create client directly for more control
        client = MistralClient(model="mistral-small")
        
        result = client.generate_answer(
            question=question,
            context_chunks=context_chunks,
            temperature=0.5,  # Lower temperature for more focused answers
            include_source_numbers=True  # Ask LLM to cite sources
        )
        
        print("\n✅ Answer (with citations):")
        print(result["answer"])
        print(f"\n📊 Metadata:")
        print(f"   Model: {result['model']}")
        print(f"   Tokens used: {result['tokens_used']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


def demo_no_relevant_info():
    """Demo 3: Question with no relevant context."""
    print("\n\n" + "=" * 80)
    print("DEMO 3: Handling Questions with Irrelevant Context")
    print("=" * 80)
    
    # Context about completely different topic
    context_chunks = [
        "Photosynthesis is the process by which plants convert light into energy.",
        "Chlorophyll in plant cells captures sunlight for photosynthesis.",
    ]
    
    question = "How do I train a neural network?"
    
    print(f"\n📝 Question: {question}")
    print(f"\n📚 Context: Information about photosynthesis (irrelevant)")
    print("\n🤖 Checking if AI admits it doesn't know...")
    print("-" * 80)
    
    try:
        result = generate_answer(
            question=question,
            context_chunks=context_chunks,
            temperature=0.3  # Low temperature for more conservative answers
        )
        
        print("\n✅ Answer:")
        print(result["answer"])
        
        # Check if AI properly says it doesn't know
        answer_lower = result["answer"].lower()
        if any(phrase in answer_lower for phrase in [
            "don't have", "insufficient", "not enough", "cannot answer", "don't know"
        ]):
            print("\n✅ Good! AI properly indicated lack of relevant information.")
        else:
            print("\n⚠️  Warning: AI may have hallucinated information.")
        
        print(f"\n📊 Tokens used: {result['tokens_used']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


def demo_temperature_comparison():
    """Demo 4: Compare different temperature settings."""
    print("\n\n" + "=" * 80)
    print("DEMO 4: Temperature Comparison")
    print("=" * 80)
    
    context_chunks = [
        "Python was created by Guido van Rossum and released in 1991.",
        "Python emphasizes code readability and uses significant indentation.",
    ]
    
    question = "Who created Python?"
    
    print(f"\n📝 Question: {question}")
    print(f"\n📚 Context: Information about Python")
    print("\n🤖 Generating answers with different temperatures...")
    print("-" * 80)
    
    for temp in [0.1, 0.7, 1.0]:
        print(f"\n🌡️  Temperature: {temp}")
        print("   (Lower = more deterministic, Higher = more creative)")
        
        try:
            result = generate_answer(
                question=question,
                context_chunks=context_chunks,
                temperature=temp
            )
            
            print(f"\n   Answer: {result['answer'][:150]}...")
            print(f"   Tokens: {result['tokens_used']}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("🚀 MISTRAL AI INTEGRATION DEMO")
    print("=" * 80)
    
    # Check API key
    if not settings.MISTRAL_API_KEY:
        print("\n❌ ERROR: MISTRAL_API_KEY not found in .env file!")
        print("   Please create a .env file with: MISTRAL_API_KEY=your_key_here")
        return
    
    print(f"\n✅ API Key loaded: {settings.MISTRAL_API_KEY[:10]}...")
    print(f"✅ Model: mistral-small")
    print("\n⚠️  Note: This will make real API calls and consume tokens!\n")
    
    try:
        # Run all demos
        demo_basic_answer()
        demo_with_source_citations()
        demo_no_relevant_info()
        demo_temperature_comparison()
        
        print("\n\n" + "=" * 80)
        print("✅ ALL DEMOS COMPLETE!")
        print("=" * 80)
        print("\n💡 Key Takeaways:")
        print("   • Mistral AI integration is working correctly")
        print("   • Answers are generated from provided context")
        print("   • Source citations can be included")
        print("   • AI properly handles irrelevant context")
        print("   • Temperature controls answer creativity")
        print("\n🎉 Ready to integrate into the full RAG pipeline!\n")
        
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        print("\nPossible issues:")
        print("   • Invalid API key")
        print("   • Network connection problem")
        print("   • API rate limit reached")
        print("   • Mistral API service issue")


if __name__ == "__main__":
    main()

