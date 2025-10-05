"""Script to download required models before running the application"""

import sys
from sentence_transformers import SentenceTransformer
from app.core.config import settings


def download_embedding_model():
    """Download the embedding model to local cache"""
    print(f"\n🔽 Downloading embedding model: {settings.embedding_model}")
    print("This may take a few minutes on first run...\n")
    
    try:
        # Clear any HuggingFace tokens that might be causing auth issues
        import os
        hf_token = os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)
        hf_token_alt = os.environ.pop('HF_TOKEN', None)
        
        if hf_token or hf_token_alt:
            print("ℹ️  Cleared HuggingFace tokens (model is public, no auth needed)\n")
        
        model = SentenceTransformer(settings.embedding_model, use_auth_token=False)
        print(f"✅ Model downloaded successfully!")
        print(f"   Model name: {settings.embedding_model}")
        print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
        print(f"   Device: {model.device}")
        print(f"\n✅ Ready to use!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nPossible solutions:")
        print("  1. Check your internet connection")
        print("  2. Try again (sometimes HuggingFace hub has temporary issues)")
        print("  3. Check if you have invalid HF tokens: echo $HUGGING_FACE_HUB_TOKEN")
        print("  4. Clear tokens: unset HUGGING_FACE_HUB_TOKEN && unset HF_TOKEN")
        print("  5. Use a different model by changing EMBEDDING_MODEL in .env")
        return False


def main():
    print("""
    ╔════════════════════════════════════════╗
    ║   Model Download Utility               ║
    ╚════════════════════════════════════════╝
    """)
    
    success = download_embedding_model()
    
    if success:
        print("🎉 All models downloaded successfully!")
        print("You can now run tests or start the application.\n")
        return 0
    else:
        print("\n⚠️  Model download failed. Please check the error above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

