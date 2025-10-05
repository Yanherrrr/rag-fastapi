"""Verification script to check Phase 1 setup"""

import sys
from pathlib import Path


def check_directories():
    """Check if all required directories exist"""
    required_dirs = [
        "app",
        "app/api",
        "app/core",
        "app/models",
        "app/storage",
        "frontend",
        "frontend/static",
        "data",
        "uploads",
        "tests"
    ]
    
    print("🔍 Checking directories...")
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - MISSING")
            all_exist = False
    
    return all_exist


def check_files():
    """Check if all required files exist"""
    required_files = [
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "run.py",
        "app/__init__.py",
        "app/main.py",
        "app/core/config.py",
        "app/models/schemas.py",
        "app/api/ingestion.py",
        "app/api/query.py",
    ]
    
    print("\n🔍 Checking files...")
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist


def check_imports():
    """Check if core modules can be imported"""
    print("\n🔍 Checking imports...")
    
    try:
        from app.core.config import settings
        print(f"  ✅ app.core.config - Settings loaded")
        print(f"     - Mistral API Key: {'*' * 20}{settings.mistral_api_key[-5:]}")
        print(f"     - Chunk Size: {settings.chunk_size}")
        print(f"     - Top K Results: {settings.top_k_results}")
    except Exception as e:
        print(f"  ❌ app.core.config - {e}")
        return False
    
    try:
        from app.models.schemas import QueryRequest, QueryResponse, Intent
        print(f"  ✅ app.models.schemas - Models imported")
    except Exception as e:
        print(f"  ❌ app.models.schemas - {e}")
        return False
    
    try:
        from app.main import app
        print(f"  ✅ app.main - FastAPI app created")
    except Exception as e:
        print(f"  ❌ app.main - {e}")
        return False
    
    return True


def main():
    """Run all verification checks"""
    print("""
    ╔════════════════════════════════════════╗
    ║   RAG FastAPI - Phase 1 Verification  ║
    ╚════════════════════════════════════════╝
    """)
    
    dirs_ok = check_directories()
    files_ok = check_files()
    imports_ok = check_imports()
    
    print("\n" + "="*50)
    if dirs_ok and files_ok and imports_ok:
        print("✅ Phase 1 Setup Complete!")
        print("\n📝 Next Steps:")
        print("   1. Create virtual environment: python -m venv venv")
        print("   2. Activate: source venv/bin/activate")
        print("   3. Install dependencies: pip install -r requirements.txt")
        print("   4. Run server: python run.py")
        print("   5. Visit: http://localhost:8000/docs")
        print("\n🚀 Ready for Phase 2: Data Ingestion Pipeline")
        return 0
    else:
        print("❌ Phase 1 Setup Incomplete")
        print("\n⚠️  Please fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

