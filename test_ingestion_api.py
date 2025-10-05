"""Test script for the ingestion API endpoint"""

import sys
import requests
import json
from pathlib import Path


def test_api_health():
    """Test if API is running"""
    print("\n" + "="*60)
    print("1. TESTING API HEALTH")
    print("="*60 + "\n")
    
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ API is running!")
            print(f"   Response: {response.json()}\n")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        print("   Start the server with: python run.py\n")
        return False


def test_api_status():
    """Test the status endpoint"""
    print("\n" + "="*60)
    print("2. TESTING STATUS ENDPOINT")
    print("="*60 + "\n")
    
    try:
        response = requests.get("http://localhost:8000/api/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ Status endpoint working!")
            print(f"\n   Current Statistics:")
            for key, value in data.get("statistics", {}).items():
                print(f"   • {key}: {value}")
            print()
            return True
        else:
            print(f"❌ Status endpoint returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_ingestion_endpoint(pdf_path: str = None):
    """Test the ingestion endpoint"""
    print("\n" + "="*60)
    print("3. TESTING INGESTION ENDPOINT")
    print("="*60 + "\n")
    
    if not pdf_path:
        print("⚠️  No PDF file provided for testing.")
        print("   Usage: python test_ingestion_api.py <path_to_pdf>")
        print("\n   To test ingestion, provide a PDF file:")
        print("   python test_ingestion_api.py sample.pdf\n")
        return False
    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    print(f"📄 Uploading PDF: {pdf_file.name}")
    print(f"   File size: {pdf_file.stat().st_size / 1024:.2f} KB\n")
    
    try:
        # Prepare file for upload
        with open(pdf_file, 'rb') as f:
            files = {'files': (pdf_file.name, f, 'application/pdf')}
            
            print("⏳ Sending request to /api/ingest...")
            response = requests.post(
                "http://localhost:8000/api/ingest",
                files=files,
                timeout=300  # 5 minute timeout for large files
            )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Ingestion successful!\n")
            print(f"   Statistics:")
            print(f"   • Status: {data.get('status')}")
            print(f"   • Files processed: {data.get('files_processed')}")
            print(f"   • Total chunks: {data.get('total_chunks')}")
            print(f"   • Processing time: {data.get('processing_time_seconds')}s")
            
            print(f"\n   Per-file breakdown:")
            for file_info in data.get('files', []):
                print(f"   • {file_info['filename']}:")
                print(f"     - Pages: {file_info['pages']}")
                print(f"     - Chunks: {file_info['chunks']}")
            print()
            return True
            
        else:
            print(f"❌ Ingestion failed with status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out. File may be too large or processing is slow.")
        return False
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        return False


def test_status_after_ingestion():
    """Check status after ingestion"""
    print("\n" + "="*60)
    print("4. CHECKING STATUS AFTER INGESTION")
    print("="*60 + "\n")
    
    try:
        response = requests.get("http://localhost:8000/api/status")
        if response.status_code == 200:
            data = response.json()
            stats = data.get("statistics", {})
            
            print("📊 Vector Store Statistics:")
            print(f"   • Total documents: {stats.get('total_documents', 0)}")
            print(f"   • Total chunks: {stats.get('total_chunks', 0)}")
            print(f"   • Embedding dimension: {stats.get('embedding_dimension', 0)}")
            print(f"   • Store size: {stats.get('vector_store_size_mb', 0)} MB")
            print(f"   • Memory usage: {stats.get('memory_usage_mb', 0)} MB")
            print()
            return True
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_api_docs():
    """Test API documentation"""
    print("\n" + "="*60)
    print("5. API DOCUMENTATION")
    print("="*60 + "\n")
    
    print("📚 Interactive API docs available at:")
    print("   • Swagger UI: http://localhost:8000/docs")
    print("   • ReDoc: http://localhost:8000/redoc")
    print()


def main():
    """Run all tests"""
    print("""
    ╔════════════════════════════════════════╗
    ║   Ingestion API Test Suite            ║
    ║   Step 2.5 Verification               ║
    ╚════════════════════════════════════════╝
    """)
    
    # Get PDF path from command line if provided
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run tests
    results = []
    
    results.append(("API Health", test_api_health()))
    results.append(("Status Endpoint", test_api_status()))
    
    if pdf_path:
        results.append(("Ingestion", test_ingestion_endpoint(pdf_path)))
        results.append(("Status After Ingestion", test_status_after_ingestion()))
    else:
        test_ingestion_endpoint(None)  # Show usage message
    
    test_api_docs()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60 + "\n")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
    
    print("\n" + "="*60)
    
    if not pdf_path:
        print("\nℹ️  To test ingestion, run:")
        print("   python test_ingestion_api.py <your_pdf_file.pdf>\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

