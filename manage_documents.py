"""
Document Management Helper Script

This script helps you manage documents in your RAG system.

Usage:
    # List documents
    python manage_documents.py list
    
    # Upload PDFs
    python manage_documents.py upload file1.pdf file2.pdf
    
    # Clear all documents
    python manage_documents.py clear
"""

import sys
import requests
from pathlib import Path


BASE_URL = "http://localhost:8000/api"


def list_documents():
    """List all documents in the system"""
    try:
        response = requests.get(f"{BASE_URL}/status")
        response.raise_for_status()
        stats = response.json()["statistics"]
        
        print("\n" + "="*70)
        print("CURRENT DOCUMENTS IN RAG SYSTEM")
        print("="*70)
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Vector Store Size: {stats['vector_store_size_mb']:.2f} MB")
        print(f"Embedding Dimension: {stats['embedding_dimension']}")
        print("="*70 + "\n")
        
        # Get actual document names
        from app.storage.vector_store import get_vector_store
        store = get_vector_store()
        if store.chunks:
            unique_files = set(chunk.source_file for chunk in store.chunks)
            print("📄 Documents:")
            for source in sorted(unique_files):
                chunks = [c for c in store.chunks if c.source_file == source]
                pages = set(c.page_number for c in chunks)
                print(f"   • {source}")
                print(f"     └─ {len(chunks)} chunks, {len(pages)} pages\n")
        else:
            print("📭 No documents loaded.\n")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def upload_documents(files):
    """Upload PDF documents"""
    if not files:
        print("❌ No files specified")
        return
    
    # Validate files exist
    valid_files = []
    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            continue
        if path.suffix.lower() != '.pdf':
            print(f"⚠️  Skipping non-PDF file: {file_path}")
            continue
        valid_files.append(path)
    
    if not valid_files:
        print("❌ No valid PDF files to upload")
        return
    
    print(f"\n📤 Uploading {len(valid_files)} PDF(s)...")
    print("="*70)
    
    try:
        files_data = [('files', (f.name, open(f, 'rb'), 'application/pdf')) 
                      for f in valid_files]
        
        response = requests.post(f"{BASE_URL}/ingest", files=files_data)
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ Upload successful!")
        print(f"   • Files processed: {result['files_processed']}")
        print(f"   • Total chunks: {result['total_chunks']}")
        print(f"   • Processing time: {result['processing_time_seconds']:.2f}s")
        print()
        
        for file_info in result['files']:
            print(f"   📄 {file_info['filename']}")
            print(f"      └─ {file_info['chunks']} chunks, {file_info['pages']} pages")
        
        print("="*70 + "\n")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Upload failed: {e}")
    finally:
        # Close file handles
        for _, (_, f, _) in files_data:
            f.close()


def clear_documents():
    """Clear all documents from the system"""
    print("\n⚠️  WARNING: This will delete ALL documents from the system!")
    response = input("Are you sure? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("❌ Cancelled")
        return
    
    try:
        response = requests.delete(f"{BASE_URL}/clear")
        response.raise_for_status()
        result = response.json()
        
        print(f"\n✅ {result['message']}")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def show_usage():
    """Show usage information"""
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        show_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        list_documents()
    elif command == "upload":
        files = sys.argv[2:]
        upload_documents(files)
    elif command == "clear":
        clear_documents()
    elif command in ["help", "-h", "--help"]:
        show_usage()
    else:
        print(f"❌ Unknown command: {command}")
        show_usage()


if __name__ == "__main__":
    main()

