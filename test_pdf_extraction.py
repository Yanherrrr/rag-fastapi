"""Demo script to test PDF extraction and chunking"""

import sys
from pathlib import Path

from app.core.chunking import (
    extract_text_from_pdf,
    chunk_pages,
    get_text_statistics
)
from app.core.config import settings


def test_with_sample_pdf(pdf_path: str):
    """
    Test PDF extraction with a sample file.
    
    Args:
        pdf_path: Path to PDF file
    """
    print(f"\n{'='*60}")
    print(f"Testing PDF Extraction: {Path(pdf_path).name}")
    print(f"{'='*60}\n")
    
    try:
        # Extract text from PDF
        print("📄 Step 1: Extracting text from PDF...")
        pages = extract_text_from_pdf(pdf_path)
        print(f"   ✅ Extracted {len(pages)} pages")
        
        # Show sample from first page
        if pages:
            first_page = pages[0]
            preview = first_page.text[:200] + "..." if len(first_page.text) > 200 else first_page.text
            print(f"\n   Preview of page 1:")
            print(f"   {preview}\n")
        
        # Get statistics
        print("📊 Step 2: Text Statistics...")
        total_text = " ".join([page.text for page in pages])
        stats = get_text_statistics(total_text)
        print(f"   - Total characters: {stats['total_characters']:,}")
        print(f"   - Total words: {stats['total_words']:,}")
        print(f"   - Total sentences: {stats['total_sentences']:,}")
        print(f"   - Avg sentence length: {stats['average_sentence_length']:.1f} words\n")
        
        # Chunk the text
        print("✂️  Step 3: Chunking text...")
        chunks = chunk_pages(
            pages=pages,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap
        )
        print(f"   ✅ Created {len(chunks)} chunks")
        print(f"   - Chunk size: {settings.chunk_size} characters")
        print(f"   - Overlap: {settings.chunk_overlap} characters\n")
        
        # Show sample chunks
        print("📝 Sample chunks:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n   Chunk {i} (ID: {chunk.chunk_id}):")
            print(f"   Source: {chunk.source_file}, Page: {chunk.page_number}")
            preview = chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text
            print(f"   Text: {preview}")
        
        if len(chunks) > 3:
            print(f"\n   ... and {len(chunks) - 3} more chunks")
        
        print(f"\n{'='*60}")
        print("✅ PDF extraction and chunking completed successfully!")
        print(f"{'='*60}\n")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ Error: PDF file not found: {pdf_path}")
        print(f"   Please provide a valid PDF file path.\n")
        return False
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    print("""
    ╔════════════════════════════════════════╗
    ║   PDF Extraction & Chunking Demo      ║
    ╚════════════════════════════════════════╝
    """)
    
    if len(sys.argv) < 2:
        print("Usage: python test_pdf_extraction.py <path_to_pdf>")
        print("\nExample:")
        print("  python test_pdf_extraction.py sample.pdf")
        print("  python test_pdf_extraction.py /path/to/document.pdf")
        print("\n📝 Note: Please provide a PDF file to test the extraction.\n")
        return 1
    
    pdf_path = sys.argv[1]
    success = test_with_sample_pdf(pdf_path)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

