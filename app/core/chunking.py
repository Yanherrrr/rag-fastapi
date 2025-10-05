"""Text extraction and chunking functionality"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

from PyPDF2 import PdfReader

from app.models.schemas import PageContent, Chunk

logger = logging.getLogger(__name__)


# ============= PDF Text Extraction =============

def extract_text_from_pdf(file_path: str) -> List[PageContent]:
    """
    Extract text from a PDF file page by page.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of PageContent objects with text and metadata
        
    Raises:
        Exception: If PDF cannot be read or is corrupted
    """
    try:
        pdf_path = Path(file_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Read PDF
        reader = PdfReader(str(pdf_path))
        pages_content = []
        
        logger.info(f"Extracting text from {pdf_path.name} ({len(reader.pages)} pages)")
        
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                # Extract text from page
                text = page.extract_text()
                
                if text.strip():  # Only add non-empty pages
                    pages_content.append(
                        PageContent(
                            page_number=page_num,
                            text=text,
                            source_file=pdf_path.name
                        )
                    )
                else:
                    logger.warning(f"Page {page_num} is empty or unreadable in {pdf_path.name}")
                    
            except Exception as e:
                logger.error(f"Error extracting text from page {page_num} in {pdf_path.name}: {e}")
                # Continue with other pages even if one fails
                continue
        
        if not pages_content:
            raise ValueError(f"No text could be extracted from {pdf_path.name}")
        
        # Remove potential headers/footers
        pages_content = remove_repeated_text(pages_content)
        
        logger.info(f"Successfully extracted {len(pages_content)} pages from {pdf_path.name}")
        return pages_content
        
    except Exception as e:
        logger.error(f"Failed to extract text from {file_path}: {e}")
        raise


def remove_repeated_text(pages: List[PageContent], threshold: int = 3) -> List[PageContent]:
    """
    Remove repeated text across pages (headers/footers).
    
    This is a heuristic approach: if the same line appears in multiple pages
    at the start or end, it's likely a header/footer.
    
    Args:
        pages: List of PageContent objects
        threshold: Minimum number of pages for text to be considered repeated
        
    Returns:
        List of PageContent with headers/footers removed
    """
    if len(pages) < threshold:
        return pages
    
    try:
        # Collect first and last lines from each page
        first_lines = []
        last_lines = []
        
        for page in pages:
            lines = page.text.split('\n')
            clean_lines = [line.strip() for line in lines if line.strip()]
            
            if clean_lines:
                first_lines.append(clean_lines[0])
                last_lines.append(clean_lines[-1])
        
        # Find repeated lines
        first_line_counts = Counter(first_lines)
        last_line_counts = Counter(last_lines)
        
        # Identify headers/footers (appear in threshold+ pages)
        headers = {line for line, count in first_line_counts.items() 
                  if count >= threshold and len(line) > 5}
        footers = {line for line, count in last_line_counts.items() 
                  if count >= threshold and len(line) > 5}
        
        # Remove headers/footers from each page
        cleaned_pages = []
        for page in pages:
            text = page.text
            
            # Remove headers
            for header in headers:
                text = text.replace(header, '', 1)
            
            # Remove footers
            for footer in footers:
                # Remove from end only
                if text.rstrip().endswith(footer):
                    text = text.rstrip()[:-len(footer)]
            
            cleaned_pages.append(
                PageContent(
                    page_number=page.page_number,
                    text=clean_text(text),
                    source_file=page.source_file
                )
            )
        
        if headers or footers:
            logger.info(f"Removed {len(headers)} headers and {len(footers)} footers")
        
        return cleaned_pages
        
    except Exception as e:
        logger.warning(f"Error removing repeated text: {e}. Returning original pages.")
        return pages


def clean_text(text: str) -> str:
    """
    Clean extracted text by normalizing whitespace and removing artifacts.
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers (common pattern: single number on its own line)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Remove bullet points and list markers that might be artifacts
    # Keep actual content but normalize spacing
    text = re.sub(r'\s*[•·○▪▫■□]\s*', ' ', text)
    
    # Normalize dashes and hyphens
    text = re.sub(r'—|–', '-', text)
    
    # Remove multiple consecutive periods (artifacts)
    text = re.sub(r'\.{4,}', '', text)
    
    # Clean up spacing around punctuation
    text = re.sub(r'\s+([.,;!?])', r'\1', text)
    
    # Ensure single space after punctuation
    text = re.sub(r'([.,;!?])([A-Za-z])', r'\1 \2', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


# ============= Text Chunking =============

def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    metadata: Dict[str, Any]
) -> List[Chunk]:
    """
    Split text into overlapping chunks with sentence awareness.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum chunk size in characters
        overlap: Number of characters to overlap between chunks
        metadata: Metadata to attach to each chunk
        
    Returns:
        List of Chunk objects
    """
    if not text.strip():
        return []
    
    # Split into sentences
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_index = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence exceeds chunk_size and we have content, save current chunk
        if current_length + sentence_length > chunk_size and current_chunk:
            # Create chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(
                Chunk(
                    chunk_id=f"{metadata.get('source_file', 'unknown')}_{metadata.get('page_number', 0)}_{chunk_index}",
                    text=chunk_text,
                    source_file=metadata.get('source_file', 'unknown'),
                    page_number=metadata.get('page_number', 0),
                    chunk_index=chunk_index,
                    metadata=metadata
                )
            )
            chunk_index += 1
            
            # Start new chunk with overlap
            # Keep last few sentences for overlap
            overlap_text = chunk_text[-overlap:] if overlap > 0 else ""
            overlap_sentences = split_into_sentences(overlap_text)
            
            current_chunk = overlap_sentences
            current_length = sum(len(s) for s in current_chunk)
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(
            Chunk(
                chunk_id=f"{metadata.get('source_file', 'unknown')}_{metadata.get('page_number', 0)}_{chunk_index}",
                text=chunk_text,
                source_file=metadata.get('source_file', 'unknown'),
                page_number=metadata.get('page_number', 0),
                chunk_index=chunk_index,
                metadata=metadata
            )
        )
    
    return chunks


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex patterns.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Handle common abbreviations that shouldn't trigger sentence breaks
    text = re.sub(r'\bDr\.', 'Dr<DOT>', text)
    text = re.sub(r'\bMr\.', 'Mr<DOT>', text)
    text = re.sub(r'\bMrs\.', 'Mrs<DOT>', text)
    text = re.sub(r'\bMs\.', 'Ms<DOT>', text)
    text = re.sub(r'\bProf\.', 'Prof<DOT>', text)
    text = re.sub(r'\bet al\.', 'et al<DOT>', text)
    text = re.sub(r'\bi\.e\.', 'i<DOT>e<DOT>', text)
    text = re.sub(r'\be\.g\.', 'e<DOT>g<DOT>', text)
    
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Restore abbreviations
    sentences = [s.replace('<DOT>', '.') for s in sentences]
    
    # Filter out very short fragments
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    return sentences


def chunk_pages(
    pages: List[PageContent],
    chunk_size: int,
    overlap: int
) -> List[Chunk]:
    """
    Chunk multiple pages of content.
    
    Args:
        pages: List of PageContent objects
        chunk_size: Maximum chunk size
        overlap: Overlap between chunks
        
    Returns:
        List of all chunks from all pages
    """
    all_chunks = []
    
    for page in pages:
        metadata = {
            'source_file': page.source_file,
            'page_number': page.page_number
        }
        
        page_chunks = chunk_text(
            text=page.text,
            chunk_size=chunk_size,
            overlap=overlap,
            metadata=metadata
        )
        
        all_chunks.extend(page_chunks)
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks


# ============= Utility Functions =============

def get_text_statistics(text: str) -> Dict[str, Any]:
    """
    Get statistics about a text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with statistics
    """
    return {
        'total_characters': len(text),
        'total_words': len(text.split()),
        'total_sentences': len(split_into_sentences(text)),
        'average_sentence_length': len(text.split()) / max(len(split_into_sentences(text)), 1)
    }

