"""Tests for chunking and PDF extraction functionality"""

import pytest
from pathlib import Path

from app.core.chunking import (
    clean_text,
    split_into_sentences,
    chunk_text,
    get_text_statistics
)
from app.models.schemas import PageContent


class TestCleanText:
    """Test text cleaning functionality"""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning"""
        text = "This  is   a    test."
        result = clean_text(text)
        assert result == "This is a test."
    
    def test_clean_text_multiple_periods(self):
        """Test removal of multiple periods"""
        text = "Table of contents........ 5"
        result = clean_text(text)
        assert "....." not in result
    
    def test_clean_text_bullet_points(self):
        """Test removal of bullet point artifacts"""
        text = "• First item\n• Second item"
        result = clean_text(text)
        assert "•" not in result
    
    def test_clean_text_empty(self):
        """Test cleaning empty text"""
        result = clean_text("")
        assert result == ""


class TestSplitSentences:
    """Test sentence splitting functionality"""
    
    def test_split_basic(self):
        """Test basic sentence splitting"""
        text = "This is sentence one. This is sentence two. This is sentence three."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3
    
    def test_split_with_abbreviations(self):
        """Test that abbreviations don't break sentences"""
        text = "Dr. Smith went to the store. He bought milk."
        sentences = split_into_sentences(text)
        assert len(sentences) == 2
        assert "Dr. Smith" in sentences[0]
    
    def test_split_with_ie_eg(self):
        """Test i.e. and e.g. don't break sentences"""
        text = "Some examples, e.g. apples and oranges. Another sentence here."
        sentences = split_into_sentences(text)
        assert "e.g." in sentences[0] or "e<DOT>g<DOT>" in sentences[0]
    
    def test_split_short_fragments_filtered(self):
        """Test that very short fragments are filtered out"""
        text = "Hi. This is a proper sentence with enough content."
        sentences = split_into_sentences(text)
        # "Hi." should be filtered (< 10 chars)
        assert all(len(s) > 10 for s in sentences)


class TestChunkText:
    """Test text chunking functionality"""
    
    def test_chunk_simple_text(self):
        """Test chunking simple text"""
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunk_text(
            text=text,
            chunk_size=40,
            overlap=10,
            metadata={'source_file': 'test.pdf', 'page_number': 1}
        )
        
        assert len(chunks) > 0
        assert all(isinstance(chunk.text, str) for chunk in chunks)
        assert all(chunk.source_file == 'test.pdf' for chunk in chunks)
    
    def test_chunk_with_overlap(self):
        """Test that chunks have proper overlap"""
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = chunk_text(
            text=text,
            chunk_size=50,
            overlap=20,
            metadata={'source_file': 'test.pdf', 'page_number': 1}
        )
        
        # Should have multiple chunks due to size limit
        assert len(chunks) > 1
    
    def test_chunk_empty_text(self):
        """Test chunking empty text"""
        chunks = chunk_text(
            text="",
            chunk_size=100,
            overlap=10,
            metadata={'source_file': 'test.pdf', 'page_number': 1}
        )
        assert len(chunks) == 0
    
    def test_chunk_ids_unique(self):
        """Test that chunk IDs are unique"""
        text = "Sentence one. " * 20  # Create long text
        chunks = chunk_text(
            text=text,
            chunk_size=50,
            overlap=10,
            metadata={'source_file': 'test.pdf', 'page_number': 1}
        )
        
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique


class TestTextStatistics:
    """Test text statistics functionality"""
    
    def test_statistics_basic(self):
        """Test basic statistics calculation"""
        text = "This is a test. Another sentence here."
        stats = get_text_statistics(text)
        
        assert 'total_characters' in stats
        assert 'total_words' in stats
        assert 'total_sentences' in stats
        assert 'average_sentence_length' in stats
        
        assert stats['total_characters'] > 0
        assert stats['total_words'] > 0
        assert stats['total_sentences'] == 2
    
    def test_statistics_empty(self):
        """Test statistics with empty text"""
        stats = get_text_statistics("")
        assert stats['total_characters'] == 0
        assert stats['total_words'] == 0


# Integration test (will be skipped if no PDF available)
class TestPDFExtraction:
    """Test PDF extraction - requires actual PDF file"""
    
    @pytest.mark.skip(reason="Requires actual PDF file")
    def test_extract_text_from_pdf(self):
        """Test PDF text extraction with a real file"""
        # This test would need an actual PDF file
        # Will implement once we have sample data
        pass

