"""
Tests for LLM integration module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.core.llm import (
    MistralClient,
    get_mistral_client,
    generate_answer,
    format_answer_with_metadata,
    ANSWER_GENERATION_PROMPT,
    ANSWER_WITH_SOURCES_PROMPT
)


class TestMistralClient:
    """Tests for MistralClient class."""
    
    def test_initialization_with_api_key(self):
        """Test client initialization with API key."""
        with patch('app.core.llm.MistralAPIClient'):
            client = MistralClient(api_key="test_key", model="mistral-small")
            
            assert client.api_key == "test_key"
            assert client.model == "mistral-small"
            assert client.max_retries == 3
    
    def test_initialization_without_api_key(self):
        """Test that missing API key raises error."""
        with patch('app.core.llm.settings') as mock_settings:
            mock_settings.MISTRAL_API_KEY = None
            
            with pytest.raises(ValueError, match="Mistral API key not found"):
                MistralClient()
    
    def test_initialization_from_settings(self):
        """Test client initialization from settings."""
        with patch('app.core.llm.settings') as mock_settings:
            mock_settings.MISTRAL_API_KEY = "settings_key"
            
            with patch('app.core.llm.MistralAPIClient'):
                client = MistralClient()
                
                assert client.api_key == "settings_key"


class TestAnswerGeneration:
    """Tests for answer generation."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mocked Mistral client."""
        with patch('app.core.llm.MistralAPIClient') as mock_api:
            client = MistralClient(api_key="test_key")
            
            # Mock the API response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "This is a test answer."
            mock_response.usage.total_tokens = 50
            
            client.client.chat = Mock(return_value=mock_response)
            
            yield client
    
    def test_generate_answer_basic(self, mock_client):
        """Test basic answer generation."""
        context_chunks = [
            "Machine learning is a method of data analysis.",
            "It automates analytical model building."
        ]
        
        result = mock_client.generate_answer(
            question="What is machine learning?",
            context_chunks=context_chunks
        )
        
        assert result["answer"] == "This is a test answer."
        assert result["tokens_used"] == 50
        assert result["model"] == "mistral-small"
        assert "error" not in result
    
    def test_generate_answer_no_context(self, mock_client):
        """Test answer generation with no context."""
        result = mock_client.generate_answer(
            question="What is machine learning?",
            context_chunks=[]
        )
        
        assert "don't have any relevant information" in result["answer"]
        assert result["tokens_used"] == 0
    
    def test_generate_answer_with_source_numbers(self, mock_client):
        """Test answer generation with source numbers."""
        context_chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        
        result = mock_client.generate_answer(
            question="Test question?",
            context_chunks=context_chunks,
            include_source_numbers=True
        )
        
        # Verify that the client was called
        assert mock_client.client.chat.called
        
        # Check that the prompt includes numbered sources
        call_args = mock_client.client.chat.call_args
        messages = call_args.kwargs['messages']
        prompt = messages[0].content
        
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt
    
    def test_generate_answer_temperature(self, mock_client):
        """Test that temperature parameter is passed correctly."""
        context_chunks = ["Test context"]
        
        mock_client.generate_answer(
            question="Test?",
            context_chunks=context_chunks,
            temperature=0.3
        )
        
        call_args = mock_client.client.chat.call_args
        assert call_args.kwargs['temperature'] == 0.3
    
    def test_generate_answer_max_tokens(self, mock_client):
        """Test that max_tokens parameter is passed correctly."""
        context_chunks = ["Test context"]
        
        mock_client.generate_answer(
            question="Test?",
            context_chunks=context_chunks,
            max_tokens=256
        )
        
        call_args = mock_client.client.chat.call_args
        assert call_args.kwargs['max_tokens'] == 256


class TestErrorHandling:
    """Tests for error handling and retries."""
    
    def test_retry_on_failure(self):
        """Test that client retries on failure."""
        with patch('app.core.llm.MistralAPIClient') as mock_api:
            client = MistralClient(api_key="test_key", max_retries=3)
            
            # Mock to fail twice, then succeed
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Success!"
            mock_response.usage.total_tokens = 10
            
            client.client.chat = Mock(side_effect=[
                Exception("API Error 1"),
                Exception("API Error 2"),
                mock_response
            ])
            
            with patch('time.sleep'):  # Don't actually sleep in tests
                result = client.generate_answer(
                    question="Test?",
                    context_chunks=["Context"]
                )
            
            assert result["answer"] == "Success!"
            assert client.client.chat.call_count == 3
    
    def test_all_retries_fail(self):
        """Test behavior when all retries fail."""
        with patch('app.core.llm.MistralAPIClient') as mock_api:
            client = MistralClient(api_key="test_key", max_retries=2)
            
            # Mock to always fail
            client.client.chat = Mock(side_effect=Exception("Persistent error"))
            
            with patch('time.sleep'):
                result = client.generate_answer(
                    question="Test?",
                    context_chunks=["Context"]
                )
            
            assert "error" in result["answer"].lower()
            assert "error" in result
            assert result["tokens_used"] == 0


class TestContextFormatting:
    """Tests for context formatting."""
    
    @pytest.fixture
    def client(self):
        with patch('app.core.llm.MistralAPIClient'):
            return MistralClient(api_key="test_key")
    
    def test_format_context(self, client):
        """Test basic context formatting."""
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        
        formatted = client._format_context(chunks)
        
        assert "Chunk 1" in formatted
        assert "Chunk 2" in formatted
        assert "Chunk 3" in formatted
        assert "---" in formatted  # Separator
    
    def test_format_context_truncation(self, client):
        """Test that very long context is truncated."""
        # Create a very long chunk
        long_chunk = "A" * 10000
        chunks = [long_chunk]
        
        formatted = client._format_context(chunks)
        
        assert len(formatted) <= 8100  # max_chars + some buffer
        assert "[Context truncated...]" in formatted
    
    def test_format_context_with_numbers(self, client):
        """Test context formatting with source numbers."""
        chunks = ["First chunk", "Second chunk", "Third chunk"]
        
        formatted = client._format_context_with_numbers(chunks)
        
        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "[3]" in formatted
        assert "First chunk" in formatted
        assert "Second chunk" in formatted


class TestSummarization:
    """Tests for text summarization."""
    
    def test_summarize_text(self):
        """Test text summarization."""
        with patch('app.core.llm.MistralAPIClient') as mock_api:
            client = MistralClient(api_key="test_key")
            
            # Mock response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "This is a summary."
            
            client.client.chat = Mock(return_value=mock_response)
            
            summary = client.summarize_text("Long text to summarize...")
            
            assert summary == "This is a summary."
            assert client.client.chat.called
    
    def test_summarize_text_error(self):
        """Test summarization error handling."""
        with patch('app.core.llm.MistralAPIClient') as mock_api:
            client = MistralClient(api_key="test_key")
            
            client.client.chat = Mock(side_effect=Exception("API error"))
            
            summary = client.summarize_text("Text")
            
            assert "Error" in summary


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_get_mistral_client_singleton(self):
        """Test that get_mistral_client returns singleton."""
        with patch('app.core.llm.MistralAPIClient'):
            with patch('app.core.llm.settings') as mock_settings:
                mock_settings.MISTRAL_API_KEY = "test_key"
                
                client1 = get_mistral_client()
                client2 = get_mistral_client()
                
                assert client1 is client2
    
    def test_generate_answer_convenience(self):
        """Test generate_answer convenience function."""
        with patch('app.core.llm.MistralAPIClient') as mock_api:
            with patch('app.core.llm.settings') as mock_settings:
                mock_settings.MISTRAL_API_KEY = "test_key"
                
                # Mock response
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "Answer"
                mock_response.usage.total_tokens = 20
                
                # Get client and mock its chat method
                client = get_mistral_client()
                client.client.chat = Mock(return_value=mock_response)
                
                result = generate_answer(
                    question="Test?",
                    context_chunks=["Context"]
                )
                
                assert result["answer"] == "Answer"
    
    def test_format_answer_with_metadata(self):
        """Test formatting answer with metadata."""
        answer_dict = {
            "answer": "Test answer",
            "model": "mistral-small",
            "tokens_used": 50
        }
        
        sources = [
            {"file": "doc1.pdf", "page": 1},
            {"file": "doc2.pdf", "page": 3}
        ]
        
        formatted = format_answer_with_metadata(answer_dict, sources)
        
        assert formatted["answer"] == "Test answer"
        assert formatted["model"] == "mistral-small"
        assert formatted["tokens_used"] == 50
        assert formatted["sources"] == sources
        assert formatted["status"] == "success"
    
    def test_format_answer_with_error(self):
        """Test formatting answer with error."""
        answer_dict = {
            "answer": "Error occurred",
            "model": "mistral-small",
            "tokens_used": 0,
            "error": "API timeout"
        }
        
        formatted = format_answer_with_metadata(answer_dict)
        
        assert formatted["status"] == "error"
        assert formatted["error"] == "API timeout"


class TestPromptTemplates:
    """Tests for prompt templates."""
    
    def test_answer_generation_prompt_format(self):
        """Test answer generation prompt formatting."""
        prompt = ANSWER_GENERATION_PROMPT.format(
            context="Test context",
            question="Test question?"
        )
        
        assert "Test context" in prompt
        assert "Test question?" in prompt
        assert "based ONLY on the provided context" in prompt
    
    def test_answer_with_sources_prompt_format(self):
        """Test answer with sources prompt formatting."""
        prompt = ANSWER_WITH_SOURCES_PROMPT.format(
            context_with_numbers="[1] Test",
            question="Test?"
        )
        
        assert "[1] Test" in prompt
        assert "Test?" in prompt
        assert "source numbers" in prompt.lower()

