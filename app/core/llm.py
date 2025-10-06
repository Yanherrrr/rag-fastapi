"""
LLM Integration Module

This module provides integration with Mistral AI for generating natural language
answers based on retrieved context. This is the final step in the RAG pipeline
that converts retrieved chunks into coherent, contextual answers.
"""

import logging
import time
from typing import List, Optional, Dict
from mistralai.client import MistralClient as MistralAPIClient
from mistralai.models.chat_completion import ChatMessage

from app.core.config import settings

logger = logging.getLogger(__name__)


# ============= Prompt Templates =============

ANSWER_GENERATION_PROMPT = """You are a helpful AI assistant. Your task is to answer the user's question based ONLY on the provided context from the knowledge base.

Important guidelines:
- Only use information from the context below
- If the context doesn't contain enough information to answer the question, say "I don't have sufficient information to answer this question based on the available documents."
- Be concise and accurate
- Cite relevant parts of the context when possible
- Do not make up or infer information not present in the context

Context:
{context}

Question: {question}

Answer:"""


ANSWER_WITH_SOURCES_PROMPT = """You are a helpful AI assistant. Answer the user's question based ONLY on the provided context.

Important guidelines:
- Use only information from the numbered context sources below
- Reference sources by their numbers (e.g., "According to source [1]...")
- If the context doesn't contain enough information, say "I don't have sufficient information to answer this question."
- Be specific and cite which sources support each claim

Context Sources:
{context_with_numbers}

Question: {question}

Answer (include source numbers in your response):"""


SUMMARIZATION_PROMPT = """Summarize the following text concisely, focusing on the key points:

Text:
{text}

Summary:"""


# ============= LLM Client =============

class MistralClient:
    """
    Client for Mistral AI LLM integration.
    
    Handles:
    - Answer generation from context
    - Error handling and retries
    - Token usage tracking
    - Multiple prompt strategies
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-small",
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize Mistral client.
        
        Args:
            api_key: Mistral API key (if None, reads from settings)
            model: Model to use (mistral-small, mistral-medium, mistral-large)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or settings.MISTRAL_API_KEY
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        
        if not self.api_key:
            logger.error("Mistral API key not provided")
            raise ValueError(
                "Mistral API key not found. Set MISTRAL_API_KEY in .env file."
            )
        
        try:
            self.client = MistralAPIClient(api_key=self.api_key)
            logger.info(f"Mistral client initialized with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize Mistral client: {e}")
            raise
    
    def generate_answer(
        self,
        question: str,
        context_chunks: List[str],
        temperature: float = 0.7,
        max_tokens: int = 512,
        include_source_numbers: bool = False
    ) -> Dict:
        """
        Generate an answer based on the question and context chunks.
        
        Args:
            question: User's question
            context_chunks: List of relevant context chunks
            temperature: Sampling temperature (0-1, lower = more deterministic)
            max_tokens: Maximum tokens in response
            include_source_numbers: Whether to include source numbers in prompt
            
        Returns:
            Dict with 'answer', 'tokens_used', and 'model'
        """
        if not context_chunks:
            logger.warning("No context chunks provided for answer generation")
            return {
                "answer": "I don't have any relevant information to answer this question.",
                "tokens_used": 0,
                "model": self.model
            }
        
        # Format context
        if include_source_numbers:
            context = self._format_context_with_numbers(context_chunks)
            prompt = ANSWER_WITH_SOURCES_PROMPT.format(
                context_with_numbers=context,
                question=question
            )
        else:
            context = self._format_context(context_chunks)
            prompt = ANSWER_GENERATION_PROMPT.format(
                context=context,
                question=question
            )
        
        logger.info(f"Generating answer for question: '{question[:50]}...'")
        logger.debug(f"Using {len(context_chunks)} context chunks")
        
        # Generate answer with retries
        for attempt in range(self.max_retries):
            try:
                response = self._call_api(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                answer = response.choices[0].message.content.strip()
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
                
                logger.info(f"Answer generated successfully (tokens: {tokens_used})")
                
                return {
                    "answer": answer,
                    "tokens_used": tokens_used,
                    "model": self.model
                }
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries} attempts failed")
                    return {
                        "answer": "I'm sorry, I encountered an error while generating an answer. Please try again.",
                        "tokens_used": 0,
                        "model": self.model,
                        "error": str(e)
                    }
    
    def _call_api(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ):
        """
        Call Mistral API.
        
        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            API response
        """
        messages = [
            ChatMessage(role="user", content=prompt)
        ]
        
        response = self.client.chat(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response
    
    def _format_context(self, chunks: List[str]) -> str:
        """
        Format context chunks into a single string.
        
        Args:
            chunks: List of context chunks
            
        Returns:
            Formatted context string
        """
        # Join chunks with clear separation
        formatted = "\n\n---\n\n".join(chunks)
        
        # Truncate if too long (approximate token limit)
        max_chars = 8000  # ~2000 tokens
        if len(formatted) > max_chars:
            logger.warning(f"Context truncated from {len(formatted)} to {max_chars} chars")
            formatted = formatted[:max_chars] + "\n\n[Context truncated...]"
        
        return formatted
    
    def _format_context_with_numbers(self, chunks: List[str]) -> str:
        """
        Format context chunks with source numbers.
        
        Args:
            chunks: List of context chunks
            
        Returns:
            Formatted context with numbers
        """
        numbered_chunks = []
        for i, chunk in enumerate(chunks, start=1):
            numbered_chunks.append(f"[{i}] {chunk}")
        
        formatted = "\n\n".join(numbered_chunks)
        
        # Truncate if too long
        max_chars = 8000
        if len(formatted) > max_chars:
            logger.warning(f"Context truncated from {len(formatted)} to {max_chars} chars")
            formatted = formatted[:max_chars] + "\n\n[Context truncated...]"
        
        return formatted
    
    def summarize_text(
        self,
        text: str,
        temperature: float = 0.5,
        max_tokens: int = 256
    ) -> str:
        """
        Summarize a given text.
        
        Args:
            text: Text to summarize
            temperature: Sampling temperature
            max_tokens: Maximum tokens in summary
            
        Returns:
            Summary text
        """
        prompt = SUMMARIZATION_PROMPT.format(text=text)
        
        logger.info("Generating summary")
        
        try:
            response = self._call_api(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info("Summary generated successfully")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return "Error generating summary."


# ============= Convenience Functions =============

# Global client instance (singleton pattern)
_mistral_client: Optional[MistralClient] = None


def get_mistral_client(
    api_key: Optional[str] = None,
    model: str = "mistral-small"
) -> MistralClient:
    """
    Get the global Mistral client instance.
    
    Args:
        api_key: Optional API key (uses settings if not provided)
        model: Model to use
        
    Returns:
        MistralClient instance
    """
    global _mistral_client
    
    # Create new client if doesn't exist or if API key/model changed
    if _mistral_client is None or api_key is not None:
        _mistral_client = MistralClient(api_key=api_key, model=model)
    
    return _mistral_client


def generate_answer(
    question: str,
    context_chunks: List[str],
    temperature: float = 0.7,
    include_source_numbers: bool = False
) -> Dict:
    """
    Generate answer using global client (convenience function).
    
    Args:
        question: User's question
        context_chunks: Relevant context chunks
        temperature: Sampling temperature
        include_source_numbers: Include source citations
        
    Returns:
        Dict with answer and metadata
    """
    client = get_mistral_client()
    return client.generate_answer(
        question=question,
        context_chunks=context_chunks,
        temperature=temperature,
        include_source_numbers=include_source_numbers
    )


def format_answer_with_metadata(
    answer_dict: Dict,
    sources: Optional[List[Dict]] = None
) -> Dict:
    """
    Format answer with additional metadata.
    
    Args:
        answer_dict: Dict from generate_answer()
        sources: Optional list of source metadata
        
    Returns:
        Formatted answer dict
    """
    formatted = {
        "answer": answer_dict["answer"],
        "model": answer_dict["model"],
        "tokens_used": answer_dict.get("tokens_used", 0)
    }
    
    if sources:
        formatted["sources"] = sources
    
    if "error" in answer_dict:
        formatted["error"] = answer_dict["error"]
        formatted["status"] = "error"
    else:
        formatted["status"] = "success"
    
    return formatted

