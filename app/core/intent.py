"""
Intent Detection Module

This module classifies user queries into different intent categories to determine
the appropriate response strategy. Intent detection helps distinguish between:
- Knowledge base searches (RAG pipeline)
- Simple conversational queries (greetings, chitchat)
- Goodbyes

The implementation uses pattern matching and heuristics for fast, deterministic
classification without requiring LLM calls.
"""

import logging
from enum import Enum
from typing import Set, List

logger = logging.getLogger(__name__)


class Intent(Enum):
    """
    User query intent categories.
    """
    SEARCH_KNOWLEDGE_BASE = "search"  # Requires RAG pipeline
    GREETING = "greeting"              # Simple greeting response
    CHITCHAT = "chitchat"             # Acknowledgments, thanks, etc.
    GOODBYE = "goodbye"                # Farewell messages
    
    def __str__(self) -> str:
        return self.value


# ============= Intent Patterns =============

# Greeting patterns (case-insensitive)
GREETING_PATTERNS: Set[str] = {
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "greetings",
    "howdy",
    "hi there",
    "hello there",
}

# Goodbye patterns (case-insensitive)
GOODBYE_PATTERNS: Set[str] = {
    "bye",
    "goodbye",
    "good bye",
    "see you",
    "see ya",
    "farewell",
    "take care",
    "catch you later",
    "talk to you later",
    "exit",
    "quit",
}

# Chitchat patterns (case-insensitive)
CHITCHAT_PATTERNS: Set[str] = {
    "thanks",
    "thank you",
    "thank you very much",
    "thanks a lot",
    "appreciate it",
    "ok",
    "okay",
    "sure",
    "alright",
    "got it",
    "understood",
    "i see",
    "makes sense",
    "cool",
    "nice",
    "awesome",
    "great",
    "perfect",
    "how are you",
    "how are you doing",
    "what's up",
    "whats up",
}

# Question words that indicate search intent
QUESTION_WORDS: Set[str] = {
    "what",
    "how",
    "why",
    "when",
    "where",
    "who",
    "which",
    "whose",
    "whom",
    "can",
    "could",
    "would",
    "should",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "will",
    "shall",
    "may",
    "might",
    "tell",
    "explain",
    "describe",
    "define",
    "show",
}


# ============= Intent Detection =============

class IntentDetector:
    """
    Detects user intent from query text.
    
    Uses a three-tier detection strategy:
    1. Pattern matching (fast, deterministic)
    2. Heuristics (for edge cases)
    3. Safe default (SEARCH_KNOWLEDGE_BASE)
    """
    
    def __init__(
        self,
        greeting_patterns: Set[str] = None,
        goodbye_patterns: Set[str] = None,
        chitchat_patterns: Set[str] = None,
        question_words: Set[str] = None
    ):
        """
        Initialize intent detector.
        
        Args:
            greeting_patterns: Custom greeting patterns (optional)
            goodbye_patterns: Custom goodbye patterns (optional)
            chitchat_patterns: Custom chitchat patterns (optional)
            question_words: Custom question words (optional)
        """
        self.greeting_patterns = greeting_patterns or GREETING_PATTERNS
        self.goodbye_patterns = goodbye_patterns or GOODBYE_PATTERNS
        self.chitchat_patterns = chitchat_patterns or CHITCHAT_PATTERNS
        self.question_words = question_words or QUESTION_WORDS
        
        logger.info(
            f"IntentDetector initialized with "
            f"{len(self.greeting_patterns)} greeting patterns, "
            f"{len(self.goodbye_patterns)} goodbye patterns, "
            f"{len(self.chitchat_patterns)} chitchat patterns"
        )
    
    def detect(self, query: str) -> Intent:
        """
        Detect intent from user query.
        
        Args:
            query: User query text
            
        Returns:
            Detected Intent
        """
        if not query or not query.strip():
            logger.warning("Empty query received, defaulting to SEARCH")
            return Intent.SEARCH_KNOWLEDGE_BASE
        
        # Normalize query
        query_lower = query.lower().strip()
        query_words = query_lower.split()
        
        logger.debug(f"Detecting intent for query: '{query}'")
        
        # Tier 1: Pattern matching
        intent = self._match_patterns(query_lower)
        if intent:
            logger.info(f"Intent detected via pattern: {intent}")
            return intent
        
        # Tier 2: Heuristics
        intent = self._apply_heuristics(query_lower, query_words)
        logger.info(f"Intent detected via heuristic: {intent}")
        
        return intent
    
    def _match_patterns(self, query_lower: str) -> Intent:
        """
        Match query against known patterns.
        
        Uses whole-phrase matching to avoid false positives
        (e.g., "hi" in "machine" should not match).
        
        Args:
            query_lower: Lowercase query text
            
        Returns:
            Intent if pattern matched, None otherwise
        """
        # For multi-word patterns, check if the pattern exists as a phrase
        # For single-word patterns, check if it's a whole word
        
        # Check greeting patterns
        for pattern in self.greeting_patterns:
            if self._pattern_matches(pattern, query_lower):
                return Intent.GREETING
        
        # Check goodbye patterns
        for pattern in self.goodbye_patterns:
            if self._pattern_matches(pattern, query_lower):
                return Intent.GOODBYE
        
        # Check chitchat patterns
        for pattern in self.chitchat_patterns:
            if self._pattern_matches(pattern, query_lower):
                return Intent.CHITCHAT
        
        return None
    
    def _pattern_matches(self, pattern: str, query_lower: str) -> bool:
        """
        Check if pattern matches in query (whole word/phrase matching).
        
        Args:
            pattern: Pattern to match
            query_lower: Lowercase query text
            
        Returns:
            True if pattern matches
        """
        # For multi-word patterns, check if the phrase exists
        if " " in pattern:
            return pattern in query_lower
        
        # For single-word patterns, use word boundary checking
        # Split query into words and check for exact match
        query_words = query_lower.split()
        
        # Remove punctuation from words for comparison
        import string
        query_words_clean = [
            word.strip(string.punctuation) for word in query_words
        ]
        
        return pattern in query_words_clean
    
    def _apply_heuristics(self, query_lower: str, query_words: List[str]) -> Intent:
        """
        Apply heuristics to determine intent.
        
        Args:
            query_lower: Lowercase query text
            query_words: List of query words
            
        Returns:
            Detected Intent
        """
        # Heuristic 1: Very short queries without questions
        # Examples: "ok", "sure", "yeah", "hmm"
        # But NOT topics like "machine learning" or "neural networks"
        if len(query_words) <= 2 and "?" not in query_lower:
            # Check if it's not a potential topic
            # Single-word or very short phrases that are clearly not topics
            if len(query_words) == 1 and len(query_words[0]) <= 4:
                # Very short single words like "ok", "sure", "yep", "hmm"
                logger.debug("Very short single-word query → CHITCHAT")
                return Intent.CHITCHAT
            # If 2 words, only classify as chitchat if both are very short
            elif len(query_words) == 2 and all(len(w) <= 3 for w in query_words):
                # Things like "oh ok", "uh huh"
                logger.debug("Very short two-word query → CHITCHAT")
                return Intent.CHITCHAT
            # Otherwise, topics like "machine learning", "neural networks" fall through
        
        # Heuristic 2: Contains question words → search
        if any(word in self.question_words for word in query_words):
            logger.debug("Contains question word → SEARCH")
            return Intent.SEARCH_KNOWLEDGE_BASE
        
        # Heuristic 3: Ends with question mark → search
        if query_lower.strip().endswith("?"):
            logger.debug("Ends with '?' → SEARCH")
            return Intent.SEARCH_KNOWLEDGE_BASE
        
        # Heuristic 4: Command-like queries → search
        # Examples: "explain X", "show me Y", "tell me about Z"
        command_verbs = {"explain", "describe", "tell", "show", "list", "give", "find"}
        if any(word in command_verbs for word in query_words):
            logger.debug("Contains command verb → SEARCH")
            return Intent.SEARCH_KNOWLEDGE_BASE
        
        # Default: SEARCH_KNOWLEDGE_BASE (safe fallback)
        # Better to search and find nothing than to dismiss a real question
        logger.debug("No pattern or heuristic matched → SEARCH (default)")
        return Intent.SEARCH_KNOWLEDGE_BASE


# ============= Convenience Functions =============

# Global detector instance (singleton pattern)
_intent_detector: IntentDetector = None


def get_intent_detector() -> IntentDetector:
    """
    Get the global intent detector instance.
    
    Returns:
        IntentDetector instance
    """
    global _intent_detector
    if _intent_detector is None:
        _intent_detector = IntentDetector()
    return _intent_detector


def detect_intent(query: str) -> Intent:
    """
    Detect intent from query (convenience function).
    
    Args:
        query: User query text
        
    Returns:
        Detected Intent
    """
    detector = get_intent_detector()
    return detector.detect(query)


def is_conversational(intent: Intent) -> bool:
    """
    Check if intent is conversational (doesn't need RAG).
    
    Args:
        intent: Intent to check
        
    Returns:
        True if conversational, False if needs RAG
    """
    return intent in {Intent.GREETING, Intent.CHITCHAT, Intent.GOODBYE}


def get_simple_response(intent: Intent) -> str:
    """
    Get a simple response for conversational intents.
    
    Args:
        intent: Conversational intent
        
    Returns:
        Simple response text
    """
    responses = {
        Intent.GREETING: "Hello! How can I help you today?",
        Intent.CHITCHAT: "You're welcome! Is there anything else I can help you with?",
        Intent.GOODBYE: "Goodbye! Feel free to come back if you have more questions."
    }
    
    return responses.get(intent, "How can I help you?")

