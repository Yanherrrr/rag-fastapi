"""
Tests for intent detection module.
"""

import pytest
from app.core.intent import (
    Intent,
    IntentDetector,
    detect_intent,
    get_intent_detector,
    is_conversational,
    get_simple_response,
    GREETING_PATTERNS,
    GOODBYE_PATTERNS,
    CHITCHAT_PATTERNS
)


class TestIntentEnum:
    """Tests for Intent enum."""
    
    def test_intent_values(self):
        """Test Intent enum values."""
        assert Intent.SEARCH_KNOWLEDGE_BASE.value == "search"
        assert Intent.GREETING.value == "greeting"
        assert Intent.CHITCHAT.value == "chitchat"
        assert Intent.GOODBYE.value == "goodbye"
    
    def test_intent_str(self):
        """Test Intent string representation."""
        assert str(Intent.SEARCH_KNOWLEDGE_BASE) == "search"
        assert str(Intent.GREETING) == "greeting"


class TestIntentDetector:
    """Tests for IntentDetector class."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = IntentDetector()
        
        assert detector.greeting_patterns == GREETING_PATTERNS
        assert detector.goodbye_patterns == GOODBYE_PATTERNS
        assert detector.chitchat_patterns == CHITCHAT_PATTERNS
        assert len(detector.question_words) > 0
    
    def test_custom_patterns(self):
        """Test initialization with custom patterns."""
        custom_greetings = {"custom hello", "custom hi"}
        detector = IntentDetector(greeting_patterns=custom_greetings)
        
        assert detector.greeting_patterns == custom_greetings
    
    def test_singleton_pattern(self):
        """Test that get_intent_detector returns singleton."""
        detector1 = get_intent_detector()
        detector2 = get_intent_detector()
        
        assert detector1 is detector2


class TestGreetingDetection:
    """Tests for greeting intent detection."""
    
    @pytest.fixture
    def detector(self):
        return IntentDetector()
    
    def test_simple_greetings(self, detector):
        """Test simple greeting detection."""
        greetings = [
            "hello",
            "hi",
            "hey",
            "Hello",
            "HI",
            "HELLO"
        ]
        
        for greeting in greetings:
            intent = detector.detect(greeting)
            assert intent == Intent.GREETING, f"Failed for: {greeting}"
    
    def test_greeting_with_punctuation(self, detector):
        """Test greetings with punctuation."""
        greetings = [
            "Hello!",
            "Hi there!",
            "Hey!!!",
        ]
        
        for greeting in greetings:
            intent = detector.detect(greeting)
            assert intent == Intent.GREETING, f"Failed for: {greeting}"
    
    def test_greeting_phrases(self, detector):
        """Test greeting phrases."""
        greetings = [
            "good morning",
            "Good afternoon",
            "GOOD EVENING",
            "Hi there",
            "Hello there"
        ]
        
        for greeting in greetings:
            intent = detector.detect(greeting)
            assert intent == Intent.GREETING, f"Failed for: {greeting}"
    
    def test_greeting_in_longer_text(self, detector):
        """Test greeting detection in longer text."""
        # Should still detect greeting even with additional text
        intent = detector.detect("Hello, can you help me?")
        assert intent == Intent.GREETING


class TestGoodbyeDetection:
    """Tests for goodbye intent detection."""
    
    @pytest.fixture
    def detector(self):
        return IntentDetector()
    
    def test_simple_goodbyes(self, detector):
        """Test simple goodbye detection."""
        goodbyes = [
            "bye",
            "goodbye",
            "good bye",
            "BYE",
            "Goodbye"
        ]
        
        for goodbye in goodbyes:
            intent = detector.detect(goodbye)
            assert intent == Intent.GOODBYE, f"Failed for: {goodbye}"
    
    def test_goodbye_phrases(self, detector):
        """Test goodbye phrases."""
        goodbyes = [
            "see you",
            "See ya",
            "take care",
            "exit",
            "quit"
        ]
        
        for goodbye in goodbyes:
            intent = detector.detect(goodbye)
            assert intent == Intent.GOODBYE, f"Failed for: {goodbye}"


class TestChitchatDetection:
    """Tests for chitchat intent detection."""
    
    @pytest.fixture
    def detector(self):
        return IntentDetector()
    
    def test_thanks_patterns(self, detector):
        """Test thank you detection."""
        thanks = [
            "thanks",
            "thank you",
            "Thanks!",
            "Thank you very much",
            "THANKS",
            "appreciate it"
        ]
        
        for thank in thanks:
            intent = detector.detect(thank)
            assert intent == Intent.CHITCHAT, f"Failed for: {thank}"
    
    def test_acknowledgments(self, detector):
        """Test acknowledgment detection."""
        acks = [
            "ok",
            "okay",
            "sure",
            "alright",
            "got it",
            "understood",
            "i see",
            "makes sense"
        ]
        
        for ack in acks:
            intent = detector.detect(ack)
            assert intent == Intent.CHITCHAT, f"Failed for: {ack}"
    
    def test_positive_feedback(self, detector):
        """Test positive feedback detection."""
        feedback = [
            "cool",
            "nice",
            "awesome",
            "great",
            "perfect"
        ]
        
        for fb in feedback:
            intent = detector.detect(fb)
            assert intent == Intent.CHITCHAT, f"Failed for: {fb}"


class TestSearchDetection:
    """Tests for search intent detection."""
    
    @pytest.fixture
    def detector(self):
        return IntentDetector()
    
    def test_question_words(self, detector):
        """Test questions with question words."""
        questions = [
            "What is machine learning?",
            "How does Python work?",
            "Why is data science important?",
            "When should I use neural networks?",
            "Where can I learn more?",
            "Who invented Python?",
            "Which algorithm is best?"
        ]
        
        for question in questions:
            intent = detector.detect(question)
            assert intent == Intent.SEARCH_KNOWLEDGE_BASE, f"Failed for: {question}"
    
    def test_questions_with_mark(self, detector):
        """Test questions ending with question mark."""
        questions = [
            "Machine learning?",
            "Tell me about Python?",
            "Deep learning applications?"
        ]
        
        for question in questions:
            intent = detector.detect(question)
            assert intent == Intent.SEARCH_KNOWLEDGE_BASE, f"Failed for: {question}"
    
    def test_command_queries(self, detector):
        """Test command-like queries."""
        commands = [
            "Explain machine learning",
            "Describe neural networks",
            "Tell me about Python",
            "Show me examples",
            "List the features"
        ]
        
        for command in commands:
            intent = detector.detect(command)
            assert intent == Intent.SEARCH_KNOWLEDGE_BASE, f"Failed for: {command}"
    
    def test_statements_as_queries(self, detector):
        """Test statements that are implicit questions."""
        statements = [
            "Machine learning applications",
            "Python for data science",
            "Deep learning basics",
            "Neural network architecture"
        ]
        
        for statement in statements:
            intent = detector.detect(statement)
            # Should default to SEARCH (safe fallback)
            assert intent == Intent.SEARCH_KNOWLEDGE_BASE, f"Failed for: {statement}"
    
    def test_multiword_topics(self, detector):
        """Test multi-word topics default to search."""
        topics = [
            "artificial intelligence",
            "natural language processing",
            "computer vision algorithms"
        ]
        
        for topic in topics:
            intent = detector.detect(topic)
            assert intent == Intent.SEARCH_KNOWLEDGE_BASE, f"Failed for: {topic}"


class TestHeuristics:
    """Tests for heuristic-based detection."""
    
    @pytest.fixture
    def detector(self):
        return IntentDetector()
    
    def test_short_non_question(self, detector):
        """Test short non-questions are classified as chitchat."""
        short_phrases = [
            "yep",
            "nope",
            "hmm",
            "uh huh"
        ]
        
        for phrase in short_phrases:
            intent = detector.detect(phrase)
            # These should be chitchat (short, no question words)
            assert intent == Intent.CHITCHAT, f"Failed for: {phrase}"
    
    def test_question_mark_triggers_search(self, detector):
        """Test that question mark triggers search."""
        intent = detector.detect("Something?")
        assert intent == Intent.SEARCH_KNOWLEDGE_BASE
    
    def test_empty_query(self, detector):
        """Test empty query handling."""
        intent = detector.detect("")
        assert intent == Intent.SEARCH_KNOWLEDGE_BASE  # Safe default
        
        intent = detector.detect("   ")
        assert intent == Intent.SEARCH_KNOWLEDGE_BASE


class TestEdgeCases:
    """Tests for edge cases."""
    
    @pytest.fixture
    def detector(self):
        return IntentDetector()
    
    def test_mixed_case(self, detector):
        """Test mixed case queries."""
        queries = [
            "HeLLo",
            "WHAT IS ML?",
            "ThAnKs"
        ]
        
        intents = [
            detector.detect(q) for q in queries
        ]
        
        assert intents[0] == Intent.GREETING
        assert intents[1] == Intent.SEARCH_KNOWLEDGE_BASE
        assert intents[2] == Intent.CHITCHAT
    
    def test_extra_whitespace(self, detector):
        """Test queries with extra whitespace."""
        intent = detector.detect("  hello  ")
        assert intent == Intent.GREETING
        
        intent = detector.detect("what   is   ml")
        assert intent == Intent.SEARCH_KNOWLEDGE_BASE
    
    def test_special_characters(self, detector):
        """Test queries with special characters."""
        intent = detector.detect("hello!!!")
        assert intent == Intent.GREETING
        
        intent = detector.detect("what is ML???")
        assert intent == Intent.SEARCH_KNOWLEDGE_BASE


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_detect_intent_function(self):
        """Test detect_intent convenience function."""
        intent = detect_intent("hello")
        assert intent == Intent.GREETING
        
        intent = detect_intent("what is machine learning?")
        assert intent == Intent.SEARCH_KNOWLEDGE_BASE
    
    def test_is_conversational(self):
        """Test is_conversational function."""
        assert is_conversational(Intent.GREETING) is True
        assert is_conversational(Intent.CHITCHAT) is True
        assert is_conversational(Intent.GOODBYE) is True
        assert is_conversational(Intent.SEARCH_KNOWLEDGE_BASE) is False
    
    def test_get_simple_response(self):
        """Test get_simple_response function."""
        response = get_simple_response(Intent.GREETING)
        assert "Hello" in response or "help" in response
        
        response = get_simple_response(Intent.CHITCHAT)
        assert "welcome" in response.lower() or "help" in response.lower()
        
        response = get_simple_response(Intent.GOODBYE)
        assert "Goodbye" in response or "bye" in response.lower()
        
        # Unknown intent should have default
        response = get_simple_response(Intent.SEARCH_KNOWLEDGE_BASE)
        assert "help" in response.lower()


class TestRealWorldQueries:
    """Tests with real-world query examples."""
    
    @pytest.fixture
    def detector(self):
        return IntentDetector()
    
    def test_knowledge_base_queries(self, detector):
        """Test realistic knowledge base queries."""
        queries = [
            "How do I train a machine learning model?",
            "What are the best practices for data preprocessing?",
            "Explain the difference between supervised and unsupervised learning",
            "Can you show me examples of deep learning?",
            "Python numpy tutorial",
            "scikit-learn random forest parameters"
        ]
        
        for query in queries:
            intent = detector.detect(query)
            assert intent == Intent.SEARCH_KNOWLEDGE_BASE, f"Failed for: {query}"
    
    def test_conversational_queries(self, detector):
        """Test realistic conversational queries."""
        queries = [
            ("Hi, can you help me?", Intent.GREETING),
            ("Thanks for the information!", Intent.CHITCHAT),
            ("That's great, thanks!", Intent.CHITCHAT),
            ("I'm done for now, bye!", Intent.GOODBYE),
        ]
        
        for query, expected_intent in queries:
            intent = detector.detect(query)
            assert intent == expected_intent, f"Failed for: {query}"

