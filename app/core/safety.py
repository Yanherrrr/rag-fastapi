"""
Query Safety Checker

Detects and handles sensitive/inappropriate queries:
- PII (Personal Identifiable Information)
- Medical advice requests
- Legal advice requests
- Financial advice requests

Uses fast pattern matching to avoid unnecessary LLM calls and protect users.
"""

import re
import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum


logger = logging.getLogger(__name__)


# ============================================================================
# Safety Categories
# ============================================================================

class SafetyCategory(str, Enum):
    """Categories of safety violations"""
    SAFE = "safe"
    PII_DETECTED = "pii_detected"
    MEDICAL_ADVICE = "medical_advice"
    LEGAL_ADVICE = "legal_advice"
    FINANCIAL_ADVICE = "financial_advice"


# ============================================================================
# Pattern Definitions
# ============================================================================

# PII Patterns (regex-based detection)
PII_PATTERNS = {
    'ssn': {
        'pattern': r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        'description': 'Social Security Number',
        'examples': ['123-45-6789', '123 45 6789', '123456789']
    },
    'credit_card': {
        'pattern': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        'description': 'Credit Card Number',
        'examples': ['1234-5678-9012-3456', '1234 5678 9012 3456']
    },
    'email': {
        'pattern': r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b',
        'description': 'Email Address',
        'examples': ['user@example.com']
    },
    'phone': {
        'pattern': r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'description': 'Phone Number',
        'examples': ['(555) 123-4567', '555-123-4567', '+1-555-123-4567']
    }
}

# Medical Keywords (case-insensitive)
MEDICAL_KEYWORDS = [
    'diagnose', 'diagnosis', 'diagnosed',
    'symptom', 'symptoms',
    'treatment', 'treat',
    'prescription', 'prescribe',
    'medication', 'medicine', 'drug',
    'disease', 'illness', 'condition',
    'cure', 'cured',
    'medical advice',
    'should i take',
    'is it safe to take',
    'doctor recommend',
    'health issue',
    'am i sick',
    'what does it mean if'
]

# Legal Keywords (case-insensitive)
LEGAL_KEYWORDS = [
    'legal advice',
    'lawyer', 'attorney', 'counsel',
    'sue', 'lawsuit', 'litigation',
    'contract', 'agreement',
    'liability', 'liable',
    'rights', 'legal rights',
    'court', 'judge',
    'illegal', 'legality',
    'can i sue',
    'should i sue',
    'legal action',
    'violation of law',
    'legal matter'
]

# Financial Advice Keywords (case-insensitive)
FINANCIAL_KEYWORDS = [
    'stock advice',
    'investment advice',
    'financial advice',
    'should i invest',
    'should i buy',
    'buy stocks',
    'sell stocks',
    'trading advice',
    'invest in',
    'portfolio advice',
    'financial planning',
    'should i purchase',
    'good investment',
    'worth investing'
]


# ============================================================================
# Result Classes
# ============================================================================

@dataclass
class SafetyCheckResult:
    """Result of a safety check"""
    is_safe: bool
    category: SafetyCategory
    violations: List[str]
    sanitized_query: Optional[str]
    refusal_message: Optional[str]
    check_time_ms: float
    
    def __str__(self):
        if self.is_safe:
            return f"Safe (checked in {self.check_time_ms:.2f}ms)"
        return f"Unsafe: {self.category} - {', '.join(self.violations)}"


# ============================================================================
# Safety Checker
# ============================================================================

class SafetyChecker:
    """
    Check queries for safety violations.
    
    Uses pattern matching for fast detection of:
    - PII (strips and warns)
    - Medical/Legal/Financial advice (refuses)
    """
    
    def __init__(self):
        """Initialize safety checker with compiled patterns"""
        # Compile PII regex patterns for efficiency
        self.pii_patterns = {
            name: re.compile(info['pattern'], re.IGNORECASE)
            for name, info in PII_PATTERNS.items()
        }
        
        # Convert keywords to lowercase for case-insensitive matching
        self.medical_keywords = [k.lower() for k in MEDICAL_KEYWORDS]
        self.legal_keywords = [k.lower() for k in LEGAL_KEYWORDS]
        self.financial_keywords = [k.lower() for k in FINANCIAL_KEYWORDS]
        
        logger.info("SafetyChecker initialized with patterns")
    
    def check_query(self, query: str) -> SafetyCheckResult:
        """
        Check query for safety violations.
        
        Args:
            query: User query string
            
        Returns:
            SafetyCheckResult with safety status and details
        """
        start_time = time.time()
        violations = []
        
        # Check for PII first (we can sanitize this)
        pii_detected, pii_types = self._detect_pii(query)
        if pii_detected:
            sanitized_query = self._sanitize_pii(query)
            violations.extend(pii_types)
            
            check_time = (time.time() - start_time) * 1000
            return SafetyCheckResult(
                is_safe=False,
                category=SafetyCategory.PII_DETECTED,
                violations=violations,
                sanitized_query=sanitized_query,
                refusal_message=self._get_pii_warning(pii_types),
                check_time_ms=check_time
            )
        
        # Check for legal advice requests (check first as it's more specific)
        if self._contains_keywords(query, self.legal_keywords):
            check_time = (time.time() - start_time) * 1000
            return SafetyCheckResult(
                is_safe=False,
                category=SafetyCategory.LEGAL_ADVICE,
                violations=['legal_advice_request'],
                sanitized_query=None,
                refusal_message=self._get_legal_disclaimer(),
                check_time_ms=check_time
            )
        
        # Check for financial advice requests
        if self._contains_keywords(query, self.financial_keywords):
            check_time = (time.time() - start_time) * 1000
            return SafetyCheckResult(
                is_safe=False,
                category=SafetyCategory.FINANCIAL_ADVICE,
                violations=['financial_advice_request'],
                sanitized_query=None,
                refusal_message=self._get_financial_disclaimer(),
                check_time_ms=check_time
            )
        
        # Check for medical advice requests
        if self._contains_keywords(query, self.medical_keywords):
            check_time = (time.time() - start_time) * 1000
            return SafetyCheckResult(
                is_safe=False,
                category=SafetyCategory.MEDICAL_ADVICE,
                violations=['medical_advice_request'],
                sanitized_query=None,
                refusal_message=self._get_medical_disclaimer(),
                check_time_ms=check_time
            )
        
        # Query is safe
        check_time = (time.time() - start_time) * 1000
        return SafetyCheckResult(
            is_safe=True,
            category=SafetyCategory.SAFE,
            violations=[],
            sanitized_query=None,
            refusal_message=None,
            check_time_ms=check_time
        )
    
    def _detect_pii(self, query: str) -> Tuple[bool, List[str]]:
        """Detect PII in query"""
        detected_types = []
        
        for pii_type, pattern in self.pii_patterns.items():
            if pattern.search(query):
                detected_types.append(pii_type)
        
        return len(detected_types) > 0, detected_types
    
    def _sanitize_pii(self, query: str) -> str:
        """Remove PII from query"""
        sanitized = query
        
        for pii_type, pattern in self.pii_patterns.items():
            replacement = f"[{PII_PATTERNS[pii_type]['description']} REMOVED]"
            sanitized = pattern.sub(replacement, sanitized)
        
        return sanitized
    
    def _contains_keywords(self, query: str, keywords: List[str]) -> bool:
        """Check if query contains any of the keywords"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in keywords)
    
    def _get_pii_warning(self, pii_types: List[str]) -> str:
        """Generate PII warning message"""
        types_str = ', '.join([PII_PATTERNS[t]['description'] for t in pii_types])
        return (
            f"⚠️ **Privacy Warning**: I detected potential personal information in your query "
            f"({types_str}). For your safety, I've removed it. Please avoid sharing sensitive "
            f"data like Social Security Numbers, credit card numbers, email addresses, or phone numbers."
        )
    
    def _get_medical_disclaimer(self) -> str:
        """Generate medical advice disclaimer"""
        return (
            "⚠️ **Medical Disclaimer**: I cannot provide medical advice, diagnosis, or treatment "
            "recommendations. The information I provide is for educational purposes only and should "
            "not be used as a substitute for professional medical advice. Please consult a qualified "
            "healthcare professional for any medical concerns."
        )
    
    def _get_legal_disclaimer(self) -> str:
        """Generate legal advice disclaimer"""
        return (
            "⚠️ **Legal Disclaimer**: I cannot provide legal advice, counsel, or representation. "
            "The information I provide is for general informational purposes only and does not "
            "constitute legal advice. For legal matters, please consult a licensed attorney in "
            "your jurisdiction."
        )
    
    def _get_financial_disclaimer(self) -> str:
        """Generate financial advice disclaimer"""
        return (
            "⚠️ **Financial Disclaimer**: I cannot provide financial, investment, or trading advice. "
            "The information I provide is for educational purposes only and should not be considered "
            "as financial guidance. Please consult a qualified financial advisor for investment decisions."
        )


# ============================================================================
# Singleton Instance & Convenience Functions
# ============================================================================

_safety_checker: Optional[SafetyChecker] = None


def get_safety_checker() -> SafetyChecker:
    """Get or create singleton SafetyChecker instance"""
    global _safety_checker
    if _safety_checker is None:
        _safety_checker = SafetyChecker()
    return _safety_checker


def check_query_safety(query: str) -> SafetyCheckResult:
    """
    Convenience function to check query safety.
    
    Args:
        query: User query string
        
    Returns:
        SafetyCheckResult
    """
    checker = get_safety_checker()
    return checker.check_query(query)


def is_query_safe(query: str) -> bool:
    """
    Quick check if query is safe.
    
    Args:
        query: User query string
        
    Returns:
        True if safe, False otherwise
    """
    result = check_query_safety(query)
    return result.is_safe

