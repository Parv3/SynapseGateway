import re
from typing import Tuple, List, Dict

class PIIFilter:
    """
    Detects and masks Personally Identifiable Information (PII)
    and sensitive developer secrets (API keys, tokens).
    """
    PATTERNS: Dict[str, re.Pattern] = {
        "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        "CREDIT_CARD": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
        "PHONE_NUMBER": re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
        "OPENAI_KEY": re.compile(r'\bsk-[A-Za-z0-9-_]{20,}\b'),
        "AWS_KEY": re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
        "BEARER_TOKEN": re.compile(r'\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b', re.IGNORECASE),
        "SSN": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    }

    @classmethod
    def sanitize(cls, text: str) -> Tuple[str, List[str]]:
        """
        Scans text, replaces sensitive entities with [REDACTED:<TYPE>],
        and returns the sanitized text along with a list of detected PII types.
        """
        detected: List[str] = []
        sanitized_text = text

        for pii_type, pattern in cls.PATTERNS.items():
            matches = pattern.findall(sanitized_text)
            if matches:
                detected.append(pii_type)
                sanitized_text = pattern.sub(f"[REDACTED:{pii_type}]", sanitized_text)

        return sanitized_text, detected
