import re
from typing import Tuple, List, Dict

class PIIFilter:
    """
    Production Guardrail for Sensitive Data & Secret Token Sanitization.
    Combines RFC-compliant regex patterns with the Luhn Algorithm
    to validate credit card checksums and prevent false positives.
    """
    SECRET_PATTERNS: Dict[str, re.Pattern] = {
        "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        "PHONE_NUMBER": re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
        "OPENAI_KEY": re.compile(r'\bsk-[A-Za-z0-9-_]{20,}\b'),
        "AWS_KEY": re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
        "BEARER_TOKEN": re.compile(r'\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b', re.IGNORECASE),
        "SSN": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    }

    # Matches candidate 13 to 19 digit sequences separated by spaces or hyphens
    CC_CANDIDATE = re.compile(r'\b(?:\d[ -]*?){13,19}\b')

    @classmethod
    def luhn_checksum_valid(cls, card_number: str) -> bool:
        """
        Validates whether a number string satisfies the Luhn algorithm (Mod 10).
        Standard for Visa, MasterCard, Amex, and Discover.
        """
        digits = [int(c) for c in card_number if c.isdigit()]
        if len(digits) < 13 or len(digits) > 19:
            return False
            
        checksum = 0
        reverse_digits = digits[::-1]
        for i, digit in enumerate(reverse_digits):
            if i % 2 == 1:
                doubled = digit * 2
                checksum += doubled - 9 if doubled > 9 else doubled
            else:
                checksum += digit
        return checksum % 10 == 0

    @classmethod
    def sanitize(cls, text: str) -> Tuple[str, List[str]]:
        """
        Scans and redacts PII/secrets, returning (sanitized_text, detected_types).
        """
        detected: List[str] = []
        sanitized_text = text

        # 1. First check candidate credit cards with Luhn verification
        for candidate in cls.CC_CANDIDATE.findall(sanitized_text):
            digits_only = re.sub(r'\D', '', candidate)
            if cls.luhn_checksum_valid(digits_only):
                if "CREDIT_CARD" not in detected:
                    detected.append("CREDIT_CARD")
                sanitized_text = sanitized_text.replace(candidate.strip(), "[REDACTED:CREDIT_CARD]")

        # 2. Check secret tokens and standard PII
        for pii_type, pattern in cls.SECRET_PATTERNS.items():
            matches = pattern.findall(sanitized_text)
            if matches:
                detected.append(pii_type)
                sanitized_text = pattern.sub(f"[REDACTED:{pii_type}]", sanitized_text)

        return sanitized_text, detected
