import re
from typing import Tuple, List

class PromptInjectionDetector:
    """
    Heuristic rule engine to detect adversarial prompt injection
    and jailbreak patterns before sending to upstream LLMs.
    """
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)", re.IGNORECASE),
        re.compile(r"disregard\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts)", re.IGNORECASE),
        re.compile(r"system\s*override", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+in\s+(developer|unrestricted|god|dan)\s+mode", re.IGNORECASE),
        re.compile(r"bypass\s+(all\s+)?(filters|safety|guardrails)", re.IGNORECASE),
        re.compile(r"output\s+the\s+(system\s+prompt|initial\s+prompt|internal\s+instructions)", re.IGNORECASE),
        re.compile(r"print\s+your\s+instructions\s+verbatim", re.IGNORECASE),
    ]

    @classmethod
    def evaluate(cls, text: str) -> Tuple[bool, List[str]]:
        """
        Returns (is_injection_detected, matched_rules)
        """
        matches = []
        for pattern in cls.INJECTION_PATTERNS:
            found = pattern.findall(text)
            if found:
                matches.append(pattern.pattern)

        return len(matches) > 0, matches
