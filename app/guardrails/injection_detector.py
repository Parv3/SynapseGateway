import re
from typing import Tuple, List, Dict

class PromptInjectionDetector:
    """
    Weighted Risk Heuristic Engine for Prompt Injections & Jailbreaks.
    Computes a risk score based on rule severity and pattern frequency.
    """
    WEIGHTED_PATTERNS: Dict[str, Tuple[re.Pattern, float]] = {
        "INSTRUCTION_OVERRIDE": (
            re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)", re.IGNORECASE),
            1.0
        ),
        "DIRECT_DISREGARD": (
            re.compile(r"disregard\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts)", re.IGNORECASE),
            1.0
        ),
        "SYSTEM_OVERRIDE": (
            re.compile(r"system\s*override", re.IGNORECASE),
            0.9
        ),
        "JAILBREAK_MODE": (
            re.compile(r"you\s+are\s+now\s+in\s+(developer|unrestricted|god|dan|sudo)\s+mode", re.IGNORECASE),
            0.95
        ),
        "FILTER_BYPASS": (
            re.compile(r"bypass\s+(all\s+)?(filters|safety|guardrails)", re.IGNORECASE),
            0.85
        ),
        "LEAK_SYSTEM_PROMPT": (
            re.compile(r"output\s+the\s+(system\s+prompt|initial\s+prompt|internal\s+instructions)", re.IGNORECASE),
            0.9
        ),
        "VERBATIM_INSTRUCTION": (
            re.compile(r"print\s+your\s+instructions\s+verbatim", re.IGNORECASE),
            0.85
        ),
    }

    RISK_THRESHOLD = 0.80

    @classmethod
    def evaluate(cls, text: str) -> Tuple[bool, List[str], float]:
        """
        Evaluates input text against weighted security heuristics.
        Returns: (is_injection_detected, matched_rules, max_risk_score)
        """
        matches = []
        max_score = 0.0

        for rule_name, (pattern, weight) in cls.WEIGHTED_PATTERNS.items():
            if pattern.search(text):
                matches.append(rule_name)
                if weight > max_score:
                    max_score = weight

        is_flagged = max_score >= cls.RISK_THRESHOLD
        return is_flagged, matches, max_score
