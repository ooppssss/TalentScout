"""
Input validators for candidate-supplied data.

Each validator returns a tuple: (is_valid, cleaned_value, error_message).
Deterministic validation (vs. LLM-based) keeps the screening fast, cheap,
and auditable — and avoids the model "interpreting" obviously invalid input
as valid.
"""

import re
from typing import Tuple


# ---------------------------------------------------------------------- #
# Exit / conversation-ending keywords                                    #
# ---------------------------------------------------------------------- #
EXIT_KEYWORDS = {
    "exit", "quit", "bye", "goodbye", "stop", "end",
    "cancel", "abort", "terminate", "leave",
    "see you", "see ya", "later",
}


def is_exit_keyword(text: str) -> bool:
    """
    True if the user's message indicates they want to end the conversation.

    We match on whole-word equality of the trimmed lowercase input, plus a
    few common multi-word phrases. We deliberately avoid substring matching
    (e.g. "exit" inside "I'd like to exit my current role" should NOT trigger).
    """
    cleaned = text.strip().lower().rstrip(".!?")
    if not cleaned:
        return False
    if cleaned in EXIT_KEYWORDS:
        return True
    # Allow phrases like "exit please" or "i want to quit"
    words = cleaned.split()
    if len(words) <= 4 and any(w in EXIT_KEYWORDS for w in words):
        return True
    return False


# ---------------------------------------------------------------------- #
# Name                                                                    #
# ---------------------------------------------------------------------- #
def validate_name(text: str) -> Tuple[bool, str, str]:
    """
    Names: 2-80 chars, letters/spaces/hyphens/apostrophes/periods only.
    Require at least 2 'words' to discourage single-word inputs like 'Hi'.
    """
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return False, "", "Name cannot be empty."
    if len(cleaned) < 2 or len(cleaned) > 80:
        return False, "", "Name must be 2-80 characters."
    if not re.match(r"^[A-Za-zÀ-ÿ\u0900-\u097F\s'\-\.]+$", cleaned):
        return False, "", "Name contains invalid characters."
    if len(cleaned.split()) < 2:
        return False, "", "Please provide your **full name** (first and last)."
    return True, cleaned.title(), ""


# ---------------------------------------------------------------------- #
# Email                                                                   #
# ---------------------------------------------------------------------- #
EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)


def validate_email(text: str) -> Tuple[bool, str, str]:
    cleaned = text.strip().lower()
    if not cleaned:
        return False, "", "Email cannot be empty."
    if not EMAIL_RE.match(cleaned):
        return False, "", "That doesn't look like a valid email address."
    return True, cleaned, ""


# ---------------------------------------------------------------------- #
# Phone                                                                   #
# ---------------------------------------------------------------------- #
def validate_phone(text: str) -> Tuple[bool, str, str]:
    """
    Phone numbers: keep digits, +, spaces, dashes, parens. After stripping
    formatting chars we expect 7-15 digits (per E.164 max).
    """
    cleaned = text.strip()
    if not cleaned:
        return False, "", "Phone number cannot be empty."

    digits = re.sub(r"[^\d]", "", cleaned)
    if len(digits) < 7 or len(digits) > 15:
        return False, "", (
            "Phone number should contain 7-15 digits. "
            "Include the country code if you're outside India."
        )
    if not re.match(r"^[\+\d\s\-\(\)\.]+$", cleaned):
        return False, "", "Phone number contains invalid characters."

    return True, cleaned, ""


# ---------------------------------------------------------------------- #
# Years of experience                                                     #
# ---------------------------------------------------------------------- #
def validate_experience(text: str) -> Tuple[bool, float, str]:
    """
    Accept a number 0-60. Tolerates inputs like '3 years', '2.5', 'five'.
    """
    cleaned = text.strip().lower().replace("years", "").replace("year", "").strip()

    # Try direct float parse
    try:
        years = float(cleaned)
    except ValueError:
        # Try word-to-number for small values (common case: "fresher", "two")
        word_map = {
            "zero": 0, "fresher": 0, "fresh": 0, "none": 0, "no": 0,
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        for word, value in word_map.items():
            if word in cleaned:
                years = float(value)
                break
        else:
            return False, 0.0, "Please enter a number (e.g., *3* or *5.5*)."

    if years < 0 or years > 60:
        return False, 0.0, "Years of experience must be between 0 and 60."
    return True, round(years, 1), ""
