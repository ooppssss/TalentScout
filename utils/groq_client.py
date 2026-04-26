"""
Groq API client wrapper for TalentScout.

Uses Groq's free-tier OpenAI-compatible chat-completions endpoint.
Get a free API key at https://console.groq.com/keys
"""

import json
import re
from typing import List

from groq import Groq

from utils.prompts import (
    SYSTEM_PROMPT,
    TECH_QUESTION_GENERATION_PROMPT,
    FALLBACK_PROMPT,
)


class GroqClient:
    """Thin wrapper around the Groq SDK with task-specific helpers."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        if not api_key:
            raise ValueError("Groq API key is required.")
        self.api_key = api_key
        self.model = model
        self._client = Groq(api_key=api_key)

    # ------------------------------------------------------------------ #
    # Core chat completion                                                #
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: List[dict],
        temperature: float = 0.4,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ) -> str:
        """Run a chat completion, optionally injecting a system prompt."""
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------ #
    # Task-specific helpers                                               #
    # ------------------------------------------------------------------ #
    def generate_technical_questions(
        self,
        tech_stack: List[str],
        years_experience: float,
        position: str,
        num_questions: int = 4,
    ) -> List[str]:
        """
        Generate 3-5 technical questions tailored to the candidate's stack
        and experience level. Returns a list of question strings.

        Robustly parses the JSON output even if the model wraps it in code
        fences or adds preamble — Groq's free models occasionally do this
        despite explicit instructions.
        """
        # Pick number of questions based on stack breadth (3-5 range)
        num_questions = max(3, min(5, len(tech_stack) if len(tech_stack) <= 5 else 4))

        prompt = TECH_QUESTION_GENERATION_PROMPT.format(
            num_questions=num_questions,
            tech_stack=", ".join(tech_stack),
            years_experience=years_experience,
            position=position or "Software Engineer",
        )

        raw = self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,  # some creativity but mostly deterministic
            max_tokens=800,
            system_prompt=(
                "You are a precise technical interviewer. You output only "
                "valid JSON arrays — no commentary, no markdown."
            ),
        )

        questions = self._parse_json_array(raw)
        if not questions:
            # Fallback: split the response by numbered lines
            questions = self._fallback_question_parse(raw)

        # Defensive: ensure 3-5 questions
        questions = [q.strip() for q in questions if q and q.strip()]
        if len(questions) < 3:
            raise ValueError(
                f"Model returned only {len(questions)} usable questions. "
                "Please retry."
            )
        return questions[:5]

    def fallback_redirect(
        self, user_input: str, current_stage: str, expected: str
    ) -> str:
        """Generate a polite redirect when the user goes off-topic."""
        prompt = FALLBACK_PROMPT.format(
            user_input=user_input,
            current_stage=current_stage,
            expected=expected,
        )
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=120,
            system_prompt=SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------ #
    # Parsing helpers                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_json_array(raw: str) -> List[str]:
        """Try to parse a JSON array, tolerating code fences and preamble."""
        # Strip common code-fence wrappers
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # Find the first `[` and last `]` and slice
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            arr = json.loads(cleaned[start : end + 1])
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return arr
        except json.JSONDecodeError:
            pass
        return []

    @staticmethod
    def _fallback_question_parse(raw: str) -> List[str]:
        """Last-ditch: extract numbered-list items from raw text."""
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        questions = []
        for ln in lines:
            # Match lines starting with a number, letter, or bullet
            m = re.match(r"^\s*(?:\d+[\.\)]|\-|\*)\s*(.+)", ln)
            if m:
                q = m.group(1).strip().strip("\"'")
                if q.endswith("?") or len(q) > 20:
                    questions.append(q)
        return questions
