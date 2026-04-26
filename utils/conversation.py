"""
Conversation state management for the TalentScout Hiring Assistant.

A simple finite-state machine drives the screening flow:
    NAME → EMAIL → PHONE → EXPERIENCE → POSITION → LOCATION
        → TECH_STACK → TECHNICAL_QA → WRAP_UP → ENDED

Why a state machine instead of pure LLM-driven flow?
    1. Reliability — the model can't accidentally skip required fields.
    2. Cost & latency — no LLM call needed for simple validation.
    3. Auditability — easy to test and debug each stage in isolation.
    4. Privacy — sensitive PII is never sent to the LLM unnecessarily.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional
from datetime import datetime, timezone


class Stage(str, Enum):
    """All conversation stages, in order."""
    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    EXPERIENCE = "experience"
    POSITION = "position"
    LOCATION = "location"
    TECH_STACK = "tech_stack"
    TECHNICAL_QA = "technical_qa"
    WRAP_UP = "wrap_up"
    ENDED = "ended"


# Order is meaningful — used by ConversationManager.advance()
STAGE_ORDER = [
    Stage.NAME,
    Stage.EMAIL,
    Stage.PHONE,
    Stage.EXPERIENCE,
    Stage.POSITION,
    Stage.LOCATION,
    Stage.TECH_STACK,
    Stage.TECHNICAL_QA,
    Stage.WRAP_UP,
    Stage.ENDED,
]


@dataclass
class Candidate:
    """Container for collected candidate information."""
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    years_experience: Optional[float] = None
    desired_position: Optional[str] = None
    location: Optional[str] = None
    tech_stack: List[str] = field(default_factory=list)
    technical_answers: List[dict] = field(default_factory=list)
    additional_notes: Optional[str] = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


class ConversationManager:
    """Holds candidate state and tracks current screening stage."""

    def __init__(self):
        self.candidate = Candidate()
        self.stage: Stage = Stage.NAME
        self.questions: List[str] = []
        self.current_q_index: int = 0  # which question to ask NEXT

    # ------------------------------------------------------------------ #
    # Stage transitions                                                   #
    # ------------------------------------------------------------------ #
    def advance(self) -> Stage:
        """Move to the next stage. Returns the new stage."""
        try:
            idx = STAGE_ORDER.index(self.stage)
            if idx + 1 < len(STAGE_ORDER):
                self.stage = STAGE_ORDER[idx + 1]
        except ValueError:
            pass
        return self.stage

    def progress(self) -> float:
        """Return progress as a float in [0.0, 1.0] for the progress bar."""
        try:
            idx = STAGE_ORDER.index(self.stage)
            # Don't count ENDED as a stage to "complete past"
            return min(1.0, idx / (len(STAGE_ORDER) - 1))
        except ValueError:
            return 0.0

    # ------------------------------------------------------------------ #
    # Technical Q&A helpers                                               #
    # ------------------------------------------------------------------ #
    def next_question(self) -> Optional[str]:
        """Get the next unanswered technical question (or None when done)."""
        if self.current_q_index < len(self.questions):
            q = self.questions[self.current_q_index]
            return q
        return None

    def record_answer(self, answer: str) -> None:
        """Store the answer to the current question and advance the pointer."""
        if self.current_q_index < len(self.questions):
            self.candidate.technical_answers.append(
                {
                    "question": self.questions[self.current_q_index],
                    "answer": answer.strip(),
                }
            )
            self.current_q_index += 1

    # ------------------------------------------------------------------ #
    # Stage descriptors (for fallback / UI)                               #
    # ------------------------------------------------------------------ #
    @property
    def expected_input(self) -> str:
        """Human-readable description of what we're currently asking for."""
        return {
            Stage.NAME: "your full name",
            Stage.EMAIL: "your email address",
            Stage.PHONE: "your phone number",
            Stage.EXPERIENCE: "your years of professional experience",
            Stage.POSITION: "the role(s) you're interested in",
            Stage.LOCATION: "your current city/country",
            Stage.TECH_STACK: "your tech stack (languages, frameworks, tools)",
            Stage.TECHNICAL_QA: "your answer to the technical question",
            Stage.WRAP_UP: "any final notes (or 'no' to finish)",
            Stage.ENDED: "(session ended)",
        }.get(self.stage, "your response")
