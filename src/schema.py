"""
Gauntlet — data models for questions, scores, and results.

Designed by SK. Built by Claude.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ── enums ───────────────────────────────────────────────────────────────
class Category(str, Enum):
    INVESTMENT = "investment"
    BIOTECH = "biotech"
    GENERAL = "general"


CATEGORY_LABELS = {
    "investment": "Investment",
    "biotech": "Biotech",
    "general": "General",
}


class Difficulty(int, Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3


# ── question ────────────────────────────────────────────────────────────
class Language(BaseModel):
    input: str = "en"
    expected_output: str = "en"
    bilingual_required: bool = False


class Message(BaseModel):
    role: str
    content: str


class ContextDocument(BaseModel):
    type: str
    content: str


class Prompt(BaseModel):
    system: str
    messages: list[Message]
    context_documents: list[ContextDocument] = []


class Evaluation(BaseModel):
    reference_answer: str
    key_facts: list[str] = []
    disqualifiers: list[str] = []


class Metadata(BaseModel):
    created: str
    tags: list[str] = []
    estimated_tokens: dict[str, int] = {}


class Question(BaseModel):
    id: str
    version: str
    category: Category
    subcategory: str
    difficulty: Difficulty
    language: Language = Language()
    prompt: Prompt
    evaluation: Evaluation
    metadata: Metadata


# ── scores ──────────────────────────────────────────────────────────────
class DimensionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str


class Scores(BaseModel):
    factual_accuracy: DimensionScore
    reasoning_depth: DimensionScore
    practical_usefulness: DimensionScore
    instruction_following: DimensionScore
    communication_quality: DimensionScore
    bilingual_competence: DimensionScore


class JudgeResult(BaseModel):
    question_id: str
    model: str
    scores: Scores
    composite: float
    judge_model: str = "opus"
    judge_version: str = "v1"


# ── results ─────────────────────────────────────────────────────────────
def _load_weights() -> dict[str, float]:
    """Load scoring weights from metrics.toml."""
    import tomllib
    from pathlib import Path
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "metrics.toml"
    if cfg_path.exists():
        with open(cfg_path, "rb") as f:
            return tomllib.load(f).get("weights", {})
    return {
        "factual_accuracy": 0.25,
        "reasoning_depth": 0.20,
        "practical_usefulness": 0.20,
        "instruction_following": 0.15,
        "communication_quality": 0.10,
        "bilingual_competence": 0.10,
    }


WEIGHTS = _load_weights()


def composite_score(scores: Scores) -> float:
    """Weighted sum normalized to 0-100."""
    total = sum(
        getattr(scores, dim).score * weight
        for dim, weight in WEIGHTS.items()
    )
    return round(total / 5 * 100, 1)


class ABResult(BaseModel):
    question_id: str
    model_a: str
    model_b: str
    winner: str  # model directory name or "tie"
    response_time_sec: float


class LeaderboardEntry(BaseModel):
    rank: int
    model: str
    overall: float
    by_category: dict[str, float]
    by_difficulty: dict[str, float]
    by_dimension: dict[str, float]
    ab_win_rate: float | None = None
