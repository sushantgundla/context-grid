"""Generation, and whether retrieval gains survived to the answer."""

from __future__ import annotations

from contextgrid.core.registry import Registry
from contextgrid.generate.answer import (
    DEFAULT_PROMPT,
    Answer,
    AnswerScore,
    ExtractiveGenerator,
    GenerationReport,
    Generator,
    LLMGenerator,
    lift,
    score_answer,
)
from contextgrid.generate.judge import METRICS as GENERATION_METRICS
from contextgrid.generate.judge import (
    GenerationJudge,
    JudgedAnswer,
    JudgeError,
    available_generation_metrics,
)

GENERATORS: Registry[Generator] = Registry(family="generator")

GENERATORS.register(
    "extractive",
    shorthand="sentences",
    doc="Return the top passage verbatim. The ceiling retrieval alone can reach.",
)(ExtractiveGenerator)

# `JudgedAnswer` is deliberately not `AnswerScore`: that one records whether an extractive
# answer matched its reference, this one records what a judge model thought of a generated one.
__all__ = [
    "DEFAULT_PROMPT",
    "GENERATION_METRICS",
    "GENERATORS",
    "Answer",
    "AnswerScore",
    "ExtractiveGenerator",
    "GenerationJudge",
    "GenerationReport",
    "Generator",
    "JudgeError",
    "JudgedAnswer",
    "LLMGenerator",
    "available_generation_metrics",
    "lift",
    "score_answer",
]
