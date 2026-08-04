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

GENERATORS: Registry[Generator] = Registry(family="generator")

GENERATORS.register(
    "extractive",
    shorthand="sentences",
    doc="Return the top passage verbatim. The ceiling retrieval alone can reach.",
)(ExtractiveGenerator)

__all__ = [
    "DEFAULT_PROMPT",
    "GENERATORS",
    "Answer",
    "AnswerScore",
    "ExtractiveGenerator",
    "GenerationReport",
    "Generator",
    "LLMGenerator",
    "lift",
    "score_answer",
]
