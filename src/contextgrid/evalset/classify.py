"""Tagging questions by type.

The single most useful thing an eval set can carry beyond the questions themselves. Overall
scores hide the interesting result: semantic chunking can win by three points and still lose
badly on every question about a table, and a leaderboard that reports one number per
configuration will never show it.

Classification is heuristic by default -- keyword patterns, no model, no cost. It is wrong
sometimes, and the review queue lets a human fix it in a keystroke. An LLM classifier can be
supplied where accuracy matters more than free.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from contextgrid.core.evalset import EvalItem, EvalSet, QuestionType

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        QuestionType.SUMMARISATION,
        re.compile(
            r"\b(summar\w*|overview|outline|what are the (main|key)|describe the|in general)\b",
            re.IGNORECASE,
        ),
    ),
    (
        QuestionType.COMPARATIVE,
        re.compile(
            r"\b(compare|comparison|difference between|versus|vs\.?"
            r"|which is (more|less|better|cheaper)"
            r"|higher than|lower than|more than|less than)\b",
            re.IGNORECASE,
        ),
    ),
    (
        QuestionType.TABULAR,
        re.compile(
            r"\b(fee|price|cost|rate|tier|column|row|table|per month|monthly|annual|"
            r"schedule of)\b",
            re.IGNORECASE,
        ),
    ),
    (
        QuestionType.NUMERIC,
        re.compile(
            r"\b(how (much|many|long|often)|what percentage|how far|number of|"
            r"within \d+|\d+\s*(days?|months?|years?|%))\b",
            re.IGNORECASE,
        ),
    ),
    (
        QuestionType.MULTI_HOP,
        re.compile(
            r"\b(and (also|then)|after .+ what|both .+ and|in each|across (all|both)|"
            r"for every)\b",
            re.IGNORECASE,
        ),
    ),
)


def classify_question(question: str, *, has_evidence: bool = True) -> str:
    """Guess a question's type from its wording.

    Checked most specific first: "which is cheaper, Standard or Premium?" is comparative
    *and* tabular *and* numeric, and comparative is the most informative of the three.
    """
    if not has_evidence:
        return QuestionType.UNANSWERABLE
    for label, pattern in _PATTERNS:
        if pattern.search(question):
            return label
    return QuestionType.FACTOID


@dataclass(frozen=True, slots=True)
class Classifier:
    """Labels questions, by heuristic or by model."""

    model: Callable[[str], str] | None = None
    overwrite: bool = False

    def label(self, item: EvalItem) -> EvalItem:
        """Return the item with a `qtype`, leaving an existing one alone unless told not to.

        A label a human set in the review queue outranks anything guessed here.
        """
        if item.qtype and not self.overwrite:
            return item

        if self.model is not None:
            guess = self.model(item.question).strip().lower()
            if guess in QuestionType.ALL:
                return replace(item, qtype=guess)

        return replace(item, qtype=classify_question(item.question, has_evidence=item.has_evidence))

    def label_all(self, items: Sequence[EvalItem]) -> list[EvalItem]:
        return [self.label(item) for item in items]

    def label_set(self, evalset: EvalSet) -> EvalSet:
        return evalset.with_items(tuple(self.label_all(list(evalset))))


def type_distribution(evalset: EvalSet) -> dict[str, int]:
    """How many questions of each type. The input to per-type metric slicing."""
    counts: dict[str, int] = {}
    for item in evalset:
        label = item.qtype or "unlabelled"
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))
