"""Filtering auto-generated questions.

Everybody in this field generates synthetic eval sets. Ragas does, AutoRAG does, RAGBuilder
does. What none of them do is throw the bad ones away, and auto-generated ground truth is
mostly bad in four specific, detectable ways:

1. **The question is general knowledge.** "What is a force majeure clause?" can be answered
   without reading anything, so it measures the model rather than the retriever, and it
   scores well for every configuration.
2. **The question refers to something that is not there.** "What does it cover?" made sense
   beside the chunk it was written from and means nothing on its own.
3. **The question is a near-duplicate of another one.** Ten rewordings of the same question
   look like ten measurements and are one, with ten times the false confidence.
4. **The question separates nothing.** If the simplest possible baseline already answers it
   at rank one, every configuration will too, and it contributes nothing but inflation.

Each filter reports what it rejected and why, because a filter that silently discards a third
of an eval set is worse than no filter at all.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from contextgrid.core.evalset import EvalItem, EvalSet
from contextgrid.core.warnings import Severity, WarningCode, WarningLog

#: Pronouns that need something earlier in the sentence to point at.
_DANGLING = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|he|she|his|her|him)\b",
    re.IGNORECASE,
)

#: A noun phrase earlier in the question can be the antecedent: either "the/a/an <word>" or
#: a proper noun. Crude and good enough -- the filter is a first pass in front of a human.
_ARTICLE_PHRASE = re.compile(r"\b(?:the|a|an|this|these)\s+\w+", re.IGNORECASE)
_PROPER_NOUN = re.compile(r"\b[A-Z][a-z]+\b")

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class Rejection:
    """One question that did not make it, and the reason."""

    item: EvalItem
    filter_name: str
    reason: str

    def __str__(self) -> str:
        return f"[{self.filter_name}] {self.item.id}: {self.reason}"


@dataclass(slots=True)
class FilterResult:
    """What survived, what did not, and anything worth knowing about the filtering itself."""

    kept: tuple[EvalItem, ...] = ()
    rejected: tuple[Rejection, ...] = ()
    warnings: WarningLog = field(default_factory=WarningLog)

    @property
    def kept_count(self) -> int:
        return len(self.kept)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    def by_filter(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rejection in self.rejected:
            counts[rejection.filter_name] = counts.get(rejection.filter_name, 0) + 1
        return counts

    def as_evalset(self, original: EvalSet) -> EvalSet:
        """The surviving items, as a new version of the original set."""
        from dataclasses import replace

        return replace(original, items=self.kept, version=original.version + 1)

    def summary(self) -> str:
        if not self.rejected:
            return f"kept all {self.kept_count} questions"
        breakdown = ", ".join(f"{name} {count}" for name, count in sorted(self.by_filter().items()))
        return (
            f"kept {self.kept_count} of {self.kept_count + self.rejected_count} "
            f"questions ({breakdown})"
        )


@runtime_checkable
class Filter(Protocol):
    """Decides which questions are worth keeping."""

    @property
    def name(self) -> str: ...

    def apply(self, items: Sequence[EvalItem]) -> tuple[list[EvalItem], list[Rejection]]: ...


# ---------------------------------------------------------------------------
# the filters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DanglingReferenceFilter:
    """Rejects questions whose pronouns have nothing to point at.

    "What does it cover?" was perfectly clear beside the chunk it was written from. On its
    own it is unanswerable by anything, and it will score zero for every configuration --
    which looks like a hard question and is actually a broken one.
    """

    name: str = "dangling-reference"

    def apply(self, items: Sequence[EvalItem]) -> tuple[list[EvalItem], list[Rejection]]:
        kept: list[EvalItem] = []
        rejected: list[Rejection] = []

        for item in items:
            match = _DANGLING.search(item.question)
            if match is None:
                kept.append(item)
                continue

            if _has_antecedent(item.question[: match.start()]):
                kept.append(item)
                continue

            rejected.append(
                Rejection(
                    item,
                    self.name,
                    f"{match.group(0)!r} has nothing to refer to. The question only made "
                    "sense beside the chunk it was written from",
                )
            )
        return kept, rejected


@dataclass(frozen=True, slots=True)
class DuplicateFilter:
    """Rejects questions too similar to one already kept.

    Similarity is Jaccard overlap on words. Ten rewordings of one question look like ten
    measurements, are one, and give ten times the confidence they have earned.
    """

    threshold: float = 0.8
    name: str = "near-duplicate"

    def apply(self, items: Sequence[EvalItem]) -> tuple[list[EvalItem], list[Rejection]]:
        kept: list[EvalItem] = []
        rejected: list[Rejection] = []
        seen: list[tuple[EvalItem, set[str]]] = []

        for item in items:
            words = _words(item.question)
            duplicate_of = next(
                (
                    other
                    for other, other_words in seen
                    if _jaccard(words, other_words) >= self.threshold
                ),
                None,
            )
            if duplicate_of is not None:
                rejected.append(
                    Rejection(
                        item,
                        self.name,
                        f"nearly the same question as {duplicate_of.id!r}: "
                        f"{duplicate_of.question!r}",
                    )
                )
                continue
            kept.append(item)
            seen.append((item, words))

        return kept, rejected


@dataclass(frozen=True, slots=True)
class UnresolvedEvidenceFilter:
    """Rejects questions whose evidence could not be found in the corpus at all.

    Distinct from a parser losing it. This catches a generator that invented a quote, which
    LLMs do, and which produces a question nothing can ever answer.

    **Only meaningful after the anchors have been resolved against a parse.** Before that,
    every item has a quote and no span, and "unresolved" is indistinguishable from
    "not yet resolved". Running anyway would reject the entire eval set, so the filter
    detects that case and stands down.
    """

    name: str = "unresolved-evidence"

    def apply(self, items: Sequence[EvalItem]) -> tuple[list[EvalItem], list[Rejection]]:
        # `is_resolved`, not `is_answerable`: this filter asks whether a parse located the
        # quote, and an unresolved item is answerable in principle -- that is the whole
        # point of the anchor. Nothing resolved yet means stand down.
        if not any(item.is_resolved for item in items):
            return list(items), []

        kept: list[EvalItem] = []
        rejected: list[Rejection] = []
        for item in items:
            if item.anchors and not item.is_resolved:
                rejected.append(
                    Rejection(
                        item,
                        self.name,
                        "the quoted evidence does not appear in the corpus. The generator "
                        "probably invented it",
                    )
                )
                continue
            kept.append(item)
        return kept, rejected


@dataclass(frozen=True, slots=True)
class ShortQuestionFilter:
    """Rejects questions too short to be specific.

    "Notice?" retrieves by luck. The cut-off is low on purpose: this is a coarse pass before
    a human sees the queue, not a judgement about writing quality.
    """

    min_words: int = 4
    name: str = "too-short"

    def apply(self, items: Sequence[EvalItem]) -> tuple[list[EvalItem], list[Rejection]]:
        kept: list[EvalItem] = []
        rejected: list[Rejection] = []
        for item in items:
            count = len(_WORD.findall(item.question))
            if count < self.min_words:
                rejected.append(
                    Rejection(item, self.name, f"only {count} words; too vague to retrieve on")
                )
                continue
            kept.append(item)
        return kept, rejected


@dataclass(frozen=True, slots=True)
class NonDiscriminatingFilter:
    """Rejects questions the baseline already answers perfectly.

    The filter nothing else has, and the one that matters most for a *comparison*. A question
    every configuration gets right at rank one is not evidence that they are all good, it is
    a constant added to every score. Twenty of them will make a real difference between two
    configurations look like a rounding error.

    Takes the per-question scores from a baseline run, which the grid already produces.
    """

    baseline_scores: dict[str, float] = field(default_factory=dict)
    threshold: float = 1.0
    name: str = "non-discriminating"

    def apply(self, items: Sequence[EvalItem]) -> tuple[list[EvalItem], list[Rejection]]:
        if not self.baseline_scores:
            return list(items), []

        kept: list[EvalItem] = []
        rejected: list[Rejection] = []
        for item in items:
            score = self.baseline_scores.get(item.id)
            if score is not None and score >= self.threshold:
                rejected.append(
                    Rejection(
                        item,
                        self.name,
                        f"the baseline already scores {score:.2f} on this. Every "
                        "configuration will, so it separates nothing and inflates them all",
                    )
                )
                continue
            kept.append(item)
        return kept, rejected


@dataclass(frozen=True, slots=True)
class GeneralKnowledgeFilter:
    """Rejects questions answerable without reading the corpus.

    The test is the obvious one: ask the model the question with no context, and if it gets
    it right, the question measures the model rather than the retriever.

    Needs an LLM. Without one it is a no-op that says so, rather than silently doing nothing
    -- an eval set full of general knowledge scores well for every configuration, and the
    user should know the check did not run.
    """

    answerer: Callable[[str], str] | None = None
    name: str = "general-knowledge"

    def apply(self, items: Sequence[EvalItem]) -> tuple[list[EvalItem], list[Rejection]]:
        if self.answerer is None:
            return list(items), []

        kept: list[EvalItem] = []
        rejected: list[Rejection] = []
        for item in items:
            if item.answer is None:
                kept.append(item)
                continue
            without_context = self.answerer(item.question)
            if _looks_like(without_context, item.answer):
                rejected.append(
                    Rejection(
                        item,
                        self.name,
                        "answerable from general knowledge, so it measures the model rather "
                        "than the retriever",
                    )
                )
                continue
            kept.append(item)
        return kept, rejected


# ---------------------------------------------------------------------------
# chaining
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FilterChain:
    """Runs filters in order, keeping a record of everything dropped."""

    filters: list[Filter] = field(default_factory=list)

    def run(self, evalset: EvalSet) -> FilterResult:
        items = list(evalset)
        rejections: list[Rejection] = []
        log = WarningLog()

        for step in self.filters:
            items, rejected = step.apply(items)
            rejections.extend(rejected)

        result = FilterResult(kept=tuple(items), rejected=tuple(rejections), warnings=log)

        if isinstance_no_answerer(self.filters):
            log.add(
                WarningCode.SMALL_EVAL_SET,
                "the general-knowledge filter had no model to ask, so it did not run. "
                "Questions answerable without reading the corpus will still be in this set, "
                "and they score well for every configuration",
                severity=Severity.CAUTION,
                stage="evalset",
            )

        if (
            result.kept_count
            and result.rejected_count / (result.kept_count + result.rejected_count) > 0.5
        ):
            log.add(
                WarningCode.SMALL_EVAL_SET,
                f"filtering removed {result.rejected_count} of "
                f"{result.kept_count + result.rejected_count} questions. That much rejection "
                "usually means the generator prompt needs work rather than the filters",
                severity=Severity.CAUTION,
                stage="evalset",
            )

        if result.kept_count < 30:
            log.add(
                WarningCode.SMALL_EVAL_SET,
                f"{result.kept_count} questions is a small eval set. Differences below about "
                "0.1 will not be distinguishable from noise on it",
                severity=Severity.CAUTION,
                stage="evalset",
                kept=result.kept_count,
            )

        return result


def default_filters(
    *,
    baseline_scores: dict[str, float] | None = None,
    answerer: Callable[[str], str] | None = None,
) -> FilterChain:
    """The filters worth running on any auto-generated set, cheapest first."""
    return FilterChain(
        [
            ShortQuestionFilter(),
            DanglingReferenceFilter(),
            UnresolvedEvidenceFilter(),
            DuplicateFilter(),
            NonDiscriminatingFilter(baseline_scores=baseline_scores or {}),
            GeneralKnowledgeFilter(answerer=answerer),
        ]
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def isinstance_no_answerer(filters: Sequence[Filter]) -> bool:
    """True when a general-knowledge filter is present but has no model to ask."""
    return any(
        isinstance(step, GeneralKnowledgeFilter) and step.answerer is None for step in filters
    )


def _has_antecedent(prefix: str) -> bool:
    """Whether anything before a pronoun could be what it refers to.

    The first word is ignored when looking for proper nouns, because every question starts
    with a capital letter and counting "What" as an antecedent would let every dangling
    reference through -- which is exactly the bug this filter exists to catch.
    """
    if _ARTICLE_PHRASE.search(prefix):
        return True
    after_first_word = prefix.split(maxsplit=1)
    return len(after_first_word) > 1 and bool(_PROPER_NOUN.search(after_first_word[1]))


def _words(text: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(text)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union)


def _looks_like(candidate: str, expected: str) -> bool:
    """Whether an answer given without context matches the expected one.

    Deliberately loose: this decides whether to *flag* a question for a human, and a false
    positive costs a keystroke while a false negative leaves a useless question in the set.
    """
    return _jaccard(_words(candidate), _words(expected)) >= 0.6
