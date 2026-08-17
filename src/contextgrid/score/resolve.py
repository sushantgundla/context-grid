"""Resolving gold spans to whatever chunks a configuration happened to produce.

This is the piece that makes the whole tool valid.

Ground truth in retrieval evaluation is normally stored as a chunk ID. That works right up
until you change the chunker -- and comparing chunkers is the entire point here. A gold chunk
ID recorded under a 512-token recursive splitter means nothing under a semantic splitter,
because the second one produced different chunks. Comparisons built that way are invalid and
nothing warns you.

So gold is stored as character spans in the source document, and resolved to chunks at scoring
time. The eval set is written once and stays correct across every configuration it scores.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from contextgrid.core.errors import ResolutionError
from contextgrid.core.types import (
    Chunk,
    EvalItem,
    EvalSet,
    GoldSpan,
    Qrels,
    RelevanceLabel,
    Span,
    coverage_fraction,
    intersection_length,
    merge_spans,
    spans_of,
    total_length,
)
from contextgrid.core.warnings import Severity, WarningCode, WarningLog


class ResolutionPolicy(str, Enum):
    """How to decide that a chunk counts as relevant to a gold span.

    COVERAGE     the chunk holds at least `threshold` of the gold span's characters.
    IOU          intersection over union of chunk and gold is at least `threshold`.
    CONTAINMENT  the chunk holds the gold span entirely. `threshold` is ignored.
    """

    COVERAGE = "coverage"
    IOU = "iou"
    CONTAINMENT = "containment"


@dataclass(frozen=True, slots=True)
class GoldResolution:
    """What happened to one gold span under one chunk set.

    Kept per gold span rather than aggregated, because "which evidence is unreachable under
    this chunker" is a diagnosis the user needs, not a number.
    """

    gold: GoldSpan
    chunk_ids: tuple[str, ...]
    best_score: float
    union_coverage: float

    @property
    def is_reachable(self) -> bool:
        """True when at least one chunk satisfies the policy on its own."""
        return bool(self.chunk_ids)

    @property
    def is_split(self) -> bool:
        """True when no single chunk qualifies but the chunks together hold the evidence.

        A real and under-reported situation: the gold sentence straddles a chunk boundary,
        every individual chunk falls below the threshold, and a per-chunk-only scorer calls
        it a miss even though retrieving both chunks would ground the answer perfectly.
        """
        return not self.chunk_ids and self.union_coverage > 0.0


@dataclass(slots=True)
class Resolution:
    """The resolved relevance judgements for one question, with diagnostics."""

    item_id: str
    labels: tuple[RelevanceLabel, ...] = ()
    per_gold: tuple[GoldResolution, ...] = ()
    warnings: WarningLog = field(default_factory=WarningLog)

    @property
    def relevant_chunk_ids(self) -> tuple[str, ...]:
        return tuple(label.chunk_id for label in self.labels)

    @property
    def has_unreachable_gold(self) -> bool:
        return any(not g.is_reachable and not g.is_split for g in self.per_gold)

    def as_qrel(self) -> dict[str, int]:
        """This question's judgements in the `{chunk_id: grade}` shape ranx expects."""
        return {label.chunk_id: label.grade for label in self.labels}


@dataclass(frozen=True, slots=True)
class SpanResolver:
    """Turns span-level ground truth into chunk-level relevance judgements.

    The default policy is COVERAGE at 0.5 -- a chunk counts when it holds at least half of
    the gold span's characters.

    Why not IoU, which is the obvious symmetric choice? Because it builds a bias into the
    measuring instrument. Take a 170-character gold span. A 2000-character chunk containing
    every character of it scores IoU 0.085 and would be called a miss at any sensible
    threshold, while a 250-character chunk containing the same evidence scores 0.68 and
    passes. IoU therefore penalises large-chunk configurations for being large -- and chunk
    size is one of the axes under test. Coverage asks the question that actually matters:
    is the evidence there?

    IoU remains available, because when you care about wasted context rather than answer
    grounding, punishing bloat is exactly right.
    """

    policy: ResolutionPolicy = ResolutionPolicy.COVERAGE
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ResolutionError(
                f"threshold must be in (0, 1], got {self.threshold}. "
                "A threshold of 0 would mark every touching chunk relevant."
            )

    # -- single-pair decision ------------------------------------------------

    def score(self, chunk_span: Span, gold_span: Span) -> float:
        """The policy's score for this pair, in [0, 1]."""
        if self.policy is ResolutionPolicy.COVERAGE:
            return chunk_span.coverage_of(gold_span)
        if self.policy is ResolutionPolicy.IOU:
            return chunk_span.iou(gold_span)
        return 1.0 if chunk_span.contains(gold_span) else 0.0

    def is_relevant(self, chunk_span: Span, gold_span: Span) -> bool:
        """Whether this chunk counts as relevant to this gold span."""
        if self.policy is ResolutionPolicy.CONTAINMENT:
            return chunk_span.contains(gold_span)
        return self.score(chunk_span, gold_span) >= self.threshold

    # -- one question --------------------------------------------------------

    def resolve_item(self, item: EvalItem, chunks: Sequence[Chunk]) -> Resolution:
        """Resolve one question's gold spans against a chunk set.

        A chunk relevant to several gold spans takes the highest grade among them, which is
        the standard convention and keeps nDCG honest.
        """
        log = WarningLog()

        # Two different things arrive here with no gold spans, and they are not the same
        # problem. A question with no gold *and* no anchors carries no ground truth at all --
        # either it is deliberately unanswerable, which is a useful thing to test, or the eval
        # set is unfinished. A question with anchors and no gold had evidence written for it
        # and *this parse* lost it, which is a fact about the parser.
        #
        # One guard used to cover both, because `is_answerable` once meant `bool(gold)`. It
        # now means "carries evidence in either form", so the parser-lost case fell straight
        # through into an empty loop and reported nothing at all.
        if not item.is_resolved:
            if item.anchors:
                # "That is a measurement of the parser, not of the retriever" was half right,
                # and the wrong half was stated the most confidently. This fires whenever the
                # anchors produced no spans -- which happens for an invented quote, a quote
                # naming the wrong `source_id`, and an `occurrence` past the last copy, none
                # of which is the parser's doing. `anchor.py` is where those causes are told
                # apart, so the honest thing here is to keep the claim that survives -- this
                # is not a retriever result -- and send the reader to the warning that knows.
                log.add(
                    WarningCode.GOLD_SPAN_UNREACHABLE,
                    (
                        f"item {item.id!r} quotes its evidence but none of it was located in "
                        f"this parse, so it is excluded from ranking metrics. Nothing here is "
                        f"a measurement of the retriever. Whether the parser lost the text or "
                        f"the eval set quotes something that is not in the document it names, "
                        f"this cannot tell -- the `anchor_not_found` warnings for "
                        f"{item.id!r} say which of those could be told apart"
                    ),
                    severity=Severity.CAUTION,
                    stage="resolve",
                    subject=item.id,
                )
            else:
                # No judgements to make. Still useful -- an unanswerable question tests
                # whether a system declines to answer -- but it cannot contribute to ranking
                # metrics, and ranx would treat an empty qrel as a bug.
                log.add(
                    WarningCode.GOLD_SPAN_UNREACHABLE,
                    f"item {item.id!r} has no gold spans and is excluded from ranking metrics",
                    severity=Severity.INFO,
                    stage="resolve",
                    subject=item.id,
                )
            return Resolution(item_id=item.id, warnings=log)

        chunk_spans = spans_of(chunks)
        grades: dict[str, int] = {}
        per_gold: list[GoldResolution] = []

        for gold in item.gold:
            matched: list[str] = []
            best = 0.0
            for chunk in chunks:
                value = self.score(chunk.span, gold.span)
                best = max(best, value)
                if self.is_relevant(chunk.span, gold.span):
                    matched.append(chunk.id)
                    grades[chunk.id] = max(grades.get(chunk.id, 0), gold.grade)

            union = coverage_fraction(gold.span, chunk_spans)
            resolution = GoldResolution(
                gold=gold,
                chunk_ids=tuple(matched),
                best_score=best,
                union_coverage=union,
            )
            per_gold.append(resolution)

            if resolution.is_split:
                log.add(
                    WarningCode.SPLIT_GOLD_SPAN,
                    (
                        f"gold span {gold.span.start}-{gold.span.end} in {gold.doc_id!r} is "
                        f"split across chunks; no single chunk reaches the {self.policy.value} "
                        f"threshold of {self.threshold:g} (best {best:.2f}), but the chunk set "
                        f"covers {union:.0%} of it"
                    ),
                    severity=Severity.CAUTION,
                    stage="resolve",
                    subject=item.id,
                    best_score=best,
                    union_coverage=union,
                )
            elif not resolution.is_reachable:
                log.add(
                    WarningCode.GOLD_SPAN_UNREACHABLE,
                    _unreachable_message(gold, chunk_spans),
                    severity=Severity.CAUTION,
                    stage="resolve",
                    subject=item.id,
                )

        if any(not chunk.offsets_exact for chunk in chunks):
            log.add(
                WarningCode.APPROXIMATE_RESOLUTION,
                (
                    "some chunks report approximate offsets, so their text is not a literal "
                    "slice of the source. Relevance judgements against them are estimates"
                ),
                severity=Severity.CAUTION,
                stage="resolve",
                subject=item.id,
            )

        labels = tuple(
            RelevanceLabel(item_id=item.id, chunk_id=chunk_id, grade=grade)
            for chunk_id, grade in sorted(grades.items())
            if grade > 0
        )
        return Resolution(item_id=item.id, labels=labels, per_gold=tuple(per_gold), warnings=log)

    # -- a whole eval set ----------------------------------------------------

    def resolve(
        self, evalset: EvalSet, chunks: Sequence[Chunk]
    ) -> tuple[dict[str, Resolution], WarningLog]:
        """Resolve every question in an eval set against one chunk set."""
        resolutions: dict[str, Resolution] = {}
        log = WarningLog()
        for item in evalset:
            resolution = self.resolve_item(item, chunks)
            resolutions[item.id] = resolution
            log.extend(resolution.warnings)
        return resolutions, log

    def qrels(self, evalset: EvalSet, chunks: Sequence[Chunk]) -> Qrels:
        """Relevance judgements in the shape `ranx` consumes.

        Questions with no resolvable gold are left out rather than included empty, because
        an empty judgement set would be scored as a legitimate zero and drag the mean down
        for a reason that has nothing to do with the retriever.
        """
        resolutions, _ = self.resolve(evalset, chunks)
        return {
            item_id: resolution.as_qrel()
            for item_id, resolution in resolutions.items()
            if resolution.labels
        }


def _unreachable_message(gold: GoldSpan, chunk_spans: Sequence[Span]) -> str:
    """Why one gold span matched no chunk, claiming only what the offsets can show.

    "This question cannot be answered under this chunking" was the only answer given, and it
    is not the only cause. A gold span at 900-950 of a 100-character document matches no chunk
    for the same reason a real chunking gap does -- `is_reachable` is `bool(chunk_ids)` and
    says nothing about why -- so an eval set carrying stale offsets sent somebody off to sweep
    a chunk size that was never the problem.

    The offsets settle it. A span outside everything this parse chunked cannot be a decision
    the chunker made; a span inside that range that still hits nothing is exactly that.
    """
    where = f"gold span {gold.span.start}-{gold.span.end} in {gold.doc_id!r}"
    same_document = [span for span in chunk_spans if span.doc_id == gold.doc_id]

    if not same_document:
        return (
            f"{where} matches no chunk, because this run produced no chunks for "
            f"{gold.doc_id!r} at all. Either the corpus does not hold that document or the "
            "eval set names it differently -- the chunker never saw it"
        )

    start = min(span.start for span in same_document)
    end = max(span.end for span in same_document)
    if gold.span.start >= end or gold.span.end <= start:
        return (
            f"{where} matches no chunk, and it lies outside the {start}-{end} this parse "
            f"chunked for {gold.doc_id!r}. That is an eval set pointing at offsets this text "
            "does not have, not a chunking decision -- span-form gold does not survive a "
            "change of parser, which is what `anchors` are for"
        )

    return (
        f"{where} matches no chunk at all, though it lies inside the {start}-{end} this parse "
        f"chunked for {gold.doc_id!r}. This question cannot be answered under this chunking, "
        "whatever the retriever does"
    )


# ---------------------------------------------------------------------------
# Character-level measures
# ---------------------------------------------------------------------------
#
# These do not depend on the resolution policy at all -- they work directly on spans, which
# is what makes them the honest check on chunk-level metrics. A configuration returning
# enormous chunks can score Recall@5 of 1.0 while filling the context window with text that
# has nothing to do with the question. Chunk-level recall applauds it; character precision
# shows what it costs.


def character_recall(item: EvalItem, retrieved: Iterable[Chunk]) -> float:
    """Fraction of the question's gold characters present anywhere in the retrieved chunks.

    Union-based, so gold split across two chunks counts as fully retrieved when both come
    back -- which is the truthful answer, since the evidence would all be in the context.
    """
    gold_spans = item.gold_spans
    if not gold_spans:
        return 0.0
    # Non-zero by construction: GoldSpan rejects empty spans, so any gold at all has length.
    gold_total = total_length(gold_spans)
    return intersection_length(gold_spans, spans_of(retrieved)) / gold_total


def character_precision(item: EvalItem, retrieved: Iterable[Chunk]) -> float:
    """Fraction of the retrieved characters that are gold.

    The metric that exposes context waste. Chunk Recall@5 of 1.0 alongside character
    precision of 0.04 means the right evidence arrived buried in 25x its weight in
    irrelevant text, and every generation call is paying for it.
    """
    retrieved_spans = spans_of(retrieved)
    retrieved_total = total_length(retrieved_spans)
    if retrieved_total == 0:
        return 0.0
    return intersection_length(item.gold_spans, retrieved_spans) / retrieved_total


def character_f1(item: EvalItem, retrieved: Iterable[Chunk]) -> float:
    """Harmonic mean of character precision and recall."""
    chunks = list(retrieved)
    precision = character_precision(item, chunks)
    recall = character_recall(item, chunks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def retrieved_character_count(retrieved: Iterable[Chunk]) -> int:
    """Characters sent downstream, counting overlapping chunks once.

    The real driver of the generation bill, and the number `k` alone never tells you.
    """
    return total_length(spans_of(retrieved))


def gold_coverage_by_chunk(item: EvalItem, chunks: Sequence[Chunk]) -> dict[str, float]:
    """Per chunk, the fraction of this question's gold characters it holds.

    Feeds the per-query inspector: it is how a chunk gets highlighted as "holds 60% of the
    evidence" rather than a flat relevant/not-relevant mark.
    """
    gold = merge_spans(item.gold_spans)
    gold_total = total_length(gold)
    if gold_total == 0:
        return {}
    return {chunk.id: intersection_length(gold, [chunk.span]) / gold_total for chunk in chunks}
