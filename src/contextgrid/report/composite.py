"""One number out of many, and an honest account of what went into it.

A leaderboard with fourteen columns is a leaderboard nobody reads to the end of. A single score
is what people actually want, and it is also the easiest thing in this package to make dishonest
-- so the rules it follows are worth stating plainly.

**Harmonic, not arithmetic.** A configuration retrieving at 0.95 and generating faithfully at
0.10 averages to 0.53, which reads as middling. It is not middling; it is a system that
confidently invents answers, and 0.53 hides that behind a good retriever. The harmonic mean puts
it at 0.18, because a chain is worth what its weakest link is worth. Every composite score built
on an arithmetic mean is a way of not noticing the worst thing about a system.

**Only what ran.** Somebody sweeping ingestion and retrieval has no generator, and scoring the
missing generation dimension as zero would punish them for a question they never asked. The
score covers the dimensions that produced numbers, and it says which ones those were -- so a 73
over three dimensions is never mistaken for a 73 over six.

**Comparable only within a run.** Two scores computed over different dimension sets are not
comparable, and the report says so rather than leaving the reader to notice. That is why the
dimensions come back attached to the number rather than in a footnote.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

#: Which metric stands for each dimension in the composite, and how to read it.
#:
#: One per dimension on purpose. Averaging four retrieval metrics into a retrieval score and
#: then averaging that with generation gives retrieval four votes and generation one, which is
#: an opinion about what matters dressed up as arithmetic.
DIMENSION_METRICS: dict[str, tuple[str, ...]] = {
    # Did the parse make the evidence findable at all? Falls out of anchor resolution, and it
    # is the only dimension whose failure makes every later number meaningless.
    "parse": ("evidence_resolvable",),
    # Of the characters returned, how many were the ones asked for? The chunker's honest score:
    # a huge chunk can hold the answer and still waste most of a context window.
    #
    # `char_precision`, not `character_precision`: the runner emits the short spelling, and this
    # table spent its whole existence asking for the long one. Nothing matched, so the chunk
    # dimension was silently dropped from every composite score ever computed -- reported as
    # "not measured" on runs that had measured it perfectly well.
    "chunk": ("char_precision",),
    # Can this embedder discriminate on this corpus at all? Measurable with no eval set, which
    # makes it the only dimension somebody choosing a model can score before writing questions.
    "embed": ("embedding_quality",),
    # Did the right passages come back?
    "retrieval": ("recall", "ndcg"),
    # Is the answer supported by them, and does it address the question?
    "generation": ("faithfulness", "answer_relevancy"),
}


@dataclass(slots=True)
class CompositeScore:
    """A 0-100 score, and everything needed to argue with it."""

    score: float
    #: Dimension -> the 0-1 value that went in.
    parts: dict[str, float] = field(default_factory=dict)
    #: Dimensions that produced no number, and why.
    missing: dict[str, str] = field(default_factory=dict)

    @property
    def dimensions(self) -> tuple[str, ...]:
        return tuple(sorted(self.parts))

    def summary(self) -> str:
        """The score, and immediately what it covers.

        Never the number alone. A 73 over retrieval and generation is a different claim from a
        73 over all four, and printing them identically invites the comparison that is wrong.
        """
        if not self.parts:
            return "no score: nothing measurable ran"

        covered = ", ".join(self.dimensions)
        line = f"{self.score:.0f}/100 over {len(self.parts)} dimension(s): {covered}"
        if self.missing:
            line += f" (not measured: {', '.join(sorted(self.missing))})"
        return line

    def as_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "parts": dict(self.parts),
            "missing": dict(self.missing),
            "dimensions": list(self.dimensions),
        }


def harmonic_mean(values: Mapping[str, float]) -> float:
    """The mean that refuses to let one bad component hide behind good ones.

    Zero anywhere gives zero, which is correct and deliberate: a system that generates nothing
    faithful has no score worth reporting, however well it retrieves.
    """
    numbers = list(values.values())
    if not numbers:
        return 0.0
    if any(value <= 0 for value in numbers):
        return 0.0
    return len(numbers) / sum(1.0 / value for value in numbers)


def composite(
    metrics: Mapping[str, float],
    *,
    k: int = 5,
    dimensions: Mapping[str, tuple[str, ...]] | None = None,
) -> CompositeScore:
    """Combine whatever this run actually measured into one 0-100 score.

    `metrics` is a run's flat metric map -- `recall@5`, `faithfulness`, `character_precision`
    and so on. Cut-offs are resolved against `k`, so `recall` finds `recall@5` without the
    caller spelling it out.
    """
    wanted = dimensions or DIMENSION_METRICS
    parts: dict[str, float] = {}
    missing: dict[str, str] = {}

    for dimension, names in wanted.items():
        found = [value for name in names if (value := _lookup(metrics, name, k)) is not None]
        if not found:
            missing[dimension] = f"no value for {' or '.join(names)}"
            continue
        # Several metrics for one dimension average arithmetically: they are two views of the
        # same thing, not two links in a chain, and one being low is not the same kind of
        # failure as a whole dimension being low.
        parts[dimension] = sum(found) / len(found)

    return CompositeScore(score=100.0 * harmonic_mean(parts), parts=parts, missing=missing)


def _lookup(metrics: Mapping[str, float], name: str, k: int) -> float | None:
    """Find a metric by name, with or without a cut-off, and only if it is usable.

    Values outside 0-1 are ignored rather than clamped. A composite is a comparison of
    like-scaled things, and something reporting 3.2 is not on that scale -- silently squashing
    it to 1.0 would put a number in the score that nothing measured.
    """
    for key in (name, f"{name}@{k}"):
        value = metrics.get(key)
        if value is not None and 0.0 <= float(value) <= 1.0:
            return float(value)
    return None
