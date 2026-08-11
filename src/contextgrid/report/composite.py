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
    # Did the chunking keep the evidence whole and reachable? A chunker that cuts a quoted
    # passage down the middle loses half of it however good the retriever is.
    #
    # `char_recall`, not `char_precision`, and that choice is the difference between a score
    # and a ranking that runs backwards. Character *precision* is bounded above by roughly
    # `gold_chars / (k * chunk_size)`: with 512-token chunks, k=5 and a 60-character quote it
    # cannot exceed about 0.005 no matter how good the chunker is. Feeding that into a
    # harmonic mean makes it the only dimension that matters, and since the bound loosens as
    # chunks get smaller, the composite rewards tiny chunks rather than good ones -- on the
    # run that exposed this, the leaderboard winner scored 2/100 and the leaderboard loser
    # scored 10/100.
    #
    # Recall has no such ceiling: 1.0 means the evidence came back intact, which is a thing a
    # chunker can actually achieve and a thing worth scoring. Precision is still computed and
    # still reported, as `char_precision@k` -- it measures wasted context, which is a real
    # cost and a genuinely useful column. It is just not a 0-1 quality score, so it does not
    # belong in a mean with things that are.
    "chunk": ("char_recall",),
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
    #: Dimension -> the exact metric keys its value came from, cut-offs included.
    #:
    #: Kept because the cut-offs need not agree. A run whose headline is `recall@1` emits
    #: every metric at `@1`, and one that mixes cut-offs would otherwise average `char_recall`
    #: over one chunk with `recall` over five and print a single number for both.
    sources: dict[str, tuple[str, ...]] = field(default_factory=dict)

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
        if len(self.cutoffs) > 1:
            # Only when they actually disagree. Naming the cut-off on every line would be
            # noise on the normal run, where every metric shares one; staying silent when they
            # differ would hide that two dimensions were measured over different top-k lists.
            attributed = "; ".join(
                f"{dimension} from {', '.join(self.sources[dimension])}"
                for dimension in self.dimensions
                if dimension in self.sources
            )
            line += f" (cut-offs differ: {attributed})"
        if self.missing:
            line += f" (not measured: {', '.join(sorted(self.missing))})"
        return line

    @property
    def cutoffs(self) -> tuple[int, ...]:
        """Every distinct cut-off the scored values came from, smallest first."""
        seen = set()
        for keys in self.sources.values():
            for key in keys:
                _, at, tail = key.rpartition("@")
                if at and tail.isdigit():
                    seen.add(int(tail))
        return tuple(sorted(seen))

    def as_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "parts": dict(self.parts),
            "missing": dict(self.missing),
            "dimensions": list(self.dimensions),
            "sources": {dimension: list(keys) for dimension, keys in self.sources.items()},
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
    k: int | None = None,
    dimensions: Mapping[str, tuple[str, ...]] | None = None,
) -> CompositeScore:
    """Combine whatever this run actually measured into one 0-100 score.

    `metrics` is a run's flat metric map -- `recall@5`, `faithfulness`, `char_recall@5` and so
    on. Cut-offs are resolved against `k`, so `recall` finds `recall@5` without the caller
    spelling it out.

    **`k=None` reads the cut-off off the metrics instead of assuming one.** It used to default
    to 5, and a run whose headline was `recall@1` therefore had every dimension looked up at a
    cut-off nothing in it had ever emitted: a run holding `char_recall@1 = 0.8824` was reported
    as `88/100 over 3 dimension(s) ... (not measured: chunk, generation)`. Rule 2 is about not
    scoring a dimension that did not run; calling a dimension that *did* run unmeasured is the
    same lie pointing the other way, and it is worse, because the number still gets printed.

    So with no `k`, the run's own dominant cut-off is used, and any metric absent at that
    cut-off falls back to the nearest cut-off it does have -- a value measured at *some* k is a
    measurement, and `.sources` records which one so a mixed run cannot hide it. Passing `k`
    explicitly still means exactly that k, because a caller asking for `k=5` is asking a
    question about the top 5 and deserves silence rather than an answer about the top 3.
    """
    wanted = dimensions or DIMENSION_METRICS
    # An explicit k is a constraint; an absent one is a question about this run.
    strict = k is not None
    resolved_k = k if k is not None else _dominant_cutoff(metrics)

    parts: dict[str, float] = {}
    missing: dict[str, str] = {}
    sources: dict[str, tuple[str, ...]] = {}

    for dimension, names in wanted.items():
        hits = [
            hit
            for name in names
            if (hit := _lookup(metrics, name, resolved_k, strict=strict)) is not None
        ]
        if not hits:
            missing[dimension] = f"no value for {' or '.join(names)}"
            continue
        # Several metrics for one dimension average arithmetically: they are two views of the
        # same thing, not two links in a chain, and one being low is not the same kind of
        # failure as a whole dimension being low.
        parts[dimension] = sum(value for _, value in hits) / len(hits)
        sources[dimension] = tuple(key for key, _ in hits)

    return CompositeScore(
        score=100.0 * harmonic_mean(parts),
        parts=parts,
        missing=missing,
        sources=sources,
    )


def _lookup(
    metrics: Mapping[str, float], name: str, k: int | None, *, strict: bool
) -> tuple[str, float] | None:
    """Find a metric by name, with or without a cut-off, and only if it is usable.

    Returns the key it used as well as the value, because "which cut-off is this?" is a
    question the reader of a mixed run has to be able to answer.

    Values outside 0-1 are ignored rather than clamped. A composite is a comparison of
    like-scaled things, and something reporting 3.2 is not on that scale -- silently squashing
    it to 1.0 would put a number in the score that nothing measured. An absent metric stays
    absent either way: nothing here ever substitutes a 0.0, which would be a measurement.
    """
    preferred = (name,) if k is None else (name, f"{name}@{k}")
    for key in preferred:
        value = _usable(metrics, key)
        if value is not None:
            return key, value
    if strict:
        return None

    # Nothing at the run's own cut-off, so take the nearest cut-off that does have a number.
    # Ties go to the smaller cut-off, purely so the choice is reproducible.
    prefix = f"{name}@"
    best: tuple[str, float] | None = None
    best_rank: tuple[int, int] | None = None
    for key in metrics:
        if not key.startswith(prefix) or not key[len(prefix) :].isdigit():
            continue
        value = _usable(metrics, key)
        if value is None:
            continue
        cutoff = int(key[len(prefix) :])
        rank = (0 if k is None else abs(cutoff - k), cutoff)
        if best_rank is None or rank < best_rank:
            best, best_rank = (key, value), rank
    return best


def _usable(metrics: Mapping[str, float], key: str) -> float | None:
    """The value under `key`, if there is one and it is on the 0-1 scale."""
    value = metrics.get(key)
    if value is None or not 0.0 <= float(value) <= 1.0:
        return None
    return float(value)


def _dominant_cutoff(metrics: Mapping[str, float]) -> int | None:
    """The cut-off most of this run's metrics were measured at, or `None` if none carry one.

    The runner computes every metric at the headline's cut-off, so the cut-off shared by the
    most keys is the one the run was actually scored on -- which beats a hardcoded 5 that no
    run ever has to agree with. Ties go to the smaller cut-off so the answer is reproducible.
    """
    counts: dict[int, int] = {}
    for key in metrics:
        _, at, tail = key.rpartition("@")
        if at and tail.isdigit():
            counts[int(tail)] = counts.get(int(tail), 0) + 1
    if not counts:
        return None
    return min(counts, key=lambda cutoff: (-counts[cutoff], cutoff))
