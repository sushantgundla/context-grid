"""Classifying failures into the Seven Failure Points.

A leaderboard says a configuration scored 0.62. It does not say *why* the other 0.38 failed,
and "improve retrieval" is not an action anybody can take.

The Seven Failure Points paper (Barnett et al., 2024) gives a vocabulary for this, drawn from
three real production systems. Every failing question here is sorted into one of them, which
turns a score into a list of things to actually do. Nobody has productised it, and it costs
almost nothing once the run data exists.

The seven, and what each one means for a retrieval sweep:

- **FP1 Missing content** -- the answer is not in the corpus at all. Not a retrieval failure.
- **FP2 Missed top-ranked** -- the evidence was retrieved, just not high enough. Rerank.
- **FP3 Not in context** -- it is in the index and did not reach the context, whether it
  ranked below k or was never returned. Widen k, or consolidate better.
- **FP4 Not extracted** -- present in the context and the generator missed it. Not retrieval.
- **FP5 Wrong format** -- the answer was there, in a shape nothing could use.
- **FP6 Wrong specificity** -- too general or too narrow to answer what was asked.
- **FP7 Incomplete** -- partially answered, with evidence spread wider than was retrieved.

FP4 to FP7 need a generator to observe, so a retrieval-only run classifies FP1 to FP3 and
says plainly that it cannot see the rest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from contextgrid.core.evalset import EvalItem, EvalSet, Qrels


class FailurePoint(str, Enum):
    """The seven, plus a success case and one for what a retrieval run cannot see."""

    NONE = "none"
    MISSING_CONTENT = "fp1_missing_content"
    MISSED_TOP_RANKED = "fp2_missed_top_ranked"
    NOT_IN_CONTEXT = "fp3_not_in_context"
    NOT_EXTRACTED = "fp4_not_extracted"
    WRONG_FORMAT = "fp5_wrong_format"
    WRONG_SPECIFICITY = "fp6_wrong_specificity"
    INCOMPLETE = "fp7_incomplete"
    UNOBSERVABLE = "needs_generation"


#: What to do about each one, in the order a person would try them.
REMEDIES: dict[FailurePoint, str] = {
    FailurePoint.MISSING_CONTENT: (
        "the evidence is not in this index at all. Either the parser lost it, the chunker "
        "dropped it, or the corpus does not contain it. No retriever can fix this"
    ),
    FailurePoint.MISSED_TOP_RANKED: (
        "the evidence was retrieved but ranked too low to be used. This is what a reranker "
        "is for, and it is the cheapest failure on this list to fix"
    ),
    FailurePoint.NOT_IN_CONTEXT: (
        "the evidence is in the index and never reached the context -- it ranked outside k, "
        "or the run did not return it at all. Raising k or feeding a reranker a deeper "
        "candidate list will recover it, at a cost in tokens"
    ),
    FailurePoint.NOT_EXTRACTED: (
        "the evidence reached the generator and the generator missed it. A retrieval change "
        "will not help; look at the prompt and at how much noise surrounds the answer"
    ),
    FailurePoint.WRONG_FORMAT: (
        "the answer was present in a shape the pipeline could not use, usually a table "
        "flattened into prose. This is a parser problem wearing a generation costume"
    ),
    FailurePoint.WRONG_SPECIFICITY: (
        "the retrieved passage is about the right topic at the wrong level of detail. "
        "Chunk size is the usual lever"
    ),
    FailurePoint.INCOMPLETE: (
        "part of the evidence was retrieved and part was not. The answer needs passages that "
        "no single chunk holds, so try parent-document retrieval or a larger k"
    ),
}


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """Why one question failed, and what would help."""

    item_id: str
    failure: FailurePoint
    detail: str
    gold_rank: int | None = None
    retrieved: int = 0

    @property
    def succeeded(self) -> bool:
        return self.failure is FailurePoint.NONE

    @property
    def remedy(self) -> str:
        return REMEDIES.get(self.failure, "")

    def __str__(self) -> str:
        return f"{self.item_id}: {self.failure.value} -- {self.detail}"


@dataclass(slots=True)
class FailureReport:
    """Every question's diagnosis, and what the pattern across them suggests."""

    diagnoses: list[Diagnosis] = field(default_factory=list)
    k: int = 5
    observed_generation: bool = False
    #: Questions that carry no ground truth at all -- no gold spans and no anchors -- and so
    #: were never scored and are not diagnosed.
    #:
    #: They used to be swept into `MISSING_CONTENT`, which told the reader their parser,
    #: chunker or corpus had lost the evidence for a question nobody ever wrote evidence for.
    #: The eval set has a hole in it; the pipeline is fine. Those are different problems with
    #: different fixes, so they are counted separately and never enter the histogram.
    no_ground_truth: list[str] = field(default_factory=list)

    @property
    def total_items(self) -> int:
        """Every question the eval set held, diagnosed or not."""
        return len(self.diagnoses) + len(self.no_ground_truth)

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for diagnosis in self.diagnoses:
            counts[diagnosis.failure.value] = counts.get(diagnosis.failure.value, 0) + 1
        return dict(sorted(counts.items()))

    def failures(self) -> list[Diagnosis]:
        return [d for d in self.diagnoses if not d.succeeded]

    def of(self, failure: FailurePoint) -> list[Diagnosis]:
        return [d for d in self.diagnoses if d.failure is failure]

    @property
    def dominant(self) -> FailurePoint | None:
        """The failure mode accounting for most of what went wrong."""
        failures = self.failures()
        if not failures:
            return None
        counts: dict[FailurePoint, int] = {}
        for diagnosis in failures:
            counts[diagnosis.failure] = counts.get(diagnosis.failure, 0) + 1
        return max(counts, key=lambda point: counts[point])

    def summary(self, *, include_unscored: bool = True) -> str:
        """What went wrong and what to do about it, as a sentence.

        Turns a score into an action. "0.62" tells you nothing; "most failures are evidence
        ranked just outside k, so try a reranker" tells you what to run next.

        `include_unscored=False` drops the eval-set-gap sentence, for callers such as
        `Results.summary` that have already said it in their own words and would otherwise
        say it twice in one paragraph.
        """
        total = len(self.diagnoses)
        failures = self.failures()
        gap = self._eval_set_gap() if include_unscored else []

        if not failures:
            return " ".join([f"All {total} questions succeeded.", *gap])

        dominant = self.dominant
        assert dominant is not None
        share = len(self.of(dominant)) / len(failures)

        lines = [
            f"{len(failures)} of {total} questions failed. "
            f"{share:.0%} of those are {dominant.value}: {REMEDIES[dominant]}."
        ]
        if not self.observed_generation:
            lines.append(
                "This was a retrieval-only run, so failure points four to seven -- the ones "
                "about what the generator did with the context -- cannot be seen from here."
            )
        lines.extend(gap)
        return " ".join(lines)

    def _eval_set_gap(self) -> list[str]:
        """The sentence for questions nobody wrote ground truth for, if there are any."""
        missing = len(self.no_ground_truth)
        if not missing:
            return []
        one = missing == 1
        return [
            f"A further {missing} question{'' if one else 's'} "
            f"({', '.join(self.no_ground_truth)}) "
            f"{'has' if one else 'have'} no ground truth -- no gold spans and no anchors -- "
            f"so {'it was' if one else 'they were'} not scored at all. "
            "That is a gap in the eval set, not a fault in this pipeline."
        ]


def diagnose(
    evalset: EvalSet,
    qrels: Qrels,
    run: Mapping[str, Sequence[str]],
    *,
    k: int = 5,
    deep_k: int = 100,
) -> FailureReport:
    """Sort every question into a failure point, from retrieval data alone.

    `deep_k` is how far down the ranking to look before concluding the evidence was never
    retrieved at all. It is what separates "ranked too low" from "not in the index", and
    those two have completely different fixes.
    """
    report = FailureReport(k=k, observed_generation=False)

    for item in evalset:
        # A question with no gold spans *and* no anchors was never given an answer to be
        # right about. Diagnosing it as FP1 blamed the parser, the chunker and the corpus
        # for a question the eval set never finished writing -- and inflated the failure
        # histogram past the number of questions actually scored.
        #
        # The fields, not `has_evidence`: that name is an alias of `is_answerable`, which has
        # already been redefined once, and the distinction this line rests on is the one the
        # whole fix turns on. An item with anchors and no gold is a parser that lost the
        # evidence -- a real FP1 -- and must not be swept in here.
        if not (item.gold or item.anchors):
            report.no_ground_truth.append(item.id)
            continue
        judgements = qrels.get(item.id, {})
        ranked = list(run.get(item.id, ()))
        report.diagnoses.append(_diagnose_one(item, judgements, ranked, k, deep_k))

    return report


def _diagnose_one(
    item: EvalItem,
    judgements: Mapping[str, int],
    ranked: Sequence[str],
    k: int,
    deep_k: int,
) -> Diagnosis:
    relevant = {chunk_id for chunk_id, grade in judgements.items() if grade > 0}

    if not relevant:
        # This item *does* carry ground truth -- `diagnose` filtered out the ones that do
        # not -- so nothing matching it in the index is a genuine FP1. Which half of the
        # pipeline lost it is worth saying: an anchor that never resolved is a parse
        # problem, gold that resolved and matched no chunk is a chunking problem.
        lost_at_parse = bool(item.anchors) and not item.gold
        return Diagnosis(
            item.id,
            FailurePoint.MISSING_CONTENT,
            (
                "the quoted evidence for this question could not be found in this parse"
                if lost_at_parse
                else "no chunk in this index holds the evidence for this question"
            ),
            retrieved=len(ranked),
        )

    rank = next(
        (position for position, chunk_id in enumerate(ranked, start=1) if chunk_id in relevant),
        None,
    )

    if rank is None:
        # FP3, not FP1. The qrels name a chunk that holds the evidence, so it survived the
        # parser and it survived the chunker -- the retriever simply never returned it. The
        # FP1 remedy sends the reader to fix a parser that did nothing wrong, and because
        # `FailureReport.summary` prints the remedy rather than the detail, that was the half
        # they read.
        #
        # It sits with the rank-past-`deep_k` case below because the fix is the same one:
        # more candidates. "Never seen at any depth we looked" is the far end of "ranked too
        # low", not a different problem.
        return Diagnosis(
            item.id,
            FailurePoint.NOT_IN_CONTEXT,
            "the evidence is in the index but was not among the "
            f"{len(ranked)} results this run returned",
            retrieved=len(ranked),
        )

    if rank <= k:
        # Everything the question needed was in the context. Whether the answer came out
        # right is a generation question, and this run did not watch the generator.
        found = sum(1 for chunk_id in ranked[:k] if chunk_id in relevant)
        if found < len(relevant):
            return Diagnosis(
                item.id,
                FailurePoint.INCOMPLETE,
                f"{found} of {len(relevant)} relevant chunks made it into the top {k}",
                gold_rank=rank,
                retrieved=len(ranked),
            )
        return Diagnosis(
            item.id,
            FailurePoint.NONE,
            f"evidence at rank {rank}",
            gold_rank=rank,
            retrieved=len(ranked),
        )

    if rank <= deep_k:
        return Diagnosis(
            item.id,
            FailurePoint.MISSED_TOP_RANKED,
            f"the evidence was retrieved at rank {rank}, just outside the top {k}",
            gold_rank=rank,
            retrieved=len(ranked),
        )

    return Diagnosis(
        item.id,
        FailurePoint.NOT_IN_CONTEXT,
        f"the evidence ranked {rank}, far below the top {k}",
        gold_rank=rank,
        retrieved=len(ranked),
    )


def cluster(report: FailureReport) -> dict[str, list[str]]:
    """Group failing questions by failure point.

    "These twelve failures are all evidence ranked just outside k" is worth more than twelve
    separate traces, because it is one fix rather than twelve investigations.
    """
    grouped: dict[str, list[str]] = {}
    for diagnosis in report.failures():
        grouped.setdefault(diagnosis.failure.value, []).append(diagnosis.item_id)
    return dict(sorted(grouped.items()))
