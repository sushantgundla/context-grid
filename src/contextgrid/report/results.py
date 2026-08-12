"""Results: one row per configuration, and the views worth looking at.

The leaderboard is the obvious one and the least interesting. What a sweep is actually for is
the two things below it: the Pareto frontier, which shows what quality costs, and the axis
effect, which shows which decision mattered.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from contextgrid.core.warnings import Severity, WarningCode, WarningLog
from contextgrid.cost.model import CostBreakdown
from contextgrid.diagnose.taxonomy import FailureReport
from contextgrid.pipeline import Config, Timings
from contextgrid.report.composite import CompositeScore
from contextgrid.score.significance import (
    Comparison,
    Interval,
    SignificanceError,
    bootstrap_interval,
)
from contextgrid.score.significance import compare as compare_scores


@dataclass(slots=True)
class RunResult:
    """Everything one configuration produced."""

    config: Config
    metrics: dict[str, float] = field(default_factory=dict)
    timings: Timings = field(default_factory=Timings)
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    warnings: WarningLog = field(default_factory=WarningLog)
    chunk_count: int = 0
    index_bytes: int = 0
    scored_queries: int = 0
    unresolved_gold: int = 0
    run: dict[str, list[str]] = field(default_factory=dict)
    per_query: dict[str, float] = field(default_factory=dict)
    #: How many searches and model calls the retrieval strategy made, across every question.
    #:
    #: Two strategies with the same recall and different `model_calls` are a decision, not a
    #: tie -- which is the entire argument for having a retrieval axis. The number was counted
    #: from the start and reached no output file, so the one figure that would let somebody
    #: check a cost claim themselves was the one they could not see.
    retrieval: dict[str, int] = field(default_factory=dict)
    #: What the generator actually said, per question, with the chunks it was given and the
    #: judge's per-question scores. Empty for a run with no generator.
    #:
    #: Kept because a sweep with a generator spends real money and used to save nothing you
    #: could read: a faithfulness of 0.83 could be one invented answer or fifteen slightly
    #: loose ones, and there was no way to tell which, or to check a single sentence the
    #: model produced.
    answers: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_type: dict[str, dict[str, float]] = field(default_factory=dict)
    failures: FailureReport | None = None
    #: The seed the run was configured with. `interval()` resamples with it by default, so the
    #: confidence interval on a leaderboard row is reproducible from the manifest rather than
    #: from a hidden zero -- the same bug `Results.seed` exists to close for significance.
    seed: int = 0

    def interval(self, *, confidence: float = 0.95, seed: int | None = None) -> Interval | None:
        """A confidence interval on the headline metric.

        A single number with no interval is an opinion. This is what turns 0.71 into
        "0.71, and it could plausibly be anywhere from 0.63 to 0.79".
        """
        if not self.per_query:
            return None
        seed = self.seed if seed is None else seed
        return bootstrap_interval(list(self.per_query.values()), confidence=confidence, seed=seed)

    @property
    def label(self) -> str:
        return self.config.label

    @property
    def is_sound(self) -> bool:
        """False when something recorded here invalidates the comparison."""
        return self.warnings.is_sound

    def metric(self, name: str, default: float = 0.0) -> float:
        """One metric, or `default` when this run never computed it.

        The default exists for sorting, where a missing value has to be *some* number. Use
        `has(name)` before treating the result as a measurement -- `metric()` cannot tell a
        measured zero from a metric nobody ran.
        """
        return self.metrics.get(name, default)

    def has(self, name: str) -> bool:
        """Whether this run actually computed a metric."""
        return name in self.metrics

    def composite(self, *, k: int | None = None) -> CompositeScore:
        """This run's 0-100 score, over the dimensions it actually measured.

        The honest path from a run to a score. Building the input by hand invites reading
        `metric()`'s zero-default as a result, which collapses the harmonic mean to nothing.

        `k` defaults to whatever cut-off this run's metrics carry, not to 5. A run with
        `headline: recall@1` emits `char_recall@1`, and the old default went looking for
        `char_recall@5`, found nothing, and printed "not measured: chunk" for a dimension that
        had scored 0.8824. Pass `k` only to ask about one particular cut-off.
        """
        from contextgrid.report.composite import composite as _composite

        return _composite(self.metrics, k=k)

    def row(self, metrics: Sequence[str]) -> dict[str, Any]:
        """One leaderboard row.

        A metric this run never computed is left **out**, not filled with zero. It used to be
        filled: asking for `character_precision` on a run that never measured it produced a
        confident `0.0`, and a `0.0` fed to `composite()` is a measurement rather than a gap --
        so a configuration with perfect recall came back as 0/100. A number nobody measured is
        the most dangerous thing this package can print.
        """
        row: dict[str, Any] = {"config": self.label}
        row.update({name: self.metrics[name] for name in metrics if name in self.metrics})
        row["p95_ms"] = self.timings.percentile(0.95)
        row["cost_per_1k"] = self.cost.query_usd_per_1k
        row["chunks"] = self.chunk_count
        interval = self.interval()
        if interval is not None:
            row["ci_low"] = interval.low
            row["ci_high"] = interval.high
        return row


@dataclass(slots=True)
class Results:
    """Every configuration a sweep ran, and the ways to read them."""

    runs: list[RunResult] = field(default_factory=list)
    warnings: WarningLog = field(default_factory=WarningLog)
    cache_summary: str = ""
    mode: str = "ofat"
    #: The seed the run was configured with. Every resampling here defaults to it, so a
    #: significance verdict is reproducible from the manifest rather than from a hidden zero.
    seed: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[RunResult]:
        return iter(self.runs)

    def __len__(self) -> int:
        return len(self.runs)

    def get(self, label: str) -> RunResult | None:
        for run in self.runs:
            if run.label == label:
                return run
        return None

    # -- the obvious view ----------------------------------------------------

    def best(self, metric: str = "recall@5") -> RunResult | None:
        return max(self.runs, key=lambda r: r.metric(metric), default=None)

    def leaderboard(
        self, metric: str = "recall@5", extra: Sequence[str] = ()
    ) -> list[dict[str, Any]]:
        """Configurations ranked by one metric, with latency and cost beside it.

        Latency and cost are not optional columns. A leaderboard that omits them invites
        exactly the mistake this tool exists to prevent.
        """
        wanted = list(extra)
        unknown = [name for name in wanted if not any(run.has(name) for run in self.runs)]
        if unknown:
            self.warnings.add(
                WarningCode.NON_DETERMINISTIC_STAGE,
                f"no run computed {', '.join(sorted(unknown))}, so those columns are absent "
                "rather than zero. A metric nobody ran is not a score of nought",
                severity=Severity.CAUTION,
                stage="report",
            )

        columns = [metric, *wanted]
        ordered = sorted(self.runs, key=lambda r: -r.metric(metric))
        return [run.row(columns) for run in ordered]

    def composite(self, metric: str = "recall@5", *, k: int | None = None) -> CompositeScore | None:
        """The leading configuration's 0-100 score, over what it actually measured.

        `k=None` reads the cut-off off the winner's own metrics. See `RunResult.composite`.
        """
        best = self.best(metric)
        return best.composite(k=k) if best is not None else None

    # -- the views worth having ----------------------------------------------

    def pareto(self, quality: str = "recall@5", cost: str = "cost_per_1k") -> list[RunResult]:
        """Configurations nothing else beats on both quality and cost.

        The frontier is the honest answer to "which should I use?": everything on it is a
        legitimate choice at some budget, and everything off it is beaten outright by
        something cheaper *and* better.
        """
        cheaper = _cost_getter(cost)
        frontier: list[RunResult] = []
        for run in sorted(self.runs, key=lambda r: (cheaper(r), -r.metric(quality))):
            if not frontier or run.metric(quality) > frontier[-1].metric(quality):
                frontier.append(run)
        return frontier

    def axis_effect(self, axis: str, metric: str = "recall@5") -> dict[str, float]:
        """Mean score for each value of one axis, across every run that used it.

        The interpretable summary of a sweep: "structural chunking averaged 0.71 against
        recursive's 0.63" is a sentence somebody can act on, where a table of 48 rows is not.
        """
        grouped: dict[str, list[float]] = {}
        for run in self.runs:
            value = getattr(run.config, axis, None)
            grouped.setdefault(str(value), []).append(run.metric(metric))
        return {value: statistics.fmean(scores) for value, scores in sorted(grouped.items())}

    def compare(self, left: str, right: str, metric: str = "recall@5") -> dict[str, Any]:
        """Two configurations, their difference, and where they disagreed.

        The per-query disagreement is the useful part. Two configs with the same mean can
        succeed on completely different questions, and that is invisible from the leaderboard.
        """
        first, second = self.get(left), self.get(right)
        if first is None or second is None:
            missing = left if first is None else right
            raise KeyError(f"no run labelled {missing!r}")

        differences = {
            query_id: first.per_query[query_id] - second.per_query.get(query_id, 0.0)
            for query_id in first.per_query
        }
        disagreed = {q: d for q, d in differences.items() if abs(d) > 1e-9}

        return {
            "left": left,
            "right": right,
            "metric": metric,
            "left_score": first.metric(metric),
            "right_score": second.metric(metric),
            "difference": first.metric(metric) - second.metric(metric),
            "queries_compared": len(differences),
            "queries_disagreed": len(disagreed),
            "left_wins": sum(1 for d in disagreed.values() if d > 0),
            "right_wins": sum(1 for d in disagreed.values() if d < 0),
            "differences": disagreed,
        }

    def significance(
        self,
        left: str,
        right: str,
        *,
        metric: str = "recall@5",
        alpha: float = 0.05,
        seed: int | None = None,
    ) -> Comparison:
        """Whether two configurations actually differ, tested question by question.

        Both ran on the same questions, so this is a paired test -- each question acts as its
        own control, which removes the large variance that comes from some questions simply
        being harder than others.
        """
        first, second = self.get(left), self.get(right)
        if first is None or second is None:
            missing = left if first is None else right
            raise KeyError(f"no run labelled {missing!r}")

        # Falls back to the run's own seed, not to zero. `run.seed` was written into the
        # manifest and used by nothing: a config setting `seed: 42` recorded 42 and resampled
        # with 0, so the manifest made a reproducibility claim the run did not honour.
        seed = self.seed if seed is None else seed

        return compare_scores(
            first.per_query,
            second.per_query,
            left=left,
            right=right,
            metric=metric,
            alpha=alpha,
            seed=seed,
        )

    def is_the_winner_real(
        self, metric: str = "recall@5", *, alpha: float = 0.05, seed: int | None = None
    ) -> Comparison | None:
        """Test the top configuration against the runner-up.

        The question a leaderboard implicitly answers and never actually checks.
        """
        ranked = sorted(self.runs, key=lambda r: -r.metric(metric))
        if len(ranked) < 2:
            return None
        return self.significance(
            ranked[0].label, ranked[1].label, metric=metric, alpha=alpha, seed=seed
        )

    def by_type(self, metric: str = "recall@5") -> dict[str, dict[str, float]]:
        """Each configuration's score on each kind of question.

        Where the interesting results live. A chunker can win overall and lose badly on
        every question about a table, and one number per configuration will never show it.
        """
        return {run.label: run.by_type.get(metric, {}) for run in self.runs if run.by_type}

    def summary(self, metric: str = "recall@5") -> str:
        """The result in one plain-English paragraph.

        Every tool in this space outputs a table and leaves the reader to interpret it.
        Writing the conclusion in words is the part a reader without an IR background
        actually takes away -- and stating what the winner is *worst* at keeps it honest.
        """
        if not self.runs:
            return "No configurations were run."

        winner = self.best(metric)
        if winner is None:  # pragma: no cover - guarded by the emptiness check above
            return "No configurations were run."

        ranked = sorted(self.runs, key=lambda r: -r.metric(metric))
        # "across 1 configurations" is the sort of thing that makes a reader wonder what else
        # was not proofread, and a sweep of one is a normal thing to run.
        swept = f"{len(self.runs)} configuration" + ("" if len(self.runs) == 1 else "s")
        lines = [
            f"{winner.label} scored best on {metric} at {winner.metric(metric):.3f}, "
            f"across {swept}, scored on {winner.scored_queries} questions."
        ]

        # Two different question counts used to land in one paragraph with nothing saying
        # which was which: "on 17 questions" from `scored_queries` and "8 of 20 questions
        # failed" from the failure report, whose total is the whole eval set. Naming the eval
        # set's size before the failure sentence arrives is what makes the 20 readable.
        #
        # And the two reasons a question goes unscored are not the same problem. "No chunk in
        # this index held their evidence" points at the parser, the chunker and the corpus.
        # A question with no gold spans and no anchors was never given evidence to lose --
        # the eval set is unfinished, and telling that user to go debug their pipeline sends
        # them looking for a bug that is not there.
        in_evalset = winner.failures.total_items if winner.failures is not None else 0
        no_gold = len(winner.failures.no_ground_truth) if winner.failures is not None else 0
        unreachable = max(in_evalset - no_gold - winner.scored_queries, 0)
        if unreachable or no_gold:
            unscored = unreachable + no_gold
            named = _named(winner.failures.no_ground_truth) if no_gold else ""  # type: ignore[union-attr]
            one_gap = no_gold == 1
            gap_reason = (
                f"{'it has' if one_gap else 'they have'} no ground truth at all -- no gold "
                f"spans and no anchors -- so there was nothing to score "
                f"{'it' if one_gap else 'them'} against. That is a gap in the eval set, not a "
                f"fault in this pipeline"
            )
            stem = (
                f"The eval set holds {in_evalset} question"
                f"{'' if in_evalset == 1 else 's'} in all; the other "
                f"{unscored} {'was' if unscored == 1 else 'were'} not scored"
            )
            evidence = (
                f"no chunk in this index held {'its' if unreachable == 1 else 'their'} evidence"
            )
            if unreachable and no_gold:
                lines.append(
                    f"{stem}: {unreachable} because {evidence}, and {no_gold} ({named}) "
                    f"because {gap_reason}."
                )
            elif unreachable:
                lines.append(f"{stem}, because {evidence}.")
            else:
                lines.append(f"{stem} ({named}), because {gap_reason}.")

        if len(ranked) > 1:
            try:
                verdict = self.is_the_winner_real(metric)
            except (KeyError, SignificanceError):
                # A configuration that answered no questions cannot be tested against one
                # that did. Falling back to the bare gap is better than losing the summary.
                verdict = None
            if verdict is not None:
                lines.append(verdict.verdict())
            else:
                gap = winner.metric(metric) - ranked[1].metric(metric)
                lines.append(f"That is {gap:+.3f} against {ranked[1].label}.")

        latency = _readable_ms(winner.timings.percentile(0.95))
        if winner.cost.query_usd_per_1k:
            lines.append(
                f"It costs ${winner.cost.query_usd_per_1k:.4f} per 1,000 queries and answers "
                f"at {latency} p95."
            )
        else:
            lines.append(f"It runs locally at no cost per query, answering at {latency} p95.")

        if winner.unresolved_gold:
            # One unresolved anchor is an ordinary result, and "1 pieces of evidence ... those
            # questions were" undoes the care taken over the rest of the paragraph.
            one = winner.unresolved_gold == 1
            pieces = "piece" if one else "pieces"
            questions = "that question was" if one else "those questions were"
            lines.append(
                f"Note that {winner.unresolved_gold} {pieces} of evidence could not be located "
                f"in this parse at all, so {questions} unanswerable whatever the retriever did."
            )

        if winner.failures is not None and winner.failures.failures():
            # The eval-set gap is already stated above, in this paragraph's own words.
            lines.append(winner.failures.summary(include_unscored=False))

        if not self.warnings.is_sound:
            marking = len(self.warnings.invalidating)
            one = marking == 1
            lines.append(
                f"{marking} warning{'' if one else 's'} "
                f"{'marks' if one else 'mark'} this comparison as unsound. "
                f"Read {'it' if one else 'them'} before acting on any of it."
            )

        return " ".join(lines)


#: The sentence `contextgrid.score.significance._sample_size_note` produces when a gap is real
#: but the eval set is too small to settle it.
def _named(item_ids: Sequence[str], limit: int = 3) -> str:
    """Name the questions, but do not paste a hundred ids into a paragraph."""
    if len(item_ids) <= limit:
        return ", ".join(item_ids)
    rest = len(item_ids) - limit
    return f"{', '.join(item_ids[:limit])} and {rest} more"


def _readable_ms(milliseconds: float) -> str:
    """Latency a human can read. "0 ms" is a rounding artefact, not a measurement."""
    if milliseconds < 1:
        return "under 1 ms"
    return f"{milliseconds:.0f} ms"


def _cost_getter(name: str) -> Any:
    """Resolve a Pareto cost axis to something callable on a run."""
    if name in {"cost_per_1k", "cost"}:
        return lambda run: run.cost.query_usd_per_1k
    if name in {"p95_ms", "latency"}:
        return lambda run: run.timings.percentile(0.95)
    if name == "build_ms":
        return lambda run: run.timings.build_ms
    if name == "index_bytes":
        return lambda run: float(run.index_bytes)
    return lambda run: run.metrics.get(name, 0.0)
