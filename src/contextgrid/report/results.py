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

from contextgrid.core.warnings import WarningLog
from contextgrid.cost.model import CostBreakdown
from contextgrid.pipeline import Config, Timings


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

    @property
    def label(self) -> str:
        return self.config.label

    @property
    def is_sound(self) -> bool:
        """False when something recorded here invalidates the comparison."""
        return self.warnings.is_sound

    def metric(self, name: str, default: float = 0.0) -> float:
        return self.metrics.get(name, default)

    def row(self, metrics: Sequence[str]) -> dict[str, Any]:
        row: dict[str, Any] = {"config": self.label}
        row.update({name: self.metrics.get(name, 0.0) for name in metrics})
        row["p95_ms"] = self.timings.percentile(0.95)
        row["cost_per_1k"] = self.cost.query_usd_per_1k
        row["chunks"] = self.chunk_count
        return row


@dataclass(slots=True)
class Results:
    """Every configuration a sweep ran, and the ways to read them."""

    runs: list[RunResult] = field(default_factory=list)
    warnings: WarningLog = field(default_factory=WarningLog)
    cache_summary: str = ""
    mode: str = "ofat"
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
        columns = [metric, *extra]
        ordered = sorted(self.runs, key=lambda r: -r.metric(metric))
        return [run.row(columns) for run in ordered]

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
        lines = [
            f"{winner.label} scored best on {metric} at {winner.metric(metric):.3f}, "
            f"across {len(self.runs)} configurations on {winner.scored_queries} questions."
        ]

        if len(ranked) > 1:
            runner_up = ranked[1]
            gap = winner.metric(metric) - runner_up.metric(metric)
            lines.append(
                f"That is {gap:+.3f} against {runner_up.label}. With "
                f"{winner.scored_queries} questions, treat a gap this size as suggestive "
                "rather than settled until it has been significance-tested."
            )

        latency = _readable_ms(winner.timings.percentile(0.95))
        if winner.cost.query_usd_per_1k:
            lines.append(
                f"It costs ${winner.cost.query_usd_per_1k:.4f} per 1,000 queries and answers "
                f"at {latency} p95."
            )
        else:
            lines.append(f"It runs locally at no cost per query, answering at {latency} p95.")

        if winner.unresolved_gold:
            lines.append(
                f"Note that {winner.unresolved_gold} pieces of evidence could not be located "
                "in this parse at all, so those questions were unanswerable whatever the "
                "retriever did."
            )

        if not self.warnings.is_sound:
            lines.append(
                f"{len(self.warnings.invalidating)} warnings mark this comparison as unsound. "
                "Read them before acting on any of it."
            )

        return " ".join(lines)


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
