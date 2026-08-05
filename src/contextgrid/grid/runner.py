"""Running a matrix.

The runner's whole job is to do less work than the matrix implies. Configurations sharing a
parser share its parse; those sharing parser, chunker and embedder share the embeddings.
A sweep over four indexes should parse once, chunk once and embed once -- and the cache
statistics are reported afterwards so that claim can be checked rather than believed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from contextgrid.cache.store import Cache, CacheStats, MemoryCache
from contextgrid.core.evalset import EvalSet
from contextgrid.core.warnings import Severity, WarningCode, WarningLog
from contextgrid.corpus import Corpus
from contextgrid.cost.model import CostModel
from contextgrid.diagnose.taxonomy import diagnose
from contextgrid.grid.matrix import AXIS_ORDER, Matrix, SweepMode
from contextgrid.pipeline import Config, build, build_qrels, resolve_evalset
from contextgrid.report.results import Results, RunResult
from contextgrid.score.anchor import AnchorResolver
from contextgrid.score.metrics import DEFAULT_KS, evaluate, per_query
from contextgrid.score.resolve import SpanResolver, character_precision, character_recall

Progress = Callable[[int, int, Config], None]


@dataclass(slots=True)
class Budget:
    """A ceiling on a sweep, in seconds or in dollars.

    `budget_seconds` has been enforced since the runner existed. `budget_usd` was accepted in
    the config, stored, written into the report -- and never checked, so a config that asked to
    spend at most five dollars would spend whatever the matrix cost. A knob that does nothing
    is worse than no knob, because somebody relies on it.

    Cost is charged after each configuration rather than predicted before it. A prediction
    would need to know what a model charges before calling it, which is exactly the thing the
    cost model cannot know for an agentic strategy that decides its own number of calls. So the
    ceiling is honoured to within one configuration, and the report says how much was spent.
    """

    seconds: float | None = None
    usd: float | None = None
    spent_usd: float = 0.0
    _started: float = 0.0

    def start(self) -> None:
        self._started = time.perf_counter()

    def charge(self, result: RunResult, queries: int) -> None:
        """Add what one configuration cost: building its index, and serving the eval set."""
        cost = getattr(result, "cost", None)
        if cost is not None:
            self.spent_usd += float(cost.total_at(queries))

    def exceeded(self) -> str | None:
        """Why the sweep should stop, or None to keep going."""
        if self.seconds is not None and time.perf_counter() - self._started > self.seconds:
            return f"the {self.seconds:g}s budget ran out"
        if self.usd is not None and self.spent_usd >= self.usd:
            return f"the ${self.usd:,.2f} budget ran out (${self.spent_usd:,.4f} spent)"
        return None


@dataclass(slots=True)
class Runner:
    """Runs configurations against a corpus and an eval set."""

    corpus: Corpus
    cache: Cache | None = None
    stats: CacheStats = field(default_factory=CacheStats)
    cost_model: CostModel = field(default_factory=CostModel)
    anchor_resolver: AnchorResolver = field(default_factory=AnchorResolver)
    span_resolver: SpanResolver = field(default_factory=SpanResolver)
    ks: tuple[int, ...] = DEFAULT_KS
    headline: str = "recall@5"

    def __post_init__(self) -> None:
        if self.cache is None:
            self.cache = MemoryCache()

        # Sorting a leaderboard on a metric nobody computed is the quietest possible failure.
        # `Runner(headline="recall@2")` reported 0.000 for every configuration, because 2 is
        # not one of the default cut-offs and `recall@2` was therefore never calculated -- an
        # empty column read as a real result. The config path had already learned this; the
        # Python API had not, so the guarantee now lives in the one place that serves both.
        _, _, cut = self.headline.partition("@")
        if cut.isdigit() and int(cut) not in self.ks:
            self.ks = tuple(sorted({*self.ks, int(cut)}))

    # -- one configuration ---------------------------------------------------

    def run_one(self, config: Config, evalset: EvalSet) -> RunResult:
        """Build a configuration, answer every question, and score it."""
        started = time.perf_counter()
        pipeline = build(config, self.corpus, cache=self.cache, stats=self.stats)

        # The evidence has to be located again in *this* parse. Two parsers produce
        # different text, so a span that was right for one is meaningless for the other.
        resolved, anchor_log = resolve_evalset(evalset, pipeline.parses, self.anchor_resolver)
        qrels, span_log = build_qrels(resolved, pipeline.chunks, self.span_resolver)

        run = pipeline.run_queries(resolved)
        metrics = evaluate(qrels, run, ks=self.ks)

        metric_name, _, k_text = self.headline.partition("@")
        headline_k = int(k_text or 5)
        scores = per_query(qrels, run, metric_name, headline_k)

        # Character-level precision is the honest check on chunk-level recall. A config
        # returning enormous chunks can score recall@5 of 1.0 while filling the context
        # window with text that has nothing to do with the question.
        metrics.update(_character_metrics(resolved, run, pipeline.chunk_by_id(), headline_k))
        by_type = _slice_by_type(resolved, scores, self.headline)
        failures = diagnose(resolved, qrels, run, k=headline_k)
        _check_strategy_did_something(config, pipeline, len(resolved.items), span_log)

        warnings = WarningLog()
        warnings.extend(pipeline.warnings)
        warnings.extend(anchor_log)
        warnings.extend(span_log)

        unresolved = sum(1 for item in resolved if item.anchors and not item.is_answerable)
        if not qrels:
            warnings.add(
                WarningCode.GOLD_SPAN_UNREACHABLE,
                f"{config.label} could not resolve a single piece of evidence, so every "
                "metric here is zero for reasons that have nothing to do with retrieval",
                severity=Severity.INVALID,
                stage="score",
                subject=config.label,
            )

        elapsed = time.perf_counter() - started
        cost = self.cost_model.estimate(
            embedder=config.embedder,
            index_tokens=pipeline.embed_tokens,
            query_tokens_per_query=_mean_query_tokens(resolved),
            compute_seconds=elapsed,
        )

        return RunResult(
            config=config,
            metrics=metrics,
            timings=pipeline.timings,
            cost=cost,
            warnings=warnings,
            chunk_count=len(pipeline.chunks),
            index_bytes=pipeline.index_bytes,
            scored_queries=len(qrels),
            unresolved_gold=unresolved,
            run=run,
            per_query=scores,
            by_type=by_type,
            failures=failures,
        )

    # -- a whole matrix ------------------------------------------------------

    def run(
        self,
        matrix: Matrix,
        evalset: EvalSet,
        *,
        mode: SweepMode | str = SweepMode.OFAT,
        budget_seconds: float | None = None,
        budget_usd: float | None = None,
        on_progress: Progress | None = None,
    ) -> Results:
        """Run a matrix in the chosen mode."""
        chosen = SweepMode(mode)
        budget = Budget(seconds=budget_seconds, usd=budget_usd)
        _warn_if_unbounded(matrix, budget)

        if chosen is SweepMode.STAGED:
            return self._staged(matrix, evalset, budget, on_progress)

        configs, dropped = matrix.expand_with_dropped(chosen)
        results = self._flat(configs, evalset, chosen, budget, on_progress)

        unbounded = matrix.meta.get("unbounded_model_calls")
        if unbounded:
            results.warnings.add(
                WarningCode.BUDGET_REACHED,
                f"the {unbounded!r} strategy calls a model -- per question at query time, or "
                "per chunk while building the index -- and this sweep has no `budget_usd` or "
                "`budget_seconds`. Nothing here can tell you the bill in advance",
                severity=Severity.CAUTION,
                stage="run",
                subject=str(unbounded),
            )

        if dropped:
            results.warnings.add(
                WarningCode.IMPOSSIBLE_COMBINATION,
                f"{dropped} combination(s) in this matrix cannot be built and were skipped -- "
                "a dense index with no embedder has nothing to search. The axes you wrote are "
                "almost certainly what you meant; this is just the product of them that is not",
                severity=Severity.INFO,
                stage="run",
                dropped=dropped,
            )
        return results

    def _flat(
        self,
        configs: Sequence[Config],
        evalset: EvalSet,
        mode: SweepMode,
        budget: Budget,
        on_progress: Progress | None,
    ) -> Results:
        results = Results(mode=mode.value)
        budget.start()

        for index, config in enumerate(configs, start=1):
            spent = budget.exceeded()
            if spent is not None:
                results.warnings.add(
                    WarningCode.BUDGET_REACHED,
                    f"stopped after {index - 1} of {len(configs)} configurations: {spent}. "
                    "The leaderboard below is partial",
                    severity=Severity.CAUTION,
                    stage="run",
                    completed=index - 1,
                    planned=len(configs),
                )
                break

            if on_progress:
                on_progress(index, len(configs), config)
            result = self.run_one(config, evalset)
            results.runs.append(result)
            results.warnings.extend(result.warnings)
            budget.charge(result, len(evalset.items))

        results.cache_summary = self.stats.summary()
        results.warnings.extend(self.cost_model.warnings)
        return results

    def _staged(
        self,
        matrix: Matrix,
        evalset: EvalSet,
        budget: Budget,
        on_progress: Progress | None,
    ) -> Results:
        """Pick the best value on each axis in turn, freezing it before moving on.

        Cheapest way to a good configuration and the one most people want. It is also
        conditional on the order the axes were swept in, and it cannot see interactions --
        so it says so, rather than presenting its answer as if it had searched the space.
        """
        results = Results(mode="staged")
        budget.start()
        current = matrix.baseline()
        seen: dict[Config, RunResult] = {}

        for axis in AXIS_ORDER:
            candidates = matrix.stage_configs(axis, current)
            if len(candidates) < 2:
                continue

            for position, config in enumerate(candidates, start=1):
                spent = budget.exceeded()
                if spent is not None:
                    results.warnings.add(
                        WarningCode.BUDGET_REACHED,
                        f"{spent} during the {axis!r} stage. Later axes were never swept at all",
                        severity=Severity.CAUTION,
                        stage="run",
                    )
                    results.cache_summary = self.stats.summary()
                    return results

                if config in seen:
                    continue
                if on_progress:
                    on_progress(position, len(candidates), config)
                result = self.run_one(config, evalset)
                seen[config] = result
                results.runs.append(result)
                results.warnings.extend(result.warnings)
                budget.charge(result, len(evalset.items))

            best = max(
                (seen[c] for c in candidates if c in seen),
                key=lambda r: r.metric(self.headline),
                default=None,
            )
            if best is not None:
                current = best.config

        varying = matrix.varying_axes
        if len(varying) > 1:
            results.warnings.add(
                WarningCode.NON_DETERMINISTIC_STAGE,
                f"staged mode fixed {', '.join(varying)} one axis at a time, in that order. "
                "It never tried the combinations it skipped, so if two of those axes "
                "interact the winner here may not be the best configuration in the matrix. "
                "Run factorial mode to find out",
                severity=Severity.CAUTION,
                stage="run",
                axes=list(varying),
            )

        results.cache_summary = self.stats.summary()
        results.warnings.extend(self.cost_model.warnings)
        results.meta["final"] = current.as_dict()
        return results


def _warn_if_unbounded(matrix: Matrix, budget: Budget) -> None:
    """Say so when a sweep can call a model an unknown number of times with no ceiling.

    An agentic strategy decides how many searches -- and therefore how many model calls -- each
    question needs. Multiply that by an eval set and a matrix and there is no number anybody can
    work out in advance. Every other axis has a cost you can estimate before starting; this one
    does not, which is exactly why it deserves a limit and a warning when it has none.

    A warning rather than a refusal: it is the user's money and they may well mean it.
    """
    if budget.usd is not None or budget.seconds is not None:
        return

    from contextgrid.ingest import get_ingester
    from contextgrid.retrieve import get_retriever

    for axis, resolve in (("retrieval", get_retriever), ("ingestion", get_ingester)):
        for value in getattr(matrix, axis):
            if value is None:
                continue
            try:
                strategy = resolve(value)
            except Exception:  # pragma: no cover - a bad spec fails later, more helpfully
                continue
            if getattr(strategy, "uses_model", False):
                matrix.meta["unbounded_model_calls"] = strategy.name
                return


def _check_strategy_did_something(
    config: Config, pipeline: Any, questions: int, log: WarningLog
) -> None:
    """Say when a retrieval strategy never actually differed from plain search.

    A strategy that decomposes multi-part questions does nothing at all on an eval set where
    every question has one part. Its row then matches `simple` exactly -- and read off a
    leaderboard that looks like a measured tie, as though the strategy had been tried and found
    not to help. It had not been tried.

    The distinction matters more than it sounds: the honest conclusion is "this eval set cannot
    tell you", and the fix is a better eval set, not a different strategy.
    """
    strategy = getattr(pipeline, "retrieval", None)
    trace = getattr(pipeline, "trace", None)
    if strategy is None or trace is None or not questions:
        return
    if getattr(strategy, "name", "simple") == "simple":
        return

    # One search per question is exactly what plain search does.
    if trace.searches <= questions and trace.model_calls == 0:
        log.add(
            WarningCode.NON_DETERMINISTIC_STAGE,
            f"the {strategy.name!r} retrieval strategy behaved identically to plain search on "
            f"all {questions} questions, so its score is the same as `simple` by construction "
            "rather than by measurement. This eval set cannot tell you whether it helps",
            severity=Severity.CAUTION,
            stage="retrieve",
            subject=strategy.name,
            questions=questions,
            searches=trace.searches,
        )


def _character_metrics(
    evalset: EvalSet,
    run: Mapping[str, Sequence[str]],
    chunks: Mapping[str, Any],
    k: int,
) -> dict[str, float]:
    """Character-level precision and recall over the retrieved context.

    Chunk-level recall can be 1.0 while character precision is 0.04, which means the right
    evidence arrived buried in twenty-five times its weight in irrelevant text. Every
    generation call then pays for that, and no chunk-level metric shows it.
    """
    precisions: list[float] = []
    recalls: list[float] = []

    for item in evalset:
        if not item.is_answerable:
            continue
        retrieved = [chunks[cid] for cid in list(run.get(item.id, ()))[:k] if cid in chunks]
        precisions.append(character_precision(item, retrieved))
        recalls.append(character_recall(item, retrieved))

    if not precisions:
        return {}
    return {
        f"char_precision@{k}": sum(precisions) / len(precisions),
        f"char_recall@{k}": sum(recalls) / len(recalls),
    }


def _slice_by_type(
    evalset: EvalSet, scores: Mapping[str, float], metric: str
) -> dict[str, dict[str, float]]:
    """The headline metric, split by kind of question."""
    grouped: dict[str, list[float]] = {}
    for item in evalset:
        if item.id not in scores:
            continue
        grouped.setdefault(item.qtype or "unlabelled", []).append(scores[item.id])
    if not grouped:
        return {}
    return {metric: {label: sum(v) / len(v) for label, v in sorted(grouped.items())}}


def _mean_query_tokens(evalset: EvalSet) -> float:
    """Rough query length, for costing the per-query side of a hosted embedder."""
    if not len(evalset):
        return 0.0
    return sum(len(item.question.split()) for item in evalset) / len(evalset)


def estimate_cost(
    matrix: Matrix,
    corpus: Corpus,
    *,
    mode: SweepMode | str = SweepMode.OFAT,
    cost_model: CostModel | None = None,
) -> dict[str, Any]:
    """What a sweep will cost before it runs.

    Deliberately crude on tokens -- a word count, not a tokenizer pass -- because the point
    is to catch "this will cost forty dollars" before somebody starts it, not to be exact.
    """
    model = cost_model or CostModel()
    configs = matrix.expand(mode)
    characters = sum(source.size_bytes or 0 for source in corpus)
    approximate_tokens = int(characters / 4)

    total = 0.0
    for config in configs:
        breakdown = model.estimate(
            embedder=config.embedder,
            index_tokens=approximate_tokens,
            query_tokens_per_query=12,
        )
        total += breakdown.index_usd

    return {
        "configurations": len(configs),
        "mode": SweepMode(mode).value,
        "shape": matrix.shape(),
        "approximate_index_tokens": approximate_tokens,
        "estimated_usd": round(total, 4),
    }
