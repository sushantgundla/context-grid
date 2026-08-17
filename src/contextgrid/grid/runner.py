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
from contextgrid.core.evalset import EvalSet, Qrels
from contextgrid.core.warnings import Severity, WarningCode, WarningLog
from contextgrid.corpus import Corpus
from contextgrid.cost.metering import MeteredLLM, exact_tokenizer_or_none
from contextgrid.cost.model import CostModel
from contextgrid.diagnose.taxonomy import diagnose
from contextgrid.generate import GenerationReport, score_answer
from contextgrid.grid.matrix import AXIS_ORDER, Matrix, SweepMode, deduplicate
from contextgrid.pipeline import BuiltPipeline, Config, build, build_qrels, resolve_evalset
from contextgrid.report.results import Results, RunResult
from contextgrid.score.anchor import AnchorResolver
from contextgrid.score.metrics import BUILTIN_METRIC_NAMES, DEFAULT_KS, evaluate, per_query
from contextgrid.score.resolve import SpanResolver, character_precision, character_recall

Progress = Callable[[int, int, Config], None]

#: Distinguishes "not looked up yet" from "looked up, and there is no exact tokenizer".
_UNSET: Any = object()


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
        """Add what one configuration actually spent.

        `spent_now(queries)`, not `total_at(queries)`. The two ask different questions:
        `total_at` is "what would this cost to serve in production"; a spending limit is
        asking "how much money has this sweep burnt". `total_at` both over-counts, by
        projecting a serving rate over queries nobody ran, and under-counts, by omitting the
        judge entirely. With a local embedder and a hosted generator it came to zero however
        many dollars had just gone through -- which is why a positive `budget_usd` could never
        fire on the one axis that spends real money.
        """
        cost = getattr(result, "cost", None)
        if cost is not None:
            self.spent_usd += float(cost.spent_now(queries))

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
    #: Extra registered metrics to compute alongside the built-ins, from `run.metrics` in the
    #: config. The headline's own metric is always computed too -- see `metric_names` below.
    extra_metrics: tuple[str, ...] = ()
    #: Shared by every stage that needs a model: transforms, agentic retrieval, LLM-backed
    #: ingestion, and the generation judge.
    llm: Any = None
    #: Carried onto the results, so significance testing resamples with the seed the run
    #: recorded rather than a hidden zero.
    seed: int = 0
    #: Cached tokenizer for cost metering. `_UNSET` rather than `None` because `None` is a
    #: real answer here -- it means "tiktoken is not installed" and must not be retried on
    #: every configuration in the sweep.
    _tokenizer: Any = _UNSET

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

    def _token_counter(self) -> Any:
        """The tokenizer used to count what a model consumed, or `None`.

        Built once per configuration and handed to both meters. `None` when `tiktoken` is not
        installed, which downgrades the cost to approximate rather than failing a run that
        would otherwise have succeeded.
        """
        if self._tokenizer is _UNSET:
            self._tokenizer = exact_tokenizer_or_none()
        return self._tokenizer

    def _embedding_quality(self, pipeline: BuiltPipeline) -> float | None:
        """Score this embedder against this corpus, or `None` when there is nothing to score.

        Never raises. `assess` refuses corpora too small to describe the shape of -- three
        points have no shape -- and a diagnostic declining to answer must not take down a run
        whose retrieval numbers are perfectly good.
        """
        vectors = getattr(pipeline, "vectors", None)
        if vectors is None or pipeline.embedder is None or pipeline.ingested is None:
            return None
        try:
            from contextgrid.embed.quality import assess

            # The *indexed* units, because those are what was embedded. For a strategy that
            # indexes something narrower than it returns, scoring the returned passages would
            # assess vectors that were never built.
            return float(assess(pipeline.ingested.indexed, vectors).score)
        except Exception:
            return None

    @property
    def metric_names(self) -> tuple[str, ...]:
        """Every metric this runner computes: the built-ins, `extra_metrics`, and the
        headline's own -- guaranteed present the same way `ks` guarantees the headline's
        cut-off just above, so `Runner(headline="weighted_recall@5")` computes
        `weighted_recall` even when nobody added it to `extra_metrics` too.
        """
        metric_name, _, _ = self.headline.partition("@")
        names = {*BUILTIN_METRIC_NAMES, *self.extra_metrics}
        if metric_name:
            names.add(metric_name)
        return tuple(sorted(names))

    # -- one configuration ---------------------------------------------------

    def run_one(self, config: Config, evalset: EvalSet) -> RunResult:
        """Build a configuration, answer every question, and score it."""
        started = time.perf_counter()

        # Every model call this configuration makes goes through here, so it can be counted
        # and priced. A fresh wrapper per configuration, because the numbers are per-run --
        # sharing one across a sweep would charge the whole sweep's spend to every row.
        #
        # Generation and the judge are metered separately: one is what serving costs, the
        # other is what evaluating cost, and adding them together would tell somebody their
        # production system needs a judge call per question.
        metered_llm = MeteredLLM(self.llm, self._token_counter()) if self.llm is not None else None
        pipeline = build(
            config, self.corpus, cache=self.cache, stats=self.stats, llm=metered_llm or self.llm
        )

        # The evidence has to be located again in *this* parse. Two parsers produce
        # different text, so a span that was right for one is meaningless for the other.
        resolved, anchor_log = resolve_evalset(evalset, pipeline.parses, self.anchor_resolver)
        qrels, span_log = build_qrels(resolved, pipeline.chunks, self.span_resolver)

        run = pipeline.run_queries(resolved)

        # Built early so `evaluate()` can log into it directly: a custom metric that raises
        # is left out of `metrics` rather than crashing the run, and this is what says why a
        # column is missing instead of the gap speaking for itself.
        warnings = WarningLog()
        metrics = evaluate(qrels, run, ks=self.ks, metrics=self.metric_names, warnings=warnings)

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
        _check_chunking_can_be_seen(config, pipeline, warnings)

        # A no-op when `config.generator` is unset: no assembly, no model call, no cost, same
        # as before this axis existed. Folds `faithfulness` and `answer_relevancy` in when a
        # judge ran, which is what lets DIMENSION_METRICS["generation"] find them later.
        judge_llm = MeteredLLM(self.llm, self._token_counter()) if self.llm is not None else None
        generation_metrics, generation_log, answers = self._score_generation(
            pipeline, resolved, run, qrels, judge_llm=judge_llm
        )
        metrics.update(generation_metrics)

        warnings.extend(pipeline.warnings)
        warnings.extend(anchor_log)
        warnings.extend(span_log)
        warnings.extend(generation_log)

        # `is_resolved`, not `is_answerable`: "answerable" now means the item carries evidence
        # in either form, so an anchor that this parse lost is still answerable and this count
        # would be zero on every run. What is wanted is spans -- evidence located *here*.
        unresolved = sum(1 for item in resolved if item.anchors and not item.is_resolved)

        # The parse dimension's score: of the evidence somebody quoted, how much could this
        # parser actually find in its own output? A parser that drops a table, mangles a
        # ligature or reorders columns loses the anchor, and every retrieval number below
        # becomes a measurement of the parse rather than of retrieval.
        #
        # `DIMENSION_METRICS["parse"]` has always asked for this name and nothing has ever
        # emitted it, so the parse dimension could not be scored at all -- the composite
        # reported "not measured" on every run, forever. Left out entirely when no item
        # carries an anchor, rather than scored 1.0: a run with nothing to resolve has not
        # demonstrated a perfect parse, it has demonstrated nothing.
        with_anchors = sum(1 for item in resolved if item.anchors)
        if with_anchors:
            metrics["evidence_resolvable"] = (with_anchors - unresolved) / with_anchors

        # The embed dimension's score: can this embedder tell anything apart on this corpus?
        # Measured from the vectors the run already built, so it costs one pass over an array
        # that is sitting in memory rather than a second embedding of anything.
        #
        # `DIMENSION_METRICS["embed"]` has always asked for this and no sweep has ever
        # produced it -- the number existed only in `contextgrid.embed.assess`, which nothing
        # in a sweep called. Absent rather than zero when there is no embedder at all, since
        # `bm25` has not embedded badly, it has not embedded.
        quality = self._embedding_quality(pipeline)
        if quality is not None:
            metrics["embedding_quality"] = quality

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
            # The object, not a name: `price_key` asks an instance what it is called, and the
            # LLM is the only place that knows which model `run.model` resolved to.
            model=self.llm,
            generation=metered_llm.usage if metered_llm is not None else None,
            judge=judge_llm.usage if judge_llm is not None else None,
            queries=len(resolved),
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
            answers=answers,
            retrieval={
                "searches": pipeline.trace.searches,
                "model_calls": pipeline.trace.model_calls,
            },
            by_type=by_type,
            failures=failures,
            seed=self.seed,
        )

    # -- generation ------------------------------------------------------------

    def _score_generation(
        self,
        pipeline: BuiltPipeline,
        evalset: EvalSet,
        run: Mapping[str, Sequence[str]],
        qrels: Qrels,
        judge_llm: object | None = None,
    ) -> tuple[dict[str, float], WarningLog, dict[str, dict[str, Any]]]:
        """Answer every question with the configured generator, and score it.

        Returns nothing at all when `pipeline.generator` is unset -- the cheapest possible
        no-op, which is what `None` on this axis has to mean.

        A generator that fails on one question must not fail the sweep, the same rule
        `retrieve.agentic._plan` already follows for a planner that refuses to cooperate: the
        failure is recorded on that question and the rest of the eval set still gets scored.
        """
        log = WarningLog()
        generator = pipeline.generator
        if generator is None:
            return {}, log, {}

        chunks_by_id = pipeline.chunk_by_id()
        report = GenerationReport(generator=generator.name)
        answers: dict[str, dict[str, Any]] = {}
        judge = self._generation_judge(judge_llm)
        judge_scores: dict[str, list[float]] = {}

        for item in evalset:
            try:
                answer, context = pipeline.answer(item.question, run.get(item.id, ()))
            except Exception as error:
                log.add(
                    WarningCode.GENERATION_FAILED,
                    f"the {generator.name!r} generator failed on {item.id!r}: {error}. That "
                    "question was skipped rather than failing the run",
                    severity=Severity.CAUTION,
                    stage="generate",
                    subject=item.id,
                )
                continue

            # Gold ids, not gold spans: the qrels are already resolved to this configuration's
            # own chunking, which a fresh span lookup would have to redo for no benefit.
            gold_ids = {cid for cid, grade in qrels.get(item.id, {}).items() if grade > 0}
            gold_chunks = [chunks_by_id[cid] for cid in gold_ids if cid in chunks_by_id]
            report.scores.append(score_answer(item, answer, context, gold_chunks))
            # Kept, not just scored. A sweep with a generator spends real money and used to
            # save nothing you could read afterwards -- no answer, no per-question judgement,
            # so there was no way to check whether a faithfulness of 0.83 meant one bad answer
            # or fifteen mediocre ones.
            answers[item.id] = {
                "answer": answer.text,
                "chunk_ids": [chunk.id for chunk in context.chunks],
            }

            if judge is None:
                continue
            try:
                judged = judge.score(
                    query_id=item.id,
                    question=item.question,
                    answer=answer.text,
                    contexts=[chunk.text for chunk in context.chunks],
                    reference=item.answer,
                )
            except Exception as error:
                log.add(
                    WarningCode.GENERATION_FAILED,
                    f"the generation judge failed on {item.id!r}: {error}. Its faithfulness "
                    "and answer_relevancy were skipped for this question",
                    severity=Severity.CAUTION,
                    stage="generate",
                    subject=item.id,
                )
                continue
            for name, value in judged.scores.items():
                judge_scores.setdefault(name, []).append(value)
            answers[item.id]["judge"] = dict(judged.scores)

        metrics = dict(report.metrics())
        metrics.update({name: sum(v) / len(v) for name, v in judge_scores.items() if v})
        return metrics, log, answers

    def _generation_judge(self, llm: object | None = None) -> Any:
        """The DeepEval judge, built once per configuration, or `None` when it cannot run.

        Checked here rather than inside the per-question loop: a missing `deepeval` install
        is one clear failure, and re-discovering it on every question would turn that one
        failure into as many warnings as there are questions. `self.llm` is `run.model` --
        the same model already paying for generation, transforms and agentic retrieval, so a
        sweep with a generator configured gets `faithfulness` and `answer_relevancy` without
        a second key or an unpriced call to whatever DeepEval defaults to.
        """
        # `llm` is the metered wrapper around `self.llm`, so judge calls are counted
        # separately from generation calls -- evaluation cost and serving cost are different
        # questions and must not be added together.
        model = llm if llm is not None else self.llm
        if model is None:
            return None
        try:
            import deepeval  # noqa: F401
        except ImportError:
            return None

        from contextgrid.generate import GenerationJudge

        return GenerationJudge(llm=model)

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

        _warn_if_approximate_alone(matrix, results)
        self._warn_if_at_ceiling(results)

        unbounded = matrix.meta.get("unbounded_model_calls")
        if unbounded:
            results.warnings.add(
                WarningCode.BUDGET_REACHED,
                f"the {unbounded!r} plugin calls a model -- per question at query time, or "
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

    def _warn_if_at_ceiling(self, results: Results) -> None:
        """Say so when every configuration scored full marks and the sweep ranked nothing.

        The most expensive way to learn nothing. If the baseline already answers every question
        perfectly, no arm has any headroom, the whole grid ties at 1.000, and the leaderboard
        looks like a clean result rather than the absence of one -- a blind evaluator hit
        exactly this and called it "the one warning this tool most needs and does not have".

        The cause is nearly always an eval set whose questions are too easy for the corpus,
        not a grid of configurations that are genuinely indistinguishable. The fix is harder
        questions or a smaller cut-off, so the warning says both.

        Only when *several* configurations tie at the top: one configuration scoring 1.000 is a
        result, not a ceiling, because nothing was being compared.
        """
        scores = [run.metric(self.headline) for run in results.runs if run.has(self.headline)]
        if len(scores) < 2 or min(scores) < 0.999:
            return

        results.warnings.add(
            WarningCode.EVALSET_AT_CEILING,
            f"every one of the {len(scores)} configurations scored "
            f"{self.headline} = {scores[0]:.3f}. Nothing here can be ranked: the eval set is "
            "answered perfectly by all of them, so the sweep measured no difference rather "
            "than finding none. Ask harder questions, or compare at a smaller cut-off "
            f"({self.headline.partition('@')[0]}@1) where there is room to separate them",
            severity=Severity.CAUTION,
            stage="score",
            configurations=len(scores),
        )

    def _flat(
        self,
        configs: Sequence[Config],
        evalset: EvalSet,
        mode: SweepMode,
        budget: Budget,
        on_progress: Progress | None,
    ) -> Results:
        results = Results(mode=mode.value, seed=self.seed)
        budget.start()

        for index, config in enumerate(configs, start=1):
            spent = budget.exceeded()
            if spent is not None:
                # Two different outcomes, and they need different words. "The leaderboard
                # below is partial" is simply false when the count is zero: there is no
                # leaderboard, nothing was compared, and the budget is the whole story.
                # `budget_usd: 0.0` -- documented as "already spent" -- lands there every
                # time, and so does any ceiling too small for a single configuration.
                completed = index - 1
                if completed:
                    detail = (
                        f"stopped after {completed} of {len(configs)} configurations: "
                        f"{spent}. The leaderboard below is partial"
                    )
                else:
                    detail = (
                        f"none of the {len(configs)} configurations ran: {spent}. Nothing was "
                        "measured, so there is no leaderboard rather than an empty one"
                    )
                results.warnings.add(
                    WarningCode.BUDGET_REACHED,
                    detail,
                    severity=Severity.CAUTION,
                    stage="run",
                    completed=completed,
                    planned=len(configs),
                )
                break

            if on_progress:
                on_progress(index, len(configs), config)
            result = self.run_one(config, evalset)
            results.runs.append(result)
            # `extend_unique`, not `extend`: half of what a run logs is a fact about the eval
            # set rather than about the configuration, and every configuration rediscovers it.
            # The run's own log still holds its full copy -- only the report's is collapsed.
            results.warnings.extend_unique(result.warnings)
            budget.charge(result, len(evalset.items))

        results.cache_summary = self.stats.summary()
        results.warnings.extend_unique(self.cost_model.warnings)
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

        Progress counts the whole sweep rather than each stage. It used to restart at 1 on
        every axis and count against that axis's own length, so a five-run sweep announced as
        five runs reported `[1/3] [2/3] [3/3] [2/2] [2/2]`: two different configurations both
        called the second of two, and a denominator that moved three times while the header's
        number never moved at all.
        """
        results = Results(mode="staged", seed=self.seed)
        budget.start()
        current = matrix.baseline()
        seen: dict[Config, RunResult] = {}

        # The same number `ExperimentConfig.describe` prints as "N to run in staged mode",
        # from the same call, so the counter and the header agree on the ordinary sweep rather
        # than agreeing by coincidence. It is a plan and not a count -- see `Matrix.expand`,
        # and the warning at the bottom of this method for what happens when it is missed.
        announced = matrix.count(SweepMode.STAGED)
        total = announced
        completed = 0

        def run_stage(candidates: Sequence[Config]) -> str | None:
            """Run whatever this stage has not run already, or say why the sweep stopped.

            The denominator is raised here, once, when the sweep learns this stage holds more
            work than the plan allowed for -- rather than once per line, which would tick it
            up under the reader as they watched.
            """
            nonlocal completed, total
            total = max(total, completed + sum(1 for c in candidates if c not in seen))
            for config in candidates:
                spent = budget.exceeded()
                if spent is not None:
                    return spent

                if config in seen:
                    continue
                completed += 1
                if on_progress:
                    on_progress(completed, total, config)
                result = self.run_one(config, evalset)
                seen[config] = result
                results.runs.append(result)
                results.warnings.extend_unique(result.warnings)
                budget.charge(result, len(evalset.items))
            return None

        for axis in AXIS_ORDER:
            candidates = matrix.stage_configs(axis, current)
            if len(candidates) < 2:
                continue

            spent = run_stage(candidates)
            if spent is not None:
                results.warnings.add(
                    WarningCode.BUDGET_REACHED,
                    f"{spent} during the {axis!r} stage. Later axes were never swept at all",
                    severity=Severity.CAUTION,
                    stage="run",
                )
                results.cache_summary = self.stats.summary()
                return results

            best = max(
                (seen[c] for c in candidates if c in seen),
                key=lambda r: r.metric(self.headline),
                default=None,
            )
            if best is not None:
                current = best.config

        if not seen:
            # Every axis holds a single configuration, so every stage above was skipped and
            # the sweep ran nothing at all -- an empty leaderboard printed directly under a
            # header promising one run. A matrix with nothing to compare still describes a
            # configuration, and measuring it is what the header said would happen.
            only, _ = deduplicate([current])
            spent = run_stage(only)
            if spent is not None:
                results.warnings.add(
                    WarningCode.BUDGET_REACHED,
                    f"{spent} before the only configuration in this matrix could run. Nothing "
                    "was measured, so there is no leaderboard rather than an empty one",
                    severity=Severity.CAUTION,
                    stage="run",
                )
                results.cache_summary = self.stats.summary()
                return results
            if only:
                current = only[0]

        if completed != announced:
            results.warnings.add(
                WarningCode.NON_DETERMINISTIC_STAGE,
                f"staged mode ran {completed} configuration(s) where {announced} were planned. "
                "The plan costs out varying each axis around the baseline; staged varies each "
                "axis around the winner of the stage before it, and an axis can be worth "
                "sweeping against one winner and meaningless against another -- candidate "
                "depth does nothing at all until a reranker is frozen in front of it. Neither "
                "number is wrong: a staged sweep cannot be counted until it has run",
                severity=Severity.INFO,
                stage="run",
                planned=announced,
                ran=completed,
            )

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
        results.warnings.extend_unique(self.cost_model.warnings)
        results.meta["final"] = current.as_dict()
        return results


def _warn_if_approximate_alone(matrix: Matrix, results: Results) -> None:
    """Say so when a sweep measures approximate search with no exact arm to judge it against.

    An approximate index returning 92% of what exhaustive search would have found is a perfectly
    good trade -- and only if somebody measured the 8%. On its own the number looks like a
    retrieval score, reads like one, and is one *plus* an unstated loss. `efSearch` and `nprobe`
    get tuned until the latency looks good, and this is the arm that would have said what it
    cost.
    """
    from contextgrid.index import get_index

    approximate: list[str] = []
    exact = False
    for value in matrix.index:
        try:
            index = get_index(value)
        except Exception:  # pragma: no cover - a bad spec fails later, more helpfully
            continue
        if getattr(index, "is_exact", True):
            exact = True
        else:
            approximate.append(str(value))

    if approximate and not exact:
        results.warnings.add(
            WarningCode.ANN_RECALL_LOSS,
            f"every index on this axis is approximate ({', '.join(sorted(set(approximate)))}) "
            "and none is exact, so nothing here measures what the approximation cost. Add "
            "`dense`, or `faiss:flat`, to find out",
            severity=Severity.CAUTION,
            stage="index",
        )


def _warn_if_unbounded(matrix: Matrix, budget: Budget) -> None:
    """Say so when a sweep can call a model an unknown number of times with no ceiling.

    An agentic strategy decides how many searches -- and therefore how many model calls -- each
    question needs. Multiply that by an eval set and a matrix and there is no number anybody can
    work out in advance. Every other axis has a cost you can estimate before starting; this one
    does not, which is exactly why it deserves a limit and a warning when it has none.

    The `generator` axis is not unbounded in the same sense -- one model call per question is
    entirely predictable -- but it is the single most expensive axis on the grid, and a model
    call per question times a whole eval set times a whole matrix is exactly the bill this
    warning exists to flag before it is spent rather than after.

    A warning rather than a refusal: it is the user's money and they may well mean it.
    """
    if budget.usd is not None or budget.seconds is not None:
        return

    from contextgrid.generate import MODEL_BACKED as GENERATOR_MODEL_BACKED
    from contextgrid.ingest import get_ingester
    from contextgrid.retrieve import RETRIEVERS, model_backed_retrievers

    # Checked by name rather than by building one. A model-backed strategy with no `run.model`
    # now refuses to build, and this ran with no model to give it -- so resolving an instance
    # here raised, the `except` below swallowed it, and the warning went quiet for the one
    # axis it exists for, on exactly the sweeps that were about to spend money.
    paid = model_backed_retrievers()
    for value in matrix.retrieval:
        if value is not None and RETRIEVERS.name_in(value) in paid:
            matrix.meta["unbounded_model_calls"] = RETRIEVERS.name_in(value)
            return

    # Ingestion strategies all build without a model, so an instance is still the honest
    # answer here: `uses_model` is read off the thing itself rather than a list to keep up to
    # date. The catch stays broad because a bad spec fails later, more helpfully than here.
    for value in matrix.ingestion:
        if value is None:
            continue
        try:
            strategy = get_ingester(value)
        except Exception:  # pragma: no cover - a bad spec fails later, more helpfully
            continue
        if getattr(strategy, "uses_model", False):
            matrix.meta["unbounded_model_calls"] = strategy.name
            return

    # Checked by name rather than by building one, for the same reason as `retrieval` above:
    # the `llm` generator cannot be constructed without a real `LLM`, so a try/except around
    # `get_generator` would just skip it, every time, silently.
    for value in matrix.generator:
        if value is None:
            continue
        name = value.partition(":")[0]
        if name in GENERATOR_MODEL_BACKED:
            matrix.meta["unbounded_model_calls"] = name
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


def _check_chunking_can_be_seen(config: Config, pipeline: Any, log: WarningLog) -> None:
    """Say when the chunker never cut anything, so the score is about documents.

    A chunk size above the length of the documents leaves every document whole. Retrieval then
    ranks documents, recall@5 out of a handful of them is close to free, and the number goes
    to 1.000 and stays there -- across every chunker, every index, every reranker. Read off a
    leaderboard, that is a clean sweep of tied configurations. It is actually a sweep in which
    the axis under test was never applied.

    `contextgrid profile` already knew: "The median document is 1,224 characters. Chunk sizes
    above that cannot differentiate, so sweep small sizes." But `profile` is a separate command
    nothing prompts you to run, and the leaderboard is what gets copied into a decision. The
    fact belongs next to the number it undermines.

    Deliberately measured against the parses rather than the corpus, because that is what the
    chunker was handed -- a document the parser dropped is not one the chunker declined to cut.
    """
    parses = getattr(pipeline, "parses", None)
    chunks = getattr(pipeline, "chunks", None)
    if not parses or not chunks:
        # No documents, or no chunks at all. `EMPTY_CHUNK_SET` and `GOLD_SPAN_UNREACHABLE`
        # cover the second case, and say more about it than this could.
        return

    documents = len(parses)
    if len(chunks) > documents:
        return

    log.add(
        WarningCode.ONE_CHUNK_PER_DOCUMENT,
        f"{config.chunker} produced {len(chunks)} chunk(s) from {documents} document(s), so "
        "each document is a single chunk and these scores rank documents rather than "
        "passages. The chunker axis cannot change a number it never touched -- sweep smaller "
        "sizes, or measure on longer documents",
        severity=Severity.CAUTION,
        # The chunker, not the whole config label. This is a fact about one chunker meeting
        # this corpus, and a five-row sweep holding four `recursive:512` arms re-derives the
        # identical fact four times -- `WarningLog.extend_unique` collapses those to one only
        # while the subject and the message agree, which is what keeps the real finding from
        # being buried under copies of itself.
        stage="chunk",
        subject=config.chunker,
        documents=documents,
        chunks=len(chunks),
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
        # Character metrics are measured against gold *spans*. An item whose anchor this parse
        # never found has none, so it has nothing to measure -- and scoring it as zero would
        # charge the retriever for a parse failure that `evidence_resolvable` already reports.
        if not item.is_resolved:
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
