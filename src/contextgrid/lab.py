"""The front door.

Everything below this is composable and explicit. This is the thing you reach for when you
just want an answer:

    import contextgrid as cg

    lab = cg.Lab(corpus="./contracts")
    lab.grid(chunker=["recursive:512", "structural:512"], index=["dense", "bm25", "hybrid"])
    print(lab.estimate())

    results = lab.run(evalset)
    print(results.summary())
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contextgrid.cache.store import Cache, MemoryCache
from contextgrid.core.documents import MediaType
from contextgrid.core.evalset import EvalSet
from contextgrid.corpus import Corpus, CorpusFingerprint, fingerprint
from contextgrid.cost.model import CostModel
from contextgrid.evalset.classify import Classifier
from contextgrid.evalset.filters import FilterChain, FilterResult, default_filters
from contextgrid.evalset.generate import (
    Generation,
    KeywordProbeGenerator,
    LLMQuestionGenerator,
    QuestionGenerator,
    generate,
)
from contextgrid.evalset.llm import LLM, get_llm
from contextgrid.evalset.quality import EvalSetQuality, assess
from contextgrid.evalset.review import ReviewQueue
from contextgrid.grid.matrix import Matrix, SweepMode, matrix
from contextgrid.grid.runner import Runner, estimate_cost
from contextgrid.pipeline import Config, build
from contextgrid.report.results import Results


@dataclass(slots=True)
class Lab:
    """A corpus, a matrix over it, and the runs that came out."""

    corpus: Corpus
    cache: Cache = field(default_factory=MemoryCache)
    cost_model: CostModel = field(default_factory=CostModel)
    llm: Any = None
    seed: int = 0
    _matrix: Matrix = field(default_factory=Matrix)

    def __init__(
        self,
        corpus: Corpus | str | Path | dict[str, str],
        *,
        cache: Cache | None = None,
        machine_usd_per_hour: float = 0.0,
        model: str | Any | None = None,
        seed: int = 0,
    ) -> None:
        """`model` and `seed` mirror the `run:` section of a config file.

        One model for the whole lab, as in the YAML: query transforms, agentic retrieval, the
        LLM-backed ingestion strategies and the generation judge all share it. The alternative
        is four places to set a key and four prices to reconcile.

        Without it, four of the six transforms, `agentic` retrieval, four of the eight ingestion
        strategies and the `llm` generator cannot be built at all -- so the Python API could
        reach barely half the plugins the YAML could, and nothing said so.
        """
        self.corpus = _as_corpus(corpus)
        # `cache or MemoryCache()` looked equivalent and was not. `DiskCache.__len__` counts
        # the entries on disk, so a cache directory that has not been written to yet is
        # falsy -- and that is every first run, on every machine. The `DiskCache` was
        # silently dropped, nothing was written, the directory stayed empty, and so the next
        # process dropped it too. `cg.Lab(corpus, cache=cg.DiskCache(root=...))` therefore
        # never reused anything across processes at all, which is the only reason to pass
        # one. Only `None` means "no cache was chosen"; an empty cache is still a choice.
        self.cache = MemoryCache() if cache is None else cache
        self.cost_model = CostModel(machine_usd_per_hour=machine_usd_per_hour)
        self.seed = seed
        self._matrix = Matrix()

        if model is None or not isinstance(model, str):
            self.llm = model
        else:
            from contextgrid.evalset.llm import get_llm

            self.llm = get_llm(model)

    # -- looking before you leap ---------------------------------------------

    def fingerprint(self, parser: str = "markdown") -> CorpusFingerprint:
        """Profile the corpus, using one parser to read it.

        Worth doing before configuring anything. The hints it produces name the axes likely
        to matter on *these* documents, which turns a blank matrix into a decision.
        """
        config = Config(parser=parser)
        parses = build(config, self.corpus, cache=self.cache).parses
        return fingerprint(self.corpus, parses)

    # -- ground truth --------------------------------------------------------

    def draft_evalset(
        self,
        *,
        llm: LLM | str | None = None,
        parser: str = "markdown",
        chunker: str = "recursive:512",
        sample: int = 50,
        questions_per_chunk: int = 1,
        seed: int = 0,
    ) -> Generation:
        """Draft an eval set from the corpus.

        With an `llm`, questions are written by a model and each one has to quote the passage
        that answers it. Without one, you get keyword probes -- useful for checking a pipeline
        is wired up and not a substitute for questions, which `KeywordProbeGenerator` says at
        length.

        The result is a draft. Run it through `filter_evalset` and then `review`.
        """
        chunks = build(Config(parser=parser, chunker=chunker), self.corpus, cache=self.cache).chunks

        generator: QuestionGenerator
        if llm is None:
            generator = KeywordProbeGenerator(seed=seed)
        else:
            generator = LLMQuestionGenerator(
                llm=get_llm(llm), questions_per_chunk=questions_per_chunk
            )

        drafted = generate(chunks, generator, sample=sample, seed=seed)
        return Generation(
            evalset=Classifier().label_set(drafted.evalset),
            warnings=drafted.warnings,
            chunks_sampled=drafted.chunks_sampled,
            chunks_skipped=drafted.chunks_skipped,
        )

    def filter_evalset(
        self,
        evalset: EvalSet,
        *,
        baseline_scores: dict[str, float] | None = None,
        llm: LLM | str | None = None,
        chain: FilterChain | None = None,
    ) -> FilterResult:
        """Drop the questions that would make a comparison meaningless.

        Pass `baseline_scores` from a first run to remove questions every configuration
        already answers, and an `llm` to remove ones answerable without reading anything.
        """
        from contextgrid.evalset.llm import answerer_from

        answerer = answerer_from(get_llm(llm)) if llm is not None else None
        filters = chain or default_filters(baseline_scores=baseline_scores, answerer=answerer)
        return filters.run(evalset)

    def review(self, evalset: EvalSet, *, skip_reviewed: bool = True) -> ReviewQueue:
        """A queue of questions to accept, fix or drop, one keystroke each."""
        return ReviewQueue.from_evalset(evalset, skip_reviewed=skip_reviewed)

    def assess(
        self, evalset: EvalSet, *, baseline_scores: dict[str, float] | None = None
    ) -> EvalSetQuality:
        """What this eval set can and cannot support, including the smallest difference it
        could detect."""
        return assess(evalset, baseline_scores=baseline_scores)

    # -- defining the experiment ---------------------------------------------

    def grid(
        self,
        parser: str | Sequence[str] = "markdown",
        chunker: str | Sequence[str] = "recursive:512",
        embedder: str | Sequence[str | None] | None = "tfidf",
        index: str | Sequence[str] = "dense",
        transform: str | Sequence[str | None] | None = None,
        reranker: str | Sequence[str | None] | None = None,
        candidates: int | Sequence[int] = 50,
        k: int = 10,
        *,
        ingestion: str | Sequence[str | None] | None = None,
        retrieval: str | Sequence[str | None] | None = None,
        generator: str | Sequence[str | None] | None = None,
    ) -> Matrix:
        """Set the axes. Any of them takes a single value or a list.

        All ten, matching `grid:` in a config file. The three keyword-only ones arrived after
        the positional signature was public, and go at the end for the same reason `ingestion`
        sits last on `Config`: shifting a positional argument silently changes what every call
        anybody has already written means.
        """
        self._matrix = matrix(
            parser=parser,
            chunker=chunker,
            embedder=embedder,
            index=index,
            transform=transform,
            reranker=reranker,
            candidates=candidates,
            k=k,
            ingestion=ingestion,
            retrieval=retrieval,
            generator=generator,
        )
        return self._matrix

    @property
    def matrix(self) -> Matrix:
        return self._matrix

    def estimate(self, mode: SweepMode | str = SweepMode.OFAT) -> dict[str, Any]:
        """How many configurations, and roughly what they will cost."""
        return estimate_cost(self._matrix, self.corpus, mode=mode, cost_model=self.cost_model)

    # -- running it ----------------------------------------------------------

    def run(
        self,
        evalset: EvalSet,
        *,
        mode: SweepMode | str = SweepMode.OFAT,
        budget_seconds: float | None = None,
        budget_usd: float | None = None,
        headline: str = "recall@5",
        metrics: Sequence[str] = (),
        on_progress: Any = None,
    ) -> Results:
        """Run the matrix and score every configuration.

        `budget_usd` is here for the same reason it is in a config file: an agentic strategy or
        an LLM generator decides its own number of model calls, so a sweep containing one has no
        ceiling anybody can work out in advance.

        `metrics` mirrors `run.metrics` in a config file -- extra registered metrics to compute
        alongside the built-ins and `headline`'s own, e.g. after registering a custom `Metric`
        into `contextgrid.score.METRICS` (see `docs/internals/extending.md`).
        """
        runner = Runner(
            corpus=self.corpus,
            cache=self.cache,
            cost_model=self.cost_model,
            headline=headline,
            extra_metrics=tuple(metrics),
            llm=self.llm,
            seed=self.seed,
        )
        return runner.run(
            self._matrix,
            evalset,
            mode=mode,
            budget_seconds=budget_seconds,
            budget_usd=budget_usd,
            on_progress=on_progress,
        )


def _as_corpus(value: Corpus | str | Path | dict[str, str]) -> Corpus:
    if isinstance(value, Corpus):
        return value
    if isinstance(value, dict):
        return Corpus.from_texts(value, media_type=MediaType.MARKDOWN)
    return Corpus.from_dir(value)
