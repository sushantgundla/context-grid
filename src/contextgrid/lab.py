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
    _matrix: Matrix = field(default_factory=Matrix)

    def __init__(
        self,
        corpus: Corpus | str | Path | dict[str, str],
        *,
        cache: Cache | None = None,
        machine_usd_per_hour: float = 0.0,
    ) -> None:
        self.corpus = _as_corpus(corpus)
        self.cache = cache or MemoryCache()
        self.cost_model = CostModel(machine_usd_per_hour=machine_usd_per_hour)
        self._matrix = Matrix()

    # -- looking before you leap ---------------------------------------------

    def fingerprint(self, parser: str = "markdown") -> CorpusFingerprint:
        """Profile the corpus, using one parser to read it.

        Worth doing before configuring anything. The hints it produces name the axes likely
        to matter on *these* documents, which turns a blank matrix into a decision.
        """
        config = Config(parser=parser)
        parses = build(config, self.corpus, cache=self.cache).parses
        return fingerprint(self.corpus, parses)

    # -- defining the experiment ---------------------------------------------

    def grid(
        self,
        parser: str | Sequence[str] = "markdown",
        chunker: str | Sequence[str] = "recursive:512",
        embedder: str | Sequence[str | None] | None = "tfidf",
        index: str | Sequence[str] = "dense",
        k: int = 10,
    ) -> Matrix:
        """Set the axes. Any of them takes a single value or a list."""
        self._matrix = matrix(parser=parser, chunker=chunker, embedder=embedder, index=index, k=k)
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
        headline: str = "recall@5",
        on_progress: Any = None,
    ) -> Results:
        """Run the matrix and score every configuration."""
        runner = Runner(
            corpus=self.corpus,
            cache=self.cache,
            cost_model=self.cost_model,
            headline=headline,
        )
        return runner.run(
            self._matrix,
            evalset,
            mode=mode,
            budget_seconds=budget_seconds,
            on_progress=on_progress,
        )


def _as_corpus(value: Corpus | str | Path | dict[str, str]) -> Corpus:
    if isinstance(value, Corpus):
        return value
    if isinstance(value, dict):
        return Corpus.from_texts(value, media_type=MediaType.MARKDOWN)
    return Corpus.from_dir(value)
