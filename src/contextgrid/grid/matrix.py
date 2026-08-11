"""The experiment matrix, and the three ways to walk it.

Selecting several values on an axis multiplies the run count, which is how a curious
afternoon becomes a four-hour job. So the matrix knows its own size before anything runs, and
offers three ways to cover it.

**Full factorial** measures everything, including interactions between axes. It is also the
one that explodes: four axes with three values each is 81 configurations.

**One-factor-at-a-time** holds a baseline and varies one axis at a time. Linear rather than
exponential, directly interpretable -- "switching the chunker gained 0.08" -- and blind to
interactions. It is the right default because most of the time the axes really are close to
independent, and when they are not, that is itself worth discovering deliberately.

**Staged** picks the winner on one axis, freezes it, and moves on. Cheapest of the three and
the one most practitioners actually want. It can be wrong whenever axes interact, which the
runner says out loud rather than burying in a footnote.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from typing import Any

from contextgrid.core.errors import ContextGridError
from contextgrid.pipeline import Config

#: The order axes are swept in staged mode: earliest first.
#:
#: Deliberate rather than alphabetical. The parser decides what text exists at all, so
#: choosing it first is the only ordering where the later choices are being made about the
#: real corpus. Reranking comes last because it operates on whatever the rest produced --
#: and `generator` comes after that, because it operates on whatever *reranking* produced.
AXIS_ORDER: tuple[str, ...] = (
    "ingestion",
    "parser",
    "chunker",
    "embedder",
    "index",
    "transform",
    "retrieval",
    "reranker",
    "candidates",
    "generator",
)


#: Written out with a real multiplication sign, because "2 x 3" reads as a variable.
_TIMES = " \u00d7 "


class SweepMode(str, Enum):
    FACTORIAL = "factorial"
    OFAT = "ofat"
    STAGED = "staged"


class MatrixError(ContextGridError, ValueError):
    """A matrix cannot be expanded as asked."""


@dataclass(frozen=True, slots=True)
class Matrix:
    """The axes of an experiment, and the baseline that OFAT and staged vary from."""

    ingestion: tuple[str | None, ...] = (None,)
    parser: tuple[str, ...] = ("markdown",)
    chunker: tuple[str, ...] = ("recursive:512",)
    embedder: tuple[str | None, ...] = ("tfidf",)
    index: tuple[str, ...] = ("dense",)
    transform: tuple[str | None, ...] = (None,)
    retrieval: tuple[str | None, ...] = (None,)
    reranker: tuple[str | None, ...] = (None,)
    candidates: tuple[int, ...] = (50,)
    generator: tuple[str | None, ...] = (None,)
    k: int = 10
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for axis in AXIS_ORDER:
            values = getattr(self, axis)
            if not values:
                raise MatrixError(
                    f"the {axis!r} axis is empty. Give it at least one value, or leave it out "
                    "to use the default."
                )
        if self.k < 1:
            raise MatrixError(f"k must be at least 1, got {self.k}")

    # -- shape ---------------------------------------------------------------

    @property
    def axes(self) -> dict[str, tuple[Any, ...]]:
        return {axis: getattr(self, axis) for axis in AXIS_ORDER}

    @property
    def varying_axes(self) -> tuple[str, ...]:
        """Axes with more than one value. The only ones a sweep can learn anything about."""
        return tuple(axis for axis, values in self.axes.items() if len(values) > 1)

    def baseline(self) -> Config:
        """The first value on every axis. What OFAT and staged start from."""
        return Config(
            ingestion=self.ingestion[0],
            parser=self.parser[0],
            chunker=self.chunker[0],
            embedder=self.embedder[0],
            index=self.index[0],
            transform=self.transform[0],
            retrieval=self.retrieval[0],
            reranker=self.reranker[0],
            candidates=self.candidates[0],
            generator=self.generator[0],
            k=self.k,
        )

    def count(self, mode: SweepMode | str = SweepMode.OFAT) -> int:
        """How many configurations this mode will run, without building any of them."""
        return len(self.expand(mode))

    def shape(self) -> str:
        """The multiplication, written out: `2 x 3 x 1 x 2 = 12`."""
        sizes = [len(values) for values in self.axes.values()]
        product_of = 1
        for size in sizes:
            product_of *= size
        return f"{_TIMES.join(str(size) for size in sizes)} = {product_of}"

    # -- expansion -----------------------------------------------------------

    def expand(self, mode: SweepMode | str = SweepMode.OFAT) -> list[Config]:
        """The configurations this mode covers, with redundant ones removed.

        Staged expands to the same set as OFAT here, because its later stages depend on
        earlier results and only the runner can know them. What staged actually runs is a
        subset, decided as it goes.
        """
        configs, _ = self.expand_with_report(mode)
        return configs

    def expand_with_dropped(
        self, mode: SweepMode | str = SweepMode.OFAT
    ) -> tuple[list[Config], int]:
        """The configurations, and how many combinations were impossible to run.

        Only the impossible ones, which is less than the matrix loses. A combination also
        disappears when it canonicalises onto a row already in the list, and that has never
        been in this number -- so `expand(...)` can be five shorter than this count explains.
        Prefer `expand_with_report` for anything a person reads.
        """
        configs, report = self.expand_with_report(mode)
        return configs, report.impossible

    def expand_with_report(
        self, mode: SweepMode | str = SweepMode.OFAT
    ) -> tuple[list[Config], DedupeReport]:
        """The configurations, and where every combination that is not among them went."""
        chosen = SweepMode(mode)
        raw = self._factorial() if chosen is SweepMode.FACTORIAL else self._ofat()
        return deduplicate_with_report(raw)

    def _factorial(self) -> list[Config]:
        return [
            Config(
                ingestion=g,
                parser=p,
                chunker=c,
                embedder=e,
                index=i,
                transform=t,
                retrieval=v,
                reranker=r,
                candidates=d,
                generator=n,
                k=self.k,
            )
            for g, p, c, e, i, t, v, r, d, n in product(
                self.ingestion,
                self.parser,
                self.chunker,
                self.embedder,
                self.index,
                self.transform,
                self.retrieval,
                self.reranker,
                self.candidates,
                self.generator,
            )
        ]

    def _ofat(self) -> list[Config]:
        base = self.baseline()
        configs = [base]
        seen = {base}
        for axis, values in self.axes.items():
            for value in values[1:]:
                candidate = base.with_(**{axis: value})
                if candidate not in seen:
                    configs.append(candidate)
                    seen.add(candidate)
        return configs

    def stage_configs(self, axis: str, base: Config) -> list[Config]:
        """The configurations for one stage of a staged sweep: one axis, everything else fixed."""
        if axis not in AXIS_ORDER:
            raise MatrixError(f"unknown axis {axis!r}. Axes are: {', '.join(AXIS_ORDER)}")
        values = getattr(self, axis)
        configs, _ = deduplicate([base.with_(**{axis: value}) for value in values])
        return configs

    def __iter__(self) -> Iterator[Config]:
        return iter(self.expand())


# ---------------------------------------------------------------------------
# removing configurations that are not actually different
# ---------------------------------------------------------------------------


def canonicalise(config: Config) -> Config:
    """Normalise away settings the configuration cannot possibly use.

    BM25 works on text and never looks at a vector, so `bm25 + tfidf` and `bm25 + hash` are
    the same run under two names. Left alone they waste two thirds of the sparse arm of a
    sweep -- and worse, they poison the embedder axis effect, which would average three
    identical BM25 scores into the embedder's record as though it had earned them.
    """
    # "plain" is the same run as naming no strategy at all, and two names for one run waste a
    # slot and dilute the axis effect.
    if config.ingestion == "plain":
        config = config.with_(ingestion=None)

    # `retrieval: "simple"` is deliberately *not* folded onto `None`, though the two do run the
    # same search -- `get_retriever(None)` returns `SimpleRetrieval()`. Folding it would make
    # `simple` unusable as an explicit baseline, which is how it is mostly written: a sweep
    # names it to say "compare against plain search", and reading the result back under a key
    # of `None` reads worse than the duplicate costs. The duplicate only appears in a sweep
    # naming both plain search *and* a `widened` arm that folds onto it, which is rare and
    # visible in the dedupe report rather than silent.

    # "none" is the identity reranker, so it is the same configuration as no reranker at all.
    if config.transform == "none":
        config = config.with_(transform=None)

    if config.reranker == "none":
        config = config.with_(reranker=None)

    # Candidate depth only means anything when something reranks the candidates. Without a
    # reranker, sweeping it would run identical configurations under different names and
    # credit the depth axis with differences it did not cause.
    if config.reranker is None and config.candidates != 50:
        config = config.with_(candidates=50)

    # `widened` asks the index for `k * factor` and hands back the top `k`, so the surplus is
    # sometimes thrown away unused and the run is plain search under another name. Sometimes,
    # not always -- `_widening_is_wasted` is where the difference is decided, and getting it
    # wrong in the other direction would delete a real result rather than a duplicate.
    if _widening_is_wasted(config):
        config = config.with_(retrieval=None)

    if config.embedder is None:
        return config
    try:
        from contextgrid.index import get_index

        if not get_index(config.index).needs_vectors:
            return config.with_(embedder=None)
    except Exception:
        # An index that cannot be built will fail loudly when the run reaches it. Silently
        # dropping the configuration here would hide the real error behind a missing row.
        return config
    return config


def _widening_is_wasted(config: Config) -> bool:
    """Whether `widened` provably returns exactly what plain search would.

    `docs/dimensions/retrieval.md` says of `widened`: "On its own this changes nothing -- the
    same top-`k` comes back... it changes a great deal once a reranker sits downstream." That
    reads like "no reranker means no effect", and a sweep with `reranker: null` really does
    burn a row on `widened:2` and another on `widened:8` to re-measure `simple`. But the
    headline is about the common case, not every case, and resetting the axis on it alone
    would delete real results rather than duplicates. Measured, not assumed: with two queries
    and `k=5`, `widened:8` returns a different top five with no reranker anywhere, because
    `fuse` combines the deeper lists by rank and a chunk lying 20th on both queries beats one
    lying 1st on only one.

    So the surplus is only provably wasted when four things hold at once:

    * **no reranker** -- with one, the wider net is precisely what it reorders
      (`BuiltPipeline.search` asks for `candidates` and cuts to `k`);
    * **no transform** -- one that returns several queries makes the fusion above possible,
      and whether it returns several is a runtime decision no matrix can read in advance;
    * **no ingestion** -- `BuiltPipeline._to_retrievable` collapses indexed units onto the
      passages they stand for and merges runs of siblings across whatever came back, so a
      deeper pool can merge passages a shallow one never had the pieces for;
    * **an exact index** -- an approximate one searches more of its structure when asked for
      more and can return a better top `k` for it, which is the whole reason `Index.is_exact`
      is a flag a configuration can read.

    `factor=1` needs none of those: the depth is `k` itself, so the searches are exactly the
    ones plain search would have made.
    """
    if config.retrieval is None:
        return False
    try:
        from contextgrid.retrieve import RETRIEVERS

        name, params = RETRIEVERS.parse_spec(config.retrieval)
    except Exception:
        # A spec that cannot be parsed will fail loudly when the run reaches it. Rewriting it
        # here would hide that error behind a row that quietly became plain search instead.
        return False

    if name != "widened":
        return False
    # Compared against 1 rather than defaulted, so the class keeps ownership of what `widened`
    # with no factor means.
    if params.get("factor") == 1:
        return True
    if config.reranker is not None or config.transform is not None:
        return False
    if config.ingestion is not None:
        return False

    try:
        from contextgrid.index import get_index

        return get_index(config.index).is_exact
    except Exception:
        # Same reasoning as the parse above, and as `is_runnable`: an index that cannot be
        # built is the run's problem to report, not a reason to rewrite the row.
        return False


def is_runnable(config: Config) -> bool:
    """Whether this combination can actually be built.

    Writing `embedder: [tfidf, null]` alongside `index: [dense, bm25]` obviously means "tfidf
    with dense, and bm25 with nothing" -- but a factorial expansion also produces `null` with
    `dense`, which cannot run at all because a dense index has no vectors to search.

    Dropping those is right. Erroring would force the user to write two configs to express one
    perfectly clear intention, and running them would fail halfway through a sweep.
    """
    if config.embedder is not None:
        return True
    try:
        from contextgrid.index import get_index

        return not get_index(config.index).needs_vectors
    except Exception:
        # An index that cannot be built at all is a different problem, and the run should
        # report it properly rather than have it silently disappear from the matrix.
        return True


@dataclass(frozen=True, slots=True)
class DedupeReport:
    """Where every combination went: run, impossible, collapsed, or listed twice.

    Written because `20 on paper, 10 to run (5 impossible combination(s) skipped)` was printed
    on one line and 20 minus 10 is 10, not 5. The missing five were combinations that
    canonicalised onto a row already in the list -- `bm25 + tfidf` onto `bm25` alone -- which
    is correct behaviour and was reported nowhere, leaving a reader to work out where five
    runs had gone.

    The two causes stay separate rather than being added together. "This cannot run" and "you
    wrote this run twice" are different facts about a sweep: the first is usually a factorial
    expansion producing a combination nobody meant, the second is usually an axis that is not
    varying as much as it looks. Merging them into one number tells the reader neither.
    """

    #: Combinations offered to `deduplicate`, before anything was removed. For a factorial
    #: sweep this is `Matrix.shape()`'s product; for OFAT it is the far smaller set OFAT
    #: actually walks, so it is the honest denominator to print against `kept`.
    considered: int
    #: Combinations that will actually run.
    kept: int
    #: Dropped because they cannot be built at all -- see `is_runnable`.
    impossible: int
    #: Removed because canonicalising made them identical to a row already kept: `bm25 + hash`
    #: onto `bm25 + tfidf`, `widened` with nothing downstream onto plain search.
    collapsed: int
    #: Removed because the very same configuration was offered twice with nothing rewritten.
    #: Only reachable by repeating a value on one axis, but counted so the four numbers always
    #: account for each other.
    repeated: int

    def __post_init__(self) -> None:
        accounted = self.kept + self.impossible + self.collapsed + self.repeated
        if accounted != self.considered:
            raise MatrixError(
                f"matrix accounting is wrong: {self.considered} combination(s) considered but "
                f"{accounted} accounted for ({self.kept} kept, {self.impossible} impossible, "
                f"{self.collapsed} collapsed, {self.repeated} repeated). This is a bug in "
                "contextgrid.grid.matrix, not in the matrix you wrote."
            )

    @property
    def removed(self) -> int:
        """Everything that will not run. Equal to `considered - kept`, by construction."""
        return self.impossible + self.collapsed + self.repeated

    def note(self) -> str:
        """The parenthetical for a "N considered, M to run" line. Empty when nothing was lost.

        Every category that fired is named, so the line the reader is looking at reconciles
        without them going and counting anything themselves.
        """
        parts = []
        if self.impossible:
            parts.append(f"{self.impossible} impossible combination(s) skipped")
        if self.collapsed:
            parts.append(f"{self.collapsed} collapsed onto an identical run")
        if self.repeated:
            parts.append(f"{self.repeated} listed more than once")
        return ", ".join(parts)


def deduplicate(configs: Sequence[Config]) -> tuple[list[Config], int]:
    """Canonicalise, drop impossible combinations and repeats, keeping the original order.

    Returns the surviving configurations and how many were **impossible** -- which is not the
    whole shrink, because canonicalising also collapses rows onto each other. Kept at this
    shape for callers that only want the impossible count; `deduplicate_with_report` is the
    one to reach for when the number is going to be printed.
    """
    kept, report = deduplicate_with_report(configs)
    return kept, report.impossible


def deduplicate_with_report(configs: Sequence[Config]) -> tuple[list[Config], DedupeReport]:
    """Canonicalise, drop what cannot or need not run, and account for every loss.

    Keeps the original order, and returns a `DedupeReport` whose numbers add up to the number
    of configurations handed in -- because a matrix that quietly shrinks is a matrix somebody
    will misread, and a matrix that shrinks by a number it half-explains is worse.
    """
    kept: list[Config] = []
    #: normalised configuration -> whether the row that claimed it had been rewritten. A later
    #: exact match on a rewritten row is still a collapse, not a configuration written twice.
    seen: dict[Config, bool] = {}
    impossible = 0
    collapsed = 0
    repeated = 0

    for config in configs:
        if not is_runnable(config):
            impossible += 1
            continue
        normalised = canonicalise(config)
        rewritten = normalised != config
        if normalised in seen:
            if rewritten or seen[normalised]:
                collapsed += 1
            else:
                repeated += 1
            continue
        seen[normalised] = rewritten
        kept.append(normalised)

    return kept, DedupeReport(
        considered=len(configs),
        kept=len(kept),
        impossible=impossible,
        collapsed=collapsed,
        repeated=repeated,
    )


def matrix(
    ingestion: str | Sequence[str | None] | None = None,
    parser: str | Sequence[str] = "markdown",
    chunker: str | Sequence[str] = "recursive:512",
    embedder: str | Sequence[str | None] | None = "tfidf",
    index: str | Sequence[str] = "dense",
    transform: str | Sequence[str | None] | None = None,
    retrieval: str | Sequence[str | None] | None = None,
    reranker: str | Sequence[str | None] | None = None,
    candidates: int | Sequence[int] = 50,
    generator: str | Sequence[str | None] | None = None,
    k: int = 10,
) -> Matrix:
    """Build a matrix, accepting a single value or a list on any axis.

    `lab.grid(parser="markdown", chunker=["recursive:512", "semantic"])` should work without
    the caller wrapping the single value in a list.

    Every axis takes a **spec string**, not a plugin instance. That is not fussiness: a
    configuration has to be writable into a leaderboard row, a cache key and a YAML file, and
    an object is none of those. `chonkie:recursive:512` survives all three; a
    `ChonkieRecursiveChunker` survives none, and a run nobody can write down is a run nobody
    can reproduce.
    """
    axes = {
        "ingestion": ingestion,
        "parser": parser,
        "chunker": chunker,
        "embedder": embedder,
        "index": index,
        "transform": transform,
        "retrieval": retrieval,
        "reranker": reranker,
        "generator": generator,
    }
    for axis, value in axes.items():
        _require_specs(axis, value)

    return Matrix(
        ingestion=_as_tuple(ingestion),
        parser=_as_tuple(parser),
        chunker=_as_tuple(chunker),
        embedder=_as_tuple(embedder),
        index=_as_tuple(index),
        transform=_as_tuple(transform),
        retrieval=_as_tuple(retrieval),
        reranker=_as_tuple(reranker),
        candidates=_as_tuple(candidates),
        generator=_as_tuple(generator),
        k=k,
    )


def _require_specs(axis: str, value: Any) -> None:
    """Reject plugin instances early, where the message can still be useful.

    Left through, an instance reaches the leaderboard as `TypeError: sequence item 2: expected
    str` from inside a report formatter -- long after the sweep has run, and pointing at a
    place that has nothing to do with the mistake.
    """
    # Not `_as_tuple`: it would call `tuple()` on the instance and raise "not iterable"
    # before this ever got a chance to say something useful.
    items = value if isinstance(value, (list, tuple)) else [value]
    for item in items:
        if item is None or isinstance(item, (str, int)):
            continue
        kind = type(item).__name__
        raise ContextGridError(
            f"{axis} was given a {kind} instance. Axes take spec strings, so that a "
            f"configuration can be written into a report, a cache key and a config file -- "
            f'try {axis}="{getattr(item, "name", kind.lower())}" instead. To use an object '
            "directly, build one Config and run it rather than sweeping over it."
        )


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None or isinstance(value, (str, int)):
        return (value,)
    return tuple(value)
