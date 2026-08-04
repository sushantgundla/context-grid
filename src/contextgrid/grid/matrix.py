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
#: real corpus. Reranking comes last because it operates on whatever the rest produced.
AXIS_ORDER: tuple[str, ...] = (
    "parser",
    "chunker",
    "embedder",
    "index",
    "transform",
    "reranker",
    "candidates",
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

    parser: tuple[str, ...] = ("markdown",)
    chunker: tuple[str, ...] = ("recursive:512",)
    embedder: tuple[str | None, ...] = ("tfidf",)
    index: tuple[str, ...] = ("dense",)
    transform: tuple[str | None, ...] = (None,)
    reranker: tuple[str | None, ...] = (None,)
    candidates: tuple[int, ...] = (50,)
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
            parser=self.parser[0],
            chunker=self.chunker[0],
            embedder=self.embedder[0],
            index=self.index[0],
            transform=self.transform[0],
            reranker=self.reranker[0],
            candidates=self.candidates[0],
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
        configs, _ = self.expand_with_dropped(mode)
        return configs

    def expand_with_dropped(
        self, mode: SweepMode | str = SweepMode.OFAT
    ) -> tuple[list[Config], int]:
        """The configurations, and how many combinations were impossible to run."""
        chosen = SweepMode(mode)
        raw = self._factorial() if chosen is SweepMode.FACTORIAL else self._ofat()
        return deduplicate(raw)

    def _factorial(self) -> list[Config]:
        return [
            Config(
                parser=p,
                chunker=c,
                embedder=e,
                index=i,
                transform=t,
                reranker=r,
                candidates=d,
                k=self.k,
            )
            for p, c, e, i, t, r, d in product(
                self.parser,
                self.chunker,
                self.embedder,
                self.index,
                self.transform,
                self.reranker,
                self.candidates,
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


def deduplicate(configs: Sequence[Config]) -> tuple[list[Config], int]:
    """Canonicalise, drop impossible combinations and repeats, keeping the original order.

    Returns the surviving configurations and how many were dropped, because a matrix that
    quietly shrinks is a matrix somebody will misread.
    """
    kept: list[Config] = []
    seen: set[Config] = set()
    dropped = 0

    for config in configs:
        if not is_runnable(config):
            dropped += 1
            continue
        normalised = canonicalise(config)
        if normalised in seen:
            continue
        seen.add(normalised)
        kept.append(normalised)

    return kept, dropped


def matrix(
    parser: str | Sequence[str] = "markdown",
    chunker: str | Sequence[str] = "recursive:512",
    embedder: str | Sequence[str | None] | None = "tfidf",
    index: str | Sequence[str] = "dense",
    transform: str | Sequence[str | None] | None = None,
    reranker: str | Sequence[str | None] | None = None,
    candidates: int | Sequence[int] = 50,
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
        "parser": parser,
        "chunker": chunker,
        "embedder": embedder,
        "index": index,
        "transform": transform,
        "reranker": reranker,
    }
    for axis, value in axes.items():
        _require_specs(axis, value)

    return Matrix(
        parser=_as_tuple(parser),
        chunker=_as_tuple(chunker),
        embedder=_as_tuple(embedder),
        index=_as_tuple(index),
        transform=_as_tuple(transform),
        reranker=_as_tuple(reranker),
        candidates=_as_tuple(candidates),
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
