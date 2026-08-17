"""The experiment file.

One YAML describes the whole thing: which documents, which questions, which values to try on
every axis, how long to spend, and what to write out. `contextgrid run config.yaml` does the
rest.

Three decisions shape this module, and each one is about a failure that would otherwise be
silent:

**Unknown keys are errors.** A config with `chunkers:` instead of `chunker:` that runs anyway,
using defaults, and reports a leaderboard, is far worse than one that refuses to start. Every
section rejects keys it does not know and suggests the nearest one it does.

**Any axis accepts one value or many.** `chunker: recursive:512` and `chunker: [a, b]` both
work. Forcing a list for a single value is the kind of friction that makes people write the
config wrong.

**Paths resolve against the config file.** A config that only works from one working directory
is not portable, and the first thing anybody does is run it from somewhere else.
"""

from __future__ import annotations

import difflib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from contextgrid.core.errors import ContextGridError
from contextgrid.core.registry import Registry, UnknownPluginError
from contextgrid.grid.matrix import AXIS_ORDER, Matrix
from contextgrid.score.metrics import DEFAULT_KS, available_metrics

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ContextGridError, ValueError):
    """A configuration file is malformed, and the message says exactly where."""


def _unknown_key(section: str, key: str, known: Sequence[str]) -> ConfigError:
    """Reject a key, and guess what was meant.

    A typo is the most common config failure by a wide margin, and "unknown key" without a
    suggestion leaves somebody comparing their file against documentation character by
    character.
    """
    close = difflib.get_close_matches(key, known, n=1, cutoff=0.6)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    return ConfigError(
        f"unknown key {key!r} in the {section!r} section.{hint} "
        f"Known keys: {', '.join(sorted(known))}"
    )


def _problems_error(problems: Sequence[str], summary: str) -> ConfigError:
    """One error carrying however many problems were found.

    A single problem is its own message -- a count and an indented list of one reads like a
    form letter. More than one gets the summary line and the list, so nothing is buried.
    """
    if len(problems) == 1:
        return ConfigError(problems[0])
    joined = "\n".join(f"  {problem}" for problem in problems)
    return ConfigError(f"{summary}:\n{joined}")


def _path_problems(corpus: Path, evalset: Path | None) -> list[str]:
    """The same input checks `ExperimentConfig.validate_paths` makes, as a list.

    Same wording, because the message a user reads should not depend on which command found
    the problem.
    """
    problems: list[str] = []
    if not corpus.exists():
        problems.append(f"corpus not found: {corpus}")
    if evalset is not None and not evalset.exists():
        problems.append(f"eval set not found: {evalset}")
    return problems


def _plugin_axes() -> dict[str, tuple[Registry[Any], tuple[str, ...]]]:
    """Every axis whose values are plugin names, and where those names come from.

    Imported here rather than at module scope so that reading a config does not import nine
    subpackages before it has decided the file is even a mapping. Called after `plugins:` has
    been loaded, so a chunker somebody registered from their own module is a known name like
    any other.

    The second element is the names that are real but not in the registry. `hyde` and `llm`
    cannot be registered because they need a model, and a config naming one is correct -- so
    validating against the registry alone would reject the very arms the docs recommend.
    """
    from contextgrid.chunk import CHUNKERS
    from contextgrid.embed import EMBEDDERS
    from contextgrid.generate import GENERATORS
    from contextgrid.generate import MODEL_BACKED as GENERATOR_MODEL_BACKED
    from contextgrid.index import INDEXES
    from contextgrid.ingest import INGESTERS
    from contextgrid.parse import PARSERS
    from contextgrid.rerank import RERANKERS
    from contextgrid.retrieve import RETRIEVERS
    from contextgrid.transform import MODEL_BACKED as TRANSFORM_MODEL_BACKED
    from contextgrid.transform import TRANSFORMS

    return {
        "ingestion": (INGESTERS, ()),
        "parser": (PARSERS, ()),
        "chunker": (CHUNKERS, ()),
        "embedder": (EMBEDDERS, ()),
        "index": (INDEXES, ()),
        "transform": (TRANSFORMS, TRANSFORM_MODEL_BACKED),
        "retrieval": (RETRIEVERS, ()),
        "reranker": (RERANKERS, ()),
        "generator": (GENERATORS, GENERATOR_MODEL_BACKED),
    }


@dataclass(frozen=True, slots=True)
class GridConfig:
    """The axes, and the values to try on each.

    Anything left out is a single-valued axis at its default, which is what makes a minimal
    config possible: naming one axis sweeps that axis and holds everything else still.
    """

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

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, check_names: bool = True) -> GridConfig:
        """Parse the axes. `check_names=False` parses without rejecting an unknown name.

        The caller that turns it off is `ExperimentConfig.from_mapping`, which wants the list
        of bad names rather than the first exception, so it can report them beside whatever
        else is wrong with the file. Everybody else wants the exception.
        """
        for key in data:
            if key not in AXIS_ORDER:
                raise _unknown_key("grid", key, AXIS_ORDER)

        grid = cls(
            ingestion=_as_optional_strings(data.get("ingestion", (None,)), "grid.ingestion"),
            parser=_as_strings(data.get("parser", ("markdown",)), "grid.parser"),
            chunker=_as_strings(data.get("chunker", ("recursive:512",)), "grid.chunker"),
            embedder=_as_optional_strings(data.get("embedder", ("tfidf",)), "grid.embedder"),
            index=_as_strings(data.get("index", ("dense",)), "grid.index"),
            transform=_as_optional_strings(data.get("transform", (None,)), "grid.transform"),
            retrieval=_as_optional_strings(data.get("retrieval", (None,)), "grid.retrieval"),
            reranker=_as_optional_strings(data.get("reranker", (None,)), "grid.reranker"),
            candidates=_as_ints(data.get("candidates", (50,)), "grid.candidates"),
            generator=_as_optional_strings(data.get("generator", (None,)), "grid.generator"),
        )
        # Raised here rather than collected with the name problems below, because it is the
        # same class of mistake as `run.k: -1` and is reported the same way: a number that
        # cannot mean anything, refused where it was written. `check` used to accept
        # `candidates: -3` and print it back as a leaderboard label -- while refusing `run.k:
        # -1` and `chunker: recursive:-5` in the same command, in the same breath.
        for depth in grid.candidates:
            if depth < 1:
                raise ConfigError(f"grid.candidates must be at least 1, got {depth}")

        if check_names:
            grid.validate_names()
        return grid

    def validate_names(self) -> None:
        """Reject a name nothing is registered under, on every axis at once.

        A typo'd axis value used to survive parsing, survive `check`, and surface partway
        through the sweep -- `chunker: [recursive:512, recursive:256, recursiv:128]` measured
        two configurations over several minutes and then died on the third, having written
        nothing. The name is knowable before any of that happens, so it is checked here.

        The *name* only. Not the parameters after the colon, and nothing is built: a name whose
        extra is not installed is a legitimate config on a laptop that is checking a file it
        will run somewhere else, and it still fails at run time with its own MissingExtraError
        saying what to install. Checking a config must not require being able to run it.
        """
        problems = self.name_problems()
        if problems:
            # Every one of them, because fixing a typo, re-running, and being told about the
            # next typo is the same wait repeated.
            raise _problems_error(problems, f"{len(problems)} unknown plugin name(s) in the grid")

    def name_problems(self) -> list[str]:
        """What `validate_names` would refuse, as a list instead of an exception.

        Split out so a caller holding other problems can report these beside them, rather than
        raising on whichever class of problem it happened to reach first.
        """
        problems: list[str] = []
        for axis, (registry, extra) in _plugin_axes().items():
            known = {*registry.names(), *extra}
            for spec in getattr(self, axis):
                if spec is None:
                    continue
                name = registry.name_in(spec)
                if name not in known:
                    # The run-time error, verbatim, with the axis in front of it -- one config
                    # can name eight plugins, and "no chunker named ..." is only obvious about
                    # which line to look at while `chunker` is the only axis you touched.
                    problem = UnknownPluginError(registry.family, name, sorted(known))
                    problems.append(f"grid.{axis}: {problem}")
        return problems

    def to_matrix(self, k: int) -> Matrix:
        return Matrix(
            ingestion=self.ingestion,
            parser=self.parser,
            chunker=self.chunker,
            embedder=self.embedder,
            index=self.index,
            transform=self.transform,
            retrieval=self.retrieval,
            reranker=self.reranker,
            candidates=self.candidates,
            generator=self.generator,
            k=k,
        )

    def as_dict(self) -> dict[str, Any]:
        return {axis: list(getattr(self, axis)) for axis in AXIS_ORDER}


@dataclass(frozen=True, slots=True)
class RunConfig:
    """How the sweep is executed."""

    mode: str = "ofat"
    k: int = 10
    headline: str = "recall@5"
    budget_seconds: float | None = None
    budget_usd: float | None = None
    seed: int = 0
    machine_usd_per_hour: float = 0.0
    resolution_policy: str = "coverage"
    resolution_threshold: float = 0.5
    cache: str = "memory"
    #: The model the pipeline reaches for when a strategy needs one: HyDE and the other query
    #: transforms, agentic retrieval, the LLM-backed ingestion strategies, and the generation
    #: judge. One name, so one key and one price.
    model: str | None = None
    #: Extra registered metrics to compute alongside the six built-ins, one name or a list --
    #: `metrics: weighted_recall` or `metrics: [weighted_recall, top1_only]`. The headline's
    #: own metric is always computed whether or not it's named here, the same way its cut-off
    #: is always added to `ks` below.
    metrics: tuple[str, ...] = ()

    KNOWN: ClassVar[tuple[str, ...]] = (
        "mode",
        "k",
        "headline",
        "budget_seconds",
        "budget_usd",
        "seed",
        "machine_usd_per_hour",
        "resolution_policy",
        "resolution_threshold",
        "cache",
        "model",
        "metrics",
    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RunConfig:
        for key in data:
            if key not in cls.KNOWN:
                raise _unknown_key("run", key, cls.KNOWN)

        config = cls(
            mode=str(data.get("mode", "ofat")),
            k=int(data.get("k", 10)),
            headline=str(data.get("headline", "recall@5")),
            budget_seconds=_as_optional_float(data.get("budget_seconds"), "run.budget_seconds"),
            budget_usd=_as_optional_float(data.get("budget_usd"), "run.budget_usd"),
            seed=int(data.get("seed", 0)),
            machine_usd_per_hour=float(data.get("machine_usd_per_hour", 0.0)),
            resolution_policy=str(data.get("resolution_policy", "coverage")),
            resolution_threshold=float(data.get("resolution_threshold", 0.5)),
            cache=str(data.get("cache", "memory")),
            model=str(data["model"]) if data.get("model") else None,
            metrics=_as_strings(data.get("metrics", ()), "run.metrics"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode not in {"factorial", "ofat", "staged"}:
            raise ConfigError(
                f"run.mode must be 'factorial', 'ofat' or 'staged', got {self.mode!r}"
            )
        if self.k < 1:
            raise ConfigError(f"run.k must be at least 1, got {self.k}")
        if "@" not in self.headline:
            raise ConfigError(
                f"run.headline must name a cut-off, like 'recall@5'. Got {self.headline!r}"
            )
        metric, _, cut = self.headline.partition("@")
        if not cut.isdigit():
            raise ConfigError(f"run.headline has a non-numeric cut-off: {self.headline!r}")
        known = available_metrics()
        if metric not in known:
            raise ConfigError(
                f"unknown metric {metric!r} in run.headline. Available: {', '.join(known)}"
            )
        unknown_extra = set(self.metrics) - set(known)
        if unknown_extra:
            raise ConfigError(
                f"unknown metric(s) in run.metrics: {', '.join(sorted(unknown_extra))}. "
                f"Available: {', '.join(known)}"
            )
        if self.resolution_policy not in {"coverage", "iou", "containment"}:
            raise ConfigError(
                f"run.resolution_policy must be 'coverage', 'iou' or 'containment', got "
                f"{self.resolution_policy!r}"
            )
        if not 0 < self.resolution_threshold <= 1:
            raise ConfigError(
                f"run.resolution_threshold must be in (0, 1], got {self.resolution_threshold}"
            )
        if self.cache not in {"memory", "disk", "none"}:
            raise ConfigError(f"run.cache must be 'memory', 'disk' or 'none', got {self.cache!r}")

    @property
    def ks(self) -> tuple[int, ...]:
        """The cut-offs to report, always including the headline's own."""
        _, _, cut = self.headline.partition("@")
        return tuple(sorted({*DEFAULT_KS, int(cut)}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "k": self.k,
            "headline": self.headline,
            "budget_seconds": self.budget_seconds,
            "budget_usd": self.budget_usd,
            "seed": self.seed,
            "machine_usd_per_hour": self.machine_usd_per_hour,
            "resolution_policy": self.resolution_policy,
            "resolution_threshold": self.resolution_threshold,
            "cache": self.cache,
            "model": self.model,
            "metrics": list(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class ReportConfig:
    """What to write out, and where."""

    out: Path | None = None
    formats: tuple[str, ...] = ("markdown", "json")
    leaderboard_limit: int = 20

    KNOWN: ClassVar[tuple[str, ...]] = ("out", "formats", "leaderboard_limit")
    VALID_FORMATS: ClassVar[tuple[str, ...]] = ("markdown", "json", "yaml", "python")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, base: Path) -> ReportConfig:
        for key in data:
            if key not in cls.KNOWN:
                raise _unknown_key("report", key, cls.KNOWN)

        formats = _as_strings(data.get("formats", ("markdown", "json")), "report.formats")
        unknown = set(formats) - set(cls.VALID_FORMATS)
        if unknown:
            raise ConfigError(
                f"unknown report format(s): {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(cls.VALID_FORMATS)}"
            )

        out = data.get("out")
        return cls(
            out=_resolve(out, base) if out else None,
            formats=formats,
            leaderboard_limit=int(data.get("leaderboard_limit", 20)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "out": str(self.out) if self.out else None,
            "formats": list(self.formats),
            "leaderboard_limit": self.leaderboard_limit,
        }


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """One file, one experiment."""

    corpus: Path
    evalset: Path | None = None
    grid: GridConfig = field(default_factory=GridConfig)
    run: RunConfig = field(default_factory=RunConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    name: str = "experiment"
    source_path: Path | None = None
    #: Modules imported before any name in this config is resolved. What was loaded, not what
    #: was asked for -- a `./plugins.py` comes back as the absolute path it resolved to.
    plugins: tuple[str, ...] = ()

    KNOWN: ClassVar[tuple[str, ...]] = (
        "corpus",
        "evalset",
        "grid",
        "run",
        "report",
        "name",
        "plugins",
    )

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, base: Path, source: Path | None = None
    ) -> ExperimentConfig:
        if not isinstance(data, Mapping):
            raise ConfigError("a config file must be a mapping at the top level")

        for key in data:
            if key not in cls.KNOWN:
                raise _unknown_key("config", key, cls.KNOWN)

        if "corpus" not in data:
            raise ConfigError(
                "every config needs a `corpus`: a directory of documents, or a list of files."
            )

        # Before anything else is parsed, and that ordering is the whole point: parsing `run`
        # is what checks `headline` against the metric registry, and parsing `grid` is what
        # checks every axis value against its own. A plugin loaded after those runs would be
        # rejected as a typo by the very validation it exists to satisfy.
        from contextgrid.config.plugins import load_plugins

        plugins = load_plugins(_as_strings(data.get("plugins", ()), "plugins"), base=base)

        corpus = _resolve(data["corpus"], base)
        evalset = _resolve(data["evalset"], base) if data.get("evalset") else None

        # A typo'd plugin name used to abort here, so `contextgrid check` never reached the
        # path checks it does afterwards: a config with both a missing corpus and a typo'd
        # chunker reported the chunker, and the missing corpus only on the next run. The two
        # are independent -- neither needs the other resolved -- so both are reported at once.
        # Paths are only looked at on this failing branch. A config still loads with its corpus
        # absent, which is what `validate_paths` being a separate step is for.
        grid = GridConfig.from_mapping(_section(data, "grid"), check_names=False)
        names = grid.name_problems()
        if names:
            paths = _path_problems(corpus, evalset)
            summary = (
                f"{len(names) + len(paths)} problems with this config"
                if paths
                else f"{len(names)} unknown plugin name(s) in the grid"
            )
            raise _problems_error([*names, *paths], summary)

        return cls(
            corpus=corpus,
            evalset=evalset,
            grid=grid,
            run=RunConfig.from_mapping(_section(data, "run")),
            report=ReportConfig.from_mapping(_section(data, "report"), base=base),
            name=str(data.get("name", source.stem if source else "experiment")),
            source_path=source,
            plugins=plugins,
        )

    def validate_paths(self) -> None:
        """Check the inputs exist before anything expensive starts.

        Separate from parsing so a config can be inspected without its corpus present -- the
        CLI's `check` command wants that, and so does a test.
        """
        if not self.corpus.exists():
            raise ConfigError(f"corpus not found: {self.corpus}")
        if self.evalset is not None and not self.evalset.exists():
            raise ConfigError(f"eval set not found: {self.evalset}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "corpus": str(self.corpus),
            "evalset": str(self.evalset) if self.evalset else None,
            # In the manifest because a result produced with a custom metric cannot be
            # reproduced, or even read, without knowing which code defined it.
            "plugins": list(self.plugins),
            "grid": self.grid.as_dict(),
            "run": self.run.as_dict(),
            "report": self.report.as_dict(),
        }

    def describe(self) -> str:
        """What this file will do, before it does it."""
        matrix = self.grid.to_matrix(self.run.k)
        configs, dropped = matrix.expand_with_dropped(self.run.mode)
        note = f" ({dropped} impossible combination(s) skipped)" if dropped else ""
        return (
            f"{self.name}: {matrix.shape()} on paper, "
            f"{len(configs)} to run in {self.run.mode} mode{note}, "
            f"scored on {self.run.headline}"
        )


# ---------------------------------------------------------------------------
# coercion
# ---------------------------------------------------------------------------


def _section(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"the {key!r} section must be a mapping, got {type(value).__name__}")
    return value


def _as_tuple(value: Any) -> tuple[Any, ...]:
    """One value or many. Forcing a list for a single value is needless friction."""
    if value is None:
        return (None,)
    if isinstance(value, (str, int, float, bool)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    raise ConfigError(f"expected a value or a list, got {type(value).__name__}")


def _as_strings(value: Any, where: str) -> tuple[str, ...]:
    items = _as_tuple(value)
    if any(item is None for item in items):
        raise ConfigError(f"{where} cannot contain an empty value")
    return tuple(str(item) for item in items)


def _as_optional_strings(value: Any, where: str) -> tuple[str | None, ...]:
    """Like `_as_strings`, but `null` means "no plugin on this axis".

    An axis where "none" is a legitimate arm -- reranker, transform, embedder with BM25 --
    needs a way to say so, and YAML's `null` is the natural one.
    """
    del where
    return tuple(None if item is None else str(item) for item in _as_tuple(value))


def _as_ints(value: Any, where: str) -> tuple[int, ...]:
    items = _as_tuple(value)
    try:
        return tuple(int(item) for item in items)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{where} must be whole numbers, got {items!r}") from error


def _as_optional_float(value: Any, where: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{where} must be a number, got {value!r}") from error


def _resolve(value: Any, base: Path) -> Path:
    """Resolve a path against the config file's own directory.

    A config that only works from one working directory is not portable, and running it from
    somewhere else is the first thing anybody does.
    """
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def expand_env(value: str) -> str:
    """Substitute `${VAR}` from the environment.

    So a config can reference a key without containing one. A config file with a secret in it
    ends up in version control, and then in a screenshot.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        found = os.environ.get(name)
        if found is None:
            raise ConfigError(
                f"the config refers to ${{{name}}} but that environment variable is not set"
            )
        return found

    return _ENV_PATTERN.sub(replace, value)
