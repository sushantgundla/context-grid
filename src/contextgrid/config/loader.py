"""Reading a config file, and running what it describes.

Loading is separated from running so a config can be inspected, validated and costed without
executing anything -- which is what `contextgrid check` does, and what anybody sensible does
before starting a sweep that might take an hour.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from contextgrid.cache.store import Cache, DiskCache, MemoryCache, NullCache
from contextgrid.config.schema import ConfigError, ExperimentConfig, expand_env
from contextgrid.core.evalset import EvalSet
from contextgrid.corpus import Corpus
from contextgrid.cost.model import CostModel
from contextgrid.grid.runner import Runner
from contextgrid.report.results import Results
from contextgrid.score.resolve import ResolutionPolicy, SpanResolver


def load(path: str | Path) -> ExperimentConfig:
    """Read a YAML or JSON config.

    The format is decided by content rather than by extension: a `.yaml` file containing JSON
    is still valid JSON, and refusing it on the strength of its name would be pedantic.
    """
    source = Path(path).expanduser()
    if not source.exists():
        raise ConfigError(f"no config file at {source}")

    text = source.read_text(encoding="utf-8")
    data = _parse(text, source)
    return ExperimentConfig.from_mapping(data, base=source.parent, source=source)


def loads(text: str, *, base: Path | None = None) -> ExperimentConfig:
    """Read a config from a string, for tests and for generated configs."""
    return ExperimentConfig.from_mapping(_parse(text, None), base=base or Path.cwd(), source=None)


def _parse(text: str, source: Path | None) -> Mapping[str, Any]:
    expanded = expand_env(text)

    try:
        parsed = json.loads(expanded)
    except json.JSONDecodeError:
        parsed = _parse_yaml(expanded, source)

    if not isinstance(parsed, Mapping):
        where = f"{source}: " if source else ""
        raise ConfigError(f"{where}a config file must be a mapping at the top level")
    return parsed


def _parse_yaml(text: str, source: Path | None) -> Any:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - yaml is a core dependency
        raise ConfigError(
            "reading YAML needs pyyaml, which should have been installed with context-grid. "
            "Try `pip install --force-reinstall context-grid`."
        ) from error

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as error:
        where = f" in {source}" if source else ""
        # PyYAML's own message carries the line and column, which is the useful part.
        raise ConfigError(f"could not parse the config{where}: {error}") from error


# ---------------------------------------------------------------------------
# running what it describes
# ---------------------------------------------------------------------------


def build_corpus(config: ExperimentConfig) -> Corpus:
    """Load the documents the config points at, whether a directory or a single file."""
    if config.corpus.is_dir():
        return Corpus.from_dir(config.corpus)
    return Corpus.from_files([config.corpus])


def build_evalset(config: ExperimentConfig) -> EvalSet:
    """Load the questions. Reading is decided by extension here, since both formats exist."""
    if config.evalset is None:
        raise ConfigError(
            "this config has no `evalset`, so there is nothing to score against. Point it at "
            "a JSONL or CSV file of questions, or draft one with `contextgrid evalset`."
        )

    from contextgrid.evalset.io import read_csv, read_jsonl

    if config.evalset.suffix.lower() == ".csv":
        return read_csv(config.evalset)
    return read_jsonl(config.evalset)


def build_cache(config: ExperimentConfig) -> Cache:
    if config.run.cache == "disk":
        root = (config.report.out or config.corpus.parent) / ".contextgrid-cache"
        return DiskCache(root)
    if config.run.cache == "none":
        return NullCache()
    return MemoryCache()


def build_llm(config: ExperimentConfig) -> object | None:
    """The model every stage that needs one shares.

    One name in the config rather than one per stage: the alternative is four places to set a
    key and four prices to reconcile, for a choice almost nobody wants to make differently per
    stage.
    """
    if not config.run.model:
        return None
    from contextgrid.evalset.llm import get_llm

    return get_llm(config.run.model)


def build_runner(config: ExperimentConfig, corpus: Corpus) -> Runner:
    return Runner(
        corpus=corpus,
        cache=build_cache(config),
        cost_model=CostModel(machine_usd_per_hour=config.run.machine_usd_per_hour),
        span_resolver=SpanResolver(
            policy=ResolutionPolicy(config.run.resolution_policy),
            threshold=config.run.resolution_threshold,
        ),
        ks=config.run.ks,
        headline=config.run.headline,
        extra_metrics=config.run.metrics,
        llm=build_llm(config),
        seed=config.run.seed,
    )


def run(
    config: ExperimentConfig,
    *,
    on_progress: Any = None,
) -> Results:
    """Execute a config end to end."""
    config.validate_paths()

    corpus = build_corpus(config)
    evalset = build_evalset(config)
    runner = build_runner(config, corpus)

    return runner.run(
        config.grid.to_matrix(config.run.k),
        evalset,
        mode=config.run.mode,
        budget_seconds=config.run.budget_seconds,
        budget_usd=config.run.budget_usd,
        on_progress=on_progress,
    )


def write_report(config: ExperimentConfig, results: Results) -> list[Path]:
    """Write whatever the config asked for."""
    if config.report.out is None:
        return []

    from contextgrid.report.export import (
        config_to_python,
        results_to_json,
        results_to_markdown,
        winning_config_to_yaml,
    )
    from contextgrid.report.manifest import build_manifest

    root = config.report.out
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    winner = results.best(config.run.headline)
    manifest = None
    if winner is not None:
        manifest = build_manifest(
            winner.config,
            build_corpus(config),
            build_evalset(config),
            seeds={"run": config.run.seed},
        )
        written.append(manifest.save(root / "manifest.json"))

    writers: dict[str, tuple[str, Callable[[], str]]] = {
        "markdown": (
            "report.md",
            lambda: results_to_markdown(
                results,
                metric=config.run.headline,
                manifest=manifest,
                limit=config.report.leaderboard_limit,
            ),
        ),
        "json": ("results.json", lambda: results_to_json(results, manifest=manifest)),
        "yaml": (
            "winning-config.yaml",
            # `config` -- the experiment this sweep came from -- is what makes the written file
            # runnable rather than a listing: it is the only thing here that knows the corpus.
            lambda: (
                winning_config_to_yaml(winner.config, config, manifest=manifest) if winner else ""
            ),
        ),
        "python": (
            "use_winning_config.py",
            lambda: config_to_python(winner.config) if winner else "",
        ),
    }

    for fmt in config.report.formats:
        filename, render = writers[fmt]
        target = root / filename
        target.write_text(render(), encoding="utf-8")
        written.append(target)

    # The config that produced this, copied in beside the results. A bundle that cannot be
    # re-run is a screenshot.
    if config.source_path and config.source_path.exists():
        copy = root / "experiment.yaml"
        copy.write_text(config.source_path.read_text(encoding="utf-8"), encoding="utf-8")
        written.append(copy)

    return written
