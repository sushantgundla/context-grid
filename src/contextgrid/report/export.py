"""Taking a result somewhere else.

The non-goal that shapes this module: context-grid is not a RAG framework and has no business
serving anybody's production traffic. What it produces is a *decision*, and a decision is only
worth making if it can leave the tool.

So the winning configuration comes out as YAML you can commit, as runnable Python, and as a
one-page report an engineer can paste into a team decision doc. The last of those is the
adoption path that matters: nobody adopts a tool, they adopt an argument that came out of one.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from contextgrid.core.warnings import Severity
from contextgrid.grid.matrix import AXIS_ORDER
from contextgrid.pipeline import Config
from contextgrid.report.manifest import Manifest
from contextgrid.report.results import Results, RunResult
from contextgrid.score.significance import SignificanceError

if TYPE_CHECKING:  # pragma: no cover - import only for the annotation
    from contextgrid.config.schema import ExperimentConfig


def config_to_yaml(config: Config, *, manifest: Manifest | None = None) -> str:
    """The configuration as YAML, hand-written rather than via a dependency.

    The core installs with numpy and nothing else, and pulling in a YAML library to emit
    twelve flat keys would be a poor trade.
    """
    lines = _header(manifest)
    lines.append("")

    for key, value in config.as_dict().items():
        lines.append(f"{key}: {_yaml_value(value)}")
    return "\n".join(lines) + "\n"


def winning_config_to_yaml(
    config: Config,
    experiment: ExperimentConfig,
    *,
    manifest: Manifest | None = None,
) -> str:
    """The winning configuration as an experiment file you can hand straight back to the tool.

    Three places in the documentation call `winning-config.yaml` "re-runnable". It was not. It
    was `config_to_yaml`'s output -- a flat block of pipeline fields with no `corpus:` and no
    `grid:` wrapper -- so feeding it to `contextgrid run` failed on the first key it read:
    `unknown key 'ingestion' in the 'config' section`. A file whose whole promise is that you
    can re-run it, that cannot be re-run, is worse than no file: somebody commits it as the
    record of a decision and finds out months later.

    So this writes the real thing. `corpus:` and `evalset:` are absolute, because the file
    lands in `report.out/`, which is normally *below* the directory the original config lived
    in, and paths resolve against the config file's own directory -- a copied relative path
    would silently point somewhere else. The axes go in `grid:`, one value each rather than a
    list, since this file names exactly one configuration. Everything else goes in `run:`,
    which is where the schema puts it -- note that `k` lives there while `candidates` is an
    axis, so the two do not travel together.

    Budgets are deliberately left out. `budget_seconds` and `budget_usd` exist to stop a sweep
    that is taking too long or costing too much; carrying them onto a single configuration
    could only ever cut short the one run this file is for.
    """
    lines = _header(manifest)
    lines += [
        "# Re-run this file directly:  contextgrid run winning-config.yaml",
        "#",
        "",
        f"name: {_yaml_value(experiment.name)}",
        "",
        # Absolute on purpose -- see the docstring. This file does not sit where the original
        # config sat, so a relative path copied from it would resolve against the wrong root.
        f"corpus: {_yaml_value(str(experiment.corpus))}",
    ]
    if experiment.evalset is not None:
        lines.append(f"evalset: {_yaml_value(str(experiment.evalset))}")
    if experiment.plugins:
        # Without these, a winner naming a plugin-provided chunker or metric is rejected as a
        # typo by the validation that runs before any plugin would otherwise be loaded.
        lines += ["", "plugins:", *(f"  - {_yaml_value(name)}" for name in experiment.plugins)]

    values = config.as_dict()
    lines += [
        "",
        "# One value per axis: this file names a single configuration, not a sweep.",
        "grid:",
    ]
    lines += [f"  {axis}: {_yaml_value(values[axis])}" for axis in AXIS_ORDER if axis in values]

    lines += ["", "run:"]
    lines += [f"  {key}: {_yaml_value(value)}" for key, value in _run_settings(config, experiment)]
    if experiment.run.metrics:
        lines += ["  metrics:", *(f"    - {_yaml_value(name)}" for name in experiment.run.metrics)]

    # No `report:` section on purpose. This file usually sits *inside* the previous run's
    # report directory, so inheriting `report.out` would have a re-run overwrite the report,
    # the results and this very file while it was being read.
    return "\n".join(lines) + "\n"


def config_to_python(config: Config) -> str:
    """Runnable Python that rebuilds this configuration.

    Not a template with holes in it -- this actually runs, which is the difference between an
    export somebody uses and one they read once and retype.

    The field list is read off `Config` rather than written out here. The version that spelled
    out six field names by hand fell behind the dataclass and exported a winner of
    `parent-document:4 · markdown · recursive:96 · ~relevance-feedback:3 · bm25 · lexical@20`
    with no `ingestion=` and no `retrieval=` line -- so the snippet built plain chunking and
    plain search while `winning-config.yaml` next to it described the real pipeline. Two files
    from one run, two different answers, and the wrong one is the one people paste.
    """
    defaults = Config()
    # Anything left out is provably at its default, so the constructor puts it back. That is
    # the only omission rule this function has; there is no name in it to forget to update.
    settings = [
        f"    {name}={getattr(config, name)!r},"
        for name in _config_field_names()
        if getattr(config, name) != getattr(defaults, name)
    ]
    arguments = "\n" + "\n".join(settings) + "\n" if settings else ""

    return f'''"""The winning configuration, as context-grid found it."""

import contextgrid as cg

# {config.label}
# Any field not named below is at its default; `winning-config.yaml` spells out all of them.
config = cg.Config({arguments})

corpus = cg.Corpus.from_dir("./documents")
pipeline = cg.build(config, corpus)

for chunk_id in pipeline.search("your question here"):
    print(chunk_id)
'''


def results_to_json(results: Results, *, manifest: Manifest | None = None) -> str:
    """Every configuration and every number, for offline analysis.

    Includes the per-question scores, so a sceptic can re-run the statistics rather than
    taking the summary on trust. That is the point of publishing a bundle at all.
    """
    payload: dict[str, Any] = {
        "mode": results.mode,
        "cache": results.cache_summary,
        "warnings": results.warnings.to_list(),
        "runs": [_run_payload(run) for run in results],
    }
    if manifest is not None:
        payload["manifest"] = manifest.to_dict()
    return json.dumps(payload, indent=2, default=str) + "\n"


def results_to_markdown(
    results: Results,
    *,
    metric: str = "recall@5",
    manifest: Manifest | None = None,
    limit: int = 15,
    name: str | None = None,
) -> str:
    """A one-page report to paste into a decision doc.

    Ordered the way somebody reads it rather than the way it was computed: the conclusion
    first, then the evidence, then the caveats. A report that opens with a methodology section
    does not get read.

    `name` is the experiment's name -- `name:` in an experiment config. Falls back to
    `results.meta["name"]` so a runner can carry it without every call site passing it.
    """
    # A fixed title meant a directory of experiments was a directory of files all called
    # "Retrieval configuration comparison", with nothing above the fold to say which sweep
    # produced which. The name is the one thing that tells them apart.
    titled = name if name is not None else results.meta.get("name")
    heading = (
        f"{titled} — retrieval configuration comparison"
        if titled
        else "Retrieval configuration comparison"
    )
    lines = [f"# {heading}", ""]

    lines += ["## What to use", "", results.summary(metric), ""]

    try:
        verdict = results.is_the_winner_real(metric)
    except (KeyError, SignificanceError):
        # A configuration that answered nothing cannot be tested against one that did.
        # Losing the caveat is better than losing the whole report.
        verdict = None
    if verdict is not None and not verdict.distinguishable:
        lines += [
            "> **The top two are not statistically distinguishable.** Either is a defensible "
            "choice on this evidence; pick on cost or latency instead.",
            "",
        ]

    score = results.composite(metric)
    if score is not None and score.parts:
        lines += [
            "## Score",
            "",
            f"**{score.summary()}**",
            "",
            "| Dimension | Score |",
            "|---|---:|",
            *(f"| `{name}` | {score.parts[name]:.3f} |" for name in score.dimensions),
            "",
            "Comparable only against another score computed over the same dimensions.",
            "",
        ]

    lines += ["## Leaderboard", ""]
    # Generation columns appear only when something generated. A retrieval-only sweep should
    # not carry five empty columns; a generator sweep that omits them -- which is what this
    # did -- tells the reader the axis they swept made no difference at all.
    generation_columns = [
        name
        for name in ("faithfulness", "answer_relevancy", "groundedness", "citation_accuracy")
        if any(run.has(name) for run in results.runs)
    ]
    header = f"| Configuration | {metric} |"
    align = "|---|---:|"
    for name in generation_columns:
        header += f" {name} |"
        align += "---:|"
    header += " p95 ms | $/1k queries | Chunks |"
    align += "---:|---:|---:|"
    lines += [header, align]
    for row in results.leaderboard(metric, extra=generation_columns)[:limit]:
        cells = f"| `{row['config']}` | {row.get(metric, 0):.3f} |"
        for name in generation_columns:
            cells += f" {row[name]:.3f} |" if name in row else " — |"
        cells += (
            f" {row.get('p95_ms', 0):.1f} | {row.get('cost_per_1k', 0):.4f} | "
            f"{row.get('chunks', 0)} |"
        )
        lines.append(cells)
    lines.append("")

    if generation_columns:
        lines += [
            "> **`p95 ms` is retrieval only.** It excludes the generator, which dominates "
            "wall-clock on anything calling a model — a row showing well under a millisecond "
            "here can still take seconds to answer. `$/1k queries` does include generation.",
            "",
        ]

    axes = [
        ("parser", "Parser"),
        ("chunker", "Chunker"),
        ("embedder", "Embedder"),
        ("index", "Index"),
        ("reranker", "Reranker"),
        # Swept as often as any of the above, and left out of this section entirely -- so a
        # sweep whose *only* varying axis was the generator got no "which decision mattered"
        # section at all.
        ("generator", "Generator"),
    ]
    effects = [(label, results.axis_effect(axis, metric)) for axis, label in axes]
    informative = [(label, effect) for label, effect in effects if len(effect) > 1]
    if informative:
        lines += ["## Which decision mattered", ""]
        for label, effect in informative:
            spread = max(effect.values()) - min(effect.values())
            best = max(effect, key=lambda value: effect[value])
            if spread < 5e-4:
                # "`markdown` was best, +0.000 over the worst" reads as a recommendation for a
                # benefit that measurably does not exist, and invites somebody to standardise
                # on a value that changed nothing.
                lines.append(f"- **{label}**: no measurable difference between the values tried.")
            else:
                lines.append(
                    f"- **{label}**: `{best}` was best, {spread:+.3f} over the worst value tried."
                )
        lines.append("")
        if results.mode == "ofat":
            lines += [
                "> **These are averages over runs, not controlled comparisons.** In `ofat` "
                "mode each value appears in a different number of configurations, so a value "
                "that happens to sit in the baseline arm is averaged over different company "
                "than one that does not. Treat this as a pointer to what to sweep properly in "
                "`factorial` mode, not as a measured effect.",
                "",
            ]

    winner = results.best(metric)
    if winner is not None and winner.failures is not None and winner.failures.failures():
        lines += ["## Why the rest failed", "", winner.failures.summary(), ""]

    # Everything that changed what this sweep covered, not only what invalidated it. A run
    # stopped by `budget_usd` produces a `budget_reached` warning at CAUTION severity, so the
    # old `is_sound` check hid it -- and the report showed "No configurations were run." with
    # an empty table and no reason, which is the one thing the docs promise it will not do.
    reportable = [
        warning
        for warning in results.warnings.entries
        if warning.severity is not Severity.INFO or not results.runs
    ]
    if reportable:
        lines += ["## Warnings", ""]
        for warning in reportable:
            lines.append(f"- **{warning.code.value}**: {warning.message}")
        lines.append("")

    if manifest is not None:
        lines += [
            "## Reproducing this",
            "",
            f"- Manifest: `{manifest.short_hash}`",
            f"- Corpus: `{manifest.corpus_hash[:12]}` ({manifest.corpus_files} files)",
            f"- Eval set: `{manifest.evalset_id}` v{manifest.evalset_version} "
            f"(`{manifest.evalset_hash[:12]}`)",
            f"- Resolution: {manifest.resolution['policy']} at {manifest.resolution['threshold']}",
            f"- context-grid {manifest.versions.get('contextgrid', '?')} on Python "
            f"{manifest.versions.get('python', '?')}",
            "",
            "Two runs with the same manifest hash must produce identical numbers.",
            "",
        ]

    return "\n".join(lines)


def write_bundle(
    results: Results,
    directory: str | Path,
    *,
    metric: str = "recall@5",
    manifest: Manifest | None = None,
    name: str | None = None,
    corpus: str | Path | None = None,
    evalset: str | Path | None = None,
) -> list[Path]:
    """Write everything: the report, the raw results, the winning config and the manifest.

    A sceptic should be able to re-derive every number in the report from the bundle without
    asking for anything else.

    Give it `corpus` -- the documents the sweep ran over -- and `winning-config.yaml` is a file
    you can hand back to `contextgrid run`. Without it there is no corpus path to write, so the
    file falls back to the flat listing of pipeline fields, which is a record and not a config.
    Pass `evalset` too and the re-run can be scored as well as executed.
    """
    root = Path(directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    report = root / "report.md"
    report.write_text(
        results_to_markdown(results, metric=metric, manifest=manifest, name=name), "utf-8"
    )
    written.append(report)

    raw = root / "results.json"
    raw.write_text(results_to_json(results, manifest=manifest), "utf-8")
    written.append(raw)

    winner = results.best(metric)
    if winner is not None:
        experiment = (
            None
            if corpus is None
            else _experiment_from_paths(corpus, evalset, headline=metric, k=winner.config.k)
        )
        config_yaml = root / "winning-config.yaml"
        config_yaml.write_text(
            config_to_yaml(winner.config, manifest=manifest)
            if experiment is None
            else winning_config_to_yaml(winner.config, experiment, manifest=manifest),
            "utf-8",
        )
        written.append(config_yaml)

        snippet = root / "use_winning_config.py"
        snippet.write_text(config_to_python(winner.config), "utf-8")
        written.append(snippet)

    if manifest is not None:
        written.append(manifest.save(root / "manifest.json"))

    return written


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run_payload(run: RunResult) -> dict[str, Any]:
    interval = run.interval()
    return {
        "config": run.config.as_dict(),
        "label": run.label,
        "metrics": run.metrics,
        "timings": run.timings.as_dict(),
        "cost": run.cost.as_dict(),
        "chunks": run.chunk_count,
        "index_bytes": run.index_bytes,
        "scored_queries": run.scored_queries,
        "unresolved_gold": run.unresolved_gold,
        "confidence_interval": (
            None if interval is None else {"low": interval.low, "high": interval.high}
        ),
        "by_type": run.by_type,
        "failures": (None if run.failures is None else run.failures.counts()),
        # The per-question scores, so somebody can re-run the statistics themselves.
        "per_query": run.per_query,
        # And what the model actually said, so a generation score can be checked rather than
        # believed. Absent entirely for a run with no generator.
        "answers": run.answers,
        # Searches and model calls the strategy made, so a cost figure can be checked
        # against something rather than taken on trust.
        "retrieval": run.retrieval,
        "warnings": run.warnings.to_list(),
    }


def _header(manifest: Manifest | None) -> list[str]:
    """The provenance block a config export opens with.

    Which run produced this file is the first thing anybody asks of a config found in a
    repository, and the hashes are the only answer that cannot be misremembered.
    """
    lines = ["# context-grid configuration", "#"]
    if manifest is not None:
        lines += [
            f"# manifest: {manifest.short_hash}",
            f"# corpus:   {manifest.corpus_hash[:12]} ({manifest.corpus_files} files)",
            f"# evalset:  {manifest.evalset_id} v{manifest.evalset_version}",
            "#",
        ]
    return lines


def _experiment_from_paths(
    corpus: str | Path, evalset: str | Path | None, *, headline: str, k: int
) -> ExperimentConfig:
    """The least an experiment can be: where the documents are, and what won.

    `write_bundle` is reached from the ad-hoc side of the tool -- `contextgrid sweep`, and the
    `Lab` API -- where there is no config file and so no `ExperimentConfig` to pass on. The
    paths are the only part that cannot be reconstructed; everything else is a default or is
    read off the winner, so this fills them in and hands the same writer the same shape.
    """
    from contextgrid.config.schema import ExperimentConfig, RunConfig

    return ExperimentConfig(
        corpus=Path(corpus).expanduser().resolve(),
        evalset=None if evalset is None else Path(evalset).expanduser().resolve(),
        run=RunConfig(headline=headline, k=k),
        name="winning-config",
    )


def _run_settings(config: Config, experiment: ExperimentConfig) -> list[tuple[str, Any]]:
    """The `run:` section, in the order somebody reads it.

    `k` comes off the winning `Config` rather than off `experiment.run`. They agree today --
    the matrix is built with `run.k` -- but the one that describes the pipeline that actually
    won is the one on the pipeline.
    """
    run = experiment.run
    return [
        ("mode", run.mode),
        ("k", config.k),
        ("headline", run.headline),
        ("seed", run.seed),
        ("resolution_policy", run.resolution_policy),
        ("resolution_threshold", run.resolution_threshold),
        ("machine_usd_per_hour", run.machine_usd_per_hour),
        ("cache", run.cache),
        ("model", run.model),
    ]


def _config_field_names() -> list[str]:
    """Every field on `Config`, in the order a person reads a pipeline.

    `as_dict()` already puts them in that order -- ingestion, then parse, chunk, embed, search
    -- so follow it, but take the set of names from the dataclass itself and append anything
    `as_dict()` has not been taught about. Both of them are hand-written orderings of the same
    fields, and the export must survive either one falling behind.
    """
    declared = [f.name for f in fields(Config)]
    ordered = [name for name in Config().as_dict() if name in declared]
    return ordered + [name for name in declared if name not in ordered]


def _yaml_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    needs_quotes = any(character in text for character in ":#{}[],&*?|-<>=!%@`") or not text
    return f'"{text}"' if needs_quotes else text


def format_leaderboard(results: Results, metric: str = "recall@5", limit: int = 20) -> str:
    """A fixed-width leaderboard for a terminal."""
    rows = results.leaderboard(metric)[:limit]
    if not rows:
        return "no results"

    width = max(len(str(row["config"])) for row in rows)
    lines = [f"{'configuration':{width}} {metric:>8} {'p95 ms':>8} {'$/1k':>8}"]
    lines.append("-" * (width + 28))
    for row in rows:
        lines.append(
            f"{row['config']:{width}} {row.get(metric, 0):8.3f} "
            f"{row.get('p95_ms', 0):8.1f} {row.get('cost_per_1k', 0):8.4f}"
        )
    return "\n".join(lines)
