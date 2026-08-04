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
from pathlib import Path
from typing import Any

from contextgrid.pipeline import Config
from contextgrid.report.manifest import Manifest
from contextgrid.report.results import Results, RunResult
from contextgrid.score.significance import SignificanceError


def config_to_yaml(config: Config, *, manifest: Manifest | None = None) -> str:
    """The configuration as YAML, hand-written rather than via a dependency.

    The core installs with numpy and nothing else, and pulling in a YAML library to emit
    twelve flat keys would be a poor trade.
    """
    lines = ["# context-grid configuration", "#"]
    if manifest is not None:
        lines += [
            f"# manifest: {manifest.short_hash}",
            f"# corpus:   {manifest.corpus_hash[:12]} ({manifest.corpus_files} files)",
            f"# evalset:  {manifest.evalset_id} v{manifest.evalset_version}",
            "#",
        ]
    lines.append("")

    for key, value in config.as_dict().items():
        lines.append(f"{key}: {_yaml_value(value)}")
    return "\n".join(lines) + "\n"


def config_to_python(config: Config) -> str:
    """Runnable Python that rebuilds this configuration.

    Not a template with holes in it -- this actually runs, which is the difference between an
    export somebody uses and one they read once and retype.
    """
    reranker = f"\n    reranker={config.reranker!r}," if config.reranker else ""
    candidates = f"\n    candidates={config.candidates}," if config.reranker else ""

    return f'''"""The winning configuration, as context-grid found it."""

import contextgrid as cg

config = cg.Config(
    parser={config.parser!r},
    chunker={config.chunker!r},
    embedder={config.embedder!r},
    index={config.index!r},{reranker}{candidates}
    k={config.k},
)

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
) -> str:
    """A one-page report to paste into a decision doc.

    Ordered the way somebody reads it rather than the way it was computed: the conclusion
    first, then the evidence, then the caveats. A report that opens with a methodology section
    does not get read.
    """
    lines = ["# Retrieval configuration comparison", ""]

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

    lines += ["## Leaderboard", ""]
    header = f"| Configuration | {metric} | p95 ms | $/1k queries | Chunks |"
    lines += [header, "|---|---:|---:|---:|---:|"]
    for row in results.leaderboard(metric)[:limit]:
        lines.append(
            f"| `{row['config']}` | {row.get(metric, 0):.3f} | "
            f"{row.get('p95_ms', 0):.1f} | {row.get('cost_per_1k', 0):.4f} | "
            f"{row.get('chunks', 0)} |"
        )
    lines.append("")

    axes = [
        ("parser", "Parser"),
        ("chunker", "Chunker"),
        ("embedder", "Embedder"),
        ("index", "Index"),
        ("reranker", "Reranker"),
    ]
    effects = [(label, results.axis_effect(axis, metric)) for axis, label in axes]
    informative = [(label, effect) for label, effect in effects if len(effect) > 1]
    if informative:
        lines += ["## Which decision mattered", ""]
        for label, effect in informative:
            spread = max(effect.values()) - min(effect.values())
            best = max(effect, key=lambda value: effect[value])
            lines.append(
                f"- **{label}**: `{best}` was best, {spread:+.3f} over the worst value tried."
            )
        lines.append("")

    winner = results.best(metric)
    if winner is not None and winner.failures is not None and winner.failures.failures():
        lines += ["## Why the rest failed", "", winner.failures.summary(), ""]

    if not results.warnings.is_sound:
        lines += ["## Warnings", ""]
        for warning in results.warnings.invalidating:
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
) -> list[Path]:
    """Write everything: the report, the raw results, the winning config and the manifest.

    A sceptic should be able to re-derive every number in the report from the bundle without
    asking for anything else.
    """
    root = Path(directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    report = root / "report.md"
    report.write_text(results_to_markdown(results, metric=metric, manifest=manifest), "utf-8")
    written.append(report)

    raw = root / "results.json"
    raw.write_text(results_to_json(results, manifest=manifest), "utf-8")
    written.append(raw)

    winner = results.best(metric)
    if winner is not None:
        config_yaml = root / "winning-config.yaml"
        config_yaml.write_text(config_to_yaml(winner.config, manifest=manifest), "utf-8")
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
        "warnings": run.warnings.to_list(),
    }


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
