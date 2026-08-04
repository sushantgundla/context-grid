"""The command line.

Where serious use happens. A sweep somebody runs by hand once is a demo; a sweep in a
Makefile, in CI, or in a shell history is a tool.

No dependency on a CLI framework. `argparse` is in the standard library, this has five
commands, and the core of this package installs with numpy and nothing else.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from contextgrid import __version__
from contextgrid.core.registry import Registry


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    handler = {
        "run": _run_config,
        "init": _init,
        "check": _check,
        "profile": _profile,
        "sweep": _sweep,
        "plugins": _plugins,
        "evalset": _evalset,
        "diff": _diff,
        "validate": _validate,
    }[args.command]

    try:
        return handler(args)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contextgrid",
        description="Sweep retrieval configurations on your own documents.",
    )
    parser.add_argument("--version", action="version", version=f"context-grid {__version__}")
    sub = parser.add_subparsers(dest="command")

    execute = sub.add_parser("run", help="Run everything a config file describes.")
    execute.add_argument("config", type=Path, help="A YAML or JSON experiment file.")
    execute.add_argument("--quiet", action="store_true", help="Suppress per-config progress.")

    starter = sub.add_parser("init", help="Write a starter config for this installation.")
    starter.add_argument("path", type=Path, nargs="?", default=Path("contextgrid.yaml"))
    starter.add_argument("--corpus", default="./documents")
    starter.add_argument("--evalset", default="./questions.jsonl")
    starter.add_argument("--force", action="store_true", help="Overwrite an existing file.")

    inspect = sub.add_parser("check", help="Validate a config and say what it would run.")
    inspect.add_argument("config", type=Path)

    profile = sub.add_parser("profile", help="Profile a corpus and say which axes will matter.")
    profile.add_argument("corpus", type=Path)
    profile.add_argument("--parser", default="markdown")

    sweep = sub.add_parser("sweep", help="Run a matrix and print the leaderboard.")
    sweep.add_argument("corpus", type=Path)
    sweep.add_argument("evalset", type=Path, help="A JSONL eval set.")
    sweep.add_argument("--parser", action="append")
    sweep.add_argument("--chunker", action="append")
    sweep.add_argument("--embedder", action="append")
    sweep.add_argument("--index", action="append")
    sweep.add_argument("--reranker", action="append")
    sweep.add_argument("--mode", default="ofat", choices=["factorial", "ofat", "staged"])
    sweep.add_argument("--metric", default="recall@5")
    sweep.add_argument("--k", type=int, default=10)
    sweep.add_argument("--budget-seconds", type=float, default=None)
    sweep.add_argument("--bundle", type=Path, help="Write a full result bundle here.")

    plugins = sub.add_parser("plugins", help="List everything registered.")
    plugins.add_argument("--family", default=None)

    evalset = sub.add_parser("evalset", help="Inspect an eval set and what it can support.")
    evalset.add_argument("path", type=Path)

    check = sub.add_parser("validate", help="Check the scorer against a published benchmark.")
    check.add_argument("benchmark", type=Path, help="A LegalBench-RAG JSON file.")
    check.add_argument("corpus", type=Path, help="The documents its spans point into.")
    check.add_argument("--limit", type=int, default=None)
    check.add_argument(
        "--recall-at-10", type=float, default=None, help="The published number to compare with."
    )

    diff = sub.add_parser("diff", help="Say what changed between two run manifests.")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)

    return parser


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def _run_config(args: argparse.Namespace) -> int:
    from contextgrid.config import load, run, write_report
    from contextgrid.report import format_leaderboard

    config = load(args.config)
    print(config.describe())

    progress = None
    if not args.quiet:

        def progress(index: int, total: int, cfg: object) -> None:
            print(f"  [{index}/{total}] {cfg.label}", file=sys.stderr)  # type: ignore[attr-defined]

    results = run(config, on_progress=progress)

    print()
    print(format_leaderboard(results, config.run.headline, config.report.leaderboard_limit))
    print()
    print(results.summary(config.run.headline))
    print()
    print(f"cache: {results.cache_summary}")

    written = write_report(config, results)
    if written:
        print(f"\nwrote {len(written)} files to {config.report.out}")
    return 0


def _init(args: argparse.Namespace) -> int:
    from contextgrid.config import render

    if args.path.exists() and not args.force:
        print(f"error: {args.path} already exists. Pass --force to overwrite.", file=sys.stderr)
        return 1

    args.path.write_text(
        render(
            filename=args.path.name, name=args.path.stem, corpus=args.corpus, evalset=args.evalset
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.path}")
    print(f"edit it, then run:  contextgrid run {args.path}")
    return 0


def _check(args: argparse.Namespace) -> int:
    """Validate a config and say what it would do, without running anything."""
    from contextgrid.config import load

    config = load(args.config)
    print(config.describe())

    problems: list[str] = []
    if not config.corpus.exists():
        problems.append(f"corpus not found: {config.corpus}")
    if config.evalset is None:
        problems.append("no evalset, so there is nothing to score against")
    elif not config.evalset.exists():
        problems.append(f"eval set not found: {config.evalset}")

    for axis, values in config.grid.as_dict().items():
        print(f"  {axis:11} {values}")

    if problems:
        print()
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1

    print("\nconfig is valid.")
    return 0


def _profile(args: argparse.Namespace) -> int:
    from contextgrid.lab import Lab

    lab = Lab(args.corpus)
    fingerprint = lab.fingerprint(parser=args.parser)
    print(fingerprint.summary())
    for hint in fingerprint.hints():
        print(f"  - {hint}")
    return 0


def _sweep(args: argparse.Namespace) -> int:
    from contextgrid.evalset import read_jsonl
    from contextgrid.lab import Lab
    from contextgrid.report import build_manifest, format_leaderboard, write_bundle

    lab = Lab(args.corpus)
    evalset = read_jsonl(args.evalset)

    lab.grid(
        parser=args.parser or "markdown",
        chunker=args.chunker or "recursive:512",
        embedder=args.embedder or "tfidf",
        index=args.index or "dense",
        reranker=args.reranker or None,
        k=args.k,
    )

    estimate = lab.estimate(args.mode)
    print(f"{estimate['shape']} on paper, {estimate['configurations']} to run ({args.mode})")

    results = lab.run(
        evalset,
        mode=args.mode,
        headline=args.metric,
        budget_seconds=args.budget_seconds,
        on_progress=lambda index, total, config: print(
            f"  [{index}/{total}] {config.label}", file=sys.stderr
        ),
    )

    print()
    print(format_leaderboard(results, args.metric))
    print()
    print(results.summary(args.metric))
    print()
    print(f"cache: {results.cache_summary}")

    if args.bundle:
        winner = results.best(args.metric)
        manifest = (
            build_manifest(winner.config, lab.corpus, evalset) if winner is not None else None
        )
        written = write_bundle(results, args.bundle, metric=args.metric, manifest=manifest)
        print(f"\nwrote {len(written)} files to {args.bundle}")

    return 0


def _plugins(args: argparse.Namespace) -> int:
    from contextgrid.chunk import CHUNKERS
    from contextgrid.embed import EMBEDDERS
    from contextgrid.index import INDEXES
    from contextgrid.parse import PARSERS
    from contextgrid.rerank import RERANKERS
    from contextgrid.tokens import TOKENIZERS

    families: dict[str, Registry[Any]] = {
        "parser": PARSERS,
        "chunker": CHUNKERS,
        "embedder": EMBEDDERS,
        "index": INDEXES,
        "reranker": RERANKERS,
        "tokenizer": TOKENIZERS,
    }

    for name, registry in families.items():
        if args.family and args.family != name:
            continue
        print(f"{name}s:")
        for plugin, description in registry.describe().items():
            print(f"  {plugin:24} {description}")
        print()
    return 0


def _evalset(args: argparse.Namespace) -> int:
    from contextgrid.evalset import assess, read_jsonl, type_distribution

    evalset = read_jsonl(args.path)
    quality = assess(evalset)

    print(f"{evalset.id} v{evalset.version} ({evalset.source})")
    print(quality.summary())
    print(f"types: {type_distribution(evalset)}")
    for warning in quality.warnings():
        print(f"  - {warning.message}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    from contextgrid.validate import load_benchmark, self_check, validate

    corpus, evalset = load_benchmark(args.benchmark, args.corpus, limit=args.limit)

    # Whether the corpus matches the annotations comes first. If it does not, nothing after
    # it means anything, and the cause is loading rather than retrieval.
    check = self_check(corpus, evalset)
    print(check["verdict"])
    if check["in_range_rate"] < 0.95:
        return 1

    print()
    reference = {"recall@10": args.recall_at_10} if args.recall_at_10 is not None else None
    print(validate(corpus, evalset, reference=reference).report())
    return 0


def _diff(args: argparse.Namespace) -> int:
    from contextgrid.report import Manifest, explain_diff

    print(explain_diff(Manifest.load(args.before), Manifest.load(args.after)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
