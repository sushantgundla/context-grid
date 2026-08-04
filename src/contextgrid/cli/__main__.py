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
        "profile": _profile,
        "sweep": _sweep,
        "plugins": _plugins,
        "evalset": _evalset,
        "diff": _diff,
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

    diff = sub.add_parser("diff", help="Say what changed between two run manifests.")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)

    return parser


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


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


def _diff(args: argparse.Namespace) -> int:
    from contextgrid.report import Manifest, explain_diff

    print(explain_diff(Manifest.load(args.before), Manifest.load(args.after)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
