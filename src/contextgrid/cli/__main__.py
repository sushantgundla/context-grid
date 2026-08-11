"""The command line.

Where serious use happens. A sweep somebody runs by hand once is a demo; a sweep in a
Makefile, in CI, or in a shell history is a tool.

No dependency on a CLI framework. `argparse` is in the standard library, this has five
commands, and the core of this package installs with numpy and nothing else.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from contextgrid import __version__
from contextgrid.core.registry import Registry

if TYPE_CHECKING:  # imported for types only, so `contextgrid --version` stays a cheap import
    from contextgrid.config.schema import ExperimentConfig
    from contextgrid.report.results import Results


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

    # Worked out before anything is printed, so the reasons can appear next to the empty
    # leaderboard they explain rather than only in the error stream. `budget_usd: 0.0` is
    # documented as "already spent -- nothing runs, and the report says why rather than
    # showing an empty leaderboard as if the matrix had been covered", and stdout is the
    # report most people read. It printed "no results" and "No configurations were run."
    # with the reason on stderr, so a redirect, a pipe or a scrollback lost it.
    reasons = _why_nothing_ran(config, results) if not results.runs else []

    print()
    print(format_leaderboard(results, config.run.headline, config.report.leaderboard_limit))
    print()
    print(results.summary(config.run.headline))
    for reason in reasons:
        print(f"  {reason}")
    print()
    print(f"cache: {results.cache_summary}")

    written = write_report(config, results)
    if written:
        print(f"\nwrote {len(written)} files to {config.report.out}")

    if not results.runs:
        # A sweep that measured nothing is a failure, not a result. `budget_usd: 0.0`, and a
        # matrix whose only cell cannot be built, both printed "No configurations were run."
        # and exited 0 -- a green CI build for an experiment that ran nothing at all.
        #
        # Only *nothing* is non-zero. A sweep that ran some of its configurations and was then
        # stopped by its budget did measure something: the leaderboard it printed is real, and
        # failing there would make `budget_seconds` unusable in CI, which is where it earns its
        # keep. The runner already marks that case CAUTION and says the table is partial.
        print("error: no configurations were run, so nothing was measured", file=sys.stderr)
        for reason in reasons:
            print(f"error: {reason}", file=sys.stderr)
        return 1
    return 0


def _why_nothing_ran(config: ExperimentConfig, results: Results) -> list[str]:
    """The reasons the sweep is empty, in the words the runner already used for them.

    The reasons are recorded as warnings, and warnings only reach the written report -- so on
    a console the exit code would have been the only signal, and "why" would have been left to
    guesswork.
    """
    from contextgrid.core.warnings import WarningCode

    reasons = [w.message for w in results.warnings.of_code(WarningCode.IMPOSSIBLE_COMBINATION)]

    # BUDGET_REACHED covers two different things: a sweep stopping, and an advisory that a
    # model-calling plugin has no ceiling. The advisory is only ever added when no budget was
    # set at all, so where the config sets one, every BUDGET_REACHED here is the sweep saying
    # where it stopped.
    if config.run.budget_seconds is not None or config.run.budget_usd is not None:
        reasons += [w.message for w in results.warnings.of_code(WarningCode.BUDGET_REACHED)]
    return reasons


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
    else:
        problems.extend(_corpus_problems(config))
    if config.evalset is None:
        problems.append("no evalset, so there is nothing to score against")
    elif not config.evalset.exists():
        problems.append(f"eval set not found: {config.evalset}")

    for axis, values in config.grid.as_dict().items():
        print(f"  {axis:11} {values}")

    problems.extend(_plugin_problems(config))

    if problems:
        print()
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1

    print("\nconfig is valid.")
    return 0


def _corpus_problems(config: ExperimentConfig) -> list[str]:
    """Whether the corpus has anything in it that can actually be read.

    `corpus.exists()` was the whole check, so a directory holding nothing -- or holding only
    files no parser is registered for -- passed `check` and failed in `run`. That is the exact
    shape of mistake this command exists to catch: an empty `./documents` is what you get from
    a clone without its data, or a `git clean`, or a path off by one directory.

    `Corpus.from_dir` is what `run` calls and what raises the message, so this reuses it rather
    than re-implementing the glob. `max_files=1` because the question is "is there anything
    readable here", and the emptiness check happens before any file is read -- so `check` never
    pulls a whole corpus into memory to answer it.
    """
    from contextgrid.corpus import Corpus

    try:
        if config.corpus.is_dir():
            Corpus.from_dir(config.corpus, max_files=1)
        else:
            Corpus.from_files([config.corpus])
    except Exception as error:
        return [str(error)]
    return []


def _plugin_problems(config: ExperimentConfig) -> list[str]:
    """Build every plugin the matrix names, and report whatever refuses to be built.

    `check` used to stop at parsing, so it caught a typo'd *key* (`chunkers:`) and missed a
    typo'd *value* -- `chunker: banana:999` and `chunker: recursive:-5` both reported "config
    is valid." and then failed in `run`, which is the wrong time and the reason somebody ran
    `check` in the first place. Spec strings are where typos actually happen.

    Construction is what raises both errors, and construction is cheap: no document is read,
    nothing is embedded, no index is built and no model is called. So build one of each and
    report the same message `run` would, just sooner.

    Values are taken from the expanded matrix rather than from `config.grid`, so a value the
    matrix drops as impossible is not reported as a problem in a sweep that will never run it.
    """
    from contextgrid.chunk import get_chunker
    from contextgrid.config.loader import build_llm
    from contextgrid.embed import get_embedder
    from contextgrid.generate import get_generator
    from contextgrid.index import get_index
    from contextgrid.ingest import get_ingester
    from contextgrid.parse import get_parser
    from contextgrid.rerank import get_reranker
    from contextgrid.retrieve import get_retriever
    from contextgrid.transform import get_transform

    # Builds a client, never calls one. The model-backed transforms, strategies and generators
    # cannot be built without it, and refusing to build them is precisely what tells the user
    # that `transform: hyde` with no `run.model` is not going to work.
    llm: Any = build_llm(config)

    builders: dict[str, Callable[[str], object]] = {
        "ingestion": get_ingester,
        "parser": get_parser,
        "chunker": get_chunker,
        "embedder": get_embedder,
        "index": get_index,
        "transform": lambda spec: get_transform(spec, llm),
        "retrieval": lambda spec: get_retriever(spec, llm),
        "reranker": get_reranker,
        "generator": lambda spec: get_generator(spec, llm),
    }

    problems: list[str] = []
    built: set[tuple[str, str]] = set()

    for candidate in config.grid.to_matrix(config.run.k).expand(config.run.mode):
        for axis, build in builders.items():
            spec = getattr(candidate, axis)
            # `None` is a real value on most axes -- no reranker, no transform -- and there is
            # nothing to build for it.
            if spec is None or (axis, spec) in built:
                continue
            built.add((axis, spec))
            try:
                build(spec)
            except Exception as error:
                # Prefixed with the axis and the spec because the messages are written for the
                # moment a plugin is built, when there is only one: "chunk size must be
                # positive, got -5" does not say which of six chunkers in the config said it.
                problems.append(f"{axis} {spec!r}: {error}")

    return problems


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
        # The corpus and eval set the sweep actually ran over, so `winning-config.yaml` in the
        # bundle is a config you can hand straight back to `contextgrid run`. Without them
        # `write_bundle` falls back to a flat listing of pipeline fields -- a record of what ran,
        # not something that can re-run it, which is the whole point of writing a bundle.
        written = write_bundle(
            results,
            args.bundle,
            metric=args.metric,
            manifest=manifest,
            corpus=args.corpus,
            evalset=args.evalset,
        )
        print(f"\nwrote {len(written)} files to {args.bundle}")

    return 0


def _plural(name: str) -> str:
    """The axis name as a heading. Only `index` is irregular."""
    return "indexes" if name == "index" else f"{name}s"


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

    # A family nobody registered printed nothing and exited 0, which reads as "this
    # installation has no such plugins" rather than "you typed a name this flag does not
    # know". The headings are plural and the flag is singular, so `--family chunkers` -- the
    # word printed one line above -- was the easiest possible way to hit it.
    #
    # Checked here rather than with argparse `choices=`, because filling those in means
    # importing all six registries to parse *any* command line, and `contextgrid --version`
    # is deliberately a cheap import.
    if args.family is not None and args.family not in families:
        raise ValueError(
            f"unknown plugin family {args.family!r}. Valid families: {', '.join(families)}"
        )

    for name, registry in families.items():
        if args.family and args.family != name:
            continue
        # Not `f"{name}s"`: that printed "indexs", and every page of the documentation calls
        # this axis `index` or `indexes`. A heading nobody can search for is a small thing that
        # makes a reference feel untrustworthy.
        print(f"{_plural(name)}:")
        for plugin, description in registry.describe().items():
            print(f"  {plugin:24} {description}")
        print()
    return 0


def _evalset(args: argparse.Namespace) -> int:
    from contextgrid.evalset import assess, type_distribution
    from contextgrid.evalset.io import read_evalset

    # `read_evalset`, not `read_jsonl`: a config's `evalset:` has always accepted either
    # format, and CSV is the one a subject-matter expert actually hands you -- so the command
    # for looking at a set they just handed you was the only place that refused it, with a
    # JSON parse error about line 1 rather than anything about formats.
    evalset = read_evalset(args.path)
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
