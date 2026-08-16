"""The command line.

Where serious use happens. A sweep somebody runs by hand once is a demo; a sweep in a
Makefile, in CI, or in a shell history is a tool.

No dependency on a CLI framework. `argparse` is in the standard library, this has five
commands, and the core of this package installs with numpy and nothing else.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from contextgrid import __version__
from contextgrid.core.registry import Registry

if TYPE_CHECKING:  # imported for types only, so `contextgrid --version` stays a cheap import
    from contextgrid.config.schema import ExperimentConfig
    from contextgrid.corpus import Corpus
    from contextgrid.report.manifest import Manifest
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

    profile = sub.add_parser(
        "profile", help="Measure a corpus and flag settings its shape rules out."
    )
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

    # `reasons` is already on the console, immediately under the leaderboard it explains.
    _print_warnings(results, already_shown=reasons)

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


def _print_warnings(results: Results, *, already_shown: Sequence[str] = ()) -> None:
    """Put the run's warnings on the console, the same ones the written report carries.

    Warnings are data on the results object, and the only things reading that data were
    `report.md` and `results.json`. `report.out` defaults to null and `sweep` writes nothing
    without `--bundle`, so a sweep stopped by its budget after 1 of 15 configurations printed a
    one-row leaderboard, exited 0 and said nothing at all: the truncation was recorded in a
    file nobody asked to be written. A partial matrix that looks complete is the failure this
    package exists to prevent, so it belongs on the console whether or not a report is written.

    The rule is the report's rule, so the two agree: INFO is background detail when there are
    results to read, and worth saying when there are none.

    On stderr, with the progress lines, because stdout here is the leaderboard people pipe.
    `--quiet` does not silence these -- it turns off per-config progress, and "this leaderboard
    is a fifteenth of the matrix you asked for" is exactly what an unattended CI run needs said.
    """
    from contextgrid.core.warnings import Severity

    # Unconditionally, before anything is decided: stdout is block-buffered whenever it is not
    # a terminal, so in the CI logs this exists for, everything written to stderr from here on
    # -- these warnings, and the `error:` lines after them -- came out *above* the leaderboard
    # they are about. The budget warning literally says "the leaderboard below is partial".
    sys.stdout.flush()

    seen = set(already_shown)
    reportable = [
        warning
        for warning in results.warnings.entries
        if (warning.severity is not Severity.INFO or not results.runs)
        and warning.message not in seen
    ]
    if not reportable:
        return

    print(file=sys.stderr)
    for warning in reportable:
        # Prefixed with the code, as `report.md` prefixes them, so a CI log can be grepped for
        # `budget_reached` and the console and the report name the same thing the same way.
        print(f"warning: {warning.code.value}: {warning.message}", file=sys.stderr)


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
    else:
        problems.extend(_evalset_problems(config))

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
    than re-implementing the glob. `max_files=0`, not 1: the emptiness check is the glob, and
    it happens before the limit is applied, so zero is enough to ask the question and it is
    the only value that reads nothing at all. `max_files=1` answered the same question and
    pulled one document's bytes into memory doing it, which is a document more than
    `cli.md:105` -- "reads no documents, embeds nothing, indexes nothing" -- allows for.

    A corpus that is a single named file is the one place this still reads bytes, because
    there is no glob to ask instead and the file is the whole corpus.
    """
    from contextgrid.corpus import Corpus

    try:
        if config.corpus.is_dir():
            Corpus.from_dir(config.corpus, max_files=0)
        else:
            Corpus.from_files([config.corpus])
    except Exception as error:
        return [str(error)]
    return []


def _evalset_problems(config: ExperimentConfig) -> list[str]:
    """Whether the questions file can actually be read, and has questions in it.

    `evalset.exists()` was the whole check, so `check` resolved the path and never opened the
    file. A JSONL of something that is not JSON, or a spreadsheet whose question column is
    called something nobody guessed, both got "config is valid." and then stopped `run` a
    second later with an error `check` was standing right next to. `cli.md` promises that
    every message `check` prints is "the one `run` would have printed" and only earlier, so
    this reuses `build_evalset` -- literally the function `run` calls -- rather than a second
    reader that could disagree with it about what a valid eval set is.

    **An eval set with no items is the third case, and it is not an error `run` raises.** An
    empty file, or a JSONL carrying only its `_evalset` header line, parses perfectly and
    yields nothing to ask. The sweep then runs the whole matrix -- parsing, chunking,
    embedding and indexing every document -- to score zero questions, and reports every metric
    as zero next to a warning about unresolvable evidence. That is the most expensive way this
    tool has of telling somebody their questions file is empty, and it is exactly the money
    `check` exists to save.

    Cheap, and deliberately kept cheap: this opens one small text file the user wrote by hand.
    It reads no document, builds no index and calls no model, so the promise at `cli.md:105`
    still holds.
    """
    from contextgrid.config.loader import build_evalset

    try:
        evalset = build_evalset(config)
    except Exception as error:
        return [str(error)]

    if not evalset.items:
        return [f"{config.evalset} has no questions in it, so a sweep over it would score nothing"]
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

    Building is necessary but not sufficient. A plugin whose optional package is missing
    usually fails to build -- importing its module is what fails -- but not when the module is
    in this tree and defers the third-party import to the method that needs it. `marker` is
    that case: `check` built it, said "config is valid.", and `run` then died on the first PDF
    with `The marker parser requires the 'parse-marker' extra`. So each spec is asked first
    whether its extra is installed, by `missing_extra`, which reads the same registration field
    `contextgrid init` reads when it decides what to write into a starter config.

    Values are taken from the expanded matrix rather than from `config.grid`, so a value the
    matrix drops as impossible is not reported as a problem in a sweep that will never run it.
    """
    from contextgrid.chunk import CHUNKERS, get_chunker
    from contextgrid.config.loader import build_llm
    from contextgrid.config.plugins import missing_extra
    from contextgrid.embed import EMBEDDERS, get_embedder
    from contextgrid.generate import GENERATORS, get_generator
    from contextgrid.index import INDEXES, get_index
    from contextgrid.ingest import INGESTERS, get_ingester
    from contextgrid.parse import PARSERS, get_parser
    from contextgrid.rerank import RERANKERS, get_reranker
    from contextgrid.retrieve import RETRIEVERS, get_retriever
    from contextgrid.transform import TRANSFORMS, get_transform

    # Builds a client, never calls one. The model-backed transforms, strategies and generators
    # cannot be built without it, and refusing to build them is precisely what tells the user
    # that `transform: hyde` with no `run.model` is not going to work.
    llm: Any = build_llm(config)

    # Every axis `check` reports on, each with the registry that knows what its names need
    # installed. All nine, deliberately: `marker` is the only plugin in this tree that hides a
    # missing package from construction today, but "only parsers are checked" would be a rule
    # nobody could infer, and the next lazily-imported plugin would walk straight through it.
    builders: dict[str, tuple[Registry[Any], Callable[[str], object]]] = {
        "ingestion": (INGESTERS, get_ingester),
        "parser": (PARSERS, get_parser),
        "chunker": (CHUNKERS, get_chunker),
        "embedder": (EMBEDDERS, get_embedder),
        "index": (INDEXES, get_index),
        "transform": (TRANSFORMS, lambda spec: get_transform(spec, llm)),
        "retrieval": (RETRIEVERS, lambda spec: get_retriever(spec, llm)),
        "reranker": (RERANKERS, get_reranker),
        "generator": (GENERATORS, lambda spec: get_generator(spec, llm)),
    }

    problems: list[str] = []
    built: set[tuple[str, str]] = set()

    for candidate in config.grid.to_matrix(config.run.k).expand(config.run.mode):
        for axis, (registry, build) in builders.items():
            spec = getattr(candidate, axis)
            # `None` is a real value on most axes -- no reranker, no transform -- and there is
            # nothing to build for it.
            if spec is None or (axis, spec) in built:
                continue
            built.add((axis, spec))

            absent = missing_extra(registry, spec)
            if absent is not None:
                # Unprefixed, unlike the errors below: this message already names the plugin
                # and its axis ("The marker parser"), and it is quoted verbatim from what `run`
                # prints, so a user who has seen one recognises the other.
                problems.append(str(absent))
                continue

            try:
                build(spec)
            except Exception as error:
                # Prefixed with the axis and the spec because the messages are written for the
                # moment a plugin is built, when there is only one: "chunk size must be
                # positive, got -5" does not say which of six chunkers in the config said it.
                problems.append(f"{axis} {spec!r}: {error}")

    return problems


def _profile(args: argparse.Namespace) -> int:
    """Measure a corpus. `corpus` means here what it means everywhere else in this tool.

    `configuration.md` defines a corpus as "a directory of documents, or a single file", and
    `check` and `run` both honour that -- a config whose `corpus:` names one Markdown file
    validates and sweeps. `profile` was handed the path straight to `Lab`, which loads a
    directory, so the one command whose entire job is looking at a corpus was the one that
    refused half of them: `error: documents/billing.md is not a directory`.
    """
    from contextgrid.lab import Lab

    lab = Lab(_corpus_at(args.corpus))
    fingerprint = lab.fingerprint(parser=args.parser)
    print(fingerprint.summary())
    for hint in fingerprint.hints():
        print(f"  - {hint}")
    return 0


def _corpus_at(path: Path) -> Corpus:
    """Load a corpus from a path that may be a directory or a single file.

    The same two-way split `contextgrid.config.loader.build_corpus` makes for a config's
    `corpus:` field, kept in the CLI layer and expressed with the public constructors, so the
    commands taking a corpus as an argument and the configs naming one mean the same thing.
    """
    from contextgrid.corpus import Corpus

    if path.is_dir():
        return Corpus.from_dir(path)
    if not path.exists():
        raise ValueError(f"no corpus at {path}")
    return Corpus.from_files([path])


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

    # `sweep --budget-seconds 0.001` was the worst case of all: no bundle is written unless
    # `--bundle` is passed, so the warning saying the leaderboard is partial had nowhere to go.
    _print_warnings(results)
    return 0


def _plural(name: str) -> str:
    """The axis name as a heading, for a family `_HEADINGS` does not name.

    A fallback rather than the rule: the rule gets `index` wrong ("indexs"), and it has no
    idea what to do with `ingestion`. It is here so that a registry added later and forgotten
    about prints an awkward heading instead of crashing the command with a `KeyError`.
    """
    return "indexes" if name == "index" else f"{name}s"


#: The heading each family prints under. Written out rather than pluralised by rule, because
#: the rule is wrong more often than it is right here: `index` gives "indexs", and `ingestion`
#: and `retrieval` are not things you can have two of. Every value is the word the
#: documentation uses for that axis, so a heading printed here can be searched for there.
_HEADINGS: dict[str, str] = {
    "parser": "parsers",
    "ingestion": "ingestion strategies",
    "chunker": "chunkers",
    "embedder": "embedders",
    "index": "indexes",
    "transform": "transforms",
    "retrieval": "retrieval strategies",
    "reranker": "rerankers",
    "generator": "generators",
    "llm": "models (for `run.model`)",
    "metric": "metrics",
    "tokenizer": "tokenizers",
}


def _plugins(args: argparse.Namespace) -> int:
    """Everything registered, on every axis. `cli.md:21`: "List everything registered".

    It listed six of the twelve. The six it skipped -- `ingestion`, `transform`, `retrieval`,
    `generator`, `metric` and the models behind `run.model` -- are not obscure: four of them
    are sweepable axes with their own tables in `plugins.md`, which calls this command the
    authoritative live list. So `contextgrid plugins --family transform` answered "unknown
    plugin family 'transform'" about an axis the config file accepts and the starter template
    writes a comment about.

    Ordered as a pipeline rather than alphabetically -- parse, ingest, chunk, embed, index,
    transform, retrieve, rerank, generate -- so reading the output top to bottom is reading
    the order the stages actually run in. The three that are not stages come last.
    """
    from contextgrid.chunk import CHUNKERS
    from contextgrid.embed import EMBEDDERS
    from contextgrid.evalset.llm import LLMS
    from contextgrid.generate import GENERATORS
    from contextgrid.index import INDEXES
    from contextgrid.ingest import INGESTERS
    from contextgrid.parse import PARSERS
    from contextgrid.rerank import RERANKERS
    from contextgrid.retrieve import RETRIEVERS
    from contextgrid.score.base import METRICS
    from contextgrid.tokens import TOKENIZERS
    from contextgrid.transform import TRANSFORMS

    families: dict[str, Registry[Any]] = {
        "parser": PARSERS,
        "ingestion": INGESTERS,
        "chunker": CHUNKERS,
        "embedder": EMBEDDERS,
        "index": INDEXES,
        "transform": TRANSFORMS,
        "retrieval": RETRIEVERS,
        "reranker": RERANKERS,
        "generator": GENERATORS,
        "llm": LLMS,
        "metric": METRICS,
        "tokenizer": TOKENIZERS,
    }

    # A family nobody registered printed nothing and exited 0, which reads as "this
    # installation has no such plugins" rather than "you typed a name this flag does not
    # know". The headings are plural and the flag is singular, so `--family chunkers` -- the
    # word printed one line above -- was the easiest possible way to hit it.
    #
    # Checked here rather than with argparse `choices=`, because filling those in means
    # importing all twelve registries to parse *any* command line, and `contextgrid --version`
    # is deliberately a cheap import.
    if args.family is not None and args.family not in families:
        raise ValueError(
            f"unknown plugin family {args.family!r}. Valid families: {', '.join(families)}"
        )

    for name, registry in families.items():
        if args.family and args.family != name:
            continue
        listing = dict(registry.describe())
        needs_model = _model_backed_in(name)
        listing.update(needs_model)

        print(f"{_HEADINGS.get(name, _plural(name))}:")
        for plugin, description in sorted(listing.items()):
            # A marker column rather than a suffix on the description: the note is the same
            # for every starred name, and repeating it on four lines out of six buries the
            # part that differs.
            mark = "*" if plugin in needs_model else " "
            print(f"  {plugin:22} {mark} {description}")
        if needs_model:
            print("  * needs a model. Set `run.model` in your config to use it.")
        print()
    return 0


def _model_backed_in(family: str) -> dict[str, str]:
    """The plugins on this axis that exist, are documented, and are not in the registry.

    `hyde` and the rest of `transform.MODEL_BACKED`, and the `llm` generator, cannot be
    registered: building one from a spec string alone would silently produce a transform with
    no model, which is the identity, and a config that looks like it is testing HyDE while
    testing nothing is worse than an error. `plugins.md:196` nevertheless says
    `available_transforms()` "is what `contextgrid plugins` and the config template actually
    print" -- and the template does print them. Only the command was short, so the axis looked
    like it had two arms when it has six.

    The descriptions are the classes' own first docstring lines, not invented here. They read
    in exactly the style of the registry's `doc=` strings because the same person wrote them
    for the same purpose; the only difference is where they are stored.
    """
    if family == "transform":
        from contextgrid.transform import MODEL_BACKED, Decompose, HyDE, MultiQuery, StepBack

        classes: dict[str, type] = {
            "hyde": HyDE,
            "multi-query": MultiQuery,
            "decompose": Decompose,
            "step-back": StepBack,
        }
        return {name: _first_line(classes[name]) for name in MODEL_BACKED if name in classes}

    if family == "generator":
        from contextgrid.generate import LLMGenerator

        return {"llm": _first_line(LLMGenerator)}

    return {}


def _first_line(cls: type) -> str:
    """A class's one-line summary, or a plain admission that it has none."""
    doc = (cls.__doc__ or "").strip()
    return doc.splitlines()[0].strip() if doc else "(no description)"


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
    from contextgrid.report import explain_diff

    print(explain_diff(_load_manifest(args.before), _load_manifest(args.after)))
    return 0


def _load_manifest(path: Path) -> Manifest:
    """Read one manifest, and on failure say which file was wrong and what was wrong with it.

    `Manifest.load` reads, parses and unpacks in one call, so whatever the standard library
    raised was what reached the user: `error: 'config'` for a JSON file that is not a manifest,
    `error: Expecting value: line 1 column 1 (char 0)` for a file that is not JSON at all,
    `error: [Errno 2] No such file or directory: 'nope.json'` for a path that does not exist.
    None of them says what a manifest is, and the first two do not even name the file -- which
    matters more here than anywhere else in this CLI, because `diff` is handed two of them and
    the message has to say which one it is complaining about.
    """
    import json

    from contextgrid.report import Manifest

    hint = "A manifest is the manifest.json written into a run's bundle."
    try:
        return Manifest.load(path)
    except FileNotFoundError:
        raise ValueError(f"no manifest at {path}. {hint}") from None
    except IsADirectoryError:
        raise ValueError(
            f"{path} is a directory, not a manifest. Try {path / 'manifest.json'}"
        ) from None
    except OSError as error:
        raise ValueError(f"could not read {path}: {error.strerror or error}") from None
    except UnicodeDecodeError:
        raise ValueError(f"{path} is not text, so it is not a manifest. {hint}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}. {hint}") from None
    except KeyError as error:
        # The manifest fields are read by name, so a JSON file of the wrong shape surfaced as a
        # bare key: `error: 'config'`, which reads like an internal fault rather than a file the
        # user pointed at.
        raise ValueError(
            f"{path} is JSON, but not a manifest: no {error.args[0]!r} in it. {hint}"
        ) from None
    except TypeError:
        # A JSON file whose top level is a list, a string or a number: indexing it by name
        # raises rather than missing a key.
        raise ValueError(
            f"{path} is JSON, but not a manifest: it is not an object. {hint}"
        ) from None


if __name__ == "__main__":
    raise SystemExit(main())
