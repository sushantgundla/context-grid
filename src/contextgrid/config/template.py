"""The starter config `contextgrid init` writes.

Deliberately shows every axis with the plugins that are *actually installed*, commented with
the ones that are not and how to get them. A template listing plugins the user cannot run
teaches them the wrong thing on their first contact with the tool.

**Three states, said apart, because they need three different things from the reader.** A
plugin can be installed and swept (it is on the axis line), installed and not swept (`also
available:` -- move the name up to the line and it runs), or not installed at all (`needs
pip install ...`, naming the extra). Rolling the last two together was the old behaviour and
it was the worst of both: `marker`, the one parser this installation genuinely cannot run,
was silently left out of the file altogether, while `text` -- which needs no extra at all --
sat in a comment that said nothing about how to use it. A comment that does not distinguish
"type this name" from "run this pip command" gives the reader no way to tell which they are
looking at.

**The axis line is a small sweep, not everything runnable.** The docs describe the chosen
values as "every plugin this installation can actually run", and that is the one thing here
deliberately not done: with every extra installed, `parser` alone would sweep seven parsers,
two of which download models on first use, before the reader has changed a single line. The
file already tells them to "start by sweeping one axis at a time", and a template that
contradicts its own advice on line one is not a good first experience. So the axis line stays
a small honest default and the comment underneath carries the rest, in the two states above.
"""

from __future__ import annotations

TEMPLATE = """\
# context-grid experiment
#
# Everything this file describes runs with:  contextgrid run {filename}
#
# Any axis takes one value or a list. A list means that axis gets swept; a single
# value means it is held still. Start by sweeping one axis at a time.

name: {name}

corpus: {corpus}
evalset: {evalset}

grid:
  # What goes into the index, and what a hit on it returns. A chunker makes those the
  # same thing; every strategy here deliberately breaks that. `parent-document` indexes
  # small chunks and returns the passage around them; `contextual` and the rest pay a
  # model call per chunk while building, and never again.
{ingesters}

  # What reads the documents. This is the axis nothing else in the field measures,
  # and on anything with tables it is usually the one that matters most.
{parsers}

  # How the text is cut up. Sizes are in tokens, and the tokenizer is recorded on
  # every chunk -- "512" means different text under different tokenizers.
{chunkers}

  # What turns text into vectors. `null` means none, which is what BM25 wants.
{embedders}

  # How the search is done.
{indexes}

  # Rewriting the question before searching with it. Each one costs a model call on
  # every query forever, so the interesting question is whether it earns that.
{transforms}

  # How the index is used, as opposed to what it is. `simple` is one search; the
  # others widen the net or split the question. Agentic retrieval goes here.
{retrievers}

  # Reordering what came back. `candidates` is how deep the reranker gets to look,
  # and it is where most of the effect lives.
{rerankers}
  candidates: [50]

  # Turning the retrieved, reranked passages into an answer. `null` means no generation at
  # all -- the sweep stops at retrieval, at no extra cost. `llm` costs a model call per
  # question, forever, same as a query transform.
{generators}

run:
  mode: ofat              # ofat | factorial | staged
  k: 10                   # how many chunks reach the generator
  headline: recall@5      # what the leaderboard sorts on
  cache: memory           # memory | disk | none

  # Extra metrics to compute alongside the six built-ins (recall, precision, hit_rate,
  # mrr, map, ndcg) and whatever `headline` names -- one value or a list, same as any
  # axis. Only useful once you've registered a custom `Metric` --
  # https://context-grid.mintlify.site/concepts/plugins
  # metrics: [weighted_recall]

  # Stop rather than run forever. Leave both out for no limit. A sweep containing a
  # strategy that decides its own number of model calls has no ceiling without them.
  # budget_seconds: 900
  # budget_usd: 5.00

  # Prices local compute by the hour, so a CPU model and a hosted API land on the
  # same chart. A local model is free per token and not free to run.
  # machine_usd_per_hour: 0.10

report:
  out: ./results
  formats: [markdown, json, yaml, python]
"""


def render(
    *,
    filename: str = "contextgrid.yaml",
    name: str = "experiment",
    corpus: str = "./documents",
    evalset: str = "./questions.jsonl",
) -> str:
    """Build a starter config listing what this installation can actually run."""
    from contextgrid.chunk import CHUNKERS
    from contextgrid.embed import EMBEDDERS
    from contextgrid.generate import GENERATORS
    from contextgrid.generate import MODEL_BACKED as GENERATOR_MODEL_BACKED
    from contextgrid.index import INDEXES
    from contextgrid.ingest import INGESTERS
    from contextgrid.parse import PARSERS
    from contextgrid.rerank import RERANKERS
    from contextgrid.retrieve import RETRIEVERS
    from contextgrid.transform import MODEL_BACKED, TRANSFORMS

    return TEMPLATE.format(
        filename=filename,
        name=name,
        corpus=corpus,
        evalset=evalset,
        ingesters=_axis("ingestion", ["plain"], INGESTERS),
        parsers=_axis("parser", ["markdown"], PARSERS),
        chunkers=_axis("chunker", ["recursive:512", "sentence:3"], CHUNKERS),
        embedders=_axis("embedder", ["tfidf", "null"], EMBEDDERS),
        indexes=_axis("index", ["dense", "bm25", "hybrid"], INDEXES),
        transforms=_axis("transform", ["null"], TRANSFORMS, extra=MODEL_BACKED, needs_model=True),
        retrievers=_axis("retrieval", ["simple"], RETRIEVERS),
        rerankers=_axis("reranker", ["null", "lexical"], RERANKERS),
        generators=_axis(
            "generator", ["null"], GENERATORS, extra=GENERATOR_MODEL_BACKED, needs_model=True
        ),
    )


def _axis(
    name: str,
    chosen: list[str],
    registry: object,
    extra: tuple[str, ...] = (),
    *,
    needs_model: bool = False,
) -> str:
    """One axis line, then a comment for each of the two states the line does not cover.

    `extra` is for values that cannot be registered but do exist -- the transforms and the
    generator needing a model, which were reachable only by somebody who already knew their
    names. `needs_model` says so out loud, because those names are installed, are not blocked
    by any extra, and still will not run until `run.model` is set: moving one up to the axis
    line without that is the one move this comment invites and `check` then refuses.
    """
    installed, blocked = _split(registry)
    unrunnable = {name for names in blocked.values() for name in names}

    # A chosen default that cannot run here would be a config `init` wrote and `check` then
    # rejected. Nothing optional is chosen today, so this never fires -- it is here so that
    # the day one of these defaults acquires an extra, the template degrades instead of lying.
    chosen = [value for value in chosen if _base(registry, value) not in unrunnable]
    already = {_base(registry, value) for value in chosen}

    lines = [f"  {name}: [{', '.join(chosen)}]"]

    # `recursive:512` is already on the line; listing `recursive` again under "also available"
    # reads as a second, different plugin.
    rest = sorted({*installed, *extra} - already)
    if rest:
        lines.append(_wrapped("also available: " + ", ".join(rest)))
    if extra and needs_model:
        names = sorted(extra)
        verb = "needs" if len(names) == 1 else "need"
        lines.append(_wrapped(f"of those, {_listed(names)} {verb} `run.model` set."))

    # Grouped by extra rather than one line per plugin: `pip install "context-grid[parse]"` is
    # the same command for three parsers, and printing it three times makes it look like three
    # different installs. The command comes first so wrapping cannot break it in half.
    for missing_extra, names in sorted(blocked.items()):
        lines.append(
            _wrapped(f'needs `pip install "context-grid[{missing_extra}]"`: {", ".join(names)}')
        )

    return "\n".join(lines)


def _split(registry: object) -> tuple[list[str], dict[str, list[str]]]:
    """The plugins that can run here, and the ones that cannot grouped by the extra they want.

    Every optional plugin is registered whether or not its package is installed, so the
    registry alone would advertise chunkers that raise on first use. Somebody's first contact
    with the tool should not be an ImportError from a file the tool wrote for them.

    Asked through `extra_missing_for`, which is what `contextgrid check` asks, so the file
    `init` writes and the verdict `check` gives on it come from one fact rather than two
    opinions. The two used to disagree, and in both directions: this module tried to import
    the *distribution* name, so `faiss` (whose distribution is `faiss-cpu` and whose module is
    `faiss`) was left out of a template on an installation that runs it perfectly well, and
    `marker` was excluded for the same wrong reason rather than for the right one.
    """
    from contextgrid.config.plugins import extra_missing_for

    installed: list[str] = []
    blocked: dict[str, list[str]] = {}
    for entry in registry:  # type: ignore[attr-defined]
        absent = extra_missing_for(entry)
        if absent is None:
            installed.append(str(entry.name))
        else:
            blocked.setdefault(str(entry.extra), []).append(str(entry.name))
    return sorted(installed), {extra: sorted(names) for extra, names in blocked.items()}


def _base(registry: object, value: str) -> str:
    """The plugin a chosen value names. `recursive:512` is the `recursive` chunker.

    Asked of the registry rather than split on the first colon, because a colon is in some
    plugin *names* too: `chonkie:recursive:512` is `chonkie:recursive` with a size, and
    splitting naively would look for a plugin called `chonkie`.
    """
    return str(registry.name_in(value))  # type: ignore[attr-defined]


def _listed(names: list[str]) -> str:
    """`a, b and c`, so a sentence about four transforms reads as a sentence."""
    if len(names) < 2:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _wrapped(text: str, width: int = 88) -> str:
    """Comment lines wrapped, because a 200-character line in a config nobody can read."""
    import textwrap

    return "\n".join(
        f"  # {line}" for line in textwrap.wrap(text, width=width, subsequent_indent="  ")
    )
