"""The starter config `contextgrid init` writes.

Deliberately shows every axis with the plugins that are *actually installed*, commented with
the ones that are not and how to get them. A template listing plugins the user cannot run
teaches them the wrong thing on their first contact with the tool.
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
        transforms=_axis("transform", ["null"], TRANSFORMS, extra=MODEL_BACKED),
        retrievers=_axis("retrieval", ["simple"], RETRIEVERS),
        rerankers=_axis("reranker", ["null", "lexical"], RERANKERS),
        generators=_axis("generator", ["null"], GENERATORS, extra=GENERATOR_MODEL_BACKED),
    )


def _axis(name: str, chosen: list[str], registry: object, extra: tuple[str, ...] = ()) -> str:
    """One axis line, plus a comment listing everything else this installation can run.

    `extra` is for values that cannot be registered but do exist -- the transforms needing a
    model, which were reachable only by somebody who already knew their names.
    """
    already = {value.split(":", 1)[0] if ":" in value else value for value in chosen}
    # `recursive:512` is already on the line; listing `recursive` again under "also available"
    # reads as a second, different plugin.
    rest = sorted({*_installed(registry), *extra} - already)

    line = f"  {name}: [{', '.join(chosen)}]"
    return line if not rest else f"{line}\n{_wrapped('also available: ' + ', '.join(rest))}"


def _installed(registry: object) -> list[str]:
    """The plugins whose dependencies are actually present.

    Every optional plugin is registered whether or not its package is installed, so the
    registry alone would advertise chunkers that raise on first use. Somebody's first contact
    with the tool should not be an ImportError from a file the tool wrote for them.
    """
    entries = list(getattr(registry, "__iter__", list)())
    names: list[str] = []
    for entry in entries:
        package = getattr(entry, "package", None) or getattr(entry, "module", None)
        if package and not _importable(str(package).split(".")[0].replace("-", "_")):
            continue
        names.append(str(entry.name))
    return sorted(names)


def _importable(module: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _wrapped(text: str, width: int = 88) -> str:
    """Comment lines wrapped, because a 200-character line in a config nobody can read."""
    import textwrap

    return "\n".join(
        f"  # {line}" for line in textwrap.wrap(text, width=width, subsequent_indent="  ")
    )
