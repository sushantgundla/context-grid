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

  # Reordering what came back. `candidates` is how deep the reranker gets to look,
  # and it is where most of the effect lives.
{rerankers}
  candidates: [50]

run:
  mode: ofat              # ofat | factorial | staged
  k: 10                   # how many chunks reach the generator
  headline: recall@5      # what the leaderboard sorts on
  cache: memory           # memory | disk | none

  # Stop rather than run forever. Leave out for no limit.
  # budget_seconds: 900

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
    from contextgrid.index import INDEXES
    from contextgrid.parse import PARSERS
    from contextgrid.rerank import RERANKERS
    from contextgrid.transform import TRANSFORMS

    return TEMPLATE.format(
        filename=filename,
        name=name,
        corpus=corpus,
        evalset=evalset,
        parsers=_axis("parser", ["markdown"], PARSERS),
        chunkers=_axis("chunker", ["recursive:512", "sentence:3"], CHUNKERS),
        embedders=_axis("embedder", ["tfidf", "null"], EMBEDDERS),
        indexes=_axis("index", ["dense", "bm25", "hybrid"], INDEXES),
        transforms=_axis("transform", ["null"], TRANSFORMS),
        rerankers=_axis("reranker", ["null", "lexical"], RERANKERS),
    )


def _axis(name: str, chosen: list[str], registry: object) -> str:
    """One axis line, plus a comment listing everything else available."""
    available = sorted(getattr(registry, "names", lambda: [])())
    rest = [plugin for plugin in available if plugin not in chosen]
    line = f"  {name}: [{', '.join(chosen)}]"
    if not rest:
        return line
    return f"{line}\n  # also available: {', '.join(rest)}"
