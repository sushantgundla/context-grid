"""Retrieval strategies, and the registry of them.

The index is *where* the vectors live. The strategy is *how* they are used. Keeping the two
apart is what turns "should I use agentic retrieval?" from a rewrite into a cell in a grid.
"""

from __future__ import annotations

from contextgrid.core.registry import Registry
from contextgrid.retrieve.agentic import AgenticRetrieval
from contextgrid.retrieve.base import (
    Lookup,
    RetrievalStrategy,
    RetrievalTrace,
    Searcher,
    fuse,
)
from contextgrid.retrieve.strategies import (
    DecomposedRetrieval,
    RelevanceFeedbackRetrieval,
    RetrievalError,
    SimpleRetrieval,
    WidenedRetrieval,
)

RETRIEVERS: Registry[RetrievalStrategy] = Registry(family="retrieval")

RETRIEVERS.register(
    "simple", doc="One search per query. The arm every other strategy has to beat."
)(SimpleRetrieval)
RETRIEVERS.register(
    "widened", shorthand="factor", doc="Search deeper than asked, then cut back. No model calls."
)(WidenedRetrieval)
# The only strategy here that calls a model. It is registered eagerly rather than behind an
# extra because it falls back to this package's own LLM protocol when agno is absent -- a
# strategy nobody can run is a strategy nobody measures, and measuring it is the whole point.
RETRIEVERS.register(
    "agentic",
    shorthand="model",
    doc="A model plans the searches, optionally over several rounds. Costs a call per query.",
)(AgenticRetrieval)
RETRIEVERS.register(
    "decomposed",
    shorthand="max_parts",
    doc="Split a multi-part question and search each. Mechanical, so it costs nothing.",
)(DecomposedRetrieval)
RETRIEVERS.register(
    "relevance-feedback",
    shorthand="terms",
    doc="Search, read the best hit, search again with its most distinctive words. No model calls.",
)(RelevanceFeedbackRetrieval)


def get_retriever(
    spec: str | RetrievalStrategy | None, llm: object | None = None
) -> RetrievalStrategy:
    """Resolve a strategy from a spec, or pass one through. `None` means plain search.

    `llm` is the model the configuration chose, and handing it over matters for more than
    tidiness. A model-backed strategy that builds its own client instead:

    * ignores `run.model` entirely -- `AgenticRetrieval` defaults to `openai:gpt-4o-mini`, so a
      sweep configured for any other model would quietly plan its searches with that one;
    * cannot be metered, because its calls never pass anything the runner can see, so the
      configuration is costed at zero and `budget_usd` can never stop it;
    * wants its own credentials, which is a second key for the same sweep.

    Optional rather than required, so `get_retriever("simple")` and every direct call in a test
    keeps working. A strategy with no use for a model ignores it.
    """
    if spec is None:
        return SimpleRetrieval()
    strategy = RETRIEVERS.create(spec) if isinstance(spec, str) else spec

    # Model-backed strategies keep their planner in a private `_llm` slot that `planner()`
    # fills lazily. Filling it here is what makes the configured model win over the built-in
    # default, without every strategy having to know that injection is a thing.
    if llm is not None and getattr(strategy, "uses_model", False):
        object.__setattr__(strategy, "_llm", llm)
    return strategy


__all__ = [
    "RETRIEVERS",
    "AgenticRetrieval",
    "DecomposedRetrieval",
    "Lookup",
    "RelevanceFeedbackRetrieval",
    "RetrievalError",
    "RetrievalStrategy",
    "RetrievalTrace",
    "Searcher",
    "SimpleRetrieval",
    "WidenedRetrieval",
    "fuse",
    "get_retriever",
]
