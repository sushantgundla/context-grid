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
    needs_model_error,
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


def _split_by_cost() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The registered strategy names, split into the paid ones and the free ones.

    Read off `uses_model` on each registered class rather than a hand-kept list, for two
    reasons. A hand-kept list goes stale the day a second paid strategy is registered, and
    it cannot see a strategy that arrived at runtime through `plugins:` -- which is exactly
    the strategy nobody has costed yet.

    Loading a registration only imports the class; it never builds one. That distinction is
    the point: a model-backed strategy with no model *refuses to build*, so anything that
    asks "does this cost money?" by constructing it gets no answer precisely when a sweep is
    about to spend.
    """
    paid: list[str] = []
    free: list[str] = []
    for name in RETRIEVERS.names():
        try:
            factory = RETRIEVERS.registration(name).load()
        except Exception:  # pragma: no cover - an uninstallable plugin is neither, usefully
            continue
        (paid if getattr(factory, "uses_model", False) else free).append(name)
    return tuple(paid), tuple(free)


def model_backed_retrievers() -> tuple[str, ...]:
    """The strategies that cost a model call per query.

    A function rather than the module-level `MODEL_BACKED` constant that `transform` and
    `generate` use, because those two families cannot be registered at all, while these can:
    a `plugins:` entry adds strategies after this module is imported, and a constant frozen
    at import time would miss every one of them.
    """
    return _split_by_cost()[0]


def model_free_retrievers() -> tuple[str, ...]:
    """The strategies that never call a model, for "use one of these instead"."""
    return _split_by_cost()[1]


def get_retriever(
    spec: str | RetrievalStrategy | None, llm: object | None = None
) -> RetrievalStrategy:
    """Resolve a strategy from a spec, or pass one through. `None` means plain search.

    `llm` is the model the configuration chose, and handing it over matters for more than
    tidiness. A model-backed strategy that builds its own client instead:

    * ignores `run.model` entirely -- `AgenticRetrieval` used to default to
      `openai:gpt-4o-mini`, so a sweep configured for any other model would quietly plan its
      searches with that one;
    * cannot be metered, because its calls never pass anything the runner can see, so the
      configuration is costed at zero and `budget_usd` can never stop it;
    * wants its own credentials, which is a second key for the same sweep.

    So a model-backed strategy with no model **refuses**, exactly as `hyde` and the `llm`
    generator do. Silently retrieving with a model nobody chose, on money nothing counts, is
    worse than an error: the numbers still look like a measurement.

    Optional rather than required, so `get_retriever("simple")` and every direct call in a test
    keeps working. A strategy with no use for a model ignores it.
    """
    if spec is None:
        return SimpleRetrieval()
    strategy = RETRIEVERS.create(spec) if isinstance(spec, str) else spec

    if not getattr(strategy, "uses_model", False):
        return strategy

    # Model-backed strategies keep their planner in a private `_llm` slot that `planner()`
    # reads. Filling it here is what makes the configured model win, without every strategy
    # having to know that injection is a thing.
    if llm is not None:
        object.__setattr__(strategy, "_llm", llm)
    elif getattr(strategy, "_llm", None) is None and getattr(strategy, "model", None) is None:
        # Nothing was passed, the strategy is not carrying a planner already (a factory that
        # wires in its own, which is how the tests run this axis with no key), and its spec
        # named no model either.
        #
        # That last clause is the line this refusal is actually drawing. The bug was never
        # "built a model without being handed one" -- it was *choosing a provider on the
        # user's behalf*: `AgenticRetrieval.model` used to default to `openai:gpt-4o-mini`, so
        # `retrieval: agentic` alone spent real money on a model nobody named. `agentic:gpt-4o-
        # mini` is the opposite of that. It is someone naming one, and it is a documented spec
        # (see docs/dimensions/retrieval.md), so refusing it would break a working feature to
        # fix a bug it never had.
        raise needs_model_error(strategy.name)
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
    "model_backed_retrievers",
    "model_free_retrievers",
]
