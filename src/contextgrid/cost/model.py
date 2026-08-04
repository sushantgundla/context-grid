"""Costing a configuration.

Every open-source tool in this space ranks configurations on quality alone. That is a
misleading axis on its own: a reranker that gains two points of recall for six times the
latency and forty times the cost is a different decision from one that gains two points for
free, and a leaderboard sorted by recall makes them look identical.

Two kinds of cost are modelled, because both are real and they behave differently.

**Token cost** applies to hosted models: dollars per million tokens, charged once at index
time and again on every query.

**Compute cost** applies to local models: they are free per token and not free per second.
Pricing a machine by the hour turns wall-clock into money and puts a local CPU model on the
same chart as a hosted API, which is the comparison people actually need to make.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contextgrid.core.warnings import Severity, WarningCode, WarningLog


@dataclass(frozen=True, slots=True)
class Pricing:
    """What one model charges, in dollars per million tokens."""

    embed_per_million: float = 0.0
    rerank_per_million: float = 0.0
    generate_input_per_million: float = 0.0
    generate_output_per_million: float = 0.0
    #: False for models that run locally, where the cost is time rather than tokens.
    metered: bool = True


#: Published list prices. Deliberately a plain table rather than a live lookup: a cost
#: comparison has to be reproducible, and a number that changes under you between runs is
#: worse than one that is three months stale and labelled as such.
#:
#: Last checked: August 2026. Local models are priced at zero per token by definition.
PRICES: dict[str, Pricing] = {
    "hash": Pricing(metered=False),
    "tfidf": Pricing(metered=False),
    "length": Pricing(metered=False),
    "bge-base-en-v1.5": Pricing(metered=False),
    "e5-base-v2": Pricing(metered=False),
    "all-MiniLM-L6-v2": Pricing(metered=False),
    "text-embedding-3-small": Pricing(embed_per_million=0.02),
    "text-embedding-3-large": Pricing(embed_per_million=0.13),
    "embed-v3": Pricing(embed_per_million=0.10),
    "embed-english-v3.0": Pricing(embed_per_million=0.10),
    "voyage-3": Pricing(embed_per_million=0.06),
    "text-embedding-ada-002": Pricing(embed_per_million=0.10),
}


def price_key(model: str | object) -> str:
    """The name a price is looked up under.

    Three shapes reach this, and the naive `split(":")[0]` is wrong for two of them:

    * `tfidf` or `tfidf:5000` -- a plain plugin, optionally with parameters. Take the name.
    * `litellm:text-embedding-3-small,dimensions=256` -- a backend and the model it is
      serving. The **model** is what costs money, so the backend prefix has to come off, or
      every hosted model in a sweep would be priced as the string "litellm", find no entry,
      and quietly cost zero.
    * an embedder instance, which the Python API allows anywhere a spec string does. Ask it.

    Provider-qualified names (`cohere/embed-english-v3.0`) keep only the last segment, since
    the price belongs to the model rather than to the route taken to reach it.
    """
    if not isinstance(model, str):
        named = getattr(model, "model", None) or getattr(model, "name", None)
        return price_key(str(named)) if named else ""

    text = model.split(",", 1)[0].strip()  # drop keyword parameters
    parts = [part for part in text.split(":") if part]
    if not parts:
        return ""

    # `litellm:model` and `tei:model` name a backend then a model; everything else is
    # `name` or `name:shorthand`, where the shorthand is a number or a size.
    tail = parts[-1] if parts[0] in _BACKENDS and len(parts) > 1 else parts[0]
    return tail.split("/")[-1].strip()


#: Backends that serve some *other* model, so the price belongs to the tail of the spec.
_BACKENDS = frozenset({"litellm", "tei"})

#: Backends that run on your own hardware. Free per token, not free per second.
_LOCAL_BACKENDS = frozenset({"tei"})


def _is_local(model: str | object) -> bool:
    """Whether this runs on the user's own machine rather than somebody's API."""
    name = model if isinstance(model, str) else str(getattr(model, "name", ""))
    return name.split(":", 1)[0] in _LOCAL_BACKENDS


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """What one configuration costs, itemised.

    Indexing is a one-off; querying recurs. Reporting them together as a single number is
    how a configuration that is cheap to build and ruinous to serve gets chosen.
    """

    index_usd: float = 0.0
    query_usd_per_1k: float = 0.0
    index_tokens: int = 0
    query_tokens_per_query: float = 0.0
    compute_seconds: float = 0.0
    metered: bool = True

    def total_at(self, queries: int) -> float:
        """Total cost of building the index and serving `queries` queries."""
        return self.index_usd + self.query_usd_per_1k * (queries / 1000)

    def as_dict(self) -> dict[str, float | bool | int]:
        return {
            "index_usd": self.index_usd,
            "query_usd_per_1k": self.query_usd_per_1k,
            "index_tokens": self.index_tokens,
            "query_tokens_per_query": self.query_tokens_per_query,
            "compute_seconds": self.compute_seconds,
            "metered": self.metered,
        }


@dataclass(slots=True)
class CostModel:
    """Turns tokens and seconds into dollars.

    `machine_usd_per_hour` is what makes local and hosted models comparable. Leave it at zero
    and a local model appears free, which is true per token and false in every other sense.
    A commodity 4-core cloud box is roughly $0.10/hour; put that in and the chart starts
    telling the truth about self-hosting.
    """

    machine_usd_per_hour: float = 0.0
    prices: dict[str, Pricing] = field(default_factory=lambda: dict(PRICES))
    warnings: WarningLog = field(default_factory=WarningLog)

    def pricing_for(self, model: str | object | None) -> Pricing:
        if model is None:
            return Pricing(metered=False)

        name = price_key(model)
        if name in self.prices:
            return self.prices[name]

        # A local server is free per token by definition. Saying "no published price" about a
        # model somebody is running on their own machine would be noise, and the cost that does
        # apply -- machine time -- is already counted from the clock.
        if _is_local(model):
            return Pricing(metered=False)

        self.warnings.add(
            WarningCode.BUDGET_REACHED,
            f"no published price for {name!r}, so it is costed at zero. Any cost comparison "
            "involving it understates what it charges",
            severity=Severity.CAUTION,
            stage="cost",
            subject=name,
        )
        return Pricing(metered=False)

    def estimate(
        self,
        *,
        embedder: str | object | None,
        index_tokens: int,
        query_tokens_per_query: float,
        compute_seconds: float = 0.0,
    ) -> CostBreakdown:
        """Cost one configuration, given what it embedded and how long it took."""
        pricing = self.pricing_for(embedder)
        machine_usd = compute_seconds * (self.machine_usd_per_hour / 3600)

        if not pricing.metered:
            # Free per token, not free to run. All of the cost is time.
            return CostBreakdown(
                index_usd=machine_usd,
                query_usd_per_1k=0.0,
                index_tokens=index_tokens,
                query_tokens_per_query=query_tokens_per_query,
                compute_seconds=compute_seconds,
                metered=False,
            )

        rate = pricing.embed_per_million / 1_000_000
        return CostBreakdown(
            index_usd=index_tokens * rate + machine_usd,
            query_usd_per_1k=query_tokens_per_query * 1000 * rate,
            index_tokens=index_tokens,
            query_tokens_per_query=query_tokens_per_query,
            compute_seconds=compute_seconds,
            metered=True,
        )
