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
    "voyage-3": Pricing(embed_per_million=0.06),
}


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

    def pricing_for(self, model: str | None) -> Pricing:
        if model is None:
            return Pricing(metered=False)

        # Spec strings carry parameters; the price belongs to the model name.
        name = model.split(":", 1)[0]
        if name in self.prices:
            return self.prices[name]

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
        embedder: str | None,
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
