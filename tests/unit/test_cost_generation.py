"""Costing the axis that actually spends money.

Found by an agent given the docs and no source, told to build a RAG pipeline with real LLM
calls. Its `->llm` run made about fifteen `gpt-4o-mini` generation calls and a hundred-odd
judge calls, and came back:

    "cost": {"index_usd": 0.0, "query_usd_per_1k": 0.0, ..., "metered": false}

Three consequences, all of them worse than a wrong number:

* the leaderboard printed `$/1k queries = 0.0000` for a configuration calling OpenAI;
* the summary paragraph said "it runs locally at no cost per query" about that same run;
* `budget_usd` compared its ceiling against a total that could never grow, so every positive
  value was unlimited. Only `budget_usd: 0.0` did anything, by stopping before the first
  configuration.

`Pricing` had `generate_input_per_million` and `generate_output_per_million` from the start.
The runner simply never fed them anything.
"""

from __future__ import annotations

import pytest

from contextgrid.core.documents import MediaType
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.corpus import Corpus
from contextgrid.cost.metering import MeteredLLM, Usage, exact_tokenizer_or_none
from contextgrid.cost.model import CostBreakdown, CostModel
from contextgrid.evalset.llm import RecordingLLM
from contextgrid.grid import Runner, matrix
from contextgrid.pipeline import Config

CORPUS = Corpus.from_texts(
    {"a.md": "# Refunds\n\nRefunds are issued within 30 days of purchase.\n"},
    media_type=MediaType.MARKDOWN,
    name="cost",
)


def evalset(count: int = 4) -> EvalSet:
    return EvalSet(
        id="cost",
        items=tuple(
            EvalItem(
                id=f"q{index}",
                question="how long do refunds take?",
                anchors=(GoldAnchor(quote="within 30 days", source_id="a.md"),),
            )
            for index in range(count)
        ),
    )


def priced_llm() -> RecordingLLM:
    """A scripted model wearing the name of a real one, so it prices against a published rate.

    No network and no key: the metering is a token count on strings this process already has,
    which is exactly what makes it testable.
    """
    llm = RecordingLLM(default="Refunds are issued within 30 days of purchase.")
    llm.name = "gpt-4o-mini"
    return llm


# ---------------------------------------------------------------------------
# the lie
# ---------------------------------------------------------------------------


def test_a_generation_run_is_not_free() -> None:
    """The whole finding, in one assertion."""
    result = Runner(corpus=CORPUS, llm=priced_llm()).run_one(
        Config(chunker="recursive:128", index="bm25", embedder=None, generator="llm"), evalset()
    )

    assert result.cost.generation_tokens > 0
    assert result.cost.evaluation_usd > 0.0
    assert result.cost.generation_usd_per_1k > 0.0


def test_a_local_embedder_does_not_make_a_hosted_generator_free() -> None:
    """The exact combination that produced the zero: `bm25` needs no embedder, so costing took
    the unmetered path and returned before generation was ever considered."""
    result = Runner(corpus=CORPUS, llm=priced_llm()).run_one(
        Config(chunker="recursive:128", index="bm25", embedder=None, generator="llm"), evalset()
    )

    assert result.cost.total_at(1000) > 0.0, "an OpenAI generator still reads as free to serve"


def test_no_generator_still_costs_nothing() -> None:
    """The other direction: metering must not invent a cost for a run that made no calls."""
    result = Runner(corpus=CORPUS, llm=priced_llm()).run_one(
        Config(chunker="recursive:128", index="bm25", embedder=None), evalset()
    )

    assert result.cost.evaluation_usd == 0.0
    assert result.cost.generation_tokens == 0


# ---------------------------------------------------------------------------
# the budget that could not fire
# ---------------------------------------------------------------------------


def test_a_positive_budget_stops_a_generation_sweep() -> None:
    """`budget_usd: 0.25` was incapable of stopping anything. Only `0.0` worked, by refusing
    to start."""
    results = Runner(corpus=CORPUS, llm=priced_llm()).run(
        matrix(
            chunker=["recursive:128", "sentence:2", "fixed:64"],
            index="bm25",
            embedder=None,
            generator="llm",
        ),
        evalset(),
        mode="factorial",
        budget_usd=0.0005,
    )

    assert len(results.runs) < 3, "the budget never fired"
    assert any("budget ran out" in warning.message for warning in results.warnings)


def test_the_same_sweep_without_a_budget_runs_everything() -> None:
    """Proves the test above stopped for the reason claimed, rather than the sweep being
    broken in some other way."""
    results = Runner(corpus=CORPUS, llm=priced_llm()).run(
        matrix(
            chunker=["recursive:128", "sentence:2", "fixed:64"],
            index="bm25",
            embedder=None,
            generator="llm",
        ),
        evalset(),
        mode="factorial",
    )

    assert len(results.runs) == 3


# ---------------------------------------------------------------------------
# serving cost and evaluation cost are different questions
# ---------------------------------------------------------------------------


def test_the_judge_is_charged_to_evaluation_not_to_serving() -> None:
    """A judge call happens once, while measuring, and never again in production. Folding it
    into serving cost would tell somebody their live system needs a judge per question."""
    result = Runner(corpus=CORPUS, llm=priced_llm()).run_one(
        Config(chunker="recursive:128", index="bm25", embedder=None, generator="llm"), evalset()
    )
    cost = result.cost

    assert cost.judge_tokens > 0
    # Serving cost is built from the generator's tokens alone.
    generation_only = CostModel().estimate(
        embedder=None,
        index_tokens=0,
        query_tokens_per_query=0.0,
        model="gpt-4o-mini",
        generation=Usage(prompt_tokens=result.cost.generation_tokens, completion_tokens=0, calls=1),
        queries=4,
    )
    assert cost.generation_usd_per_1k == pytest.approx(
        generation_only.generation_usd_per_1k, rel=0.5
    )


def test_spent_now_counts_everything_actually_paid_for() -> None:
    """Index build, embedding the questions asked, generating, and judging."""
    cost = CostBreakdown(
        index_usd=0.30, query_usd_per_1k=200.0, generation_usd_per_1k=50.0, evaluation_usd=0.75
    )

    # 0.30 index + 200.0 for 1000 embedded questions + 0.75 judge. The generator's *serving*
    # rate is not spent money, so it is absent.
    assert cost.spent_now(1000) == pytest.approx(201.05)
    # Whereas serving 1000 queries projects both per-query rates and no judge at all.
    assert cost.total_at(1000) == pytest.approx(250.30)


# ---------------------------------------------------------------------------
# the meter itself
# ---------------------------------------------------------------------------


def test_the_meter_counts_both_sides_of_a_call() -> None:
    """Input and output are priced differently -- output runs several times input on most
    providers, so one blended token count would misprice every short question with a long
    answer."""
    llm = MeteredLLM(RecordingLLM(default="a much longer reply than the prompt"), None)
    llm.complete("hi")

    assert llm.usage.calls == 1
    assert llm.usage.prompt_tokens > 0
    assert llm.usage.completion_tokens > llm.usage.prompt_tokens


def test_the_meter_does_not_change_what_a_model_returns() -> None:
    """It is a proxy. Every existing `LLM`, test double and `transport=` hook has to keep
    working untouched."""
    inner = RecordingLLM(default="the answer")
    assert MeteredLLM(inner, None).complete("q") == inner.complete("q")


def test_an_approximate_count_is_not_reported_as_measured() -> None:
    """Without `tiktoken` the count is characters over four. That is worth having, and it is
    not a measurement -- `metered` says which one you got."""
    approximate = MeteredLLM(RecordingLLM(default="reply"), None)
    approximate.complete("question")

    assert not approximate.usage.exact
    costed = CostModel().estimate(
        embedder=None,
        index_tokens=0,
        query_tokens_per_query=0.0,
        model="gpt-4o-mini",
        generation=approximate.usage,
        queries=1,
    )
    assert not costed.metered


def test_the_exact_tokenizer_is_used_when_available() -> None:
    tokenizer = exact_tokenizer_or_none()
    if tokenizer is None:  # pragma: no cover - only when tiktoken is absent
        pytest.skip("tiktoken not installed")

    metered = MeteredLLM(RecordingLLM(default="reply"), tokenizer)
    metered.complete("question")

    assert metered.usage.exact
