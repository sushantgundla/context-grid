"""Agentic retrieval: a model plans the searches.

No model is called for real. The planner is replaced with a scripted one, which is the whole
reason `AgenticRetrieval` takes a planner it can be handed rather than building one privately --
a strategy that can only be exercised with an API key is a strategy that never gets tested.

What matters here is the part that is easy to get wrong and impossible to notice: that a model
which fails, refuses, or returns prose instead of JSON still produces results rather than
scoring as a retrieval failure, and that what it cost is recorded rather than inferred.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from contextgrid.index.base import Scored
from contextgrid.retrieve import RETRIEVERS, AgenticRetrieval, RetrievalTrace, get_retriever
from contextgrid.retrieve.agentic import _parse_queries
from contextgrid.retrieve.strategies import RetrievalError


class ScriptedPlanner:
    """Returns prepared replies and remembers the prompts it was given."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int = 256) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "[]"


class FailingPlanner:
    def __init__(self, message: str = "the provider is down") -> None:
        self.message = message
        self.calls = 0

    def complete(self, prompt: str, *, max_tokens: int = 256) -> str:
        self.calls += 1
        raise RuntimeError(self.message)


class FakeIndex:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __call__(self, text: str, k: int) -> Sequence[Scored]:
        self.calls.append((text, k))
        return [Scored(f"{text}:{i}", 1.0 - i * 0.01) for i in range(min(k, 3))]

    @property
    def queries(self) -> list[str]:
        return [text for text, _ in self.calls]


def with_planner(strategy: AgenticRetrieval, planner: object) -> AgenticRetrieval:
    object.__setattr__(strategy, "_llm", planner)
    return strategy


# ---------------------------------------------------------------------------
# the plan becomes the searches
# ---------------------------------------------------------------------------


def test_the_model_decides_what_to_search_for() -> None:
    index = FakeIndex()
    planner = ScriptedPlanner('["refund window", "digital goods refundable"]')
    strategy = with_planner(AgenticRetrieval(), planner)

    strategy.retrieve("can I get money back?", ["ignored"], index, 5, RetrievalTrace())

    assert index.queries == ["refund window", "digital goods refundable"]


def test_the_question_reaches_the_prompt() -> None:
    planner = ScriptedPlanner('["a"]')
    with_planner(AgenticRetrieval(), planner).retrieve(
        "how long do refunds take?", [], FakeIndex(), 5, RetrievalTrace()
    )
    assert "how long do refunds take?" in planner.prompts[0]


def test_the_transforms_output_is_ignored() -> None:
    """The model plans from the question as asked. Searching a paraphrase *and* everything the
    model thought of would double the cost to measure two things at once."""
    index = FakeIndex()
    with_planner(AgenticRetrieval(), ScriptedPlanner('["planned"]')).retrieve(
        "question", ["some paraphrase"], index, 5, RetrievalTrace()
    )
    assert index.queries == ["planned"]


def test_results_from_several_searches_are_fused_not_concatenated() -> None:
    index = FakeIndex()
    found = with_planner(AgenticRetrieval(), ScriptedPlanner('["a", "b"]')).retrieve(
        "q", [], index, 5, RetrievalTrace()
    )
    ids = [scored.chunk_id for scored in found]
    assert len(ids) == len(set(ids))
    assert [s.score for s in found] == sorted((s.score for s in found), reverse=True)


def test_too_many_planned_queries_are_capped() -> None:
    """A model asked for "the queries" will happily write nine, and nine searches per question
    across a sweep is how an afternoon becomes a week."""
    index = FakeIndex()
    planner = ScriptedPlanner('["a","b","c","d","e","f","g"]')
    with_planner(AgenticRetrieval(max_queries=3), planner).retrieve(
        "q", [], index, 5, RetrievalTrace()
    )
    assert len(index.queries) == 3


# ---------------------------------------------------------------------------
# rounds
# ---------------------------------------------------------------------------


def test_one_round_is_one_model_call() -> None:
    trace = RetrievalTrace()
    with_planner(AgenticRetrieval(rounds=1), ScriptedPlanner('["a"]')).retrieve(
        "q", [], FakeIndex(), 5, trace
    )
    assert trace.model_calls == 1
    assert trace.notes["rounds"] == 1


def test_a_second_round_sees_what_the_first_one_found() -> None:
    """Which is the whole difference between agentic retrieval and a fancy query rewriter."""
    planner = ScriptedPlanner('["first"]', '["second"]')
    index = FakeIndex()

    with_planner(AgenticRetrieval(rounds=2), planner).retrieve("q", [], index, 5, RetrievalTrace())

    assert index.queries == ["first", "second"]
    assert "'first'" in planner.prompts[1]
    assert "result(s)" in planner.prompts[1]


def test_an_empty_plan_stops_early() -> None:
    """A model that says it has enough is a real signal, and stopping is cheaper than another
    round of nothing."""
    trace = RetrievalTrace()
    planner = ScriptedPlanner('["first"]', "[]")
    index = FakeIndex()

    with_planner(AgenticRetrieval(rounds=4), planner).retrieve("q", [], index, 5, trace)

    assert index.queries == ["first"]
    assert trace.notes["rounds"] == 1
    assert trace.model_calls == 2  # it asked, and was told there was nothing more


def test_rounds_below_one_is_refused() -> None:
    with pytest.raises(RetrievalError, match="at least 1"):
        AgenticRetrieval(rounds=0)


def test_max_queries_below_one_is_refused() -> None:
    with pytest.raises(RetrievalError, match="at least 1"):
        AgenticRetrieval(max_queries=0)


def test_an_unknown_backend_lists_the_real_ones() -> None:
    with pytest.raises(RetrievalError, match="auto, agno, llm"):
        AgenticRetrieval(backend="magic")


# ---------------------------------------------------------------------------
# when the model lets you down
# ---------------------------------------------------------------------------


def test_a_planner_that_raises_still_returns_results() -> None:
    """Returning nothing would score as a retrieval failure when what failed was the planner.
    A sweep must not turn a provider outage into a claim about a corpus."""
    index = FakeIndex()
    trace = RetrievalTrace()
    strategy = with_planner(AgenticRetrieval(), FailingPlanner("503 service unavailable"))

    found = strategy.retrieve("how long do refunds take?", [], index, 5, trace)

    assert found
    assert index.queries == ["how long do refunds take?"]
    assert trace.notes["fell_back"] is True
    assert "503" in str(trace.notes["planner_error"])


def test_a_failed_planner_is_still_counted_as_a_model_call() -> None:
    """It was billed. A cost column that omits failed calls understates what the run cost."""
    trace = RetrievalTrace()
    with_planner(AgenticRetrieval(), FailingPlanner()).retrieve("q", [], FakeIndex(), 5, trace)
    assert trace.model_calls == 1


def test_a_model_that_returns_prose_falls_back_to_the_question() -> None:
    index = FakeIndex()
    with_planner(
        AgenticRetrieval(), ScriptedPlanner("I'm sorry, I can't help with that.")
    ).retrieve("the original question", [], index, 5, RetrievalTrace())
    assert index.queries == ["the original question"]


# ---------------------------------------------------------------------------
# reading whatever the model actually returned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('["a", "b"]', ["a", "b"]),
        ('```json\n["a", "b"]\n```', ["a", "b"]),
        ('Here are the queries:\n["a", "b"]', ["a", "b"]),
        ("1. first query\n2. second query", ["first query", "second query"]),
        ("- first query\n- second query", ["first query", "second query"]),
        # Prose is not a plan. Without this the strategy searches for the apology.
        ("I'm sorry, I can't help with that.", []),
        ("The refund window is 30 days.", []),
        ("[]", []),
        ("", []),
        ('["a", "A", "  a  "]', ["a"]),
        ('["a", 42, null, "b"]', ["a", "b"]),
        ('{"queries": ["a"]}', []),
    ],
)
def test_queries_are_read_out_of_whatever_the_model_returned(
    reply: str, expected: list[str]
) -> None:
    """Models wrap JSON in fences, prefix it with a sentence, and sometimes give up on JSON and
    write a numbered list. Discarding those would make the strategy look worse than it is."""
    assert _parse_queries(reply, limit=10) == expected


# ---------------------------------------------------------------------------
# cost, and saying so
# ---------------------------------------------------------------------------


def test_it_declares_that_it_calls_a_model() -> None:
    """Read by the runner, which warns when a sweep containing one has no spending limit."""
    assert AgenticRetrieval().uses_model
    assert not get_retriever("simple").uses_model


def test_a_sweep_with_no_limit_says_the_bill_is_unknowable() -> None:
    """Every other axis has a cost you can estimate before starting. This one decides its own
    number of calls, so nothing can tell you the bill in advance."""
    from contextgrid.grid import matrix
    from contextgrid.grid.runner import Budget, _warn_if_unbounded

    grid = matrix(retrieval="agentic")

    _warn_if_unbounded(grid, Budget())
    assert grid.meta["unbounded_model_calls"] == "agentic"

    grid.meta.clear()
    _warn_if_unbounded(grid, Budget(usd=1.0))
    assert "unbounded_model_calls" not in grid.meta

    # A strategy that costs nothing never triggers it, whatever the budget.
    free = matrix(retrieval="decomposed")
    _warn_if_unbounded(free, Budget())
    assert "unbounded_model_calls" not in free.meta


# ---------------------------------------------------------------------------
# reachable from a config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        "agentic",
        "agentic:gpt-4o-mini",
        "agentic:gpt-4o-mini,rounds=2",
        "agentic:rounds=3,max_queries=2",
        "agentic:backend=llm",
    ],
)
def test_it_is_reachable_from_one_config_line(spec: str) -> None:
    assert get_retriever(spec).name == "agentic"


def test_it_is_registered_and_documented() -> None:
    assert "agentic" in RETRIEVERS
    assert "model" in RETRIEVERS.describe()["agentic"].lower()


def test_it_sits_on_the_same_axis_as_the_free_strategies() -> None:
    """Which is the comparison the whole axis exists for: does agentic beat decomposed, and is
    the difference worth a model call per question?"""
    from contextgrid.grid import matrix

    configs = matrix(retrieval=["simple", "decomposed", "agentic"]).expand("factorial")
    assert {config.retrieval for config in configs} == {"simple", "decomposed", "agentic"}


# ---------------------------------------------------------------------------
# does it actually beat the free strategies?
# ---------------------------------------------------------------------------


def test_a_good_plan_beats_plain_search_on_two_part_questions() -> None:
    """The claim the axis exists to check, with the model's contribution held fixed.

    Two questions, each answered by two documents. Plain search ranks whichever half BM25
    favoured; a planner that writes one query per part finds both. The planner is scripted, so
    this measures the *mechanism* rather than a particular model's mood on a particular day.
    """
    from contextgrid.core.documents import MediaType
    from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
    from contextgrid.corpus import Corpus
    from contextgrid.grid import Runner, matrix

    plans = {
        "what is the refund window and are digital goods refundable?": (
            '["refund window 30 days purchase", "digital goods not refundable downloaded"]'
        ),
        "how long is standard shipping and when does express arrive?": (
            '["standard shipping business days", "express shipping next business day"]'
        ),
    }

    class Planner:
        def complete(self, prompt: str, *, max_tokens: int = 256) -> str:
            return next((plan for q, plan in plans.items() if q in prompt), "[]")

    name = "scripted-agentic"
    if name not in RETRIEVERS:

        def build() -> AgenticRetrieval:
            strategy = AgenticRetrieval()
            object.__setattr__(strategy, "_llm", Planner())
            return strategy

        RETRIEVERS.register(name, doc="agentic with a scripted planner, for tests")(build)

    corpus = Corpus.from_texts(
        {
            "refunds.md": "# Refunds\n\nRefunds are issued within 30 days of purchase.\n",
            "digital.md": "# Digital goods\n\nDigital goods are not refundable once downloaded.\n",
            "shipping.md": "# Shipping\n\nStandard shipping takes 5 to 7 business days.\n",
            "express.md": "# Express\n\nExpress shipping arrives the next business day.\n",
            "holidays.md": "# Holidays\n\nThe office is closed on public holidays.\n",
            "returns.md": "# Returns\n\nReturns must be posted within 14 days of delivery.\n",
            "privacy.md": "# Privacy\n\nWe keep your data for seven years.\n",
        },
        media_type=MediaType.MARKDOWN,
        name="two-part",
    )
    evalset = EvalSet(
        id="two-part",
        items=(
            EvalItem(
                id="q1",
                question="what is the refund window and are digital goods refundable?",
                anchors=(
                    GoldAnchor(quote="within 30 days of purchase", source_id="refunds.md"),
                    GoldAnchor(quote="not refundable once downloaded", source_id="digital.md"),
                ),
            ),
            EvalItem(
                id="q2",
                question="how long is standard shipping and when does express arrive?",
                anchors=(
                    GoldAnchor(quote="5 to 7 business days", source_id="shipping.md"),
                    GoldAnchor(quote="the next business day", source_id="express.md"),
                ),
            ),
        ),
    )

    results = Runner(corpus=corpus, headline="recall@2").run(
        matrix(
            chunker="recursive:128",
            index="bm25",
            embedder=None,
            retrieval=["simple", name],
            k=2,
        ),
        evalset,
        mode="factorial",
    )

    scores = {run.config.retrieval: run.metric("recall@2") for run in results.runs}
    assert scores[name] > scores["simple"]
    assert scores[name] == pytest.approx(1.0)


def test_a_headline_cutoff_outside_the_defaults_is_still_computed() -> None:
    """`Runner(headline="recall@2")` reported 0.000 for every configuration, because 2 is not
    one of the default cut-offs and the metric was never calculated -- an empty column read as
    a real result, and a leaderboard sorted on nothing."""
    from contextgrid.corpus import Corpus
    from contextgrid.grid import Runner

    runner = Runner(corpus=Corpus.from_texts({"a.md": "text"}), headline="recall@2")
    assert 2 in runner.ks

    assert 7 in Runner(corpus=Corpus.from_texts({"a.md": "t"}), headline="ndcg@7").ks
