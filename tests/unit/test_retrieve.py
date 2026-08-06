"""Retrieval strategies: how the index is used, as opposed to what it is.

The split is the point. `dense`, `bm25` and `pgvector` are stores; `simple`, `widened` and
`decomposed` are what sits on top of them. Keeping the two apart turns "does agentic retrieval
help on my corpus?" from a rewrite into a cell in a grid.

So these tests are mostly about the seam: that a strategy never sees the index, that every
strategy works with every store, that what a strategy *did* is recorded rather than inferred
from a recall number, and that `simple` is byte-for-byte what the package did before the axis
existed.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from contextgrid.index.base import Scored
from contextgrid.retrieve import (
    RETRIEVERS,
    DecomposedRetrieval,
    RelevanceFeedbackRetrieval,
    RetrievalError,
    RetrievalTrace,
    SimpleRetrieval,
    WidenedRetrieval,
    get_retriever,
)


class FakeIndex:
    """Stands in for any store. Records every search it was asked to run."""

    def __init__(self, ranking: dict[str, list[str]] | None = None) -> None:
        self.ranking = ranking or {}
        self.calls: list[tuple[str, int]] = []

    def __call__(self, text: str, k: int) -> Sequence[Scored]:
        self.calls.append((text, k))
        ids = self.ranking.get(text, [f"{text}:{i}" for i in range(k)])
        return [Scored(chunk_id, 1.0 - index * 0.01) for index, chunk_id in enumerate(ids[:k])]

    @property
    def depths(self) -> list[int]:
        return [k for _, k in self.calls]

    @property
    def queries(self) -> list[str]:
        return [text for text, _ in self.calls]


class TextLookup:
    """Stands in for `BuiltPipeline.chunk_by_id().get` -- a `Lookup` backed by a plain dict of
    chunk id -> text, wrapped in something with a `.text` attribute since that is all a
    strategy is promised about what `lookup` returns."""

    def __init__(self, texts: dict[str, str]) -> None:
        self.texts = texts

    def __call__(self, chunk_id: str) -> object | None:
        text = self.texts.get(chunk_id)
        return None if text is None else SimpleNamespace(text=text)


# `RelevanceFeedbackRetrieval` belongs in the seam tests below: called without a `lookup` (as
# every one of them does), its default finds nothing to expand with and behaves exactly like
# `simple` -- one search, fused, no crash. Its own behaviour, which only shows up once a real
# `lookup` is handed in, is tested separately further down.
ALL = [
    SimpleRetrieval(),
    WidenedRetrieval(factor=3),
    DecomposedRetrieval(),
    RelevanceFeedbackRetrieval(),
]
IDS = [strategy.name for strategy in ALL]


# ---------------------------------------------------------------------------
# the seam
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ALL, ids=IDS)
def test_a_strategy_never_sees_the_index(strategy: object) -> None:
    """It is handed a function that runs one search. That is what makes every strategy work
    with every store, and lets a new store land without touching any strategy."""
    index = FakeIndex()
    found = strategy.retrieve("a question", ["a question"], index, 5, RetrievalTrace())  # type: ignore[attr-defined]

    assert found
    assert index.calls


@pytest.mark.parametrize("strategy", ALL, ids=IDS)
def test_no_more_than_k_results(strategy: object) -> None:
    index = FakeIndex()
    found = strategy.retrieve("q", ["q"], index, 3, RetrievalTrace())  # type: ignore[attr-defined]
    assert len(found) <= 3


@pytest.mark.parametrize("strategy", ALL, ids=IDS)
def test_scores_descend(strategy: object) -> None:
    index = FakeIndex()
    found = strategy.retrieve("q", ["q"], index, 5, RetrievalTrace())  # type: ignore[attr-defined]
    assert [s.score for s in found] == sorted((s.score for s in found), reverse=True)


@pytest.mark.parametrize("strategy", ALL, ids=IDS)
def test_no_duplicate_chunks_come_back(strategy: object) -> None:
    """Strategies that search several times must fuse, not concatenate. A duplicate in the
    results is a slot the generator wasted on something it already had."""
    index = FakeIndex()
    found = strategy.retrieve(  # type: ignore[attr-defined]
        "refunds and shipping", ["refunds and shipping"], index, 10, RetrievalTrace()
    )
    ids = [scored.chunk_id for scored in found]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("strategy", ALL, ids=IDS)
def test_the_same_question_twice_gives_the_same_ranking(strategy: object) -> None:
    """A leaderboard that moves when nothing changed destroys trust in every number on it."""
    first = strategy.retrieve("q", ["q"], FakeIndex(), 5, RetrievalTrace())  # type: ignore[attr-defined]
    second = strategy.retrieve("q", ["q"], FakeIndex(), 5, RetrievalTrace())  # type: ignore[attr-defined]
    assert [s.chunk_id for s in first] == [s.chunk_id for s in second]


# ---------------------------------------------------------------------------
# simple: unchanged behaviour
# ---------------------------------------------------------------------------


def test_simple_runs_one_search_per_query() -> None:
    index = FakeIndex()
    SimpleRetrieval().retrieve("q", ["q"], index, 5, RetrievalTrace())
    assert index.calls == [("q", 5)]


def test_simple_fuses_what_the_transform_produced() -> None:
    """A transform that returns several queries means several searches. Their scores are never
    compared: a cosine from one query and a cosine from another are not on the same scale,
    however similar the numbers look."""
    index = FakeIndex({"first": ["a", "b"], "second": ["b", "c"]})
    found = SimpleRetrieval().retrieve("q", ["first", "second"], index, 3, RetrievalTrace())

    assert index.queries == ["first", "second"]
    # `b` appears in both lists, so rank fusion should put it first.
    assert found[0].chunk_id == "b"


def test_simple_makes_no_model_calls() -> None:
    trace = RetrievalTrace()
    SimpleRetrieval().retrieve("q", ["q"], FakeIndex(), 5, trace)
    assert trace.model_calls == 0
    assert not SimpleRetrieval().uses_model


# ---------------------------------------------------------------------------
# widened
# ---------------------------------------------------------------------------


def test_widened_asks_the_index_for_more_than_it_returns() -> None:
    """The cheapest way to find out whether a configuration is limited by the retriever's
    ordering or by its reach."""
    index = FakeIndex()
    found = WidenedRetrieval(factor=4).retrieve("q", ["q"], index, 5, RetrievalTrace())

    assert index.depths == [20]
    assert len(found) == 5


def test_widened_records_the_depth_it_used() -> None:
    trace = RetrievalTrace()
    WidenedRetrieval(factor=3).retrieve("q", ["q"], FakeIndex(), 10, trace)
    assert trace.notes["depth"] == 30


def test_a_factor_of_one_is_plain_search() -> None:
    index = FakeIndex()
    WidenedRetrieval(factor=1).retrieve("q", ["q"], index, 7, RetrievalTrace())
    assert index.depths == [7]


def test_a_factor_below_one_is_refused() -> None:
    with pytest.raises(RetrievalError, match="at least 1"):
        WidenedRetrieval(factor=0)


# ---------------------------------------------------------------------------
# decomposed
# ---------------------------------------------------------------------------


def test_a_two_part_question_becomes_two_searches_plus_the_whole() -> None:
    """ "What is the refund window and does it cover digital goods?" has two answers, usually in
    two different chunks. One search ranks whichever half the embedding favoured and loses the
    other entirely."""
    parts = DecomposedRetrieval().parts(
        "what is the refund window and does it cover digital goods?"
    )
    assert parts[0] == "what is the refund window and does it cover digital goods?"
    assert "what is the refund window" in parts
    assert "does it cover digital goods" in parts


def test_a_single_part_question_is_not_split() -> None:
    """A trailing "?" splits a one-part question into itself, and searching the same words
    twice costs a round trip to buy nothing."""
    assert DecomposedRetrieval().parts("how long do refunds take?") == ["how long do refunds take?"]


def test_a_three_part_question_splits_into_three() -> None:
    parts = DecomposedRetrieval().parts(
        "what is the notice period, and who must it be sent to, and by when?"
    )
    assert len(parts) == 3


def test_fragments_too_short_to_be_a_search_are_dropped() -> None:
    """ "and by when" carries one meaningful word. A search for it returns noise that then
    dilutes the fusion."""
    parts = DecomposedRetrieval().parts("what is the notice period and by when?")
    assert "by when" not in parts


def test_the_whole_question_always_leads() -> None:
    """Decomposition is meant to add recall, not replace the search that was already working."""
    query = "refunds and shipping and returns and exchanges"
    assert DecomposedRetrieval().parts(query)[0] == query


def test_the_number_of_parts_is_capped() -> None:
    query = "a first thing and a second thing and a third thing and a fourth thing"
    assert len(DecomposedRetrieval(max_parts=2).parts(query)) == 2


def test_max_parts_below_one_is_refused() -> None:
    with pytest.raises(RetrievalError, match="at least 1"):
        DecomposedRetrieval(max_parts=0)


def test_decomposition_works_from_the_question_not_the_transform() -> None:
    """A transform's paraphrases are already whole questions. Splitting those as well would
    multiply the searches without adding anything."""
    index = FakeIndex()
    DecomposedRetrieval().retrieve(
        "refunds and shipping", ["some paraphrase entirely"], index, 5, RetrievalTrace()
    )
    assert "some paraphrase entirely" not in index.queries


def test_decomposition_records_how_many_parts_it_used() -> None:
    trace = RetrievalTrace()
    DecomposedRetrieval().retrieve(
        "what is the refund window and does it cover digital goods?", ["x"], FakeIndex(), 5, trace
    )
    assert trace.notes["parts"] == 3


# ---------------------------------------------------------------------------
# relevance-feedback
# ---------------------------------------------------------------------------


def test_relevance_feedback_searches_again_with_the_best_hits_rarest_words() -> None:
    """The whole mechanism: search, read the best hit through `lookup`, search again with
    words from it the question did not already have."""
    index = FakeIndex({"find gamma things": ["top"]})
    lookup = TextLookup({"top": "alpha beta beta gamma gamma gamma delta"})
    trace = RetrievalTrace()

    RelevanceFeedbackRetrieval(terms=2).retrieve(
        "find gamma things", ["find gamma things"], index, 5, trace, lookup
    )

    # "gamma" is already in the question, so it is never a candidate even though it is the
    # commonest word in the hit. Of what is left, "alpha" and "delta" occur once each --
    # rarer than "beta", which occurs twice -- so they are what the second search asks for.
    assert index.queries == ["find gamma things", "find gamma things alpha delta"]
    assert trace.notes["expansion_terms"] == ["alpha", "delta"]
    assert trace.searches == 2


def test_relevance_feedback_without_a_lookup_behaves_like_simple() -> None:
    """A strategy that has not been handed a real `lookup` -- every call site written before
    this parameter existed -- has nothing to read, and must not crash for lacking it."""
    index = FakeIndex()
    trace = RetrievalTrace()

    found = RelevanceFeedbackRetrieval().retrieve("q", ["q"], index, 5, trace)

    assert index.queries == ["q"]
    assert trace.searches == 1
    assert trace.notes["expansion_terms"] == []
    assert found


def test_relevance_feedback_with_no_initial_results_does_not_expand() -> None:
    index = FakeIndex({"q": []})
    trace = RetrievalTrace()

    found = RelevanceFeedbackRetrieval().retrieve(
        "q", ["q"], index, 5, trace, TextLookup({"top": "whatever"})
    )

    assert found == []
    assert trace.searches == 1
    assert trace.notes["expansion_terms"] == []


def test_relevance_feedback_with_nothing_new_to_say_does_not_expand() -> None:
    """The best hit is real text but every one of its words is already in the question --
    there is nothing left to search for, and a second, identical search would only cost a
    round trip to learn nothing."""
    index = FakeIndex({"refund policy": ["top"]})
    trace = RetrievalTrace()

    RelevanceFeedbackRetrieval().retrieve(
        "refund policy",
        ["refund policy"],
        index,
        5,
        trace,
        TextLookup({"top": "refund policy refund policy refund"}),
    )

    assert trace.searches == 1
    assert trace.notes["expansion_terms"] == []


def test_relevance_feedback_makes_no_model_calls() -> None:
    trace = RetrievalTrace()
    RelevanceFeedbackRetrieval().retrieve("q", ["q"], FakeIndex(), 5, trace)
    assert trace.model_calls == 0
    assert not RelevanceFeedbackRetrieval().uses_model


def test_relevance_feedback_terms_below_one_is_refused() -> None:
    with pytest.raises(RetrievalError, match="at least 1"):
        RelevanceFeedbackRetrieval(terms=0)


# ---------------------------------------------------------------------------
# the trace
# ---------------------------------------------------------------------------


def test_searches_are_counted() -> None:
    """Two strategies with the same recall and a different number of searches are a decision,
    not a tie -- and a recall column cannot carry that."""
    trace = RetrievalTrace()
    DecomposedRetrieval().retrieve("refunds and shipping costs", ["q"], FakeIndex(), 5, trace)
    assert trace.searches == len(trace.queries) > 1


def test_a_trace_can_be_merged_across_queries() -> None:
    total, one = RetrievalTrace(), RetrievalTrace()
    SimpleRetrieval().retrieve("q", ["q"], FakeIndex(), 5, one)
    one.record_model_call(2)
    total.merge(one)

    assert total.searches == 1
    assert total.model_calls == 2


# ---------------------------------------------------------------------------
# reachable from a config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        "simple",
        "widened",
        "widened:8",
        "decomposed",
        "decomposed:2",
        "decomposed:min_words=3",
        "relevance-feedback",
        "relevance-feedback:3",
    ],
)
def test_every_strategy_is_reachable_from_one_config_line(spec: str) -> None:
    assert get_retriever(spec).name in {
        "simple",
        "widened",
        "decomposed",
        "relevance-feedback",
    }


def test_none_means_plain_search() -> None:
    """So an existing config that never heard of this axis keeps behaving identically."""
    assert get_retriever(None).name == "simple"


def test_the_registry_documents_every_strategy() -> None:
    for name, description in RETRIEVERS.describe().items():
        assert description, name


def test_the_axis_multiplies_the_matrix() -> None:
    from contextgrid.grid import matrix

    configs = matrix(retrieval=["simple", "widened:4", "decomposed"]).expand("factorial")
    assert {config.retrieval for config in configs} == {"simple", "widened:4", "decomposed"}


def test_the_label_names_the_strategy_unless_it_is_plain() -> None:
    """A leaderboard row for the default arm should not carry a word that adds nothing, and a
    row for a strategy that costs money must say so."""
    from contextgrid.pipeline import Config

    assert "~" not in Config(retrieval="simple").label
    assert "~decomposed" in Config(retrieval="decomposed").label


# ---------------------------------------------------------------------------
# does it help when there is something to help with?
# ---------------------------------------------------------------------------


def build_two_part_workspace() -> tuple[object, object]:
    """A corpus where the two halves of each question live in different documents.

    Which is the only situation decomposition can help in, and therefore the only honest way
    to test that it does.
    """
    from contextgrid.core.documents import MediaType
    from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
    from contextgrid.corpus import Corpus

    corpus = Corpus.from_texts(
        {
            "refunds.md": "# Refunds\n\nRefunds are issued within 30 days of purchase.\n",
            "digital.md": "# Digital goods\n\nDigital goods are not refundable once downloaded.\n",
            "shipping.md": "# Shipping\n\nStandard shipping takes 5 to 7 business days.\n",
            "express.md": "# Express\n\nExpress shipping arrives the next business day.\n",
            "holidays.md": "# Holidays\n\nThe office is closed on public holidays.\n",
            "returns.md": "# Returns\n\nReturns must be posted within 14 days of delivery.\n",
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
    return corpus, evalset


def test_decomposition_is_measurably_different_on_two_part_questions() -> None:
    """The claim the axis exists to test. Not asserting that it *wins* -- on a corpus this
    small either could -- only that it genuinely does something different, which is what
    separates a measured tie from a strategy that never fired."""
    from contextgrid.grid import Runner, matrix

    corpus, evalset = build_two_part_workspace()
    results = Runner(corpus=corpus, headline="recall@2").run(
        matrix(
            chunker="recursive:128",
            index="bm25",
            embedder=None,
            retrieval=["simple", "decomposed"],
            k=2,
        ),
        evalset,
        mode="factorial",
    )

    assert len(results.runs) == 2
    # The "did nothing" warning must *not* fire here: these questions have two parts each.
    assert not any("identically to plain search" in w.message for w in results.warnings)


def test_the_warning_fires_when_no_question_can_be_split() -> None:
    """An eval set of single-part questions cannot tell you whether decomposition helps, and
    saying so is more useful than a row that looks like a measured tie."""
    from contextgrid.core.documents import MediaType
    from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
    from contextgrid.corpus import Corpus
    from contextgrid.grid import Runner, matrix

    corpus = Corpus.from_texts(
        {"refunds.md": "# Refunds\n\nRefunds are issued within 30 days of purchase.\n"},
        media_type=MediaType.MARKDOWN,
        name="one-part",
    )
    evalset = EvalSet(
        id="one-part",
        items=(
            EvalItem(
                id="q1",
                question="how long do refunds take?",
                anchors=(GoldAnchor(quote="within 30 days", source_id="refunds.md"),),
            ),
        ),
    )

    results = Runner(corpus=corpus, headline="recall@2").run(
        matrix(chunker="recursive:128", index="bm25", embedder=None, retrieval="decomposed", k=2),
        evalset,
        mode="factorial",
    )
    assert any("identically to plain search" in w.message for w in results.warnings)


def test_plain_search_never_triggers_the_warning() -> None:
    """`simple` behaving like plain search is not a finding."""
    from contextgrid.core.documents import MediaType
    from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
    from contextgrid.corpus import Corpus
    from contextgrid.grid import Runner, matrix

    corpus = Corpus.from_texts(
        {"a.md": "# A\n\nRefunds are issued within 30 days.\n"},
        media_type=MediaType.MARKDOWN,
        name="plain",
    )
    evalset = EvalSet(
        id="plain",
        items=(
            EvalItem(
                id="q1",
                question="refunds?",
                anchors=(GoldAnchor(quote="within 30 days", source_id="a.md"),),
            ),
        ),
    )
    results = Runner(corpus=corpus, headline="recall@2").run(
        matrix(chunker="recursive:128", index="bm25", embedder=None, retrieval="simple", k=2),
        evalset,
        mode="factorial",
    )
    assert not any("identically to plain search" in w.message for w in results.warnings)


# ---------------------------------------------------------------------------
# does relevance feedback find what plain search cannot rank?
# ---------------------------------------------------------------------------


def build_vocabulary_gap_workspace() -> tuple[object, object]:
    """A corpus where the second relevant document shares *no words at all* with the
    question -- only with the best hit's own vocabulary.

    Which is the only situation relevance feedback can help in. Plain BM25 cannot rank a
    document it shares zero terms with, however relevant it is; a second search built from
    words the best hit actually used can reach it.
    """
    from contextgrid.core.documents import MediaType
    from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
    from contextgrid.corpus import Corpus

    corpus = Corpus.from_texts(
        {
            "security.md": (
                "# Security\n\nSecurity measures at the company include SOC2 Type II audits "
                "and PCI-DSS certification for protecting customer data.\n"
            ),
            "audits.md": (
                "# Audits\n\nSOC2 Type II and PCI-DSS certification renewals happen annually "
                "through an independent assessor.\n"
            ),
            "shipping.md": (
                "# Shipping\n\nStandard shipping takes five business days for domestic orders.\n"
            ),
            "refunds.md": (
                "# Refunds\n\nRefunds are processed within thirty days of the original "
                "purchase date.\n"
            ),
            "privacy.md": (
                "# Privacy\n\nWe retain user records for seven years as required by regulation.\n"
            ),
        },
        media_type=MediaType.MARKDOWN,
        name="vocabulary-gap",
    )
    evalset = EvalSet(
        id="vocabulary-gap",
        items=(
            EvalItem(
                id="q1",
                question="what security measures does the company use?",
                anchors=(
                    GoldAnchor(
                        quote="SOC2 Type II audits and PCI-DSS certification",
                        source_id="security.md",
                    ),
                    GoldAnchor(
                        quote="renewals happen annually through an independent assessor",
                        source_id="audits.md",
                    ),
                ),
            ),
        ),
    )
    return corpus, evalset


def test_relevance_feedback_finds_what_plain_search_cannot_rank() -> None:
    """Measured, not asserted: `simple` finds only `security.md` (the only document sharing a
    word with the question at all); `relevance-feedback` reads that hit, searches again with
    its distinctive words, and reaches `audits.md` too -- a document plain search has no way
    to distinguish from the three unrelated ones in this corpus."""
    from contextgrid.grid import Runner, matrix

    corpus, evalset = build_vocabulary_gap_workspace()
    results = Runner(corpus=corpus, headline="recall@2").run(
        matrix(
            chunker="recursive:128",
            index="bm25",
            embedder=None,
            retrieval=["simple", "relevance-feedback"],
            k=2,
        ),
        evalset,
        mode="factorial",
    )

    scores = {run.config.retrieval: run.metric("recall@2") for run in results.runs}
    assert scores["simple"] < scores["relevance-feedback"]
    assert scores["relevance-feedback"] == pytest.approx(1.0)
    # The "did nothing" warning must not fire: the second search genuinely ran and changed
    # what came back.
    assert not any("identically to plain search" in w.message for w in results.warnings)
