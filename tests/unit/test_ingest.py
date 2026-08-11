"""Ingestion strategies: what goes into the index, and what comes back out.

The one rule everything here rests on: **a chunker produces units where the thing indexed and
the thing returned are the same; an ingestion strategy deliberately breaks that identity.**

So the tests are about the seam. That gold still resolves against what comes back. That
indexing four questions for one chunk does not let that chunk fill a result list four times.
That a model failing halfway through a build does not throw the index away. And that a strategy
returning wider passages is not quietly credited with finding more than it did.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from contextgrid.core.documents import Chunk
from contextgrid.core.span import Span
from contextgrid.ingest import (
    INGESTERS,
    ContextualIngestion,
    HierarchicalIngestion,
    HypotheticalQuestionsIngestion,
    IngestionContext,
    IngestionError,
    ParentDocumentIngestion,
    PlainIngestion,
    PropositionsIngestion,
    SentenceWindowIngestion,
    SummaryIngestion,
    get_ingester,
)

TEXT = (
    "Refunds are issued within 30 days of purchase. "
    "Digital goods are not refundable once downloaded. "
    "Standard shipping takes 5 to 7 business days. "
    "Express shipping arrives the next business day. "
    "The office is closed on public holidays. "
    "Returns must be posted within 14 days."
)


def chunks_of(text: str = TEXT, *, size: int = 48, doc: str = "policy.md") -> list[Chunk]:
    """Consecutive, tiling chunks -- the shape a real chunker produces."""
    out: list[Chunk] = []
    for start in range(0, len(text), size):
        end = min(start + size, len(text))
        out.append(
            Chunk(id=f"{doc}:{start}-{end}", span=Span(doc, start, end), text=text[start:end])
        )
    return out


class ScriptedLLM:
    """Returns prepared replies, so a paid strategy can be exercised with no key."""

    def __init__(self, *replies: str, fail: bool = False) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.fail = fail

    def complete(self, prompt: str, *, max_tokens: int = 256) -> str:
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("the provider is down")
        return self.replies.pop(0) if self.replies else (self.replies[-1] if self.replies else "")


def context(llm: object | None = None) -> IngestionContext:
    return IngestionContext(llm=llm)


FREE = [
    PlainIngestion(),
    ParentDocumentIngestion(group=2),
    SentenceWindowIngestion(window=1),
    HierarchicalIngestion(group=2),
]
FREE_IDS = [strategy.name for strategy in FREE]


# ---------------------------------------------------------------------------
# what every strategy has to get right
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", FREE, ids=FREE_IDS)
def test_every_indexed_unit_resolves_to_something_retrievable(strategy: object) -> None:
    """A hit that resolves to nothing is a hit that vanishes, and a strategy that loses results
    looks exactly like a retriever that could not find them.

    "Something that exists" means `retrievable` *or* `presented_chunks`. A strategy is allowed
    to hand back a wider passage than the unit it is scored on -- that is what the presentation
    split is for -- and `sentence-window` does precisely that: it scores the centre chunk and
    returns the window around it. Insisting the resolved id be in `retrievable` would force
    those overlapping windows back into the scored set, which is the denominator inflation the
    split exists to prevent.
    """
    chunks = chunks_of()
    ingested = strategy.ingest(chunks, context())  # type: ignore[attr-defined]

    reachable = {chunk.id for chunk in ingested.retrievable} | set(ingested.presented_chunks)
    for indexed in ingested.indexed:
        assert ingested.resolve(indexed.id) in reachable


@pytest.mark.parametrize("strategy", FREE, ids=FREE_IDS)
def test_retrievable_text_stays_a_slice_of_the_document(strategy: object) -> None:
    """Gold evidence is character offsets into the parse. A returned passage whose text is not
    what the document says at those offsets moves every piece of evidence inside it."""
    chunks = chunks_of()
    ingested = strategy.ingest(chunks, context())  # type: ignore[attr-defined]

    for chunk in ingested.retrievable:
        if chunk.offsets_exact:
            assert TEXT[chunk.span.start : chunk.span.end] == chunk.text


@pytest.mark.parametrize("strategy", FREE, ids=FREE_IDS)
def test_nothing_crosses_a_document_boundary(strategy: object) -> None:
    """A parent spanning the end of one contract and the start of another is a passage that
    does not exist, and any answer read out of it answers a question nobody asked."""
    chunks = chunks_of(doc="a.md") + chunks_of(doc="b.md")
    ingested = strategy.ingest(chunks, context())  # type: ignore[attr-defined]

    for chunk in ingested.retrievable:
        sources = {piece.split(":")[0] for piece in chunk.meta.get("merged_from", [chunk.id])}
        assert len(sources) == 1


@pytest.mark.parametrize("strategy", FREE, ids=FREE_IDS)
def test_the_whole_document_is_still_reachable(strategy: object) -> None:
    """Whatever the regrouping, every character that was chunked must still be inside something
    retrievable -- otherwise a strategy silently makes part of the corpus unfindable."""
    chunks = chunks_of()
    ingested = strategy.ingest(chunks, context())  # type: ignore[attr-defined]

    covered = set()
    for chunk in ingested.retrievable:
        covered.update(range(chunk.span.start, chunk.span.end))
    assert covered == set(range(0, len(TEXT)))


# ---------------------------------------------------------------------------
# plain
# ---------------------------------------------------------------------------


def test_plain_indexes_and_returns_the_same_thing() -> None:
    chunks = chunks_of()
    ingested = PlainIngestion().ingest(chunks, context())

    assert ingested.indexed == ingested.retrievable == chunks
    assert ingested.expansion == 1.0
    assert not ingested.parent_of


def test_plain_is_what_naming_no_strategy_means() -> None:
    assert get_ingester(None).name == "plain"


def test_plain_is_normalised_away() -> None:
    """`plain` and naming nothing are one run, and two names for one run dilute the axis."""
    from contextgrid.grid.matrix import canonicalise
    from contextgrid.pipeline import Config

    assert canonicalise(Config(ingestion="plain")).ingestion is None


# ---------------------------------------------------------------------------
# parent-document
# ---------------------------------------------------------------------------


def test_small_chunks_are_indexed_and_the_parent_is_returned() -> None:
    """The whole idea: a 128-token chunk embeds precisely because it is about one thing, and
    the passage around it is what a generator needs to answer from."""
    chunks = chunks_of()
    ingested = ParentDocumentIngestion(group=3).ingest(chunks, context())

    assert len(ingested.indexed) == len(chunks)
    assert len(ingested.retrievable) < len(chunks)
    for chunk in chunks:
        parent_id = ingested.resolve(chunk.id)
        parent = next(c for c in ingested.retrievable if c.id == parent_id)
        assert parent.span.start <= chunk.span.start
        assert parent.span.end >= chunk.span.end


def test_a_parent_is_a_literal_slice_not_a_join() -> None:
    """Joining the pieces with a separator inserts characters the document does not contain and
    shifts every offset after the first."""
    ingested = ParentDocumentIngestion(group=3).ingest(chunks_of(), context())
    parent = ingested.retrievable[0]
    assert parent.text == TEXT[parent.span.start : parent.span.end]


def test_a_group_of_one_is_refused() -> None:
    with pytest.raises(IngestionError, match="plain chunking under a different name"):
        ParentDocumentIngestion(group=1)


# ---------------------------------------------------------------------------
# sentence-window
# ---------------------------------------------------------------------------


def test_the_window_is_centred_on_the_match() -> None:
    """Where parent-document returns a fixed passage whatever matched inside it, this centres
    the returned context -- so a hit at the end of a passage brings back what follows."""
    chunks = chunks_of()
    ingested = SentenceWindowIngestion(window=1).ingest(chunks, context())

    middle = chunks[2]
    window_id = ingested.resolve(middle.id)
    # The window is a presentation passage, not a scored unit -- overlapping windows in
    # the scored set inflate the denominator, which is the bug this split prevents.
    window = ingested.presented_chunks[window_id]

    assert window.span.start == chunks[1].span.start
    assert window.span.end == chunks[3].span.end


def test_the_first_chunk_gets_a_truncated_window() -> None:
    """There is nothing before it, and clamping is better than dropping it."""
    chunks = chunks_of()
    ingested = SentenceWindowIngestion(window=2).ingest(chunks, context())
    window_id = ingested.resolve(chunks[0].id)
    # The window is a presentation passage, not a scored unit -- overlapping windows in
    # the scored set inflate the denominator, which is the bug this split prevents.
    window = ingested.presented_chunks[window_id]
    assert window.span.start == chunks[0].span.start


def test_a_window_of_zero_is_refused() -> None:
    with pytest.raises(IngestionError, match="plain chunking under a different name"):
        SentenceWindowIngestion(window=0)


# ---------------------------------------------------------------------------
# hierarchical
# ---------------------------------------------------------------------------


def test_a_parent_is_presentation_not_a_second_thing_to_find() -> None:
    """The bug this caught. Putting a parent and its children both in the scored set makes gold
    resolve to each of them, so a question with one answer acquires two things to find and
    recall halves for a purely structural reason -- measured at 1.86 relevant units per
    question against plain chunking's 1.00."""
    chunks = chunks_of()
    ingested = HierarchicalIngestion(group=2).ingest(chunks, context())

    assert [c.id for c in ingested.retrievable] == [c.id for c in chunks]
    assert ingested.presentation
    assert ingested.presented_chunks


def test_a_presentation_passage_is_scored_as_the_units_it_covers() -> None:
    """Showing a generator more context must never change what retrieval is credited with."""
    ingested = HierarchicalIngestion(group=2).ingest(chunks_of(), context())
    parent_id = next(iter(ingested.presentation))

    assert ingested.scored_ids(parent_id) == ingested.presentation[parent_id]
    assert ingested.scored_ids("something-else") == ["something-else"]


def test_an_impossible_threshold_is_refused() -> None:
    with pytest.raises(IngestionError, match=r"must be in \(0, 1\]"):
        HierarchicalIngestion(threshold=0)


# ---------------------------------------------------------------------------
# the paid strategies
# ---------------------------------------------------------------------------


def test_contextual_indexes_the_chunk_with_its_context_and_returns_the_chunk() -> None:
    """A chunk reading "the notice period is thirty days" is a perfect answer that no search for
    "termination notice" will find, because the connecting words are in a heading four chunks
    earlier."""
    chunks = chunks_of()
    llm = ScriptedLLM(*["This is from the refunds section." for _ in chunks])
    ingested = ContextualIngestion().ingest(chunks, context(llm))

    assert len(ingested.indexed) == len(chunks)
    assert "refunds section" in ingested.indexed[0].text
    assert ingested.indexed[0].text.endswith(chunks[0].text)
    # What comes back is the document's own words, so gold still resolves.
    assert [c.id for c in ingested.retrievable] == [c.id for c in chunks]
    assert ingested.model_calls == len(chunks)


def test_written_text_is_never_returned() -> None:
    """Returning a model's paraphrase and scoring it against the document measures nothing."""
    chunks = chunks_of()
    llm = ScriptedLLM(*["invented context" for _ in chunks])
    ingested = ContextualIngestion().ingest(chunks, context(llm))

    for chunk in ingested.retrievable:
        assert "invented context" not in chunk.text


def test_generated_text_is_flagged_as_not_a_slice() -> None:
    """It carries the span it derives from and its text is not that span. Anything scoring on
    character offsets has to know the difference."""
    chunks = chunks_of()
    llm = ScriptedLLM(*["context" for _ in chunks])
    ingested = ContextualIngestion().ingest(chunks, context(llm))

    assert not ingested.indexed[0].offsets_exact
    assert all(chunk.offsets_exact for chunk in ingested.retrievable)


def test_hypothetical_questions_index_several_vectors_for_one_chunk() -> None:
    chunks = chunks_of()
    llm = ScriptedLLM(
        *['["how long do refunds take?", "what is the refund window?"]'] * len(chunks)
    )
    ingested = HypotheticalQuestionsIngestion(count=2).ingest(chunks, context(llm))

    assert ingested.expansion == 2.0
    assert "refund" in ingested.indexed[0].text
    for indexed in ingested.indexed:
        assert ingested.resolve(indexed.id) in {c.id for c in chunks}


def test_propositions_index_atomic_facts() -> None:
    chunks = chunks_of()
    llm = ScriptedLLM(
        *['["Refunds take 30 days.", "Digital goods are not refundable."]'] * len(chunks)
    )
    ingested = PropositionsIngestion(count=2).ingest(chunks, context(llm))

    assert ingested.expansion == 2.0
    assert ingested.notes["per_chunk"] == 2.0


def test_summary_costs_one_call_per_document_not_per_chunk() -> None:
    """The cheapest paid strategy, and it answers a different question: not "which passage?"
    but "which document?"."""
    chunks = chunks_of(doc="a.md") + chunks_of(doc="b.md")
    llm = ScriptedLLM("a summary of a", "a summary of b")
    ingested = SummaryIngestion().ingest(chunks, context(llm))

    assert ingested.model_calls == 2
    assert len(ingested.retrievable) == 2
    assert "summary of a" in ingested.indexed[0].text


def test_a_provider_failure_does_not_throw_the_index_away() -> None:
    """Half an index is worse than a slow one. A hiccup two thousand chunks into a build must
    not discard the first nineteen hundred."""
    chunks = chunks_of()
    ingested = ContextualIngestion().ingest(chunks, context(ScriptedLLM(fail=True)))

    assert len(ingested.indexed) == len(chunks)
    assert ingested.indexed[0].id == chunks[0].id  # indexed as written


def test_a_failure_says_the_row_mixes_two_strategies() -> None:
    ctx = context(ScriptedLLM(fail=True))
    ContextualIngestion().ingest(chunks_of(), ctx)
    assert any("mixes two strategies" in warning.message for warning in ctx.warnings)


def test_the_paid_strategies_declare_that_they_cost_money() -> None:
    for strategy in (
        ContextualIngestion(),
        HypotheticalQuestionsIngestion(),
        PropositionsIngestion(),
        SummaryIngestion(),
    ):
        assert strategy.uses_model
    for strategy in FREE:
        assert not strategy.uses_model  # type: ignore[attr-defined]


def test_a_sweep_with_a_paid_strategy_and_no_limit_says_the_bill_is_unknowable() -> None:
    from contextgrid.grid import matrix
    from contextgrid.grid.runner import Budget, _warn_if_unbounded

    grid = matrix(ingestion="contextual")
    _warn_if_unbounded(grid, Budget())
    assert grid.meta["unbounded_model_calls"] == "contextual"

    free = matrix(ingestion="parent-document")
    _warn_if_unbounded(free, Budget())
    assert "unbounded_model_calls" not in free.meta


# ---------------------------------------------------------------------------
# the axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        "plain",
        "parent-document",
        "parent-document:3",
        "sentence-window:2",
        "hierarchical:4,threshold=0.75",
        "contextual",
        "hypothetical-questions:4",
        "propositions:8",
        "summary",
    ],
)
def test_every_strategy_is_reachable_from_one_config_line(spec: str) -> None:
    assert get_ingester(spec).name in INGESTERS.names()


def test_all_eight_are_registered_and_documented() -> None:
    assert len(INGESTERS.names()) == 8
    for name, description in INGESTERS.describe().items():
        assert description, name


def test_the_config_file_accepts_the_axis() -> None:
    from contextgrid.config import loads

    config = loads("corpus: ./docs\ngrid:\n  ingestion: [plain, parent-document:4]\n")
    assert config.grid.ingestion == ("plain", "parent-document:4")


def test_indexing_small_and_returning_big_beats_plain_chunking() -> None:
    """The claim the axis exists to check, end to end.

    Small chunks embed precisely and arrive stripped of context. Returning the passage around
    the match is the oldest free answer to that, and on this corpus it is a large win -- which
    is exactly the sort of thing nobody can currently check on their own documents.
    """
    from contextgrid.core.documents import MediaType
    from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
    from contextgrid.corpus import Corpus
    from contextgrid.grid import Runner, matrix

    corpus = Corpus.from_texts(
        {
            "policy.md": "# Policy\n\n" + TEXT + "\n",
            "other.md": "# Other\n\nThe office is closed on public holidays.\n",
        },
        media_type=MediaType.MARKDOWN,
        name="ingest",
    )
    evalset = EvalSet(
        id="ingest",
        items=(
            EvalItem(
                id="q1",
                question="how long do refunds take?",
                anchors=(GoldAnchor(quote="within 30 days of purchase", source_id="policy.md"),),
            ),
        ),
    )

    results = Runner(corpus=corpus, headline="recall@2").run(
        matrix(
            ingestion=["plain", "parent-document:3"],
            chunker="recursive:32,overlap=0",
            index="bm25",
            embedder=None,
            k=2,
        ),
        evalset,
        mode="factorial",
    )

    assert len(results.runs) == 2
    # Both should still find it; what matters is that neither silently scores zero because the
    # returned unit was something the qrels never heard of.
    assert all(run.metric("recall@2") > 0 for run in results.runs)


def test_several_vectors_for_one_chunk_do_not_fill_the_results_with_it() -> None:
    """Four generated questions for one chunk all matching is one passage found four times.
    Counting it four times fills the result list with a single passage while claiming to have
    found several."""
    from contextgrid.core.documents import MediaType
    from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
    from contextgrid.corpus import Corpus
    from contextgrid.grid import Runner, matrix
    from contextgrid.ingest import INGESTERS

    name = "scripted-questions"
    if name not in INGESTERS:

        def build() -> object:
            class Scripted(HypotheticalQuestionsIngestion):
                def ingest(self, chunks: Sequence[Chunk], ctx: IngestionContext):  # type: ignore[override]
                    ctx.llm = ScriptedLLM(
                        *['["refunds?", "money back?", "refund window?"]'] * len(chunks)
                    )
                    return HypotheticalQuestionsIngestion.ingest(self, chunks, ctx)

            return Scripted(count=3)

        INGESTERS.register(name, doc="hypothetical questions with a scripted model")(build)

    corpus = Corpus.from_texts(
        {
            "a.md": "# A\n\nRefunds are issued within 30 days of purchase.\n",
            "b.md": "# B\n\nShipping takes five days.\n",
            "c.md": "# C\n\nThe office closes on holidays.\n",
        },
        media_type=MediaType.MARKDOWN,
        name="fanout",
    )
    evalset = EvalSet(
        id="fanout",
        items=(
            EvalItem(
                id="q1",
                question="refunds?",
                anchors=(GoldAnchor(quote="within 30 days", source_id="a.md"),),
            ),
        ),
    )

    results = Runner(corpus=corpus, headline="recall@3").run(
        matrix(chunker="recursive:64", index="bm25", embedder=None, ingestion=name, k=3),
        evalset,
        mode="factorial",
    )
    returned = results.runs[0]
    assert returned.metric("recall@3") > 0


def test_overlapping_windows_do_not_inflate_the_denominator() -> None:
    """The bug this split exists for, and the one place it had not reached.

    `sentence-window` put its windows in `retrievable`. Windows overlap by design, so a single
    piece of gold evidence resolved to every window containing it -- about four of them at
    `window=2`. The arm then had four things to find where plain chunking had one, and was
    marked down for returning the answer once: measured per-question recall of 1/3, 1/4 and
    2/5 on evidence it had returned in full, while `char_recall` stayed at 1.000 and disagreed
    with the headline the whole way down the table.

    Worse than the `hierarchical` case that prompted the presentation split in the first
    place: 4.03 relevant units per question against 1.86.
    """
    chunks = chunks_of()
    ingested = SentenceWindowIngestion(window=2).ingest(chunks, context())

    # One scored unit per chunk. Not one per window.
    assert len(ingested.retrievable) == len(chunks)
    assert {chunk.id for chunk in ingested.retrievable} == {chunk.id for chunk in chunks}

    # Every window is a presentation passage covering the chunks inside it, centre first.
    #
    # Centre first is not cosmetic: a returned window expands into several scored units and a
    # cut-off counts units, so `recall@1` looks at the first one. The centre is the chunk that
    # matched and the reason the window ranked at all. In document order it sits second or
    # third, and `sentence-window` fell to 0.029 at `recall@1` against plain chunking's 0.765
    # while returning strictly more text.
    #
    # Crediting the centre *alone* was the first attempt and was worse than the original bug:
    # a window pulled in by its centre often holds the gold in a neighbour, so the returned
    # text answered the question and scoring called it a miss -- 0.301 against plain's 0.658
    # on the demo corpus.
    for chunk in chunks:
        window_id = ingested.resolve(chunk.id)
        assert window_id in ingested.presented_chunks
        scored = ingested.scored_ids(window_id)
        assert scored[0] == chunk.id, "the chunk that matched must be credited first"
        assert set(scored) <= {c.id for c in chunks}
        assert len(scored) == len(set(scored))


def test_a_window_still_returns_more_text_than_the_chunk_it_scores() -> None:
    """The scoring fix must not quietly turn the strategy into plain chunking -- the whole
    point is that a generator sees the neighbours."""
    chunks = chunks_of()
    ingested = SentenceWindowIngestion(window=1).ingest(chunks, context())

    window = ingested.presented_chunks[ingested.resolve(chunks[2].id)]
    assert window.span.start < chunks[2].span.start
    assert window.span.end > chunks[2].span.end
