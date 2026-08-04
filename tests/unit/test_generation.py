"""Unit tests for context assembly and the generation panel."""

from __future__ import annotations

import pytest

from contextgrid.assemble import AssembledContext, ContextAssembler, Ordering, tokens_sent
from contextgrid.core.documents import Chunk
from contextgrid.core.evalset import EvalItem, GoldSpan
from contextgrid.core.span import Span
from contextgrid.evalset.llm import RecordingLLM
from contextgrid.generate import (
    Answer,
    ExtractiveGenerator,
    GenerationReport,
    Generator,
    LLMGenerator,
    lift,
    score_answer,
)

PASSAGES = [
    "The notice period for termination is thirty days in writing.",
    "Fees are payable within thirty days of the invoice date.",
    "The API key belongs in the X-Api-Key header of every request.",
    "Unknown widget identifiers cause the endpoint to return 404.",
]


def chunks(count: int = 4, doc: str = "d") -> list[Chunk]:
    return [
        Chunk(id=f"{doc}:{i}", span=Span(doc, i * 100, i * 100 + len(text)), text=text)
        for i, text in enumerate(PASSAGES[:count])
    ]


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def test_assembling_nothing_is_safe() -> None:
    assembled = ContextAssembler().assemble([])
    assert assembled.text == ""
    assert assembled.used == 0
    assert assembled.tokens == 0


def test_every_chunk_reaches_the_context_by_default() -> None:
    assembled = ContextAssembler().assemble(chunks())
    assert assembled.used == 4
    for passage in PASSAGES:
        assert passage in assembled.text


def test_relevance_order_is_the_order_it_was_given() -> None:
    assembled = ContextAssembler(ordering=Ordering.RELEVANCE).assemble(chunks())
    assert [c.id for c in assembled.chunks] == ["d:0", "d:1", "d:2", "d:3"]


def test_the_ends_ordering_puts_the_weakest_evidence_in_the_middle() -> None:
    """Long-context models attend most strongly to the start and end of their context, so
    the least useful passage belongs where it will be missed if anything is."""
    assembled = ContextAssembler(ordering=Ordering.ENDS).assemble(chunks())
    order = [c.id for c in assembled.chunks]
    assert order[0] == "d:0"  # best first
    assert order[-1] == "d:1"  # second-best last
    assert "d:3" in order[1:-1]  # weakest in the middle


def test_document_ordering_restores_reading_order() -> None:
    shuffled = list(reversed(chunks()))
    assembled = ContextAssembler(ordering=Ordering.DOCUMENT).assemble(shuffled)
    assert [c.char_start for c in assembled.chunks] == [0, 100, 200, 300]


def test_reversed_ordering_puts_the_best_closest_to_the_question() -> None:
    assembled = ContextAssembler(ordering=Ordering.REVERSED).assemble(chunks())
    assert [c.id for c in assembled.chunks][-1] == "d:0"


def test_a_token_budget_drops_chunks_rather_than_truncating_one() -> None:
    """Half a passage reads as a complete one, and a model given half an answer will
    confidently give half an answer back."""
    assembled = ContextAssembler(budget_tokens=15).assemble(chunks())
    assert assembled.used < 4
    assert assembled.dropped > 0
    for chunk in assembled.chunks:
        assert chunk.text in assembled.text  # whole passages only


def test_a_dropped_chunk_is_reported_as_possibly_fatal() -> None:
    assembled = ContextAssembler(budget_tokens=15).assemble(chunks())
    assert any("cannot answer" in w.message for w in assembled.warnings)


def test_at_least_one_chunk_survives_an_impossible_budget() -> None:
    """Sending nothing guarantees a wrong answer; sending one thing might not."""
    assembled = ContextAssembler(budget_tokens=1).assemble(chunks())
    assert assembled.used == 1


def test_a_generous_budget_drops_nothing() -> None:
    assert ContextAssembler(budget_tokens=10_000).assemble(chunks()).dropped == 0


def test_a_chunk_already_covered_by_a_better_one_is_dropped() -> None:
    """Overlap is a chunker parameter, not a bug -- but sending the same sentence to a
    generator twice is paying twice for it."""
    outer = Chunk(id="a", span=Span("d", 0, 200), text=PASSAGES[0] + " " + PASSAGES[1])
    inner = Chunk(id="b", span=Span("d", 0, 60), text=PASSAGES[0])

    assembled = ContextAssembler().assemble([outer, inner])
    assert [c.id for c in assembled.chunks] == ["a"]
    assert assembled.duplicate_characters > 0


def test_deduplication_can_be_switched_off() -> None:
    outer = Chunk(id="a", span=Span("d", 0, 200), text="text")
    inner = Chunk(id="b", span=Span("d", 0, 60), text="text")
    assert ContextAssembler(deduplicate=False).assemble([outer, inner]).used == 2


def test_partial_overlap_is_counted_but_not_dropped() -> None:
    first = Chunk(id="a", span=Span("d", 0, 100), text=PASSAGES[0])
    second = Chunk(id="b", span=Span("d", 50, 200), text=PASSAGES[1])
    assembled = ContextAssembler().assemble([first, second])
    assert assembled.used == 2
    assert assembled.duplicate_characters == 50


def test_sources_are_labelled_so_a_model_can_cite_them() -> None:
    """Without a label a model cannot cite anything, and citation accuracy is the metric
    enterprises care about most."""
    assembled = ContextAssembler(include_source=True).assemble(chunks(2))
    assert "[1] d" in assembled.text
    assert "[2] d" in assembled.text


def test_heading_paths_can_be_included() -> None:
    chunk = Chunk(
        id="c",
        span=Span("d", 0, 20),
        text="body",
        meta={"heading_path": ("Termination", "Notice")},
    )
    assembled = ContextAssembler(include_heading=True).assemble([chunk])
    assert "Termination > Notice" in assembled.text


def test_tokens_sent_is_the_number_k_is_a_poor_proxy_for() -> None:
    """Five structural chunks can be four times the text of five sentence windows."""
    assert tokens_sent(chunks()) > tokens_sent(chunks(1))


def test_the_assembly_serialises() -> None:
    payload = ContextAssembler().assemble(chunks()).as_dict()
    assert set(payload) == {
        "tokens",
        "characters",
        "chunks_used",
        "chunks_dropped",
        "duplicate_characters",
    }


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------


GENERATORS: list[Generator] = [ExtractiveGenerator()]


@pytest.fixture(params=GENERATORS, ids=[g.name for g in GENERATORS])
def generator(request: pytest.FixtureRequest) -> Generator:
    return request.param  # type: ignore[no-any-return]


def context() -> AssembledContext:
    return ContextAssembler().assemble(chunks())


def test_a_generator_answers(generator: Generator) -> None:
    assert generator.answer("How long is notice?", context()).text


def test_a_generator_with_no_context_declines(generator: Generator) -> None:
    """Answering with nothing to go on is the failure this measures."""
    empty = ContextAssembler().assemble([])
    assert generator.answer("How long is notice?", empty).is_abstention


def test_the_extractive_generator_is_the_ceiling_retrieval_alone_can_reach() -> None:
    """An answer score against it separates "the retriever found the evidence" from "the
    generator did something useful with it"."""
    answer = ExtractiveGenerator().answer("How long is notice?", context())
    assert answer.text.startswith("The notice period")
    assert answer.citations == (1,)


def test_an_llm_generator_fills_the_prompt_and_reads_citations() -> None:
    llm = RecordingLLM(replies=["The notice period is thirty days [1]."])
    answer = LLMGenerator(llm=llm).answer("How long is notice?", context())

    assert answer.citations == (1,)
    assert "How long is notice?" in llm.prompts[0]
    assert "thirty days" in llm.prompts[0]  # the context went in


def test_the_default_prompt_asks_the_model_to_decline_rather_than_guess() -> None:
    llm = RecordingLLM(replies=["..."])
    LLMGenerator(llm=llm).answer("q", context())
    assert "say so plainly rather than guessing" in llm.prompts[0]


def test_the_prompt_is_a_sweepable_axis() -> None:
    """Prompt changes routinely beat retrieval changes, which is uncomfortable and worth
    knowing before spending a quarter on an embedding migration."""
    llm = RecordingLLM(replies=["x"])
    LLMGenerator(llm=llm, prompt="ONLY: {context}\nQ: {question}").answer("q", context())
    assert llm.prompts[0].startswith("ONLY:")


@pytest.mark.parametrize(
    "text",
    [
        "I don't know.",
        "The context does not say.",
        "There is not enough information to answer.",
        "That is not mentioned in the passages.",
    ],
)
def test_refusals_are_recognised(text: str) -> None:
    assert Answer(text=text).is_abstention


def test_a_real_answer_is_not_a_refusal() -> None:
    assert not Answer(text="The notice period is thirty days.").is_abstention


# ---------------------------------------------------------------------------
# scoring an answer
# ---------------------------------------------------------------------------


def item(answerable: bool = True) -> EvalItem:
    gold = (GoldSpan(Span("d", 0, 60)),) if answerable else ()
    return EvalItem(id="q1", question="How long is the notice period?", gold=gold)


def test_an_answer_drawn_from_the_context_is_grounded() -> None:
    answer = Answer(text="The notice period for termination is thirty days in writing.")
    score = score_answer(item(), answer, context())
    assert score.groundedness > 0.9


def test_an_invented_answer_is_flagged() -> None:
    """Content words that are not in the context came from somewhere else, and that is a
    reason to trust the answer less."""
    answer = Answer(text="Napoleon conquered Egypt during a lengthy military campaign.")
    score = score_answer(item(), answer, context())
    assert score.groundedness < 0.5
    assert any("came from somewhere else" in w for w in score.warnings)


def test_a_citation_outside_the_context_is_caught() -> None:
    answer = Answer(text="Thirty days [9].", citations=(9,))
    score = score_answer(item(), answer, context())
    assert score.citation_accuracy == 0.0
    assert any("not in the context" in w for w in score.warnings)


def test_valid_citations_score_full_marks() -> None:
    answer = Answer(text="Thirty days [1][2].", citations=(1, 2))
    assert score_answer(item(), answer, context()).citation_accuracy == 1.0


def test_declining_when_there_is_no_evidence_is_correct() -> None:
    """A correct refusal is a success, not a zero. Marking it as a zero teaches exactly the
    wrong lesson."""
    score = score_answer(item(answerable=False), Answer(text="I don't know."), context())
    assert score.abstained
    assert score.should_have_abstained
    assert score.abstention_correct


def test_answering_anyway_when_the_evidence_is_absent_is_not() -> None:
    score = score_answer(item(answerable=False), Answer(text="It is ninety days."), context())
    assert not score.abstention_correct


def test_an_empty_answer_is_reported() -> None:
    score = score_answer(item(), Answer(text="   "), context())
    assert "returned nothing" in score.warnings[0]


def test_overlap_with_the_gold_evidence_is_measured() -> None:
    gold = chunks(1)
    answer = Answer(text="The notice period for termination is thirty days in writing.")
    assert score_answer(item(), answer, context(), gold_chunks=gold).evidence_overlap > 0.8


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


def test_the_report_names_the_questions_answered_without_evidence() -> None:
    """The failure worth naming. A system that does this is worse than one scoring lower and
    declining, and no retrieval metric will ever show it."""
    report = GenerationReport(generator="test")
    report.scores.append(
        score_answer(item(answerable=False), Answer(text="It is ninety days."), context())
    )
    report.scores.append(score_answer(item(), Answer(text="Thirty days."), context()))

    assert report.confident_when_it_should_not_be == ["q1"]
    assert "answered anyway" in report.summary()


def test_a_well_behaved_system_is_credited_for_declining() -> None:
    report = GenerationReport(generator="test")
    report.scores.append(
        score_answer(item(answerable=False), Answer(text="I don't know."), context())
    )
    assert report.abstention_accuracy == 1.0
    assert "almost never measured" in report.summary()


def test_an_empty_report_says_so() -> None:
    assert GenerationReport().summary() == "No answers were generated."


def test_the_report_averages_what_it_has() -> None:
    report = GenerationReport(generator="test")
    report.scores.append(
        score_answer(item(), Answer(text="Thirty days [1].", citations=(1,)), context())
    )
    metrics = report.metrics()
    assert set(metrics) == {
        "groundedness",
        "citation_accuracy",
        "evidence_overlap",
        "abstention_accuracy",
    }


# ---------------------------------------------------------------------------
# the lift chart
# ---------------------------------------------------------------------------


def test_a_retrieval_gain_that_reached_the_answer() -> None:
    assert "survived to the answer" in lift(0.80, 0.75, 0.60)


def test_a_retrieval_gain_the_generator_would_have_compensated_for() -> None:
    """The question the whole project implicitly promises to answer, and which nothing in
    the field plots."""
    assert "bought nothing" in lift(0.80, 0.70, 0.70)


def test_better_retrieval_that_produced_worse_answers() -> None:
    verdict = lift(0.80, 0.55, 0.70)
    assert "fell" in verdict
    assert "character precision" in verdict
