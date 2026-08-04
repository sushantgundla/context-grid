"""Unit tests for generating, filtering, reviewing and importing eval sets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextgrid.core.documents import Chunk
from contextgrid.core.errors import EvalSetError
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor, GoldSpan, QuestionType
from contextgrid.core.span import Span
from contextgrid.evalset import (
    Classifier,
    DanglingReferenceFilter,
    DuplicateFilter,
    FilterChain,
    GeneralKnowledgeFilter,
    KeywordProbeGenerator,
    LLMError,
    LLMQuestionGenerator,
    NonDiscriminatingFilter,
    RecordingLLM,
    ReviewQueue,
    ShortQuestionFilter,
    UnresolvedEvidenceFilter,
    Verdict,
    answerer_from,
    assess,
    classify_question,
    default_filters,
    generate,
    minimum_detectable_difference,
    parse_json_reply,
    read_beir,
    read_csv,
    read_jsonl,
    read_legalbench_rag,
    review_summary,
    write_csv,
    write_jsonl,
)


def item(
    iid: str = "q1",
    question: str = "How long is the notice period under this agreement?",
    *,
    quote: str | None = "thirty days",
    **kwargs: object,
) -> EvalItem:
    anchors = (GoldAnchor(source_id="contract.md", quote=quote),) if quote else ()
    return EvalItem(id=iid, question=question, anchors=anchors, **kwargs)  # type: ignore[arg-type]


def evalset(*items: EvalItem, eid: str = "es") -> EvalSet:
    return EvalSet(id=eid, items=items)


# ---------------------------------------------------------------------------
# JSONL round trip
# ---------------------------------------------------------------------------


def test_jsonl_round_trips_everything(tmp_path: Path) -> None:
    original = EvalSet(
        id="es",
        items=(
            EvalItem(
                id="q1",
                question="How long is notice?",
                gold=(GoldSpan(Span("contract.md", 10, 40), grade=1),),
                anchors=(GoldAnchor(source_id="contract.md", quote="thirty days", page_hint=2),),
                qtype=QuestionType.FACTOID,
                answer="Thirty days.",
                meta={"reviewed": True},
            ),
        ),
        version=3,
        source="auto",
    )
    path = write_jsonl(original, tmp_path / "es.jsonl")
    assert read_jsonl(path) == original


def test_jsonl_survives_a_file_with_no_header(tmp_path: Path) -> None:
    path = tmp_path / "bare.jsonl"
    path.write_text(json.dumps(item().to_dict()) + "\n")
    assert len(read_jsonl(path)) == 1


def test_a_missing_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(EvalSetError, match="no eval set at"):
        read_jsonl(tmp_path / "absent.jsonl")


def test_malformed_json_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(item().to_dict()) + "\nnot json at all\n")
    with pytest.raises(EvalSetError, match=":2 is not valid JSON"):
        read_jsonl(path)


# ---------------------------------------------------------------------------
# CSV -- what a domain expert actually hands you
# ---------------------------------------------------------------------------


def test_csv_accepts_the_column_names_people_actually_use(tmp_path: Path) -> None:
    path = tmp_path / "questions.csv"
    path.write_text(
        "query,document,evidence\n"
        "How long is notice?,contract.md,thirty days\n"
        "Which header carries the key?,api.md,X-Api-Key\n"
    )
    loaded = read_csv(path)
    assert len(loaded) == 2
    assert loaded.items[0].anchors[0].source_id == "contract.md"


def test_csv_without_evidence_still_loads_the_questions(tmp_path: Path) -> None:
    path = tmp_path / "questions.csv"
    path.write_text("question\nHow long is notice?\n")
    loaded = read_csv(path)
    assert len(loaded) == 1
    assert not loaded.items[0].is_portable


def test_csv_without_a_question_column_says_what_it_looked_for(tmp_path: Path) -> None:
    path = tmp_path / "questions.csv"
    path.write_text("thing,other\na,b\n")
    with pytest.raises(EvalSetError, match="no question column"):
        read_csv(path)


def test_csv_skips_blank_rows(tmp_path: Path) -> None:
    path = tmp_path / "questions.csv"
    path.write_text("question\nHow long is notice?\n\n,\n")
    assert len(read_csv(path)) == 1


def test_csv_round_trips_for_hand_editing(tmp_path: Path) -> None:
    original = evalset(item(qtype="factoid", answer="Thirty days."))
    reloaded = read_csv(write_csv(original, tmp_path / "out.csv"))
    assert reloaded.items[0].question == original.items[0].question
    assert reloaded.items[0].anchors[0].quote == "thirty days"


# ---------------------------------------------------------------------------
# BEIR and LegalBench-RAG
# ---------------------------------------------------------------------------


def test_beir_import_says_it_is_document_level(tmp_path: Path) -> None:
    """It compares retrievers fairly and cannot compare chunkers fairly. Stating that on the
    imported set is better than letting somebody discover it from a strange leaderboard."""
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"_id": "1", "text": "how long is notice"}\n')
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\n1\tdoc7\t1\n")

    loaded = read_beir(queries, qrels)
    assert len(loaded) == 1
    assert loaded.meta["granularity"] == "document"
    assert "cannot compare chunkers fairly" in loaded.meta["note"]


def test_beir_drops_queries_with_no_judgements(tmp_path: Path) -> None:
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"_id": "1", "text": "a"}\n{"_id": "2", "text": "b"}\n')
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\n1\tdoc7\t1\n2\tdoc8\t0\n")
    assert [i.id for i in read_beir(queries, qrels)] == ["1"]


def test_legalbench_rag_imports_character_spans(tmp_path: Path) -> None:
    """The only public benchmark that anchors evidence the way this package does, which
    makes it the natural set to validate the whole scoring chain against."""
    path = tmp_path / "lb.json"
    path.write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "query": "What is the notice period?",
                        "snippets": [{"file_path": "contract.txt", "span": [840, 1010]}],
                    }
                ]
            }
        )
    )
    loaded = read_legalbench_rag(path)
    assert loaded.meta["granularity"] == "span"
    assert loaded.items[0].gold[0].span == Span("contract.txt", 840, 1010)


def test_legalbench_rag_skips_malformed_snippets(tmp_path: Path) -> None:
    path = tmp_path / "lb.json"
    path.write_text(
        json.dumps({"tests": [{"query": "q", "snippets": [{"file_path": "c.txt"}, {}]}]})
    )
    assert read_legalbench_rag(path).items[0].gold == ()


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------


def test_a_dangling_pronoun_is_rejected() -> None:
    """ "What does it cover?" was clear beside its chunk and is unanswerable alone."""
    kept, rejected = DanglingReferenceFilter().apply([item(question="What does it cover?")])
    assert not kept
    assert "nothing to refer to" in rejected[0].reason


def test_a_pronoun_with_an_antecedent_survives() -> None:
    kept, _ = DanglingReferenceFilter().apply(
        [item(question="Under the agreement, how long is its notice period?")]
    )
    assert len(kept) == 1


def test_near_duplicates_are_rejected_and_named() -> None:
    """Ten rewordings look like ten measurements and are one."""
    first = item("q1", "How long is the notice period for termination?")
    second = item("q2", "How long is the notice period for termination")
    kept, rejected = DuplicateFilter().apply([first, second])
    assert [i.id for i in kept] == ["q1"]
    assert "q1" in rejected[0].reason


def test_different_questions_are_not_duplicates() -> None:
    kept, _ = DuplicateFilter().apply(
        [item("q1", "How long is the notice period?"), item("q2", "Which header carries the key?")]
    )
    assert len(kept) == 2


def test_a_question_that_is_too_short_is_rejected() -> None:
    kept, rejected = ShortQuestionFilter().apply([item(question="Notice?")])
    assert not kept
    assert "too vague" in rejected[0].reason


def test_invented_evidence_is_rejected() -> None:
    """LLMs fabricate quotes. The anchor fails to resolve, which makes it detectable."""
    resolved = EvalItem(
        id="q1",
        question="How long is the notice period?",
        gold=(GoldSpan(Span("contract.md", 10, 40)),),
        anchors=(GoldAnchor(source_id="contract.md", quote="thirty days"),),
    )
    invented = EvalItem(
        id="q2",
        question="What is the termination fee?",
        anchors=(GoldAnchor(source_id="contract.md", quote="not in the corpus"),),
    )
    kept, rejected = UnresolvedEvidenceFilter().apply([resolved, invented])
    assert [i.id for i in kept] == ["q1"]
    assert "probably invented it" in rejected[0].reason


def test_the_filter_stands_down_before_resolution_has_run() -> None:
    """Every freshly generated item has a quote and no span. Running anyway would reject the
    entire eval set, because "unresolved" is indistinguishable from "not yet resolved"."""
    fresh = [item("q1"), item("q2")]
    kept, rejected = UnresolvedEvidenceFilter().apply(fresh)
    assert len(kept) == 2
    assert not rejected


def test_a_question_the_baseline_aces_separates_nothing() -> None:
    """The filter nothing else has. Twenty of these make a real difference between two
    configurations look like a rounding error."""
    kept, rejected = NonDiscriminatingFilter(baseline_scores={"q1": 1.0, "q2": 0.4}).apply(
        [item("q1"), item("q2")]
    )
    assert [i.id for i in kept] == ["q2"]
    assert "separates nothing" in rejected[0].reason


def test_without_a_baseline_the_discriminating_filter_does_nothing() -> None:
    kept, rejected = NonDiscriminatingFilter().apply([item("q1"), item("q2")])
    assert len(kept) == 2
    assert not rejected


def test_general_knowledge_is_rejected_when_a_model_can_answer_it() -> None:
    llm = RecordingLLM(replies=["Thirty days is the standard notice period."])
    kept, rejected = GeneralKnowledgeFilter(answerer=answerer_from(llm)).apply(
        [item(answer="Thirty days is the standard notice period.")]
    )
    assert not kept
    assert "measures the model rather than the retriever" in rejected[0].reason


def test_a_question_the_model_cannot_answer_blind_survives() -> None:
    llm = RecordingLLM(replies=["Ninety days, probably."])
    kept, _ = GeneralKnowledgeFilter(answerer=answerer_from(llm)).apply(
        [item(answer="Thirty days.")]
    )
    assert len(kept) == 1


def test_without_a_model_the_general_knowledge_filter_says_it_did_not_run() -> None:
    """Silently doing nothing would leave an eval set full of questions that score well for
    every configuration, with nothing to indicate why."""
    result = FilterChain([GeneralKnowledgeFilter()]).run(evalset(item()))
    assert any("no model to ask" in w.message for w in result.warnings)


def test_the_chain_reports_what_each_filter_dropped() -> None:
    result = default_filters().run(
        evalset(
            item("q1"),
            item("q2", "What does it cover?"),
            item("q3", "Notice?"),
        )
    )
    assert result.kept_count == 1
    assert set(result.by_filter()) == {"dangling-reference", "too-short"}
    assert "kept 1 of 3" in result.summary()


def test_heavy_rejection_is_flagged_as_a_generator_problem() -> None:
    result = default_filters().run(
        evalset(*[item(f"q{i}", "What does it cover?") for i in range(10)], item("ok"))
    )
    assert any("generator prompt needs work" in w.message for w in result.warnings)


def test_the_filtered_set_is_a_new_version_of_the_original() -> None:
    original = EvalSet(id="es", items=(item("q1"), item("q2", "Notice?")), version=2)
    result = default_filters().run(original)
    filtered = result.as_evalset(original)
    assert filtered.version == 3
    assert len(filtered) == 1


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the notice period?", QuestionType.FACTOID),
        ("How many days of notice are required?", QuestionType.NUMERIC),
        ("What is the Premium monthly fee?", QuestionType.TABULAR),
        ("Which is cheaper, Standard or Premium?", QuestionType.COMPARATIVE),
        ("Summarise the termination provisions.", QuestionType.SUMMARISATION),
        ("What notice applies in each of the two schedules?", QuestionType.MULTI_HOP),
    ],
)
def test_question_types_are_recognised(question: str, expected: str) -> None:
    assert classify_question(question) == expected


def test_a_question_with_no_evidence_is_unanswerable_by_type() -> None:
    assert classify_question("Anything?", has_evidence=False) == QuestionType.UNANSWERABLE


def test_a_label_set_by_a_human_is_not_overwritten() -> None:
    """A review-queue correction outranks anything guessed here."""
    labelled = item(qtype="my-own-label")
    assert Classifier().label(labelled).qtype == "my-own-label"
    assert Classifier(overwrite=True).label(labelled).qtype != "my-own-label"


def test_a_model_classifier_is_used_when_given_one() -> None:
    assert Classifier(model=lambda q: "tabular").label(item()).qtype == QuestionType.TABULAR


def test_a_nonsense_label_from_a_model_falls_back_to_the_heuristic() -> None:
    assert Classifier(model=lambda q: "banana").label(item()).qtype in QuestionType.ALL


# ---------------------------------------------------------------------------
# eval-set quality
# ---------------------------------------------------------------------------


def test_a_bigger_eval_set_detects_smaller_differences() -> None:
    assert minimum_detectable_difference(1000) < minimum_detectable_difference(50)


def test_a_tiny_eval_set_detects_nothing() -> None:
    assert minimum_detectable_difference(1) == 1.0


def test_quality_reports_the_smallest_difference_it_could_support() -> None:
    """ "You can detect 0.28" lands harder than "n is small"."""
    quality = assess(evalset(*[item(f"q{i}") for i in range(20)]))
    assert quality.answerable == 20
    assert 0.2 < quality.detectable_difference < 0.6
    assert not quality.can_support(0.05)
    assert quality.can_support(0.8)


def test_a_small_set_warns_that_small_differences_are_noise() -> None:
    warnings = assess(evalset(item())).warnings()
    assert any("is noise" in w.message for w in warnings)


def test_an_unreviewed_set_says_the_review_queue_is_the_cheapest_fix() -> None:
    warnings = assess(evalset(*[item(f"q{i}") for i in range(10)])).warnings()
    assert any("looked at by a human" in w.message for w in warnings)


def test_a_span_only_set_warns_that_the_parser_axis_is_unavailable() -> None:
    span_only = evalset(
        EvalItem(id="q1", question="How long is notice?", gold=(GoldSpan(Span("c", 0, 10)),))
    )
    warnings = assess(span_only).warnings()
    assert any("parser axis is not available" in w.message for w in warnings)


def test_a_portable_set_does_not_warn_about_the_parser_axis() -> None:
    warnings = assess(evalset(item())).warnings()
    assert not any("parser axis" in w.message for w in warnings)


def test_a_one_sided_set_says_results_will_only_describe_that_kind() -> None:
    lopsided = evalset(
        *[item(f"q{i}", f"What is the {i} monthly fee for the tier?") for i in range(12)]
    )
    quality = assess(Classifier().label_set(lopsided))
    assert any("and little else" in w.message for w in quality.warnings())


def test_discriminating_power_is_highest_in_the_middle() -> None:
    scores = {"q0": 1.0, "q1": 0.5, "q2": 0.0}
    quality = assess(evalset(item("q0"), item("q1"), item("q2")), baseline_scores=scores)
    assert quality.non_discriminating == 1
    assert quality.mean_discriminating_power == pytest.approx((0.0 + 1.0 + 0.0) / 3)


# ---------------------------------------------------------------------------
# the review queue
# ---------------------------------------------------------------------------


def queue_of(*items: EvalItem) -> ReviewQueue:
    return ReviewQueue.from_evalset(evalset(*items))


def test_accepting_advances_the_queue() -> None:
    queue = queue_of(item("q1"), item("q2"))
    assert queue.current is not None
    queue.accept()
    assert queue.current is not None
    assert queue.current.id == "q2"
    assert queue.remaining == 1


def test_a_finished_queue_says_so_rather_than_crashing() -> None:
    queue = queue_of(item("q1"))
    queue.accept()
    assert queue.is_done
    assert queue.current is None
    with pytest.raises(EvalSetError, match="nothing to decide"):
        queue.accept()


def test_undo_is_non_negotiable_at_five_seconds_an_item() -> None:
    """The reviewer will hit the wrong key, and without an undo they slow down to avoid it."""
    queue = queue_of(item("q1"), item("q2"))
    queue.reject("mistake")
    assert queue.counts()["rejected"] == 1

    back = queue.undo()
    assert back is not None
    assert back.id == "q1"
    assert queue.counts()["rejected"] == 0


def test_undo_on_an_untouched_queue_does_nothing() -> None:
    assert queue_of(item("q1")).undo() is None


def test_editing_the_quote_is_the_common_fix() -> None:
    """The generator picked a passage that half answers the question; moving it is a
    two-second fix that turns a rejection into a good question."""
    queue = queue_of(item("q1", quote="thirty days"))
    edited = queue.edit(quote="thirty days written notice")
    assert edited.anchors[0].quote == "thirty days written notice"


def test_editing_a_question_with_no_evidence_says_what_to_do() -> None:
    queue = queue_of(item("q1", quote=None))
    with pytest.raises(EvalSetError, match="Reject it, or add an anchor"):
        queue.edit(quote="something")


def test_marking_a_type_is_one_keystroke() -> None:
    queue = queue_of(item("q1"))
    assert queue.mark(QuestionType.TABULAR).qtype == QuestionType.TABULAR


def test_the_result_keeps_accepted_and_edited_and_drops_the_rest() -> None:
    original = evalset(item("q1"), item("q2"), item("q3"), item("q4"))
    queue = ReviewQueue.from_evalset(original)
    queue.accept()
    queue.reject("general knowledge")
    queue.edit(question="A better question about the notice period?")
    queue.skip()

    reviewed = queue.result(original)
    assert [i.id for i in reviewed] == ["q1", "q3"]
    assert all(i.meta["reviewed"] for i in reviewed)
    assert reviewed.version == original.version + 1


def test_stopping_halfway_loses_nothing() -> None:
    original = evalset(item("q1"), item("q2"), item("q3"))
    queue = ReviewQueue.from_evalset(original)
    queue.accept()
    assert len(queue.result(original)) == 3  # the untouched two are kept as they were


def test_already_reviewed_questions_are_skipped_by_default() -> None:
    original = evalset(item("q1", meta={"reviewed": True}), item("q2"))
    assert [i.id for i in ReviewQueue.from_evalset(original)] == ["q2"]
    assert len(ReviewQueue.from_evalset(original, skip_reviewed=False)) == 2


def test_rejections_are_kept_with_their_reasons() -> None:
    queue = queue_of(item("q1"), item("q2"))
    queue.reject("answerable without the corpus")
    queue.accept()
    assert [d.note for d in queue.rejections()] == ["answerable without the corpus"]


def test_the_session_summary_reports_the_pace() -> None:
    queue = queue_of(item("q1"), item("q2"))
    queue.accept()
    queue.reject()
    summary = review_summary(queue, elapsed_seconds=8.0)
    assert "2 reviewed" in summary
    assert "4.0s per question" in summary


def test_progress_reads_plainly() -> None:
    queue = queue_of(item("q1"), item("q2"))
    queue.accept()
    assert queue.progress() == "1 of 2 · 1 left"


def test_verdicts_are_named() -> None:
    assert {v.value for v in Verdict} == {
        "pending",
        "accepted",
        "rejected",
        "edited",
        "skipped",
    }


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def chunks(count: int = 4, doc: str = "contract.md") -> list[Chunk]:
    text = (
        "Either party may terminate this agreement for convenience by giving thirty days "
        "written notice to the address set out in Schedule A of this agreement."
    )
    return [
        Chunk(id=f"{doc}:{i}", span=Span(doc, i * 200, i * 200 + len(text)), text=text)
        for i in range(count)
    ]


def test_an_llm_generator_produces_questions_with_quoted_evidence() -> None:
    llm = RecordingLLM(
        replies=[
            json.dumps(
                [
                    {
                        "question": "How much notice is needed to terminate for convenience?",
                        "quote": "thirty days written notice",
                        "answer": "Thirty days.",
                    }
                ]
            )
        ]
    )
    items = LLMQuestionGenerator(llm=llm).generate(chunks(1)[0])
    assert len(items) == 1
    assert items[0].anchors[0].quote == "thirty days written notice"
    assert items[0].is_portable


def test_the_generator_prompt_demands_a_verbatim_quote() -> None:
    """The quote requirement is what makes an invented question detectable rather than
    merely suspected."""
    llm = RecordingLLM(replies=["[]"])
    LLMQuestionGenerator(llm=llm).generate(chunks(1)[0])
    assert "Copy it verbatim" in llm.prompts[0]


def test_an_unusable_model_reply_yields_nothing_rather_than_raising() -> None:
    assert (
        LLMQuestionGenerator(llm=RecordingLLM(replies=["sorry, no"])).generate(chunks(1)[0]) == []
    )


def test_records_without_a_quote_are_dropped() -> None:
    llm = RecordingLLM(replies=[json.dumps([{"question": "A question?"}])])
    assert LLMQuestionGenerator(llm=llm).generate(chunks(1)[0]) == []


def test_the_keyword_probe_generator_needs_no_model() -> None:
    generator = KeywordProbeGenerator()
    generator.fit(chunks())
    produced = generator.generate(chunks(1)[0])
    assert produced
    assert produced[0].meta["probe"] is True


def test_generation_samples_across_documents_rather_than_within_one() -> None:
    """A corpus of one long document and several short ones would otherwise produce an eval
    set almost entirely about the long one."""
    long_document = chunks(20, doc="long.md")
    short_documents = [chunks(1, doc=f"short{i}.md")[0] for i in range(3)]

    drafted = generate(
        [*long_document, *short_documents], KeywordProbeGenerator(), sample=4, seed=1
    )
    documents = {i.anchors[0].source_id for i in drafted.evalset}
    assert len(documents) > 1


def test_generation_says_the_draft_is_not_ground_truth_yet() -> None:
    drafted = generate(chunks(), KeywordProbeGenerator(), sample=2)
    assert any("not ground truth yet" in w.message for w in drafted.warnings)


def test_chunks_too_short_to_ask_about_are_skipped() -> None:
    tiny = [Chunk(id="c0", span=Span("d", 0, 5), text="short")]
    drafted = generate(tiny, KeywordProbeGenerator())
    assert drafted.count == 0
    assert drafted.chunks_skipped == 1
    assert not drafted.warnings.is_sound  # nothing to build an eval set from


def test_generation_is_deterministic_for_a_seed() -> None:
    first = generate(chunks(10), KeywordProbeGenerator(), sample=3, seed=7)
    second = generate(chunks(10), KeywordProbeGenerator(), sample=3, seed=7)
    assert [i.id for i in first.evalset] == [i.id for i in second.evalset]


# ---------------------------------------------------------------------------
# talking to a model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        '[{"question": "q", "quote": "x"}]',
        '```json\n[{"question": "q", "quote": "x"}]\n```',
        'Here is the JSON:\n[{"question": "q", "quote": "x"}]',
        '```\n[{"question": "q", "quote": "x"}]\n```',
    ],
)
def test_json_is_recovered_from_however_the_model_wrapped_it(reply: str) -> None:
    """Insisting on clean output would mean discarding usable replies."""
    assert parse_json_reply(reply) == [{"question": "q", "quote": "x"}]


def test_an_empty_reply_is_an_error() -> None:
    with pytest.raises(LLMError, match="returned nothing"):
        parse_json_reply("   ")


def test_a_reply_with_no_json_says_what_it_saw() -> None:
    with pytest.raises(LLMError, match="could not find JSON"):
        parse_json_reply("I would rather not.")


def test_the_closed_book_answerer_forbids_declining() -> None:
    """A model that says "I don't know" would let every question through and quietly
    disable the general-knowledge filter."""
    llm = RecordingLLM(replies=["Thirty days."])
    answerer_from(llm)("How long is notice?")
    assert "give your best guess" in llm.prompts[0]
