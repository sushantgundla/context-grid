"""Unit tests for documents, chunks, gold spans and eval sets."""

from __future__ import annotations

import pytest

from contextgrid import (
    Chunk,
    Document,
    DocumentError,
    EvalItem,
    EvalSet,
    EvalSetError,
    GoldSpan,
    RetrievedChunk,
    Span,
)
from contextgrid.core.types import chunks_of, spans_of

TEXT = "The notice period is thirty days. Either party may terminate for convenience."
DOC = Document(id="contract", text=TEXT, source="contract.pdf")


def span(start: int, end: int, doc: str = "contract") -> Span:
    return Span(doc, start, end)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def test_document_span_covers_everything() -> None:
    assert DOC.span() == span(0, len(TEXT))
    assert DOC.length == len(TEXT)


def test_slice_returns_the_referenced_text() -> None:
    assert DOC.slice(span(4, 17)) == "notice period"


def test_slice_rejects_a_span_from_another_document() -> None:
    with pytest.raises(DocumentError, match="belongs to document"):
        DOC.slice(span(0, 5, "other"))


def test_slice_rejects_a_span_past_the_end() -> None:
    # Silently returning a short string here would let truncated text be scored as evidence.
    with pytest.raises(DocumentError, match="runs past the end"):
        DOC.slice(span(0, len(TEXT) + 10))


def test_contains_span() -> None:
    assert DOC.contains_span(span(0, len(TEXT)))
    assert not DOC.contains_span(span(0, len(TEXT) + 1))
    assert not DOC.contains_span(span(0, 5, "other"))


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


def test_chunk_exposes_its_source_position() -> None:
    chunk = Chunk(id="c0", span=span(4, 17), text="notice period")
    assert chunk.doc_id == "contract"
    assert chunk.char_start == 4
    assert chunk.char_end == 17


def test_offset_exact_chunk_matches_its_source() -> None:
    chunk = Chunk(id="c0", span=span(0, 32), text=TEXT[0:32])
    assert chunk.matches_source(DOC)


def test_mismatched_chunk_is_caught() -> None:
    chunk = Chunk(id="c0", span=span(0, 32), text="something else entirely")
    assert not chunk.matches_source(DOC)


def test_chunk_running_past_the_document_does_not_match() -> None:
    chunk = Chunk(id="c0", span=span(0, len(TEXT) + 5), text=TEXT)
    assert not chunk.matches_source(DOC)


def test_token_counts_are_per_tokenizer() -> None:
    """Chunk size is meaningless without naming the tokenizer that measured it."""
    chunk = Chunk(
        id="c0",
        span=span(0, 32),
        text=TEXT[0:32],
        token_counts={"cl100k_base": 7, "bert-base-uncased": 9},
    )
    assert chunk.token_count("cl100k_base") == 7
    assert chunk.token_count("bert-base-uncased") == 9
    assert chunk.token_count("never-measured") is None


def test_offsets_exact_defaults_to_true_and_can_be_denied() -> None:
    assert Chunk(id="c0", span=span(0, 5), text=TEXT[:5]).offsets_exact
    rewritten = Chunk(id="c1", span=span(0, 5), text="an LLM wrote this", offsets_exact=False)
    assert not rewritten.offsets_exact


# ---------------------------------------------------------------------------
# GoldSpan
# ---------------------------------------------------------------------------


def test_gold_span_defaults_to_fully_answering() -> None:
    assert GoldSpan(span(4, 17)).grade == 2


def test_gold_span_must_cover_at_least_one_character() -> None:
    with pytest.raises(EvalSetError, match="at least one character"):
        GoldSpan(span(5, 5))


def test_gold_grade_cannot_be_negative() -> None:
    with pytest.raises(EvalSetError, match="grade must be >= 0"):
        GoldSpan(span(0, 5), grade=-1)


def test_gold_span_round_trips_through_dict() -> None:
    gold = GoldSpan(span(4, 17), grade=1)
    assert GoldSpan.from_dict(gold.to_dict()) == gold


# ---------------------------------------------------------------------------
# EvalItem
# ---------------------------------------------------------------------------


def test_eval_item_rejects_an_empty_question() -> None:
    with pytest.raises(EvalSetError, match="empty question"):
        EvalItem(id="q1", question="   ")


def test_item_with_gold_is_answerable() -> None:
    item = EvalItem(id="q1", question="How long is notice?", gold=(GoldSpan(span(4, 32)),))
    assert item.is_answerable


def test_item_without_gold_is_deliberately_unanswerable() -> None:
    item = EvalItem(id="q1", question="What is the CEO's shoe size?")
    assert not item.is_answerable
    assert item.gold_length == 0


def test_gold_length_counts_overlapping_gold_once() -> None:
    item = EvalItem(
        id="q1",
        question="How long is notice?",
        gold=(GoldSpan(span(0, 30)), GoldSpan(span(20, 40))),
    )
    assert item.gold_length == 40


def test_gold_documents() -> None:
    item = EvalItem(
        id="q1",
        question="q",
        gold=(GoldSpan(span(0, 10)), GoldSpan(span(0, 10, "annex"))),
    )
    assert item.gold_documents() == {"contract", "annex"}


def test_eval_item_round_trips_through_dict() -> None:
    item = EvalItem(
        id="q1",
        question="How long is the notice period?",
        gold=(GoldSpan(span(4, 32), grade=2),),
        qtype="factoid",
        answer="Thirty days.",
        meta={"reviewed": True},
    )
    assert EvalItem.from_dict(item.to_dict()) == item


# ---------------------------------------------------------------------------
# EvalSet
# ---------------------------------------------------------------------------


def make_set() -> EvalSet:
    return EvalSet(
        id="es1",
        items=(
            EvalItem(id="q1", question="a", qtype="factoid", gold=(GoldSpan(span(0, 5)),)),
            EvalItem(id="q2", question="b", qtype="tabular", gold=(GoldSpan(span(5, 9)),)),
            EvalItem(id="q3", question="c", qtype="factoid"),
        ),
    )


def test_duplicate_item_ids_are_rejected() -> None:
    with pytest.raises(EvalSetError, match="duplicate eval item id"):
        EvalSet(
            id="es1",
            items=(EvalItem(id="q1", question="a"), EvalItem(id="q1", question="b")),
        )


def test_eval_set_slicing() -> None:
    es = make_set()
    assert len(es) == 3
    assert len(es.answerable) == 2
    assert [i.id for i in es.by_type("factoid")] == ["q1", "q3"]
    assert es.types() == {"factoid", "tabular"}


def test_eval_set_lookup() -> None:
    es = make_set()
    found = es.get("q2")
    assert found is not None
    assert found.question == "b"
    assert es.get("nope") is None


def test_eval_set_is_iterable() -> None:
    assert [item.id for item in make_set()] == ["q1", "q2", "q3"]


# ---------------------------------------------------------------------------
# RetrievedChunk
# ---------------------------------------------------------------------------


def test_rank_movement_from_reranking() -> None:
    chunk = Chunk(id="c0", span=span(0, 5), text=TEXT[:5])
    moved_up = RetrievedChunk(chunk=chunk, score=0.9, rank=1, rank_before_rerank=7)
    assert moved_up.moved == 6

    moved_down = RetrievedChunk(chunk=chunk, score=0.4, rank=9, rank_before_rerank=2)
    assert moved_down.moved == -7


def test_rank_movement_is_unknown_without_reranking() -> None:
    chunk = Chunk(id="c0", span=span(0, 5), text=TEXT[:5])
    assert RetrievedChunk(chunk=chunk, score=0.9, rank=1).moved is None


def test_retrieved_chunk_borrows_its_chunk_id() -> None:
    chunk = Chunk(id="c0", span=span(0, 5), text=TEXT[:5])
    assert RetrievedChunk(chunk=chunk, score=0.9, rank=1).id == "c0"


def test_unwrapping_helpers() -> None:
    chunks = [
        Chunk(id="c0", span=span(0, 5), text=TEXT[:5]),
        Chunk(id="c1", span=span(5, 9), text=TEXT[5:9]),
    ]
    retrieved = [RetrievedChunk(chunk=c, score=1.0, rank=i) for i, c in enumerate(chunks)]
    assert chunks_of(retrieved) == chunks
    assert spans_of(chunks) == [span(0, 5), span(5, 9)]
