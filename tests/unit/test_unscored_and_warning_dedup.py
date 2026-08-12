"""Two ways a report told a user something that was not true.

1. A question with no ground truth at all was counted as `fp1_missing_content`, so a run that
   retrieved everything perfectly still told its user the parser, the chunker or the corpus had
   lost the evidence -- for a question nobody had written evidence for. The failure histogram
   also summed to the size of the eval set rather than to the number of questions scored.

2. Facts about the *eval set* are rediscovered by every configuration that resolves it, so a
   seven-way sweep printed the same two warnings six times each. Facts about one configuration
   must survive that collapse, which is why sameness includes `detail`.
"""

from __future__ import annotations

from contextgrid.core.documents import Chunk
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor, GoldSpan
from contextgrid.core.span import Span
from contextgrid.core.warnings import Severity, WarningCode, WarningLog
from contextgrid.diagnose.taxonomy import FailurePoint, cluster, diagnose
from contextgrid.pipeline import Config
from contextgrid.report.results import Results, RunResult
from contextgrid.score.resolve import SpanResolver

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _gold(item_id: str) -> EvalItem:
    """A question whose evidence has been located in this parse."""
    return EvalItem(
        id=item_id,
        question=f"Question {item_id}?",
        gold=(GoldSpan(span=Span("d", 0, 10)),),
    )


def _anchored(item_id: str) -> EvalItem:
    """A question that quotes its evidence but whose quote was never found in this parse."""
    return EvalItem(
        id=item_id,
        question=f"Question {item_id}?",
        anchors=(GoldAnchor(source_id="d", quote="a sentence this parse does not hold"),),
    )


def _no_ground_truth(item_id: str) -> EvalItem:
    """A question the eval set never finished: no gold spans, no anchors."""
    return EvalItem(id=item_id, question=f"Question {item_id}?")


def _evalset(*items: EvalItem) -> EvalSet:
    return EvalSet(id="es", items=tuple(items))


# ---------------------------------------------------------------------------
# BUG A -- the failure histogram
# ---------------------------------------------------------------------------


def test_a_question_with_no_ground_truth_is_not_a_retrieval_failure() -> None:
    """It never entered scoring, so it cannot have failed at retrieval."""
    report = diagnose(
        _evalset(_gold("q1"), _no_ground_truth("q2")),
        {"q1": {"c1": 2}},
        {"q1": ["c1"], "q2": ["c1"]},
        k=5,
    )

    assert [d.item_id for d in report.diagnoses] == ["q1"]
    assert report.no_ground_truth == ["q2"]
    assert report.counts() == {"none": 1}
    assert "fp1_missing_content" not in cluster(report)


def test_the_histogram_sums_to_the_questions_that_were_scored() -> None:
    """`failures` used to sum to the size of the eval set, which is a different number."""
    scored = 3
    report = diagnose(
        _evalset(*(_gold(f"q{i}") for i in range(scored)), _no_ground_truth("gap")),
        {f"q{i}": {"c1": 2} for i in range(scored)},
        {f"q{i}": ["c1"] for i in range(scored)} | {"gap": []},
        k=5,
    )

    assert sum(report.counts().values()) == scored
    assert report.total_items == scored + 1


def test_missing_evidence_with_an_anchor_is_still_fp1() -> None:
    """The genuine case must not be suppressed by the fix for the false one."""
    report = diagnose(_evalset(_anchored("q1")), {}, {"q1": ["c1", "c2"]}, k=5)

    diagnosis = report.diagnoses[0]
    assert diagnosis.failure is FailurePoint.MISSING_CONTENT
    assert report.no_ground_truth == []
    assert "could not be found in this parse" in diagnosis.detail
    assert "No retriever can fix this" in diagnosis.remedy


def test_gold_that_matches_no_chunk_is_still_fp1() -> None:
    """The other genuine case: evidence located in the parse, lost by the chunker."""
    report = diagnose(_evalset(_gold("q1")), {}, {"q1": ["c1"]}, k=5)

    assert report.diagnoses[0].failure is FailurePoint.MISSING_CONTENT
    assert "no chunk in this index holds" in report.diagnoses[0].detail


def test_both_reasons_at_once_are_counted_separately() -> None:
    """A run can hold a real FP1 and an unfinished question. They are not the same problem."""
    report = diagnose(
        _evalset(_gold("ok"), _anchored("lost"), _no_ground_truth("gap")),
        {"ok": {"c1": 2}},
        {"ok": ["c1"], "lost": ["c9"], "gap": []},
        k=5,
    )

    assert report.counts() == {"fp1_missing_content": 1, "none": 1}
    assert report.no_ground_truth == ["gap"]
    assert cluster(report) == {"fp1_missing_content": ["lost"]}


def test_the_taxonomy_summary_names_the_eval_set_gap() -> None:
    report = diagnose(
        _evalset(_gold("ok"), _no_ground_truth("gap")),
        {"ok": {"c1": 2}},
        {"ok": ["c1"], "gap": []},
        k=5,
    )
    summary = report.summary()

    assert "All 1 questions succeeded." in summary
    assert "gap" in summary
    assert "gap in the eval set, not a fault in this pipeline" in summary
    assert "no chunk in this index" not in summary


# ---------------------------------------------------------------------------
# BUG A -- the paragraph the user reads
# ---------------------------------------------------------------------------


def _results(*items: EvalItem, qrels: dict[str, dict[str, int]]) -> Results:
    evalset = _evalset(*items)
    run = {item.id: ["c1"] for item in items}
    report = diagnose(evalset, qrels, run, k=5)
    return Results(
        runs=[
            RunResult(
                config=Config(k=5),
                metrics={"recall@5": 1.0},
                scored_queries=len(qrels),
                failures=report,
                per_query=dict.fromkeys(qrels, 1.0),
            )
        ]
    )


def test_the_paragraph_does_not_blame_the_pipeline_for_a_missing_question() -> None:
    """The sentence this whole fix exists for."""
    results = _results(_gold("q1"), _no_ground_truth("q2"), qrels={"q1": {"c1": 2}})
    summary = results.summary("recall@5")

    assert "no chunk in this index held" not in summary
    assert "the other 1 was not scored (q2)" in summary
    assert "no gold spans and no anchors" in summary
    assert "gap in the eval set, not a fault in this pipeline" in summary
    assert "fp1_missing_content" not in summary


def test_the_paragraph_still_says_when_the_index_really_lost_the_evidence() -> None:
    results = _results(_gold("q1"), _anchored("q2"), qrels={"q1": {"c1": 2}})
    summary = results.summary("recall@5")

    assert "no chunk in this index held its evidence" in summary
    assert "no gold spans and no anchors" not in summary
    assert "fp1_missing_content" in summary


def test_the_paragraph_reports_the_split_when_both_happen() -> None:
    results = _results(
        _gold("q1"), _anchored("q2"), _no_ground_truth("q3"), qrels={"q1": {"c1": 2}}
    )
    summary = results.summary("recall@5")

    assert "the other 2 were not scored" in summary
    assert "1 because no chunk in this index held its evidence" in summary
    assert "1 (q3) because it has no ground truth at all" in summary
    # Said once, by the paragraph, not twice by the paragraph and the taxonomy.
    assert summary.count("gap in the eval set") == 1


# ---------------------------------------------------------------------------
# the same distinction, one stage earlier
# ---------------------------------------------------------------------------


def _resolve(item: EvalItem) -> WarningLog:
    chunks = [Chunk(id="c1", span=Span("d", 0, 20), text="x" * 20)]
    return SpanResolver().resolve_item(item, chunks).warnings


def test_a_question_with_no_evidence_at_all_is_reported_as_such() -> None:
    warning = next(iter(_resolve(_no_ground_truth("q1"))))

    assert warning.code is WarningCode.GOLD_SPAN_UNREACHABLE
    assert warning.severity is Severity.INFO
    assert "has no gold spans" in warning.message


def test_evidence_this_parse_lost_is_reported_as_a_parse_failure() -> None:
    """It stopped being reported at all when `is_answerable` was widened to cover anchors."""
    warning = next(iter(_resolve(_anchored("q1"))))

    assert warning.code is WarningCode.GOLD_SPAN_UNREACHABLE
    assert warning.severity is Severity.CAUTION
    assert "none of it was located in this parse" in warning.message
    assert "measurement of the parser, not of the retriever" in warning.message


# ---------------------------------------------------------------------------
# BUG B -- warning deduplication
# ---------------------------------------------------------------------------


def _log(**detail: object) -> WarningLog:
    log = WarningLog()
    log.add(
        WarningCode.ANCHOR_NORMALISED,
        "evidence found after collapsing whitespace",
        severity=Severity.INFO,
        stage="anchor",
        subject="nw13",
        **detail,
    )
    return log


def test_the_same_eval_set_fact_is_collected_once() -> None:
    """Seven configurations resolving one eval set found one thing, not seven."""
    report = WarningLog()
    for _ in range(7):
        report.extend_unique(_log())

    assert len(report) == 1
    assert report.counts() == {"anchor_normalised": 1}


def test_extend_unique_reports_what_it_dropped() -> None:
    report = WarningLog()

    assert report.extend_unique(_log()) == 0
    assert report.extend_unique(_log()) == 1


def test_a_per_configuration_warning_survives_deduplication() -> None:
    """Two parsers reflowing the same quote is two findings, not one repeated."""
    report = WarningLog()
    report.extend_unique(_log(parser="markdown"))
    report.extend_unique(_log(parser="text"))
    report.extend_unique(_log(parser="markdown"))  # the second markdown configuration

    assert len(report) == 2
    assert [w.detail["parser"] for w in report] == ["markdown", "text"]


def test_messages_that_differ_are_two_facts() -> None:
    report = WarningLog()
    first = WarningLog()
    first.add(WarningCode.GOLD_SPAN_UNREACHABLE, "item 'a' has no gold spans", subject="a")
    second = WarningLog()
    second.add(WarningCode.GOLD_SPAN_UNREACHABLE, "item 'b' has no gold spans", subject="b")

    report.extend_unique(first)
    report.extend_unique(second)

    assert len(report) == 2


def test_plain_extend_still_keeps_everything() -> None:
    """A run's own log is a record of what that run did, duplicates and all."""
    log = WarningLog()
    log.extend(_log())
    log.extend(_log())

    assert len(log) == 2
