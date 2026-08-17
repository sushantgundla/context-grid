"""Three arithmetic and diagnosis faults a stranger driving 0.9.2 found.

All three share a shape: the tool produced a confident number or sentence that a person
doing the same sum by hand would not have produced.

1. A chunk returned twice was counted twice, so `recall`, `map` and `ndcg` could rise above
   1.0 -- a scale that has no above-1.0 -- and a broken retriever scored better for being
   broken.
2. `diagnose()` put a question in FP1 "missing content" while its own `detail` said the
   evidence was in the index. `summary()` prints the remedy, not the detail, so the reader
   got the half that told them to go and fix a parser that had done nothing wrong.
3. "No relevant chunks here" written as `{"c1": 0}` and written as `{}` meant the same thing
   and scored differently -- one halved the mean, the other did not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from contextgrid.assemble.context import AssembledContext
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor, GoldSpan
from contextgrid.core.span import Span, coverage_fraction
from contextgrid.core.types import Chunk
from contextgrid.diagnose.taxonomy import FailurePoint, diagnose
from contextgrid.generate.answer import Answer, score_answer
from contextgrid.index.base import Scored
from contextgrid.index.quantize import recall_against_exact
from contextgrid.score.metrics import (
    BUILTIN_METRIC_NAMES,
    available_metrics,
    average_precision,
    evaluate,
    mean_rank_of_first_relevant,
    ndcg_at_k,
    per_query,
    precision_at_k,
    recall_at_k,
)

METRICS_PAGE = Path(__file__).resolve().parents[2] / "docs-site" / "scoring" / "metrics.mdx"

# ---------------------------------------------------------------------------
# 1. the same chunk returned three times is not three retrievals
# ---------------------------------------------------------------------------

#: Two relevant chunks, and a run that returns the first of them three times over. The top
#: three holds one distinct relevant chunk, so recall is 0.5 by hand.
DUPLICATE_JUDGEMENTS = {"c1": 2, "c2": 2}
DUPLICATE_RANKING = ["c1", "c1", "c1", "c2"]


def test_recall_counts_distinct_chunks_not_repeats() -> None:
    assert recall_at_k(DUPLICATE_JUDGEMENTS, DUPLICATE_RANKING, 3) == 0.5


def test_precision_does_not_reward_padding_the_top_k_with_repeats() -> None:
    """One distinct relevant chunk in three slots is a third of the window, not all of it."""
    assert precision_at_k(DUPLICATE_JUDGEMENTS, DUPLICATE_RANKING, 3) == pytest.approx(1 / 3)
    # and a repeat scores no better than an irrelevant chunk in the same slot
    assert precision_at_k({"c1": 2}, ["c1", "c1", "x"], 3) == precision_at_k(
        {"c1": 2}, ["c1", "y", "x"], 3
    )


def test_no_metric_can_exceed_one_when_a_chunk_repeats() -> None:
    for k in (1, 2, 3, 4, 10):
        assert 0.0 <= recall_at_k(DUPLICATE_JUDGEMENTS, DUPLICATE_RANKING, k) <= 1.0
        assert 0.0 <= precision_at_k(DUPLICATE_JUDGEMENTS, DUPLICATE_RANKING, k) <= 1.0
        assert 0.0 <= average_precision(DUPLICATE_JUDGEMENTS, DUPLICATE_RANKING, k) <= 1.0
        assert 0.0 <= ndcg_at_k(DUPLICATE_JUDGEMENTS, DUPLICATE_RANKING, k) <= 1.0


def test_average_precision_scores_a_repeated_hit_once() -> None:
    """c1 hits at rank 1 and never again. 1.0 of precision, over two relevant chunks."""
    assert average_precision(DUPLICATE_JUDGEMENTS, DUPLICATE_RANKING, 3) == 0.5


def test_ndcg_gives_a_repeat_no_gain() -> None:
    """A chunk already in the context teaches the generator nothing the second time."""
    assert ndcg_at_k(DUPLICATE_JUDGEMENTS, DUPLICATE_RANKING, 3) == pytest.approx(0.6131471927)


def test_evaluate_refuses_a_run_that_returns_the_same_chunk_twice() -> None:
    """Rejected, not quietly repaired: duplicate ids mean the retriever is broken."""
    with pytest.raises(ValueError) as caught:
        evaluate({"qa": DUPLICATE_JUDGEMENTS}, {"qa": DUPLICATE_RANKING}, ks=[3])
    message = str(caught.value)
    assert "qa" in message
    assert "c1" in message


def test_per_query_refuses_a_duplicated_ranking_too() -> None:
    with pytest.raises(ValueError):
        per_query({"qa": DUPLICATE_JUDGEMENTS}, {"qa": DUPLICATE_RANKING}, "recall", 3)


def test_duplicates_in_an_unscored_query_are_nobodys_business() -> None:
    """A query the qrels never judged is never averaged, so its ranking is not checked."""
    scores = evaluate({"qa": {"c9": 2}}, {"qa": ["c9"], "spare": ["c1", "c1"]}, ks=[3])
    assert scores["recall@3"] == 1.0


# ---------------------------------------------------------------------------
# 2. a diagnosis that argued with itself
# ---------------------------------------------------------------------------


def _one_item_set(item: EvalItem) -> EvalSet:
    return EvalSet(id="t", items=(item,))


GOLD_ITEM = EvalItem(id="d", question="d", gold=(GoldSpan(Span("doc1", 0, 50)),))


def test_evidence_in_the_index_but_never_returned_is_fp3_not_fp1() -> None:
    """The qrels name a chunk holding the evidence, so the parser and chunker are innocent."""
    run = {"d": [f"x{i}" for i in range(20)]}
    report = diagnose(_one_item_set(GOLD_ITEM), {"d": {"gold": 2}}, run, k=5, deep_k=100)
    diagnosis = report.diagnoses[0]

    assert diagnosis.failure is FailurePoint.NOT_IN_CONTEXT
    assert diagnosis.gold_rank is None
    assert diagnosis.retrieved == 20


def test_the_remedy_does_not_contradict_the_detail() -> None:
    """`summary()` prints the remedy, so the remedy is the half that has to be right."""
    run = {"d": [f"x{i}" for i in range(20)]}
    diagnosis = diagnose(
        _one_item_set(GOLD_ITEM), {"d": {"gold": 2}}, run, k=5, deep_k=100
    ).diagnoses[0]

    assert "index" in diagnosis.detail
    assert "the parser lost it" not in diagnosis.remedy
    assert "chunker" not in diagnosis.remedy


def test_a_run_that_returned_nothing_is_still_not_a_content_failure() -> None:
    report = diagnose(_one_item_set(GOLD_ITEM), {"d": {"gold": 2}}, {}, k=5, deep_k=100)
    assert report.diagnoses[0].failure is FailurePoint.NOT_IN_CONTEXT


#: The rest of the taxonomy, checked by hand against 0.9.2 and pinned here so this fix
#: cannot quietly move any of it.
TAXONOMY_CASES = [
    ("gold at rank 5 is inside k", GOLD_ITEM, {"d": {"g": 2}}, ["a", "b", "c", "e", "g"], "none"),
    (
        "gold at rank 100 is within deep_k",
        GOLD_ITEM,
        {"d": {"g": 2}},
        [f"x{i}" for i in range(99)] + ["g"],
        "fp2_missed_top_ranked",
    ),
    (
        "gold at rank 101 is past deep_k",
        GOLD_ITEM,
        {"d": {"g": 2}},
        [f"x{i}" for i in range(100)] + ["g"],
        "fp3_not_in_context",
    ),
    (
        "one of two relevant chunks made the top k",
        GOLD_ITEM,
        {"d": {"g1": 2, "g2": 2}},
        ["g1", "a", "b", "c", "e"],
        "fp7_incomplete",
    ),
    (
        "anchors that never resolved to a span",
        EvalItem(id="d", question="d", anchors=(GoldAnchor("doc1", "a quote nothing matched"),)),
        {"d": {}},
        ["a", "b"],
        "fp1_missing_content",
    ),
    ("gold that matched no chunk", GOLD_ITEM, {"d": {}}, ["a", "b"], "fp1_missing_content"),
]


@pytest.mark.parametrize(
    ("what", "item", "qrels", "ranking", "expected"),
    TAXONOMY_CASES,
    ids=[case[0] for case in TAXONOMY_CASES],
)
def test_the_rest_of_the_taxonomy_is_unchanged(
    what: str, item: EvalItem, qrels: dict, ranking: list[str], expected: str
) -> None:
    del what
    report = diagnose(_one_item_set(item), qrels, {"d": ranking}, k=5, deep_k=100)
    assert report.diagnoses[0].failure.value == expected


def test_the_two_fp1_root_causes_still_name_the_right_stage() -> None:
    anchored = EvalItem(id="d", question="d", anchors=(GoldAnchor("doc1", "unresolvable"),))
    parse_loss = diagnose(_one_item_set(anchored), {"d": {}}, {"d": []}).diagnoses[0]
    chunk_loss = diagnose(_one_item_set(GOLD_ITEM), {"d": {}}, {"d": []}).diagnoses[0]

    assert "parse" in parse_loss.detail
    assert "chunk" in chunk_loss.detail


# ---------------------------------------------------------------------------
# 3. two spellings of "nothing relevant here"
# ---------------------------------------------------------------------------

RUN_FOR_TWO = {"q1": ["x"], "q2": ["c9"]}


def test_a_grade_zero_only_question_scores_the_same_as_an_empty_one() -> None:
    grade_zero = evaluate(
        {"q1": {"c1": 0}, "q2": {"c9": 2}}, RUN_FOR_TWO, ks=[3], metrics=["recall"]
    )
    empty = evaluate({"q1": {}, "q2": {"c9": 2}}, RUN_FOR_TWO, ks=[3], metrics=["recall"])

    assert grade_zero == empty
    assert grade_zero["recall@3"] == 1.0


def test_per_query_leaves_out_a_question_with_nothing_relevant() -> None:
    assert per_query({"q1": {"c1": 0}, "q2": {"c9": 2}}, RUN_FOR_TWO, "recall", 3) == {"q2": 1.0}


def test_mean_rank_counts_only_questions_that_could_be_got_right() -> None:
    grade_zero = mean_rank_of_first_relevant({"q1": {"c1": 0}, "q2": {"c9": 2}}, RUN_FOR_TWO)
    empty = mean_rank_of_first_relevant({"q1": {}, "q2": {"c9": 2}}, RUN_FOR_TWO)

    assert grade_zero == empty == (1.0, 0)


def test_a_qrels_with_nothing_relevant_in_it_at_all_scores_nothing() -> None:
    assert evaluate({"q1": {"c1": 0}}, {"q1": ["c1"]}, ks=[3], metrics=["recall"]) == {}


def test_a_negative_cut_off_is_rejected_rather_than_sliced() -> None:
    """`ranked[:-1]` is a valid Python slice and a meaningless cut-off."""
    with pytest.raises(ValueError) as caught:
        evaluate({"q1": {"c1": 2}}, {"q1": ["c1"]}, ks=[-1], metrics=["recall"])
    assert "-1" in str(caught.value)


def test_a_zero_cut_off_is_rejected_too() -> None:
    with pytest.raises(ValueError):
        evaluate({"q1": {"c1": 2}}, {"q1": ["c1"]}, ks=[0], metrics=["recall"])
    with pytest.raises(ValueError):
        per_query({"q1": {"c1": 2}}, {"q1": ["c1"]}, "recall", 0)


# ---------------------------------------------------------------------------
# 4. public scoring names the metrics page never mentioned
# ---------------------------------------------------------------------------
#
# `coverage_fraction`, `recall_against_exact`, `score_answer` and `AnswerScore` are exported
# from `contextgrid` and listed in `/reference/api`, and each is documented in full on the
# page owning what it measures -- spans, indexes, generation. None of them was named on
# `/scoring/metrics`, which is where a reader looking for "what can this thing score" lands,
# so a reader who started there had a public name and no definition. These pin the signpost
# and the numbers the page now prints.


#: What the page prints for `cg.available_metrics()`, and what `evaluate()` scores by default.
SIX = ("hit_rate", "map", "mrr", "ndcg", "precision", "recall")


def test_the_six_built_ins_are_the_six_the_page_names() -> None:
    """`BUILTIN_METRIC_NAMES`, not `available_metrics()`.

    `available_metrics()` reads the live `METRICS` registry, so it *grows* the moment anything
    registers a custom metric -- which the page says, and which any test module registering one
    proves. The six that `evaluate()` defaults to are the fixed fact worth pinning.
    """
    assert sorted(BUILTIN_METRIC_NAMES) == list(SIX)
    assert set(SIX) <= set(available_metrics())


def test_the_metrics_page_lists_those_six_and_no_others() -> None:
    page = METRICS_PAGE.read_text(encoding="utf-8")
    table = page.split("## The six built-in metrics", 1)[1].split("<Warning>", 1)[0]
    listed = re.findall(r"^\| `([a-z_]+)` \|", table, flags=re.MULTILINE)
    assert sorted(listed) == list(SIX)


@pytest.mark.parametrize(
    "name", ["coverage_fraction", "recall_against_exact", "score_answer", "AnswerScore"]
)
def test_every_exported_scoring_name_is_reachable_from_the_metrics_page(name: str) -> None:
    assert name in METRICS_PAGE.read_text(encoding="utf-8")


GOLD_STRETCH = Span("refund.md", 100, 200)


def test_coverage_fraction_prints_what_the_page_says() -> None:
    first, second = Span("refund.md", 80, 150), Span("refund.md", 150, 190)
    assert coverage_fraction(GOLD_STRETCH, [first, second]) == 0.9
    assert coverage_fraction(GOLD_STRETCH, [first]) == 0.5
    # same offsets in a different document cover nothing
    assert coverage_fraction(GOLD_STRETCH, [Span("other.md", 100, 200)]) == 0.0


def test_recall_against_exact_prints_what_the_page_says() -> None:
    exact = [Scored("c1", 0.9), Scored("c2", 0.8), Scored("c3", 0.7)]
    approx = [Scored("c1", 0.9), Scored("c9", 0.75), Scored("c3", 0.7)]

    assert recall_against_exact(approx, exact, k=3) == pytest.approx(2 / 3)
    assert recall_against_exact(approx, exact, k=1) == 1.0
    # the trap the page warns about: nothing to compare against reads as perfect
    assert recall_against_exact(approx, [], k=3) == 1.0


#: The phrase `ExtractiveGenerator` emits when it has nothing to work from.
REFUSAL = "The passages do not contain the answer."

REFUND_TEXT = "Refunds are available within 30 days of purchase."
REFUND_ITEM = EvalItem(
    id="q1", question="What is the refund window?", gold=(GoldSpan(Span("refund.md", 0, 46)),)
)
REFUND_CHUNK = Chunk(id="g1", span=Span("refund.md", 0, 46), text=REFUND_TEXT)
REFUND_CONTEXT = AssembledContext(text=REFUND_TEXT, chunks=(REFUND_CHUNK,), tokens=12)


def test_score_answer_prints_what_the_page_says_for_a_good_answer() -> None:
    good = Answer(text="Refunds are available within 30 days.", citations=(1,))
    score = score_answer(REFUND_ITEM, good, REFUND_CONTEXT, gold_chunks=[REFUND_CHUNK])

    assert score.groundedness == 1.0
    assert score.citation_accuracy == 1.0
    assert score.evidence_overlap == 0.75
    assert score.warnings == []
    assert score.abstention_correct


def test_score_answer_prints_what_the_page_says_for_an_invented_one() -> None:
    invented = Answer(text="Contact the vendor by fax immediately.", citations=(1, 4))
    score = score_answer(REFUND_ITEM, invented, REFUND_CONTEXT, gold_chunks=[REFUND_CHUNK])

    assert score.groundedness == 0.0
    assert score.citation_accuracy == 0.5
    assert len(score.warnings) == 2
    assert "cited passage(s) that were not in the context: [4]" in score.warnings


def test_declining_when_there_was_nothing_to_answer_from_is_a_success() -> None:
    empty = AssembledContext(text="", chunks=(), tokens=0)
    declined = score_answer(REFUND_ITEM, Answer(text=REFUSAL), empty)

    assert declined.abstained
    assert declined.should_have_abstained
    assert declined.abstention_correct
    assert declined.citation_accuracy is None
