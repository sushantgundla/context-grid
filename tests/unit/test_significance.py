"""Unit tests for confidence intervals, significance testing and failure diagnosis.

A significance test that says everything is significant is worse than no test at all, so the
important cases here are the negative ones: identical configurations must come back
indistinguishable, and small differences on small samples must not be dressed up as findings.
"""

from __future__ import annotations

import random

import pytest

from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.diagnose import FailurePoint, cluster, diagnose
from contextgrid.score.significance import (
    SignificanceError,
    _sample_size_note,
    bootstrap_interval,
    compare,
    paired_bootstrap,
    randomisation_test,
)


def scores(values: list[float], prefix: str = "q") -> dict[str, float]:
    return {f"{prefix}{i}": value for i, value in enumerate(values)}


# ---------------------------------------------------------------------------
# confidence intervals
# ---------------------------------------------------------------------------


def test_an_interval_brackets_its_own_estimate() -> None:
    interval = bootstrap_interval([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    assert interval.low <= interval.estimate <= interval.high


def test_more_questions_give_a_tighter_interval() -> None:
    """The whole reason eval-set size matters, stated as a test."""
    rng = random.Random(0)
    small = [rng.choice([0.0, 1.0]) for _ in range(20)]
    large = [rng.choice([0.0, 1.0]) for _ in range(2000)]
    assert bootstrap_interval(large).width < bootstrap_interval(small).width


def test_an_interval_is_deterministic_for_a_seed() -> None:
    values = [1.0, 0.0, 1.0, 0.5, 0.0]
    assert bootstrap_interval(values, seed=7) == bootstrap_interval(values, seed=7)


def test_identical_scores_give_a_zero_width_interval() -> None:
    interval = bootstrap_interval([0.5] * 30)
    assert interval.low == interval.high == 0.5
    assert interval.width == 0.0


def test_a_single_question_gives_no_range() -> None:
    interval = bootstrap_interval([1.0])
    assert interval.estimate == interval.low == interval.high == 1.0


def test_an_empty_set_of_scores_is_an_error() -> None:
    with pytest.raises(SignificanceError, match="no scores"):
        bootstrap_interval([])


def test_an_interval_knows_whether_it_crosses_zero() -> None:
    assert paired_bootstrap([1.0] * 20, [0.0] * 20).excludes_zero
    assert not paired_bootstrap([1.0, 0.0] * 10, [0.0, 1.0] * 10).excludes_zero


# ---------------------------------------------------------------------------
# the paired test
# ---------------------------------------------------------------------------


def test_identical_configurations_are_never_significant() -> None:
    """The failure mode that matters. A test that says everything differs is useless."""
    values = [1.0, 0.0, 1.0, 1.0, 0.0] * 20
    assert randomisation_test(values, values) == 1.0


def test_a_large_consistent_difference_is_detected() -> None:
    better = [1.0] * 50
    worse = [0.0] * 50
    assert randomisation_test(better, worse) < 0.01


def test_a_tiny_difference_on_a_small_sample_is_not_detected() -> None:
    """0.71 against 0.68 on 40 questions is not a finding, and the test must say so."""
    rng = random.Random(1)
    left = [rng.choice([0.0, 1.0]) for _ in range(40)]
    right = list(left)
    right[0] = 1.0 - right[0]  # one question different out of forty
    assert randomisation_test(left, right) > 0.05


def test_the_p_value_is_never_zero() -> None:
    """The observed arrangement is itself one of the possible ones. Reporting p=0 would
    claim a certainty that no resampling test can have."""
    assert randomisation_test([1.0] * 100, [0.0] * 100) > 0.0


def test_the_test_is_two_sided() -> None:
    left, right = [1.0] * 40, [0.0] * 40
    assert randomisation_test(left, right) == randomisation_test(right, left)


def test_mismatched_score_counts_are_refused() -> None:
    """Pairing is the whole point; unequal lists mean the pairing is already broken."""
    with pytest.raises(SignificanceError, match="one score per question"):
        randomisation_test([1.0, 0.0], [1.0])


# ---------------------------------------------------------------------------
# comparing two configurations
# ---------------------------------------------------------------------------


def test_a_clear_winner_is_reported_as_one() -> None:
    result = compare(
        scores([1.0] * 60), scores([0.0] * 60), left="good", right="bad", metric="recall@5"
    )
    assert result.distinguishable
    assert result.winner == "good"
    assert result.wins == 60
    assert "beats" in result.verdict()


def test_two_similar_configurations_are_not_distinguishable() -> None:
    rng = random.Random(3)
    left = [rng.choice([0.0, 1.0]) for _ in range(30)]
    right = list(left)
    right[0], right[1] = 1.0 - right[0], 1.0 - right[1]

    result = compare(scores(left), scores(right), left="a", right="b")
    assert not result.distinguishable
    assert result.winner is None
    assert "not distinguishable" in result.verdict()
    assert f"n={result.n}" in result.verdict()


def test_identical_configurations_say_they_are_behaving_the_same_way() -> None:
    """Not "too close to call" -- they scored the same on every question, which is a
    different and more useful statement."""
    values = scores([1.0, 0.0, 1.0] * 10)
    verdict = compare(values, values, left="a", right="b").verdict()
    assert "behaving the same way" in verdict


def test_configurations_that_tie_on_average_while_disagreeing_say_so() -> None:
    left = scores([1.0, 0.0] * 15)
    right = scores([0.0, 1.0] * 15)
    verdict = compare(left, right, left="a", right="b").verdict()
    assert "disagreeing on individual questions" in verdict


def test_a_small_gap_says_how_many_questions_would_settle_it() -> None:
    rng = random.Random(11)
    left = [rng.choice([0.0, 1.0]) for _ in range(400)]
    right = [0.0 if i < 220 else 1.0 for i in range(400)]
    result = compare(scores(left), scores(right), left="a", right="b")
    assert not result.distinguishable
    assert "would take roughly" in result.verdict()
    # Two significant figures, and the assumptions named beside the number.
    assert "alpha 0.05" in result.verdict()


@pytest.mark.parametrize(
    ("difference", "n", "ties", "expected"),
    [
        (0.0, 30, 30, "behaving the same way"),
        (0.0, 30, 0, "disagreeing on individual questions"),
        (0.02, 40, 10, "would take roughly"),
        (0.5, 400, 10, "defeated by how much the scores vary"),
    ],
)
def test_the_sample_size_note_covers_its_three_situations(
    difference: float, n: int, ties: int, expected: str
) -> None:
    """Three genuinely different things to say, tested directly.

    Stitching one phrase into a shared template produced sentences like "About many more
    than questions would be needed", which is the kind of thing that makes a reader stop
    trusting everything around it.
    """
    assert expected in _sample_size_note(difference, n, ties)


def test_only_questions_both_configurations_answered_are_compared() -> None:
    """Comparing on different question sets would confound the configurations with the
    questions, which is exactly what pairing exists to prevent."""
    result = compare({"q1": 1.0, "q2": 0.0, "q3": 1.0}, {"q1": 0.0, "q2": 1.0})
    assert result.n == 2


def test_configurations_with_no_questions_in_common_cannot_be_compared() -> None:
    with pytest.raises(SignificanceError, match="no questions in common"):
        compare({"q1": 1.0}, {"q2": 1.0})


def test_a_comparison_serialises() -> None:
    payload = compare(scores([1.0] * 30), scores([0.0] * 30)).as_dict()
    assert payload["distinguishable"] is True
    assert set(payload) >= {"p_value", "ci_low", "ci_high", "wins", "losses", "ties"}


def test_wins_losses_and_ties_add_up() -> None:
    result = compare(scores([1.0, 0.0, 0.5]), scores([0.0, 1.0, 0.5]))
    assert result.wins + result.losses + result.ties == result.n


# ---------------------------------------------------------------------------
# the failure taxonomy
# ---------------------------------------------------------------------------


def evalset_of(*ids: str) -> EvalSet:
    return EvalSet(
        id="es",
        items=tuple(
            EvalItem(
                id=item_id,
                question=f"Question {item_id}?",
                anchors=(GoldAnchor(source_id="d", quote="x"),),
            )
            for item_id in ids
        ),
    )


def test_evidence_in_the_top_k_is_a_success() -> None:
    report = diagnose(evalset_of("q1"), {"q1": {"c1": 2}}, {"q1": ["c1", "c2"]}, k=5)
    assert report.diagnoses[0].failure is FailurePoint.NONE
    assert report.diagnoses[0].gold_rank == 1


def test_evidence_just_outside_k_points_at_a_reranker() -> None:
    """The cheapest failure on the list to fix, and worth naming as such."""
    ranked = [f"c{i}" for i in range(20)] + ["gold"]
    report = diagnose(evalset_of("q1"), {"q1": {"gold": 2}}, {"q1": ranked}, k=5)
    diagnosis = report.diagnoses[0]
    assert diagnosis.failure is FailurePoint.MISSED_TOP_RANKED
    assert "reranker" in diagnosis.remedy


def test_evidence_far_down_the_ranking_is_a_different_problem() -> None:
    ranked = [f"c{i}" for i in range(150)] + ["gold"]
    report = diagnose(evalset_of("q1"), {"q1": {"gold": 2}}, {"q1": ranked}, k=5, deep_k=100)
    assert report.diagnoses[0].failure is FailurePoint.NOT_IN_CONTEXT


def test_evidence_that_is_in_no_chunk_is_not_a_retrieval_failure() -> None:
    report = diagnose(evalset_of("q1"), {}, {"q1": ["c1"]}, k=5)
    assert report.diagnoses[0].failure is FailurePoint.MISSING_CONTENT
    assert "No retriever can fix this" in report.diagnoses[0].remedy


def test_partial_evidence_is_reported_as_incomplete() -> None:
    """The answer needs passages no single chunk holds."""
    report = diagnose(evalset_of("q1"), {"q1": {"a": 2, "b": 2}}, {"q1": ["a", "c", "d"]}, k=3)
    assert report.diagnoses[0].failure is FailurePoint.INCOMPLETE


def test_the_report_names_the_dominant_failure_and_what_to_do() -> None:
    """Turns a score into an action. "0.62" tells you nothing."""
    ranked = [f"c{i}" for i in range(20)] + ["gold"]
    report = diagnose(
        evalset_of("q1", "q2", "q3"),
        {"q1": {"gold": 2}, "q2": {"gold": 2}, "q3": {"gold": 2}},
        {"q1": ranked, "q2": ranked, "q3": ["gold"]},
        k=5,
    )
    summary = report.summary()
    assert "2 of 3 questions failed" in summary
    assert "reranker" in summary


def test_the_report_says_what_a_retrieval_run_cannot_see() -> None:
    report = diagnose(evalset_of("q1"), {}, {"q1": []}, k=5)
    assert "failure points four to seven" in report.summary()


def test_a_perfect_run_says_so() -> None:
    report = diagnose(evalset_of("q1"), {"q1": {"c1": 2}}, {"q1": ["c1"]}, k=5)
    assert report.summary() == "All 1 questions succeeded."
    assert report.dominant is None


def test_failures_cluster_by_cause() -> None:
    """Twelve failures with one cause is one fix, not twelve investigations."""
    ranked = [f"c{i}" for i in range(20)] + ["gold"]
    report = diagnose(
        evalset_of("q1", "q2"),
        {"q1": {"gold": 2}, "q2": {}},
        {"q1": ranked, "q2": []},
        k=5,
    )
    grouped = cluster(report)
    assert grouped["fp2_missed_top_ranked"] == ["q1"]
    assert grouped["fp1_missing_content"] == ["q2"]


def test_counts_cover_every_question() -> None:
    report = diagnose(evalset_of("q1", "q2"), {"q1": {"c1": 2}}, {"q1": ["c1"], "q2": []}, k=5)
    assert sum(report.counts().values()) == 2
