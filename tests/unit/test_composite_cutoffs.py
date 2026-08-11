"""The composite, the cut-offs it resolves against, and the sentences the summary prints.

Two failures from one blind evaluation run, both of the same kind: a number on the screen
saying something nobody could defend. The first called a measured dimension unmeasured; the
second quoted a sample size to four significant figures from seventeen questions.
"""

from __future__ import annotations

import pytest

from contextgrid.diagnose.taxonomy import Diagnosis, FailurePoint, FailureReport
from contextgrid.pipeline import Config
from contextgrid.report.composite import composite
from contextgrid.report.results import Results, RunResult

# The blind run's own metrics: `headline: recall@1`, so the runner emitted every metric at
# `@1`, and `k: 3` in the config is the retrieval depth rather than the scoring cut-off.
BLIND_RUN = {
    "recall@1": 0.8824,
    "ndcg@1": 0.8824,
    "char_recall@1": 0.8824,
    "char_precision@1": 0.0231,
    "evidence_resolvable": 1.0,
    "embedding_quality": 0.71,
}


# ---------------------------------------------------------------------------
# the cut-off the run actually used
# ---------------------------------------------------------------------------


def test_a_dimension_measured_at_another_cutoff_is_not_reported_as_unmeasured() -> None:
    """The bug. `char_recall@1 = 0.8824` was measured, and the report said it was not.

    Scoring a dimension that did not run is one failure; saying a dimension that ran did not
    is the same failure pointing the other way, and it still prints a score.
    """
    result = composite(BLIND_RUN)

    assert "chunk" in result.parts
    assert "chunk" not in result.missing
    assert result.parts["chunk"] == pytest.approx(0.8824)
    assert "chunk" not in result.summary().partition("not measured")[2]


def test_the_cutoff_comes_from_the_run_rather_than_a_default_of_five() -> None:
    assert composite(BLIND_RUN).sources["retrieval"] == ("recall@1", "ndcg@1")
    assert composite(BLIND_RUN).cutoffs == (1,)


def test_an_explicit_cutoff_still_means_that_cutoff() -> None:
    """A caller asking about the top 5 is asking a question, not offering a hint. Answering
    it with a number measured over the top 3 would be a different answer to a different
    question."""
    assert "retrieval" in composite({"recall@3": 1.0}, k=5).missing
    assert composite({"recall@3": 1.0}, k=3).parts["retrieval"] == 1.0


def test_the_nearest_cutoff_wins_when_the_runs_own_is_absent() -> None:
    """`recall` at 1 and 5, `char_recall` only at 5: the dominant cut-off is 5, and chunk
    still has to be found even though it is the odd one out."""
    metrics = {"recall@5": 0.6, "ndcg@5": 0.6, "char_recall@2": 0.9, "char_recall@9": 0.4}
    result = composite(metrics)

    assert result.sources["chunk"] == ("char_recall@2",)  # |2-5| = 3 beats |9-5| = 4


def test_a_tie_between_cutoffs_resolves_the_same_way_every_time() -> None:
    metrics = {"recall@5": 0.6, "ndcg@5": 0.6, "char_recall@3": 0.9, "char_recall@7": 0.4}
    assert composite(metrics).sources["chunk"] == ("char_recall@3",)


def test_a_summary_says_so_when_the_cutoffs_disagree() -> None:
    """A score averaging character recall over two chunks with recall over five is still a
    fair thing to print -- but not without saying so."""
    summary = composite({"recall@5": 0.6, "ndcg@5": 0.6, "char_recall@2": 0.9}).summary()

    assert "cut-offs differ" in summary
    assert "chunk from char_recall@2" in summary
    assert "retrieval from recall@5, ndcg@5" in summary


def test_one_shared_cutoff_is_not_worth_a_sentence() -> None:
    assert "cut-offs differ" not in composite(BLIND_RUN).summary()


# ---------------------------------------------------------------------------
# what must not change: an absent metric is not a zero
# ---------------------------------------------------------------------------


def test_an_absent_metric_is_still_absent_and_not_a_measured_zero() -> None:
    """The distinction the whole module rests on. Falling back across cut-offs must never
    become falling back to 0.0, which the harmonic mean would read as a verdict."""
    result = composite({"recall@1": 0.9, "ndcg@1": 0.9})

    assert result.parts == {"retrieval": pytest.approx(0.9)}
    assert result.missing["generation"] == "no value for faithfulness or answer_relevancy"
    assert "chunk" not in result.sources


def test_a_measured_zero_is_still_a_measurement() -> None:
    result = composite({"recall@1": 0.0, "ndcg@1": 0.0, "char_recall@1": 0.5})

    assert result.parts["retrieval"] == 0.0
    assert "retrieval" not in result.missing
    assert result.score == 0.0


def test_a_value_off_the_scale_is_not_rescued_from_another_cutoff() -> None:
    """3.2 is not a score, and neither the exact cut-off nor the fallback may take it."""
    assert "retrieval" in composite({"recall@5": 3.2}).missing
    assert "retrieval" in composite({"recall@5": 3.2, "recall@1": -1.0}).missing


def test_a_run_with_no_cutoffs_anywhere_still_scores() -> None:
    result = composite({"evidence_resolvable": 1.0, "faithfulness": 0.8})

    assert result.dimensions == ("generation", "parse")
    assert result.cutoffs == ()


def test_the_sources_survive_serialisation() -> None:
    payload = composite(BLIND_RUN).as_dict()
    assert payload["sources"] == {
        "parse": ["evidence_resolvable"],
        "chunk": ["char_recall@1"],
        "embed": ["embedding_quality"],
        "retrieval": ["recall@1", "ndcg@1"],
    }


def test_the_run_object_takes_the_same_path() -> None:
    run = RunResult(config=Config(k=3), metrics=dict(BLIND_RUN))

    assert "chunk" in run.composite().parts
    assert "chunk" in run.composite(k=5).missing  # asked about the top 5; nothing measured it
    assert Results(runs=[run]).composite("recall@1") is not None


# ---------------------------------------------------------------------------
# the summary paragraph
# ---------------------------------------------------------------------------


def _one_run_results() -> Results:
    """One configuration, 17 scored questions out of an eval set of 20, 8 of them failing."""
    report = FailureReport(k=3)
    for index in range(20):
        failed = index < 8
        report.diagnoses.append(
            Diagnosis(
                f"q{index}",
                FailurePoint.NOT_IN_CONTEXT if failed else FailurePoint.NONE,
                "ranked outside k" if failed else "found",
            )
        )
    run = RunResult(
        config=Config(k=3),
        metrics=dict(BLIND_RUN),
        scored_queries=17,
        failures=report,
        per_query={f"q{index}": float(index % 2) for index in range(17)},
    )
    return Results(runs=[run])


def test_one_configuration_is_not_configurations() -> None:
    assert "across 1 configuration," in _one_run_results().summary("recall@1")
    assert "configurations" not in _one_run_results().summary("recall@1")


def test_the_two_question_counts_say_which_is_which() -> None:
    """17 and 20 both appeared in one paragraph with nothing distinguishing them."""
    summary = _one_run_results().summary("recall@1")

    assert "scored on 17 questions" in summary
    assert "eval set holds 20 questions in all" in summary
    assert "8 of 20 questions failed" in summary


def test_a_sample_size_estimate_is_rounded_and_its_assumptions_stated() -> None:
    """ "About 4532 questions would be needed" from n=17 is four significant figures of
    precision that seventeen questions cannot support."""
    from contextgrid.report.results import _honest_sample_size

    original = (
        "a and b are not distinguishable on this eval set (n=17). "
        "About 4532 questions would be needed to settle a gap this size. Then more."
    )
    rewritten = _honest_sample_size(original)

    assert "4532" not in rewritten
    assert "roughly 4,500 questions" in rewritten
    assert "alpha 0.05" in rewritten
    assert "80% power" in rewritten
    assert rewritten.endswith("Then more.")
    assert rewritten.startswith("a and b are not distinguishable on this eval set (n=17). ")


@pytest.mark.parametrize(
    ("needed", "expected"),
    [(4532, 4500), (20890, 21000), (80, 80), (175, 180), (7, 7), (999, 1000)],
)
def test_rounding_keeps_two_significant_figures(needed: int, expected: int) -> None:
    from contextgrid.report.results import _two_significant_figures

    assert _two_significant_figures(needed) == expected


def test_a_verdict_with_no_sample_size_claim_is_left_alone() -> None:
    from contextgrid.report.results import _honest_sample_size

    verdict = "a beats b by 0.200 on recall@1 (95% CI +0.100 to +0.300, p=0.010, n=17)."
    assert _honest_sample_size(verdict) == verdict
