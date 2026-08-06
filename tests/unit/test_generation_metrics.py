"""Generation metrics through DeepEval, and the 0-100 composite.

Every dimension before generation is scored on whether the right passages came back. This one
asks the question they were retrieved *for*, and it fails differently: a configuration can
retrieve perfectly and generate a confident falsehood.

No API key is used. `ScriptedJudge` answers DeepEval's prompts in the shape they ask for, so the
metrics really run -- real DeepEval code, real scores -- with nothing on the network.
"""

from __future__ import annotations

import pytest

from contextgrid.report.composite import (
    DIMENSION_METRICS,
    CompositeScore,
    composite,
    harmonic_mean,
)

deepeval = pytest.importorskip("deepeval")

from contextgrid.generate.judge import (  # noqa: E402
    METRICS,
    GenerationJudge,
    JudgeError,
    available_generation_metrics,
    build_judge,
)


class ScriptedJudge:
    """Answers every DeepEval prompt with one reply carrying all the keys it might ask for."""

    name = "scripted"

    def __init__(self, verdict: str = "yes") -> None:
        self.verdict = verdict
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return (
            '{"truths": ["Refunds are issued within 30 days."],'
            ' "claims": ["Refunds take 30 days."],'
            ' "statements": ["Refunds take 30 days."],'
            f' "verdicts": [{{"verdict": "{self.verdict}", "reason": "because"}}],'
            ' "reason": "a stated reason"}'
        )


class BrokenJudge:
    name = "broken"

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        return "I am not going to answer that."


def score_one(judge: GenerationJudge, **overrides: object) -> object:
    kwargs: dict[str, object] = {
        "query_id": "q1",
        "question": "How long do refunds take?",
        "answer": "Refunds take 30 days.",
        "contexts": ["Refunds are issued within 30 days of purchase."],
    }
    kwargs.update(overrides)
    return judge.score(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# the judge runs, and it is ours
# ---------------------------------------------------------------------------


def test_deepeval_metrics_really_run_and_return_scores() -> None:
    """Real DeepEval code paths, with the model replaced rather than the metric."""
    result = score_one(GenerationJudge(llm=ScriptedJudge(), metrics=("faithfulness",)))

    assert result.scores["faithfulness"] == 1.0  # type: ignore[attr-defined]
    assert result.reasons["faithfulness"]  # type: ignore[attr-defined]
    assert not result.failed  # type: ignore[attr-defined]


def test_an_unsupported_answer_scores_lower_than_a_supported_one() -> None:
    """The metric has to actually discriminate, or it is an expensive constant."""
    supported = score_one(GenerationJudge(llm=ScriptedJudge("yes"), metrics=("faithfulness",)))
    invented = score_one(GenerationJudge(llm=ScriptedJudge("no"), metrics=("faithfulness",)))

    assert invented.scores["faithfulness"] < supported.scores["faithfulness"]  # type: ignore[attr-defined]


def test_the_judge_is_the_model_the_config_chose() -> None:
    """DeepEval reaches for its own OpenAI configuration by default, which would put a second,
    unpriced model call in a package whose whole argument is that cost belongs on the chart."""
    llm = ScriptedJudge()
    score_one(GenerationJudge(llm=llm, metrics=("faithfulness",)))
    assert llm.calls > 0


def test_the_judges_calls_are_counted() -> None:
    """A judge grading a thousand answers is a real expense, and one this package refuses to
    leave off the chart."""
    result = score_one(GenerationJudge(llm=ScriptedJudge(), metrics=("faithfulness",)))
    assert result.model_calls > 0  # type: ignore[attr-defined]


def test_the_judge_reports_the_model_it_wraps() -> None:
    judge = build_judge(ScriptedJudge())
    assert judge.get_model_name() == "scripted"
    assert judge.load_model() is not None


# ---------------------------------------------------------------------------
# when the judge lets you down
# ---------------------------------------------------------------------------


def test_a_judge_that_returns_prose_is_recorded_not_raised() -> None:
    """A judge model refusing one awkward question must not discard the nine hundred answers it
    graded perfectly well."""
    result = score_one(GenerationJudge(llm=BrokenJudge(), metrics=("faithfulness",)))

    assert not result.scores  # type: ignore[attr-defined]
    assert "faithfulness" in result.failed  # type: ignore[attr-defined]


def test_one_metric_failing_does_not_stop_the_others() -> None:
    judge = GenerationJudge(llm=ScriptedJudge(), metrics=("faithfulness", "contextual_recall"))
    result = score_one(judge, reference=None)

    assert "faithfulness" in result.scores  # type: ignore[attr-defined]
    assert "contextual_recall" in result.failed  # type: ignore[attr-defined]


def test_a_metric_needing_a_reference_says_so_rather_than_scoring_zero() -> None:
    """Zero would read as "the context contained nothing useful", which is a claim about the
    retriever. The truth is that nobody wrote an answer to compare against."""
    result = score_one(
        GenerationJudge(llm=ScriptedJudge(), metrics=("contextual_recall",)), reference=None
    )
    assert "reference" in result.failed["contextual_recall"]  # type: ignore[attr-defined]


def test_an_unknown_metric_lists_the_real_ones() -> None:
    with pytest.raises(JudgeError, match="Available: answer_relevancy"):
        GenerationJudge(llm=ScriptedJudge(), metrics=("faithfulness", "vibes"))


def test_the_four_metrics_fail_differently() -> None:
    """Not all fifty DeepEval offers. These four catch a wrong answer, an unsupported answer, an
    evasive answer, and a retrieval problem wearing a generation problem's clothes."""
    assert set(available_generation_metrics()) == {
        "answer_relevancy",
        "contextual_recall",
        "contextual_relevancy",
        "faithfulness",
    }
    assert set(METRICS) == set(available_generation_metrics())


# ---------------------------------------------------------------------------
# the composite
# ---------------------------------------------------------------------------


def test_a_weak_link_cannot_hide_behind_strong_ones() -> None:
    """The whole reason for a harmonic mean. A system retrieving at 0.95 and generating
    faithfully at 0.10 is not middling -- it confidently invents answers -- and an arithmetic
    mean of 0.53 says middling."""
    metrics = {"recall@5": 0.95, "ndcg@5": 0.95, "faithfulness": 0.10, "answer_relevancy": 0.10}
    result = composite(metrics)

    arithmetic = 100 * sum(result.parts.values()) / len(result.parts)
    assert result.score < arithmetic / 2
    assert result.score < 20


def test_a_zero_anywhere_is_a_zero_overall() -> None:
    """A system that generates nothing faithful has no score worth reporting, however well it
    retrieves."""
    assert composite({"recall@5": 1.0, "faithfulness": 0.0}).score == 0.0


def test_only_the_dimensions_that_ran_are_scored() -> None:
    """Someone sweeping ingestion and retrieval has no generator, and scoring the missing
    dimension as zero would punish them for a question they never asked."""
    result = composite({"recall@5": 0.8, "ndcg@5": 0.7})

    assert result.dimensions == ("retrieval",)
    assert "generation" in result.missing
    assert result.score == pytest.approx(75.0)


def test_the_score_never_appears_without_what_it_covers() -> None:
    """A 73 over two dimensions is a different claim from a 73 over four, and printing them
    identically invites exactly the comparison that is wrong."""
    summary = composite({"recall@5": 0.8, "faithfulness": 0.9}).summary()

    assert "dimension(s)" in summary
    assert "retrieval" in summary
    assert "generation" in summary
    assert "not measured" in summary


def test_nothing_measurable_says_so_rather_than_scoring_zero() -> None:
    result = composite({"latency_ms": 12.0})
    assert result.score == 0.0
    assert "no score" in result.summary()


def test_a_cutoff_is_found_without_being_spelled_out() -> None:
    assert composite({"recall@5": 1.0}, k=5).parts["retrieval"] == 1.0
    assert composite({"recall@3": 1.0}, k=3).parts["retrieval"] == 1.0
    assert "retrieval" in composite({"recall@3": 1.0}, k=5).missing


def test_a_value_off_the_zero_to_one_scale_is_ignored_not_clamped() -> None:
    """A composite compares like-scaled things. Squashing 3.2 to 1.0 would put a number in the
    score that nothing measured."""
    assert "retrieval" in composite({"recall@5": 3.2}).missing
    assert "retrieval" in composite({"recall@5": -1.0}).missing


def test_two_metrics_for_one_dimension_average_rather_than_compound() -> None:
    """They are two views of the same thing, not two links in a chain. Compounding them would
    give retrieval two votes and generation one."""
    result = composite({"recall@5": 1.0, "ndcg@5": 0.0})
    assert result.parts["retrieval"] == pytest.approx(0.5)


def test_every_dimension_has_exactly_one_way_in() -> None:
    """Averaging four retrieval metrics and then averaging that against generation gives
    retrieval four votes -- an opinion about what matters dressed as arithmetic."""
    assert set(DIMENSION_METRICS) == {"parse", "chunk", "embed", "retrieval", "generation"}
    for names in DIMENSION_METRICS.values():
        assert names


def test_the_harmonic_mean_of_nothing_is_zero_not_a_crash() -> None:
    assert harmonic_mean({}) == 0.0


def test_the_score_is_serialisable() -> None:
    """It goes into the report bundle, so it has to survive being written down."""
    import json

    payload = composite({"recall@5": 0.8, "faithfulness": 0.6}).as_dict()
    assert json.loads(json.dumps(payload))["score"] > 0


def test_a_perfect_run_scores_one_hundred() -> None:
    """Every metric name here must be one the runner really emits.

    This test used to say `character_precision`, which is what `DIMENSION_METRICS` asked for
    and what nothing has ever produced -- so it agreed with the typo instead of with the
    runner, and passed while the chunk dimension was unscoreable in every real run.
    """
    result = composite(
        {
            "recall@5": 1.0,
            "ndcg@5": 1.0,
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
            "char_precision@5": 1.0,
            "evidence_resolvable": 1.0,
            "embedding_quality": 1.0,
        }
    )
    assert result.score == pytest.approx(100.0)
    assert not result.missing


def test_a_composite_is_a_dataclass_worth_inspecting() -> None:
    result = CompositeScore(score=50.0, parts={"retrieval": 0.5})
    assert result.dimensions == ("retrieval",)
