"""The `generator` axis: wiring retrieve -> assemble -> generate -> score into the grid.

No model is called for real anywhere in this file. `RecordingLLM` stands in for the model that
answers questions and, in the DeepEval tests, for the judge as well -- the same trick
`tests/unit/test_transforms.py` and `tests/unit/test_generation_metrics.py` already use, so this
runs in CI with no key.

What matters here is not that generation works (`tests/unit/test_generation.py` and
`test_generation_metrics.py` already cover the library layer in isolation) but that it is
actually *reachable* from a `Config`, a `Matrix` and a `Runner` the way every other axis is, and
that `generator=None` -- the default -- costs nothing at all, exactly as it did before this axis
existed.
"""

from __future__ import annotations

import pytest

from contextgrid.core.documents import MediaType
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.core.warnings import WarningCode
from contextgrid.corpus import Corpus
from contextgrid.evalset.llm import LLMError, RecordingLLM
from contextgrid.generate import (
    ExtractiveGenerator,
    LLMGenerator,
    available_generators,
    get_generator,
)
from contextgrid.grid import matrix
from contextgrid.grid.matrix import AXIS_ORDER, Matrix
from contextgrid.grid.runner import Budget, Runner, _warn_if_unbounded
from contextgrid.pipeline import Config, build
from tests.support import API_DOCS, CONTRACT

QUESTION = "How much notice is needed to terminate for convenience?"

#: A judge reply that satisfies every DeepEval prompt shape a faithfulness or answer-relevancy
#: metric might ask for, the same blob `test_generation_metrics.py`'s `ScriptedJudge` returns.
JUDGE_JSON = (
    '{"truths": ["Thirty days notice is required."], '
    '"claims": ["Thirty days notice is required."], '
    '"statements": ["Thirty days notice is required."], '
    '"verdicts": [{"verdict": "yes", "reason": "supported"}], "reason": "ok"}'
)


@pytest.fixture
def corpus() -> Corpus:
    return Corpus.from_texts(
        {"contract.md": CONTRACT, "api.md": API_DOCS}, media_type=MediaType.MARKDOWN
    )


@pytest.fixture
def evalset() -> EvalSet:
    return EvalSet(
        id="es",
        items=(
            EvalItem(
                id="q1",
                question=QUESTION,
                anchors=(GoldAnchor(source_id="contract.md", quote="thirty days"),),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Config: a real field, last, meaning nothing changes for callers who never set it
# ---------------------------------------------------------------------------


def test_generator_defaults_to_none() -> None:
    assert Config().generator is None


def test_generator_is_the_last_field() -> None:
    """`Config("markdown", "recursive:512", "tfidf", "dense")` is public API. A new field ahead
    of an existing one would silently shift every positional argument anybody has written."""
    assert Config("markdown", "recursive:512", "tfidf", "dense").generator is None


def test_the_label_only_shows_a_generator_when_one_is_set() -> None:
    assert "->" not in Config().label
    assert Config(generator="extractive").label.endswith("->extractive")


def test_as_dict_carries_the_generator() -> None:
    assert Config(generator="extractive").as_dict()["generator"] == "extractive"
    assert Config().as_dict()["generator"] is None


# ---------------------------------------------------------------------------
# get_generator: the same "model-backed plugins are not in the registry" rule as transforms
# ---------------------------------------------------------------------------


def test_none_means_no_generation_at_all() -> None:
    """Unlike every other axis, `None` here has nothing to be the identity of."""
    assert get_generator(None) is None


def test_extractive_needs_no_model() -> None:
    assert isinstance(get_generator("extractive"), ExtractiveGenerator)


def test_llm_generator_needs_a_model() -> None:
    with pytest.raises(LLMError, match="needs a model"):
        get_generator("llm")


def test_llm_generator_is_reachable_with_a_model() -> None:
    llm = RecordingLLM(default="an answer")
    generator = get_generator("llm", llm)
    assert isinstance(generator, LLMGenerator)
    assert generator.llm is llm


def test_an_unknown_generator_names_the_real_ones() -> None:
    with pytest.raises(LLMError, match="Available: extractive, llm"):
        get_generator("nonsense", RecordingLLM())


def test_available_generators_lists_both_kinds() -> None:
    assert available_generators() == ("extractive", "llm")


# ---------------------------------------------------------------------------
# the matrix
# ---------------------------------------------------------------------------


def test_generator_is_a_real_axis() -> None:
    assert "generator" in AXIS_ORDER
    assert AXIS_ORDER[-1] == "generator"


def test_a_single_value_does_not_need_wrapping() -> None:
    assert matrix(generator="extractive").generator == ("extractive",)


def test_generator_multiplies_the_matrix_like_any_other_axis() -> None:
    configs = matrix(chunker=["a", "b"], generator=["extractive", "llm"]).expand("factorial")
    assert len(configs) == 4
    assert {c.generator for c in configs} == {"extractive", "llm"}


def test_the_baseline_carries_the_first_generator_value() -> None:
    assert Matrix(generator=("extractive", "llm")).baseline().generator == "extractive"


# ---------------------------------------------------------------------------
# BuiltPipeline.answer: retrieve already happened, this is assemble + generate
# ---------------------------------------------------------------------------


def test_a_pipeline_with_no_generator_refuses_to_answer(corpus: Corpus) -> None:
    pipeline = build(Config(), corpus)
    with pytest.raises(Exception, match="no generator"):
        pipeline.answer(QUESTION, [])


def test_the_extractive_generator_answers_from_retrieved_chunks(corpus: Corpus) -> None:
    pipeline = build(Config(generator="extractive"), corpus)
    chunk_ids = pipeline.search(QUESTION)
    answer, context = pipeline.answer(QUESTION, chunk_ids)
    assert answer.text
    assert context.chunks


def test_the_llm_generator_reaches_the_model_the_config_named(corpus: Corpus) -> None:
    """Same story as a model-backed transform: `run.model` -- passed here as `llm` -- has to
    reach the generator, or naming `llm` in a config file is unreachable from that file."""
    llm = RecordingLLM(replies=["Thirty days written notice is required [1]."])
    pipeline = build(Config(generator="llm"), corpus, llm=llm)
    chunk_ids = pipeline.search(QUESTION)
    answer, _ = pipeline.answer(QUESTION, chunk_ids)
    assert answer.text == "Thirty days written notice is required [1]."
    assert answer.citations == (1,)


# ---------------------------------------------------------------------------
# Runner: the whole thing, end to end
# ---------------------------------------------------------------------------


def test_none_costs_nothing(corpus: Corpus, evalset: EvalSet) -> None:
    """The default. No assembly, no model call, nothing folded into the metrics that was not
    there before this axis existed."""
    runner = Runner(corpus=corpus, headline="recall@5")
    result = runner.run_one(Config(), evalset)
    for name in ("groundedness", "citation_accuracy", "faithfulness", "answer_relevancy"):
        assert name not in result.metrics
    assert not any(w.stage == "generate" for w in result.warnings.entries)


def test_extractive_generation_needs_no_model_and_folds_lexical_scores(
    corpus: Corpus, evalset: EvalSet
) -> None:
    runner = Runner(corpus=corpus, headline="recall@5")
    result = runner.run_one(Config(generator="extractive"), evalset)
    assert 0.0 <= result.metrics["groundedness"] <= 1.0
    assert "abstention_accuracy" in result.metrics
    # No model was configured, so no judge could run.
    assert "faithfulness" not in result.metrics


def test_the_llm_generator_folds_faithfulness_and_answer_relevancy(
    corpus: Corpus, evalset: EvalSet
) -> None:
    """The metrics `report/composite.py`'s `DIMENSION_METRICS["generation"]` looks for. Without
    them a sweep that names a generator still has no `generation` dimension in the composite."""
    pytest.importorskip("deepeval")
    llm = RecordingLLM(replies=["Thirty days written notice is required [1]."], default=JUDGE_JSON)
    runner = Runner(corpus=corpus, headline="recall@5", llm=llm)
    result = runner.run_one(Config(generator="llm"), evalset)

    assert result.metrics["faithfulness"] == pytest.approx(1.0)
    assert result.metrics["answer_relevancy"] == pytest.approx(1.0)

    from contextgrid.report.composite import composite

    covered = composite(result.metrics).dimensions
    assert "generation" in covered


def test_the_judge_needs_a_model_to_exist_at_all(corpus: Corpus) -> None:
    """With no `run.model`, there is nothing to build a judge from -- the same reason a
    model-backed transform cannot be built without one either."""
    runner = Runner(corpus=corpus, headline="recall@5")
    assert runner._generation_judge() is None


def test_a_model_with_no_judge_extra_still_runs_generation(
    corpus: Corpus, evalset: EvalSet
) -> None:
    """A missing `[judge]` extra should degrade the DeepEval metrics, not the run. Simulated by
    generating without the LLM axis at all, which is the same code path `_score_generation`
    takes when `_generation_judge()` returns `None`."""
    result = Runner(corpus=corpus, headline="recall@5").run_one(
        Config(generator="extractive"), evalset
    )
    assert "faithfulness" not in result.metrics
    assert result.metrics["groundedness"] >= 0.0


def test_a_generator_that_fails_on_one_question_does_not_fail_the_run(
    corpus: Corpus, evalset: EvalSet
) -> None:
    """The pattern `retrieve.agentic` already follows for a planner that refuses to
    cooperate: the failure is recorded on that question, not raised through the sweep."""

    class BrokenGenerator:
        name = "broken"

        def answer(self, question: str, context: object) -> object:
            del question, context
            raise RuntimeError("the model refused")

    pipeline = build(Config(generator="extractive"), corpus)
    pipeline.generator = BrokenGenerator()  # type: ignore[assignment]

    runner = Runner(corpus=corpus, headline="recall@5")
    run = pipeline.run_queries(evalset)
    metrics, log, answers = runner._score_generation(pipeline, evalset, run, {})

    # Every question failed, so nothing was scored -- but the call itself did not raise,
    # which is the point.
    assert metrics["groundedness"] == 0.0
    assert any(w.stage == "generate" and "broken" in w.message for w in log.entries)
    assert all(w.code == WarningCode.GENERATION_FAILED for w in log.entries)
    # Nothing is saved as an answer either. An empty string recorded for a question the
    # generator refused would read afterwards as a model that replied with silence.
    assert answers == {}


# ---------------------------------------------------------------------------
# cost, and saying so
# ---------------------------------------------------------------------------


def test_it_declares_that_the_llm_generator_calls_a_model() -> None:
    """Read by the runner, which warns when a sweep containing one has no spending limit --
    the same treatment `AgenticRetrieval.uses_model` already gets."""
    grid = matrix(generator="llm")
    _warn_if_unbounded(grid, Budget())
    assert grid.meta["unbounded_model_calls"] == "llm"


def test_a_budget_silences_the_warning() -> None:
    grid = matrix(generator="llm")
    _warn_if_unbounded(grid, Budget(usd=5.0))
    assert "unbounded_model_calls" not in grid.meta


def test_extractive_never_triggers_it() -> None:
    grid = matrix(generator="extractive")
    _warn_if_unbounded(grid, Budget())
    assert "unbounded_model_calls" not in grid.meta


def test_no_generator_never_triggers_it() -> None:
    grid = matrix()
    _warn_if_unbounded(grid, Budget())
    assert "unbounded_model_calls" not in grid.meta
