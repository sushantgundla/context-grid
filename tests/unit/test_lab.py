"""End-to-end tests through the public front door.

These are the shape of the README. If they break, the example in the docs is a lie.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.corpus import Corpus
from contextgrid.lab import Lab
from tests.support import API_DOCS, CONTRACT

QUESTIONS = [
    ("q1", "How much notice to terminate for convenience?", "contract.md", "thirty days"),
    ("q2", "Which header carries the API key?", "api.md", "X-Api-Key"),
    ("q3", "What is the Premium monthly fee?", "contract.md", "$3,400"),
    ("q4", "What happens on a material breach?", "contract.md", "fifteen days of written notice"),
    ("q5", "What does GET /widgets return?", "api.md", "Returns 404 when the id is unknown"),
]


@pytest.fixture
def evalset() -> EvalSet:
    return EvalSet(
        id="es",
        items=tuple(
            EvalItem(id=i, question=q, anchors=(GoldAnchor(source_id=s, quote=t),))
            for i, q, s, t in QUESTIONS
        ),
    )


@pytest.fixture
def lab() -> Lab:
    return Lab({"contract.md": CONTRACT, "api.md": API_DOCS})


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_a_lab_can_be_built_from_a_dict_of_texts(lab: Lab) -> None:
    assert len(lab.corpus) == 2


def test_a_lab_can_be_built_from_a_directory(tmp_path: Path) -> None:
    (tmp_path / "contract.md").write_text(CONTRACT)
    assert len(Lab(tmp_path).corpus) == 1


def test_a_lab_can_be_built_from_a_corpus() -> None:
    corpus = Corpus.from_texts({"a": "text"})
    assert Lab(corpus).corpus is corpus


# ---------------------------------------------------------------------------
# looking before you leap
# ---------------------------------------------------------------------------


def test_the_fingerprint_profiles_the_corpus_before_anything_is_configured(lab: Lab) -> None:
    profile = lab.fingerprint()
    assert profile.is_parsed
    assert profile.heading_count > 0
    assert profile.hints()  # it has something to say about these documents


def test_the_estimate_says_how_big_the_sweep_is_before_running_it(lab: Lab) -> None:
    lab.grid(chunker=["sentence:1", "fixed:20,overlap=0"], index=["dense", "bm25", "hybrid"])
    estimate = lab.estimate("factorial")
    assert estimate["configurations"] == 6  # 2 chunkers x 3 indexes, one embedder
    assert "\u00d7" in estimate["shape"]


def test_a_second_embedder_does_not_multiply_the_sparse_arm(lab: Lab) -> None:
    """BM25 never looks at a vector, so `bm25 + tfidf` and `bm25 + hash` are one run.

    Left alone they would waste two thirds of the sparse arm and, worse, average three
    identical BM25 scores into the embedder axis effect as though it had earned them.
    """
    lab.grid(embedder=["tfidf", "hash:64"], index=["dense", "bm25"])
    assert lab.estimate("factorial")["configurations"] == 3  # not 4


# ---------------------------------------------------------------------------
# the whole loop
# ---------------------------------------------------------------------------


def test_the_readme_example_actually_works(lab: Lab, evalset: EvalSet) -> None:
    lab.grid(
        chunker=["sentence:1", "structural:60,min_size=8"],
        embedder=["tfidf", "hash:128"],
        index=["dense", "bm25"],
        k=3,
    )
    results = lab.run(evalset, mode="factorial", headline="recall@3")

    assert len(results) > 1
    leaderboard = results.leaderboard("recall@3")
    assert leaderboard[0]["recall@3"] >= leaderboard[-1]["recall@3"]
    assert results.summary("recall@3")
    assert results.pareto("recall@3", "p95_ms")


def test_every_axis_gets_an_effect(lab: Lab, evalset: EvalSet) -> None:
    lab.grid(chunker=["sentence:1", "fixed:12,overlap=0"], index=["dense", "bm25"], k=3)
    results = lab.run(evalset, mode="factorial", headline="recall@3")
    assert len(results.axis_effect("chunker", "recall@3")) == 2
    assert len(results.axis_effect("index", "recall@3")) == 2


def test_a_deliberately_useless_embedder_scores_near_chance(lab: Lab, evalset: EvalSet) -> None:
    """The control that proves the scoring chain is measuring anything at all.

    `length` embeds text as its own token count, so it cannot know what a document is about.
    If it were to score like TF-IDF, something upstream would be leaking the answer.
    """
    lab.grid(chunker="fixed:12,overlap=0", embedder=["tfidf", "length"], index="dense", k=3)
    results = lab.run(evalset, mode="factorial", headline="recall@3")

    scores = {run.config.embedder: run.metric("recall@3") for run in results}
    assert scores["tfidf"] > scores["length"]


def test_a_sweep_reuses_work_and_says_how_much(lab: Lab, evalset: EvalSet) -> None:
    lab.grid(chunker="sentence:1", index=["dense", "bm25", "hybrid"], k=3)
    results = lab.run(evalset, mode="factorial")
    assert "reused" in results.cache_summary


def test_staged_mode_is_cheaper_and_says_what_it_gave_up(lab: Lab, evalset: EvalSet) -> None:
    lab.grid(
        chunker=["sentence:1", "fixed:12,overlap=0"],
        embedder=["tfidf", "hash:64"],
        index=["dense", "hybrid"],
        k=3,
    )
    factorial = lab.run(evalset, mode="factorial", headline="recall@3")
    staged = lab.run(evalset, mode="staged", headline="recall@3")

    assert len(staged) < len(factorial)
    assert any(w.code.value == "non_deterministic_stage" for w in staged.warnings)


def test_costing_a_hosted_model_puts_it_on_the_same_chart_as_a_local_one() -> None:
    """`machine_usd_per_hour` is what makes the comparison honest: a local model is free per
    token and not free to run."""
    lab = Lab({"contract.md": CONTRACT}, machine_usd_per_hour=0.10)
    assert lab.cost_model.machine_usd_per_hour == 0.10
