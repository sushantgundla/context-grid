"""What a stranger found in 0.9.3, installed from PyPI into a container.

Four things, and the first three share a shape: a guard that one entry point applies and the
one beside it does not. `evaluate()` refuses a ranking with a repeated chunk id and refuses a
cut-off below 1; `diagnose()`, reading the same two arguments, accepted both and turned the
second into a verdict. `significance()` takes a `metric` argument, stamps it on the result,
and tested a different metric entirely -- because the per-question scores it reads were only
ever kept for the headline.

The last one is a falsy zero: `sample=0` meant "sample nothing" everywhere except in the one
`if` that decided whether to sample at all, where it meant "sample everything".
"""

from __future__ import annotations

import pytest

from contextgrid.core.documents import MediaType
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor, GoldSpan
from contextgrid.core.types import Chunk, Span
from contextgrid.corpus import Corpus
from contextgrid.diagnose.taxonomy import FailurePoint, diagnose
from contextgrid.evalset.generate import KeywordProbeGenerator, generate
from contextgrid.grid import Runner, matrix
from contextgrid.pipeline import Config
from contextgrid.report.results import Results, RunResult
from contextgrid.score.metrics import evaluate
from contextgrid.score.significance import SignificanceError
from tests.support import API_DOCS, CONTRACT

# ---------------------------------------------------------------------------
# 1. a significance test that answered about a different metric
#
# `significance(left, right, metric="recall@1")` returned the recall@5 means, the recall@5
# difference and the recall@5 confidence interval, with `metric='recall@1'` written on it.
# `RunResult.per_query` held one flat `{question: score}` dict for the headline metric alone,
# and `metric` travelled beside it as a label that nothing read.
#
# The damage is not cosmetic. On the drive corpus the recall@1 gap was +0.524 and the verdict
# said "+0.000 ... consistent with no difference at all", so the tool reported two
# configurations as indistinguishable while one retrieved nearly three times as well.
# ---------------------------------------------------------------------------


QUESTIONS = [
    ("q1", "How much notice is needed to terminate for convenience?", "contract.md", "thirty days"),
    ("q2", "Which header carries the API key?", "api.md", "X-Api-Key"),
    ("q3", "What is the Premium monthly fee?", "contract.md", "$3,400"),
    ("q4", "What happens on a material breach?", "contract.md", "fifteen days of written notice"),
    ("q5", "What does GET /widgets return?", "api.md", "Returns 404 when the id is unknown"),
]


@pytest.fixture
def corpus() -> Corpus:
    return Corpus.from_texts(
        {"contract.md": CONTRACT, "api.md": API_DOCS}, media_type=MediaType.MARKDOWN
    )


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
def swept(corpus: Corpus, evalset: EvalSet) -> Results:
    """A real sweep, so the per-question scores come from the runner rather than by hand."""
    return Runner(corpus=corpus, seed=0).run(
        matrix(chunker=["recursive:512", "fixed:20,overlap=0"], index=["dense", "bm25"]),
        evalset,
        mode="factorial",
    )


def test_significance_reports_the_metric_it_was_asked_for(swept: Results) -> None:
    """The means in the `Comparison` must be the means of the metric named on it.

    Read straight off the leaderboard for the same metric, which was always right -- the two
    outputs disagreed with each other, and only one of them can have been the score.
    """
    for metric in ("recall@1", "recall@3", "recall@5", "map@5", "precision@5"):
        board = {row["config"]: row[metric] for row in swept.leaderboard(metric=metric)}
        ranked = list(board)
        left, right = ranked[0], ranked[-1]
        verdict = swept.significance(left, right, metric=metric)
        assert verdict.metric == metric
        assert verdict.left_mean == pytest.approx(board[left]), metric
        assert verdict.right_mean == pytest.approx(board[right]), metric


def test_significance_and_compare_agree_on_the_gap(swept: Results) -> None:
    """`compare()` and `significance()` are two views of one comparison.

    `compare()` read the aggregate off `metrics` and was right; `significance()` recomputed it
    from the headline's per-question scores and was wrong. They must not be able to disagree.
    """
    for metric in ("recall@1", "recall@5", "mrr@5"):
        board = [row["config"] for row in swept.leaderboard(metric=metric)]
        left, right = board[0], board[-1]
        gap = swept.compare(left, right, metric=metric)["difference"]
        assert swept.significance(left, right, metric=metric).difference.estimate == pytest.approx(
            gap
        ), metric


def test_the_disagreeing_questions_are_that_metrics_disagreements(swept: Results) -> None:
    """`compare()['differences']` named the questions two configs split on -- at recall@5,
    whatever metric was asked for. At k=1 and k=5 the same pair splits on different questions.
    """
    board = [row["config"] for row in swept.leaderboard(metric="recall@1")]
    left, right = board[0], board[-1]
    at_one = swept.compare(left, right, metric="recall@1")
    per_query_gap = sum(at_one["differences"].values()) / at_one["queries_compared"]
    assert per_query_gap == pytest.approx(at_one["difference"])


def test_is_the_winner_real_tests_the_metric_it_ranked_on(swept: Results) -> None:
    """It picks the top two rows by `metric`, so it must also test them on `metric`."""
    verdict = swept.is_the_winner_real("recall@1")
    assert verdict is not None
    board = {row["config"]: row["recall@1"] for row in swept.leaderboard(metric="recall@1")}
    assert verdict.left_mean == pytest.approx(board[verdict.left])
    assert verdict.right_mean == pytest.approx(board[verdict.right])


def test_the_summary_paragraph_does_not_contradict_its_own_headline(swept: Results) -> None:
    """`summary(metric="recall@1")` printed the winner's real recall@1 and then, one sentence
    later, the recall@5 gap -- two numbers about different metrics in one paragraph."""
    board = [row["config"] for row in swept.leaderboard(metric="recall@1")]
    paragraph = swept.summary(metric="recall@1")
    gap = swept.compare(board[0], board[1], metric="recall@1")["difference"]
    assert f"{gap:+.3f}" in paragraph


def test_a_metric_with_no_per_question_scores_is_refused_not_guessed() -> None:
    """Hand-built runs carry the headline only. Asking for anything else must raise rather
    than quietly answer about the headline -- the whole defect in one line."""
    runs = [
        RunResult(config=Config(chunker="recursive:512"), per_query={"q1": 1.0, "q2": 0.0}),
        RunResult(config=Config(chunker="sentence:3"), per_query={"q1": 0.0, "q2": 0.0}),
    ]
    results = Results(runs=runs)
    left, right = (run.label for run in results.runs)
    # The headline still works, so nothing that already relied on this breaks.
    assert results.significance(left, right, metric="recall@5").left_mean == pytest.approx(0.5)
    with pytest.raises(SignificanceError, match="recall@1"):
        results.significance(left, right, metric="recall@1")


def test_a_leaderboard_row_carries_that_metrics_confidence_interval(swept: Results) -> None:
    """`ci_low`/`ci_high` sat beside the metric column and described the headline instead."""
    rows = swept.leaderboard(metric="recall@1")
    for row in rows:
        if "ci_low" in row:
            assert row["ci_low"] <= row["recall@1"] <= row["ci_high"], row["config"]


# ---------------------------------------------------------------------------
# 2. diagnose() scored a ranking that evaluate() refuses
#
# One duplicate id earlier in the ranking pushes everything after it down a place, so the
# evidence at rank 5 was read as rank 6 and a success was reported as "buy a reranker".
# ---------------------------------------------------------------------------


_ONE_ITEM = EvalSet(
    id="e",
    items=(EvalItem(id="a", question="a", gold=(GoldSpan(Span("d", 0, 10)),)),),
)
_QRELS = {"a": {"gold": 2}}
_CLEAN = {"a": ["f1", "f2", "f3", "f4", "gold"]}
_DUPLICATED = {"a": ["f1", "f2", "f3", "f1", "f4", "gold"]}


def test_the_clean_ranking_really_is_a_success() -> None:
    """Guard the fixture: without the duplicate the evidence is inside the top 5."""
    only = diagnose(_ONE_ITEM, _QRELS, _CLEAN, k=5).diagnoses[0]
    assert only.failure is FailurePoint.NONE
    assert only.gold_rank == 5


def test_diagnose_refuses_a_ranking_with_a_repeated_chunk_id() -> None:
    """`evaluate()` calls this a retriever's bug rather than a score. So must `diagnose()`."""
    with pytest.raises(ValueError, match="same chunk id more than once"):
        evaluate(_QRELS, _DUPLICATED, ks=[5])
    with pytest.raises(ValueError, match="same chunk id more than once"):
        diagnose(_ONE_ITEM, _QRELS, _DUPLICATED, k=5)


def test_a_duplicate_cannot_turn_a_success_into_a_reranker_recommendation() -> None:
    """The consequence, pinned separately from the guard: this returned
    `fp2_missed_top_ranked` at rank 6 and told the reader to go and buy a reranker."""
    with pytest.raises(ValueError):
        diagnose(_ONE_ITEM, _QRELS, _DUPLICATED, k=5)


# ---------------------------------------------------------------------------
# 3. diagnose() accepted a cut-off of zero and below
#
# "the evidence was retrieved at rank 1, just outside the top 0" -- a rank-1 hit called a
# failure, and at k=-1 a sentence about "the top -1".
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, -1, -5])
def test_diagnose_refuses_a_cutoff_below_one(k: int) -> None:
    """Same guard and same wording as `evaluate(ks=[0])`, which has always refused it."""
    with pytest.raises(ValueError, match="cut-off k must be at least 1"):
        evaluate(_QRELS, _CLEAN, ks=[k])
    with pytest.raises(ValueError, match="cut-off k must be at least 1"):
        diagnose(_ONE_ITEM, _QRELS, _CLEAN, k=k)


def test_diagnose_refuses_a_deep_k_below_one() -> None:
    """`deep_k` is the other cut-off on the same call and needs the same floor."""
    with pytest.raises(ValueError, match="cut-off k must be at least 1"):
        diagnose(_ONE_ITEM, _QRELS, _CLEAN, k=5, deep_k=0)


# ---------------------------------------------------------------------------
# 4. generate(sample=0) drafted from every chunk
#
# `if sample` is false for 0 and for None alike, so "sample nothing" and "sample everything"
# were the same branch. `sample=-1` meanwhile returned nothing, so the two ends of the
# nonsense disagreed with each other.
# ---------------------------------------------------------------------------


def _long(doc: str, index: int, word: str) -> Chunk:
    text = " ".join(f"{word}{n}" for n in range(30))
    return Chunk(id=f"{doc}::{index}", span=Span(doc_id=doc, start=0, end=len(text)), text=text)


_THREE = [_long("a.pdf", 0, "alpha"), _long("a.pdf", 1, "beta"), _long("b.pdf", 0, "gamma")]


def test_sampling_none_of_the_chunks_drafts_no_questions() -> None:
    """`sample=0` is a number of chunks like any other, and that number is none of them."""
    draft = generate(_THREE, KeywordProbeGenerator(), sample=0)
    assert draft.chunks_sampled == 0
    assert draft.count == 0


def test_sample_none_still_means_every_usable_chunk() -> None:
    """The documented meaning of `None`, which `0` had been sharing."""
    draft = generate(_THREE, KeywordProbeGenerator(), sample=None)
    assert draft.chunks_sampled == len(_THREE)


def test_a_negative_sample_is_refused_rather_than_silently_emptied() -> None:
    """`sample=-1` returned an empty draft that looked exactly like a corpus of short chunks."""
    with pytest.raises(ValueError, match="sample"):
        generate(_THREE, KeywordProbeGenerator(), sample=-1)
