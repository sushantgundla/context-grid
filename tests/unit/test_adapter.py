"""Unit tests for query-side adapters.

The assertions that matter are about restraint. An adapter fitted on a few dozen pairs can
destroy a working retriever, and the tests below pin down the settings where that happens so
the defaults cannot drift back towards them.
"""

from __future__ import annotations

import numpy as np
import pytest

from contextgrid.core.documents import Chunk
from contextgrid.core.evalset import EvalItem, EvalSet
from contextgrid.core.span import Span
from contextgrid.embed import (
    AdaptedEmbedder,
    AdapterError,
    HashEmbedder,
    LinearAdapter,
    TfidfEmbedder,
    Triplet,
    fit_adapter,
    mine_triplets,
    split_triplets,
)

PASSAGES = [
    "Either party may terminate on thirty days written notice.",
    "Fees are payable within thirty days of the invoice date.",
    "Send your key in the X-Api-Key header of every request.",
    "Unknown identifiers cause the endpoint to return 404.",
    "Backups are retained for twelve months and encrypted.",
    "Incidents must be reported within one hour of discovery.",
]

TRIPLETS = [
    Triplet(query="how much notice to leave", positive=PASSAGES[0], negatives=(PASSAGES[1],)),
    Triplet(query="when are invoices due", positive=PASSAGES[1], negatives=(PASSAGES[0],)),
    Triplet(query="which header holds the key", positive=PASSAGES[2], negatives=(PASSAGES[3],)),
    Triplet(query="what happens for a bad id", positive=PASSAGES[3], negatives=(PASSAGES[2],)),
    Triplet(query="how long are backups kept", positive=PASSAGES[4], negatives=(PASSAGES[5],)),
    Triplet(query="how fast to report an incident", positive=PASSAGES[5], negatives=(PASSAGES[4],)),
]


def embedder() -> HashEmbedder:
    model = HashEmbedder(dimensions=64)
    model.prepare(PASSAGES)
    return model


def vectors(count: int = 6, dimensions: int = 16) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(count, dimensions)).astype(np.float32)


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


def test_an_adapter_fits_and_applies() -> None:
    adapter = LinearAdapter()
    adapter.fit(vectors(), vectors())
    assert adapter.is_fitted
    assert adapter.apply(vectors()).shape == (6, 16)


def test_applying_before_fitting_is_an_error() -> None:
    with pytest.raises(AdapterError, match="not been fitted"):
        LinearAdapter().apply(vectors())


def test_output_stays_on_the_unit_sphere() -> None:
    """Document vectors are normalised, so an adapted query that is not would be comparing
    against them on a different scale."""
    adapter = LinearAdapter()
    adapter.fit(vectors(), vectors())
    norms = np.linalg.norm(adapter.apply(vectors()), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_mismatched_shapes_are_refused() -> None:
    with pytest.raises(AdapterError, match="exactly one positive"):
        LinearAdapter().fit(vectors(6), vectors(3))


def test_a_single_pair_cannot_fit_anything() -> None:
    with pytest.raises(AdapterError, match="at least two pairs"):
        LinearAdapter().fit(vectors(1), vectors(1))


def test_negatives_from_another_model_are_refused() -> None:
    with pytest.raises(AdapterError, match="different model"):
        LinearAdapter().fit(vectors(), vectors(), vectors(6, 8))


def test_the_ridge_term_cannot_be_removed() -> None:
    """With fewer pairs than dimensions the plain least-squares problem is underdetermined
    and the solution is nonsense."""
    with pytest.raises(AdapterError, match="ridge must be positive"):
        LinearAdapter(ridge=0.0)


def test_strength_is_bounded() -> None:
    with pytest.raises(AdapterError, match="between 0 and 1"):
        LinearAdapter(strength=1.5)


def test_fitting_is_deterministic() -> None:
    queries, positives = vectors(), vectors()
    first, second = LinearAdapter(), LinearAdapter()
    first.fit(queries, positives)
    second.fit(queries, positives)
    assert np.allclose(first.apply(queries), second.apply(queries))


# ---------------------------------------------------------------------------
# restraint
# ---------------------------------------------------------------------------


def test_the_default_strength_is_low_on_purpose() -> None:
    """Measured on a held-out split, 0.1 gained +0.08 recall@5 on a dense embedder while
    1.0 lost 0.22. A fit on a few dozen pairs should nudge rather than decide."""
    assert LinearAdapter().strength <= 0.2


def test_a_lower_strength_moves_vectors_less() -> None:
    queries, positives = vectors(), vectors()
    light, heavy = LinearAdapter(strength=0.1), LinearAdapter(strength=1.0)
    light.fit(queries, positives)
    heavy.fit(queries, positives)

    reference = light.apply(queries)
    assert np.linalg.norm(light.apply(queries) - _unit(queries)) < np.linalg.norm(
        heavy.apply(queries) - _unit(queries)
    )
    assert reference.shape == queries.shape


def test_a_heavy_strength_warns_that_it_usually_loses() -> None:
    adapter = LinearAdapter(strength=0.9)
    report = adapter.fit(vectors(), vectors(), held_out=True)
    assert any("nudge rather than decide" in note for note in report.warnings())


def test_fitting_on_the_scored_questions_says_the_score_is_optimistic() -> None:
    report = LinearAdapter().fit(vectors(), vectors(), held_out=False)
    assert any("optimistic" in note for note in report.warnings())


def test_a_held_out_fit_at_a_light_strength_has_nothing_to_warn_about() -> None:
    report = LinearAdapter(strength=0.15).fit(vectors(40, 16), vectors(40, 16), held_out=True)
    assert report.warnings() == []


def test_a_thin_fit_is_flagged() -> None:
    """The ridge term is carrying most of the solution, and the gain may not survive contact
    with queries unlike these."""
    report = LinearAdapter().fit(vectors(4, 128), vectors(4, 128), held_out=True)
    assert any("thin fit" in note for note in report.warnings())


def test_the_report_reads_plainly() -> None:
    report = LinearAdapter().fit(vectors(), vectors(), vectors(), held_out=True)
    summary = report.summary()
    assert "hard negatives" in summary
    assert "held-out split" in summary


# ---------------------------------------------------------------------------
# the wrapper
# ---------------------------------------------------------------------------


def test_the_document_side_is_provably_untouched() -> None:
    """The property that means the index does not have to be rebuilt when the adapter is."""
    base = embedder()
    adapter = fit_adapter(base, TRIPLETS)
    wrapped = AdaptedEmbedder(base=base, adapter=adapter)

    assert np.array_equal(
        wrapped.embed_documents(PASSAGES).vectors, base.embed_documents(PASSAGES).vectors
    )


def test_the_query_side_is_moved() -> None:
    base = embedder()
    wrapped = AdaptedEmbedder(base=base, adapter=fit_adapter(base, TRIPLETS))
    assert not np.allclose(
        wrapped.embed_queries(["how much notice"]).vectors,
        base.embed_queries(["how much notice"]).vectors,
    )


def test_an_unfitted_adapter_passes_queries_through_unchanged() -> None:
    base = embedder()
    wrapped = AdaptedEmbedder(base=base, adapter=LinearAdapter())
    assert np.array_equal(
        wrapped.embed_queries(["anything"]).vectors, base.embed_queries(["anything"]).vectors
    )


def test_the_wrapper_names_itself_so_the_label_shows_it() -> None:
    base = embedder()
    assert AdaptedEmbedder(base=base, adapter=LinearAdapter()).name.endswith("+adapter")


def test_embedding_nothing_is_safe() -> None:
    base = embedder()
    wrapped = AdaptedEmbedder(base=base, adapter=fit_adapter(base, TRIPLETS))
    assert wrapped.embed_queries([]).count == 0


# ---------------------------------------------------------------------------
# mining what already exists
# ---------------------------------------------------------------------------


def chunks() -> dict[str, Chunk]:
    return {
        f"c{i}": Chunk(id=f"c{i}", span=Span("d", i * 100, i * 100 + 60), text=text)
        for i, text in enumerate(PASSAGES)
    }


def test_triplets_are_mined_from_a_completed_run() -> None:
    """The training data is the measurement. Neither costs anything extra."""
    evalset = EvalSet(
        id="es",
        items=(
            EvalItem(id="q1", question="how much notice"),
            EvalItem(id="q2", question="which header"),
        ),
    )
    qrels = {"q1": {"c0": 2}, "q2": {"c2": 2}}
    run = {"q1": ["c1", "c0", "c3"], "q2": ["c2", "c4"]}

    triplets = mine_triplets(evalset, qrels, run, chunks())
    assert len(triplets) == 2
    assert triplets[0].positive == PASSAGES[0]
    # c1 ranked above the answer and was not it -- exactly the near miss worth training on.
    assert PASSAGES[1] in triplets[0].negatives


def test_negatives_are_capped_per_query() -> None:
    evalset = EvalSet(id="es", items=(EvalItem(id="q1", question="q"),))
    run = {"q1": ["c1", "c2", "c3", "c4", "c0"]}
    triplets = mine_triplets(evalset, {"q1": {"c0": 2}}, run, chunks(), negatives_per_query=2)
    assert len(triplets[0].negatives) == 2


def test_questions_with_no_gold_are_skipped() -> None:
    evalset = EvalSet(id="es", items=(EvalItem(id="q1", question="q"),))
    assert mine_triplets(evalset, {}, {"q1": ["c0"]}, chunks()) == []


def test_a_question_the_run_missed_still_yields_a_positive() -> None:
    """The answer is known even when the retriever did not find it, and those are the most
    useful pairs to train on."""
    evalset = EvalSet(id="es", items=(EvalItem(id="q1", question="q"),))
    triplets = mine_triplets(evalset, {"q1": {"c0": 2}}, {"q1": ["c3", "c4"]}, chunks())
    assert triplets[0].positive == PASSAGES[0]
    assert len(triplets[0].negatives) == 2


def test_fitting_needs_more_than_one_pair_and_says_where_to_get_them() -> None:
    with pytest.raises(AdapterError, match="Run a sweep"):
        fit_adapter(embedder(), TRIPLETS[:1])


def test_splitting_holds_data_back() -> None:
    """Fitted on everything and scored on everything, an adapter reports a gain it has not
    earned, and that gain does not survive contact with a real query."""
    train, held = split_triplets(TRIPLETS, fraction=0.5, seed=0)
    assert len(train) + len(held) == len(TRIPLETS)
    assert train
    assert held
    assert not {t.query for t in train} & {t.query for t in held}


def test_splitting_is_deterministic_for_a_seed() -> None:
    first, _ = split_triplets(TRIPLETS, seed=3)
    second, _ = split_triplets(TRIPLETS, seed=3)
    assert [t.query for t in first] == [t.query for t in second]


@pytest.mark.parametrize("fraction", [0.0, 1.0, 1.5])
def test_a_nonsense_split_is_refused(fraction: float) -> None:
    with pytest.raises(AdapterError, match="between 0 and 1"):
        split_triplets(TRIPLETS, fraction=fraction)


def test_it_works_on_a_sparse_embedder_without_crashing() -> None:
    """It measurably hurts on TF-IDF -- a dense linear map destroys the sparsity that made
    lexical matching work -- but it must not fail."""
    sparse = TfidfEmbedder()
    sparse.prepare(PASSAGES)
    adapter = fit_adapter(sparse, TRIPLETS)
    assert adapter.is_fitted


def _unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
