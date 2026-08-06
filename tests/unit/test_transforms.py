"""Unit tests for query transformation.

Two things matter here. That each transform does what it claims, and that when the model
fails -- which it does -- the transform degrades to searching with the original question
rather than with nothing.
"""

from __future__ import annotations

import json

import pytest

from contextgrid.core.documents import MediaType
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.corpus import Corpus
from contextgrid.evalset.llm import LLMError, RecordingLLM
from contextgrid.grid import Runner, matrix
from contextgrid.pipeline import Config
from contextgrid.transform import (
    TRANSFORMS,
    Decompose,
    ExpandAcronyms,
    HyDE,
    MultiQuery,
    NoTransform,
    QueryTransform,
    StepBack,
    describe_cost,
    get_transform,
)
from tests.support import API_DOCS, CONTRACT

QUESTION = "How much notice is needed to terminate for convenience?"


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------


def transforms_with_a_model() -> list[QueryTransform]:
    return [
        HyDE(llm=RecordingLLM(default="Termination requires thirty days written notice.")),
        MultiQuery(llm=RecordingLLM(default=json.dumps(["a", "b", "c"]))),
        Decompose(llm=RecordingLLM(default=json.dumps(["part one", "part two"]))),
        StepBack(llm=RecordingLLM(default="What do the termination clauses say?")),
    ]


ALL = [NoTransform(), ExpandAcronyms(), *transforms_with_a_model()]


@pytest.fixture(params=ALL, ids=[t.name for t in ALL])
def transform(request: pytest.FixtureRequest) -> QueryTransform:
    return request.param  # type: ignore[no-any-return]


def test_satisfies_the_protocol(transform: QueryTransform) -> None:
    assert isinstance(transform, QueryTransform)


def test_always_produces_at_least_one_query(transform: QueryTransform) -> None:
    assert transform.transform(QUESTION).fan_out >= 1


def test_the_original_question_is_recorded(transform: QueryTransform) -> None:
    assert transform.transform(QUESTION).original == QUESTION


def test_no_query_is_blank(transform: QueryTransform) -> None:
    assert all(text.strip() for text in transform.transform(QUESTION).queries)


# ---------------------------------------------------------------------------
# what each one does
# ---------------------------------------------------------------------------


def test_the_identity_transform_searches_with_the_question_as_asked() -> None:
    """The arm every transform has to beat, and without it none of this can be judged."""
    result = NoTransform().transform(QUESTION)
    assert result.queries == (QUESTION,)
    assert result.is_identity
    assert result.llm_calls == 0


def test_hyde_searches_with_an_invented_answer() -> None:
    llm = RecordingLLM(replies=["Either party may terminate on thirty days written notice."])
    result = HyDE(llm=llm).transform(QUESTION)

    assert "thirty days written notice" in result.queries[1]
    assert result.llm_calls == 1
    assert "as if quoting a document" in llm.prompts[0]


def test_hyde_keeps_the_question_alongside_the_invention() -> None:
    """Hedging the failure mode: when the model invents nonsense for a corpus it does not
    know, the real question is still in the fused results."""
    llm = RecordingLLM(default="invented passage")
    assert HyDE(llm=llm).transform(QUESTION).queries[0] == QUESTION
    assert HyDE(llm=llm, include_question=False).transform(QUESTION).queries[0] != QUESTION


def test_multi_query_paraphrases_and_keeps_the_original() -> None:
    llm = RecordingLLM(replies=[json.dumps(["how long is notice", "what is the notice period"])])
    result = MultiQuery(llm=llm, variants=2).transform(QUESTION)

    assert result.queries[0] == QUESTION
    assert result.fan_out == 3
    assert result.llm_calls == 1


def test_multi_query_honours_its_variant_cap() -> None:
    llm = RecordingLLM(replies=[json.dumps(["a", "b", "c", "d", "e"])])
    assert MultiQuery(llm=llm, variants=2).transform(QUESTION).fan_out == 3


def test_decomposition_splits_a_question_into_its_parts() -> None:
    """The only transform that addresses a structural failure rather than a vocabulary one:
    a question no single passage can answer has to become two."""
    llm = RecordingLLM(
        replies=[json.dumps(["Which vendor has the shortest notice?", "What is their fee?"])]
    )
    result = Decompose(llm=llm).transform("Which vendor is cheapest and quickest to leave?")
    assert result.fan_out == 2


def test_decomposition_removes_duplicate_parts() -> None:
    llm = RecordingLLM(replies=[json.dumps(["same", "same", "other"])])
    assert Decompose(llm=llm).transform(QUESTION).fan_out == 2


def test_step_back_adds_a_more_general_question() -> None:
    llm = RecordingLLM(replies=["What do the termination clauses say?"])
    result = StepBack(llm=llm).transform(QUESTION)
    assert result.queries == (QUESTION, "What do the termination clauses say?")


def test_step_back_does_nothing_when_the_model_repeats_the_question() -> None:
    assert StepBack(llm=RecordingLLM(replies=[QUESTION])).transform(QUESTION).is_identity


def test_expanding_acronyms_needs_no_model_at_all() -> None:
    """Unglamorous, free, and it moves BM25 more than most of the clever transforms. No
    embedding fixes a term the model has never seen."""
    transform = ExpandAcronyms(expansions={"RPO": "recovery point objective"})
    result = transform.transform("What is our RPO?")
    assert "recovery point objective" in result.queries[0]
    assert result.llm_calls == 0


def test_expansion_with_nothing_to_expand_is_the_identity() -> None:
    transform = ExpandAcronyms(expansions={"RPO": "recovery point objective"})
    assert transform.transform("What is the notice period?").is_identity
    assert ExpandAcronyms().transform("What is our RPO?").is_identity


# ---------------------------------------------------------------------------
# degrading when the model does
# ---------------------------------------------------------------------------


class BrokenLLM:
    name = "broken"

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        raise LLMError("the model is unavailable")


@pytest.mark.parametrize(
    "build",
    [
        lambda llm: HyDE(llm=llm),
        lambda llm: MultiQuery(llm=llm),
        lambda llm: Decompose(llm=llm),
        lambda llm: StepBack(llm=llm),
    ],
)
def test_a_failed_model_call_falls_back_to_the_original_question(build) -> None:  # type: ignore[no-untyped-def]
    """Searching with nothing would score zero and look like a retrieval result."""
    assert build(BrokenLLM()).transform(QUESTION).is_identity


@pytest.mark.parametrize("reply", ["", "   ", "not json at all"])
def test_an_unusable_reply_falls_back(reply: str) -> None:
    llm = RecordingLLM(default=reply)
    assert MultiQuery(llm=llm).transform(QUESTION).is_identity
    assert HyDE(llm=RecordingLLM(default="")).transform(QUESTION).is_identity


def test_a_reply_wrapped_in_an_object_is_still_read() -> None:
    llm = RecordingLLM(replies=[json.dumps({"questions": ["one", "two"]})])
    assert MultiQuery(llm=llm).transform(QUESTION).fan_out == 3


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def test_model_free_transforms_resolve_without_a_model() -> None:
    assert isinstance(get_transform(None), NoTransform)
    assert isinstance(get_transform("none"), NoTransform)
    assert isinstance(get_transform("expand"), ExpandAcronyms)


def test_asking_for_a_model_backed_transform_without_one_is_an_error() -> None:
    """A transform with no model would silently become the identity, and a configuration
    that looks like it is testing HyDE while testing nothing is worse than an error."""
    with pytest.raises(LLMError, match="needs a model"):
        get_transform("hyde")


def test_a_model_backed_transform_resolves_when_given_one() -> None:
    llm = RecordingLLM()
    assert isinstance(get_transform("hyde", llm), HyDE)
    assert get_transform("multi-query:5", llm).variants == 5  # type: ignore[union-attr]


def test_an_unknown_transform_lists_the_real_ones() -> None:
    with pytest.raises(LLMError, match="unknown transform"):
        get_transform("telepathy", RecordingLLM())


def test_the_registry_holds_the_model_free_ones() -> None:
    assert set(TRANSFORMS.names()) == {"none", "expand"}


# ---------------------------------------------------------------------------
# cost, which is the whole point
# ---------------------------------------------------------------------------


def test_the_cost_description_says_it_is_paid_on_every_query() -> None:
    """Index-time cost is paid once. This is paid forever, which is why a transform has to
    earn considerably more than it appears to."""
    llm = RecordingLLM(default=json.dumps(["a", "b"]))
    transformed = [MultiQuery(llm=llm).transform(QUESTION) for _ in range(4)]
    description = describe_cost(transformed)
    assert "model calls" in description
    assert "every query forever" in description


def test_a_model_free_transform_says_it_costs_no_calls() -> None:
    transformed = [NoTransform().transform(QUESTION)]
    assert "no model calls" in describe_cost(transformed)


def test_describing_nothing_is_safe() -> None:
    assert describe_cost([]) == "no queries transformed"


# ---------------------------------------------------------------------------
# through the grid
# ---------------------------------------------------------------------------


def test_a_transform_is_an_axis_like_any_other() -> None:
    grid = matrix(transform=["none", "expand"], index="bm25")
    configs = grid.expand("factorial")
    assert len(configs) == 2
    assert sum(1 for c in configs if c.transform is None) == 1  # "none" collapses


def test_the_label_shows_the_transform() -> None:
    assert "+expand" in Config(transform="expand").label
    assert "+" not in Config().label


def test_a_fan_out_transform_searches_more_than_once_and_fuses() -> None:
    """Scores are never compared across the searches: a cosine from one query and a cosine
    from another are not on the same scale, however similar they look."""
    corpus = Corpus.from_texts(
        {"contract.md": CONTRACT, "api.md": API_DOCS}, media_type=MediaType.MARKDOWN
    )
    evalset = EvalSet(
        id="es",
        items=(
            EvalItem(
                id="q1",
                question="What is our RPO for termination?",
                anchors=(GoldAnchor(source_id="contract.md", quote="thirty days"),),
            ),
        ),
    )
    result = Runner(corpus=corpus, headline="recall@3").run_one(
        Config(chunker="sentence:1", transform="expand", index="bm25", k=3), evalset
    )
    assert result.scored_queries == 1


# ---------------------------------------------------------------------------
# reachable from a config file
# ---------------------------------------------------------------------------


def test_a_model_backed_transform_works_from_a_config() -> None:
    """`transform: hyde` could not work. The pipeline never passed a model to `get_transform`,
    so four of the five transforms raised "needs a model" from a place the user had no way to
    influence -- unreachable from the config file, which is the primary interface."""
    import contextgrid as cg
    from contextgrid.core.documents import MediaType
    from contextgrid.evalset.llm import LiteLLMChat
    from contextgrid.pipeline import Config, build

    corpus = cg.Corpus.from_texts(
        {"a.md": "# Refunds\n\nRefunds are issued within 30 days of purchase.\n"},
        media_type=MediaType.MARKDOWN,
        name="hyde",
    )
    scripted = LiteLLMChat(
        model="scripted", transport=lambda prompt, limit: "Refunds take thirty days."
    )

    pipeline = build(
        Config(chunker="recursive:128", index="bm25", embedder=None, transform="hyde"),
        corpus,
        llm=scripted,
    )
    assert pipeline.transform.name == "hyde"
    assert pipeline.search("how long do refunds take?")


def test_the_config_names_one_model_for_every_stage_that_needs_one() -> None:
    """One name rather than one per stage: the alternative is four places to set a key and four
    prices to reconcile, for a choice almost nobody wants to make differently per stage."""
    from contextgrid.config import loads
    from contextgrid.config.loader import build_llm

    config = loads("corpus: ./docs\nrun:\n  model: openai:gpt-4o-mini\n")
    assert config.run.model == "openai:gpt-4o-mini"
    assert build_llm(config) is not None
    assert build_llm(loads("corpus: ./docs\n")) is None


def test_a_transform_needing_a_model_without_one_says_which_are_free() -> None:
    """Better than silently becoming the identity, which would look like testing HyDE while
    testing nothing."""
    from contextgrid.evalset.llm import LLMError
    from contextgrid.transform import get_transform

    with pytest.raises(LLMError, match="expand, none"):
        get_transform("hyde", None)


def test_every_transform_is_discoverable_even_when_it_needs_a_model() -> None:
    """They were invisible: reachable if you already knew the name, mentioned nowhere the tool
    prints -- so the axis appeared to have two arms when it has six."""
    from contextgrid.transform import MODEL_BACKED, available_transforms

    names = available_transforms()
    assert set(MODEL_BACKED) <= set(names)
    assert {"hyde", "multi-query", "decompose", "step-back", "expand", "none"} == set(names)


def test_the_starter_config_mentions_the_model_backed_transforms() -> None:
    from contextgrid.config import render

    text = render()
    line = next(line for line in text.splitlines() if "also available" in line and "hyde" in line)
    assert "hyde" in line
