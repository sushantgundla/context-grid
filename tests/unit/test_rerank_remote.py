"""Cross-encoder rerankers: TEI locally, litellm for hosted.

Neither is called for real -- CI has no server and no key. What is tested is the part that
decides whether the reranker axis means anything: that every candidate comes back, that ties
do not shuffle between runs, and that a backend dropping documents is an error rather than a
silent demotion.
"""

from __future__ import annotations

import json
import urllib.error
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest

from contextgrid.core.documents import Chunk
from contextgrid.core.span import Span
from contextgrid.rerank import get_reranker
from contextgrid.rerank.remote import LiteLLMReranker, RerankerError, TEIReranker


def chunks(*texts: str) -> list[Chunk]:
    return [
        Chunk(
            id=f"doc:{index}",
            span=Span("doc", index * 100, index * 100 + len(text)),
            text=text,
        )
        for index, text in enumerate(texts)
    ]


CANDIDATES = chunks(
    "Shipping takes five to seven business days.",
    "Refunds are issued within thirty days of purchase.",
    "The office is closed on public holidays.",
    "Digital goods are not refundable once downloaded.",
)


def by_keyword(word: str) -> Any:
    """A stand-in cross-encoder: scores a passage by whether it mentions a word."""

    def transport(query: str, passages: Sequence[str]) -> list[tuple[int, float]]:
        return [(i, 1.0 if word in p.lower() else 0.0) for i, p in enumerate(passages)]

    return transport


@pytest.fixture
def tei() -> TEIReranker:
    return TEIReranker(model="bge-reranker-base", transport=by_keyword("refund"))


# ---------------------------------------------------------------------------
# the ordering
# ---------------------------------------------------------------------------


def test_the_best_candidate_comes_first(tei: TEIReranker) -> None:
    top = tei.rerank("do I get a refund?", CANDIDATES, k=2)
    assert {scored.chunk_id for scored in top} == {"doc:1", "doc:3"}


def test_only_k_results_come_back(tei: TEIReranker) -> None:
    assert len(tei.rerank("refund", CANDIDATES, k=1)) == 1


def test_scores_descend(tei: TEIReranker) -> None:
    scores = [scored.score for scored in tei.rerank("refund", CANDIDATES, k=4)]
    assert scores == sorted(scores, reverse=True)


def test_ties_keep_the_retrievers_order() -> None:
    """Without a stable tie-break, a rerun reorders equally-scored passages and a diff shows a
    change that did not happen."""
    flat = TEIReranker(model="m", transport=lambda q, ps: [(i, 0.5) for i in range(len(ps))])
    order = [scored.chunk_id for scored in flat.rerank("q", CANDIDATES, k=4)]
    assert order == [chunk.id for chunk in CANDIDATES]


def test_no_candidates_means_no_call() -> None:
    called = False

    def transport(query: str, passages: Sequence[str]) -> list[tuple[int, float]]:
        nonlocal called
        called = True
        return []

    assert TEIReranker(model="m", transport=transport).rerank("q", [], k=5) == []
    assert not called


# ---------------------------------------------------------------------------
# batching, and not losing anybody
# ---------------------------------------------------------------------------


def test_candidates_are_split_into_batches() -> None:
    seen: list[int] = []

    def transport(query: str, passages: Sequence[str]) -> list[tuple[int, float]]:
        seen.append(len(passages))
        return [(i, float(i)) for i in range(len(passages))]

    reranker = TEIReranker(model="m", batch_size=3, transport=transport)
    reranker.rerank("q", chunks(*[f"passage {i}" for i in range(7)]), k=7)
    assert seen == [3, 3, 1]


def test_a_score_in_a_later_batch_is_attached_to_the_right_candidate() -> None:
    """Backends index within the batch they were given. Forget to add the offset and every
    score after the first batch lands on the wrong passage."""

    def transport(query: str, passages: Sequence[str]) -> list[tuple[int, float]]:
        # Only the last passage of each batch is relevant.
        return [(i, 1.0 if i == len(passages) - 1 else 0.0) for i in range(len(passages))]

    docs = chunks(*[f"passage {i}" for i in range(6)])
    top = TEIReranker(model="m", batch_size=2, transport=transport).rerank("q", docs, k=3)
    assert {scored.chunk_id for scored in top} == {"doc:1", "doc:3", "doc:5"}


def test_a_backend_that_returns_fewer_scores_is_an_error() -> None:
    """A dropped candidate looks exactly like a candidate the model judged irrelevant, and it
    is a completely different claim."""
    short = TEIReranker(model="m", transport=lambda q, ps: [(0, 1.0)])
    with pytest.raises(RerankerError, match="Every candidate must come back"):
        short.rerank("q", CANDIDATES, k=4)


def test_an_index_outside_the_batch_is_an_error() -> None:
    wrong = TEIReranker(model="m", transport=lambda q, ps: [(99, 1.0) for _ in ps])
    with pytest.raises(RerankerError, match="wrong passages"):
        wrong.rerank("q", CANDIDATES, k=4)


def test_a_very_long_passage_is_trimmed() -> None:
    seen: list[str] = []

    def transport(query: str, passages: Sequence[str]) -> list[tuple[int, float]]:
        seen.extend(passages)
        return [(i, 0.0) for i in range(len(passages))]

    reranker = TEIReranker(model="m", max_chars=50, transport=transport)
    reranker.rerank("q", chunks("word " * 200), k=1)
    assert len(seen[0]) == 50


def test_a_model_name_is_required() -> None:
    with pytest.raises(RerankerError, match="needs a model name"):
        TEIReranker()


def test_batch_size_must_be_at_least_one() -> None:
    with pytest.raises(RerankerError, match="batch_size"):
        TEIReranker(model="m", batch_size=0)


# ---------------------------------------------------------------------------
# failure
# ---------------------------------------------------------------------------


def test_a_transient_failure_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    attempts = {"n": 0}

    def flaky(query: str, passages: Sequence[str]) -> list[tuple[int, float]]:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RerankerError("rate limit exceeded")
        return [(i, 0.0) for i in range(len(passages))]

    assert TEIReranker(model="m", transport=flaky).rerank("q", CANDIDATES, k=2)


def test_a_bad_key_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    attempts = {"n": 0}

    def refuse(query: str, passages: Sequence[str]) -> list[tuple[int, float]]:
        attempts["n"] += 1
        raise RerankerError("invalid api key")

    with pytest.raises(RerankerError, match="api key"):
        TEIReranker(model="m", retries=3, transport=refuse).rerank("q", CANDIDATES, k=2)
    assert attempts["n"] == 1


def test_a_failure_names_the_candidates_it_happened_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    def refuse(query: str, passages: Sequence[str]) -> list[tuple[int, float]]:
        raise RerankerError("unauthorized")

    reranker = TEIReranker(model="m", batch_size=2, retries=0, transport=refuse)
    with pytest.raises(RerankerError, match=r"candidates 0-1"):
        reranker.rerank("q", CANDIDATES, k=4)


# ---------------------------------------------------------------------------
# TEI over HTTP
# ---------------------------------------------------------------------------


def respond(payload: object) -> Any:
    class Response:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

    return lambda request, timeout=None: Response()


def test_tei_posts_the_query_and_the_passages(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    def urlopen(request: Any, timeout: float | None = None) -> Any:
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data.decode("utf-8"))
        return respond([{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}])(request)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    reranker = TEIReranker(model="bge-reranker-base", api_base="http://localhost:8081/")
    top = reranker.rerank("refund?", CANDIDATES[:2], k=2)

    assert sent["url"] == "http://localhost:8081/rerank"
    assert sent["body"]["query"] == "refund?"
    assert sent["body"]["truncate"] is True
    assert top[0].chunk_id == "doc:1"


def test_a_missing_server_says_it_needs_its_own_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """One TEI process serves one model, so reranking needs a second server. The natural
    assumption is otherwise and the failure is an unhelpful 400."""

    def refuse(request: Any, timeout: float | None = None) -> Any:
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(RerankerError, match="its own server"):
        TEIReranker(model="bge-reranker-base", retries=0).rerank("q", CANDIDATES, k=2)


def test_an_embedding_model_answering_a_rerank_request_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pointing at the embedding server by mistake is the likeliest misconfiguration here."""
    monkeypatch.setattr(urllib.request, "urlopen", respond([[0.1, 0.2, 0.3]]))
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(RerankerError, match="not a reranker"):
        TEIReranker(model="m", retries=0).rerank("q", CANDIDATES[:1], k=1)


def test_an_http_error_carries_the_servers_message(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    def fail(request: Any, timeout: float | None = None) -> Any:
        raise urllib.error.HTTPError(
            "http://localhost:8081/rerank",
            413,
            "Too Large",
            {},  # type: ignore[arg-type]
            io.BytesIO(b'{"error":"input too long"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(RerankerError, match="input too long"):
        TEIReranker(model="m", retries=0).rerank("q", CANDIDATES, k=2)


# ---------------------------------------------------------------------------
# litellm
# ---------------------------------------------------------------------------


def install_fake_litellm(monkeypatch: pytest.MonkeyPatch, **behaviour: Any) -> dict[str, Any]:
    import sys
    import types

    seen: dict[str, Any] = {}
    module = types.ModuleType("litellm")

    def rerank(**kwargs: Any) -> Any:
        seen.update(kwargs)
        if behaviour.get("raises"):
            raise behaviour["raises"]
        rows = [
            {"index": i, "relevance_score": 1.0 - i * 0.1} for i in range(len(kwargs["documents"]))
        ]
        return {"results": behaviour.get("results", rows)}

    module.rerank = rerank  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", module)
    return seen


def test_litellm_is_asked_for_every_passage_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """`top_n` defaults to a handful at most providers. Accept that and the rest of the
    candidates vanish from the ranking without a word."""
    seen = install_fake_litellm(monkeypatch)
    LiteLLMReranker(model="cohere/rerank-english-v3.0").rerank("q", CANDIDATES, k=2)

    assert seen["top_n"] == len(CANDIDATES)
    assert seen["query"] == "q"


def test_litellm_results_are_ordered(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_litellm(
        monkeypatch,
        results=[
            {"index": 2, "relevance_score": 0.2},
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.5},
            {"index": 3, "relevance_score": 0.1},
        ],
    )
    top = LiteLLMReranker(model="cohere/rerank-english-v3.0").rerank("q", CANDIDATES, k=2)
    assert [scored.chunk_id for scored in top] == ["doc:0", "doc:1"]


def test_a_missing_key_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_litellm(monkeypatch, raises=Exception("AuthenticationError: no api key"))
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(RerankerError, match="COHERE_API_KEY"):
        LiteLLMReranker(model="cohere/rerank-english-v3.0", retries=0).rerank("q", CANDIDATES, k=2)


def test_an_unqualified_model_explains_the_provider_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_litellm(monkeypatch, raises=Exception("LLM Provider NOT provided: not found"))
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(RerankerError, match="provider/model"):
        LiteLLMReranker(model="rerank-english-v3.0", retries=0).rerank("q", CANDIDATES, k=2)


def test_the_key_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = install_fake_litellm(monkeypatch)
    monkeypatch.setenv("CG_RERANK_KEY", "sk-secret")
    LiteLLMReranker(model="m", api_key_env="CG_RERANK_KEY").rerank("q", CANDIDATES, k=1)
    assert seen["api_key"] == "sk-secret"


def test_results_without_scores_say_what_was_expected(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_litellm(monkeypatch, results=[{"document": "text"}])
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(RerankerError, match="relevance"):
        LiteLLMReranker(model="m", retries=0).rerank("q", CANDIDATES[:1], k=1)


# ---------------------------------------------------------------------------
# reachable from a config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        "tei-rerank:bge-reranker-base",
        "tei-rerank:bge-reranker-v2-m3,api_base=http://elsewhere:9000",
        "litellm-rerank:cohere/rerank-english-v3.0",
    ],
)
def test_every_backend_is_reachable_from_one_config_line(spec: str) -> None:
    reranker = get_reranker(spec)
    assert reranker.model
    assert reranker.name in {"tei-rerank", "litellm-rerank"}


def test_the_reranker_axis_still_has_a_do_nothing_arm() -> None:
    """Half of "use a reranker" advice is untested. The baseline has to be on the same chart,
    with the same latency and cost columns."""
    from contextgrid.rerank import RERANKERS

    assert "none" in RERANKERS


def test_a_stand_in_reranker_plugs_into_a_whole_sweep() -> None:
    from contextgrid.core.documents import MediaType as Media
    from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
    from contextgrid.corpus import Corpus
    from contextgrid.grid import Runner, matrix
    from contextgrid.rerank import RERANKERS

    name = "stand-in-rerank"
    if name not in RERANKERS:
        RERANKERS.register(name, doc="offline stand-in for a TEI reranker")(
            lambda: TEIReranker(model="bge-reranker-base", transport=by_keyword("refund"))
        )

    corpus = Corpus.from_texts(
        {
            "refunds.md": "# Refunds\n\nRefunds are issued within 30 days of purchase.\n",
            "shipping.md": "# Shipping\n\nStandard shipping takes 5 to 7 business days.\n",
        },
        media_type=Media.MARKDOWN,
        name="stand-in",
    )
    evalset = EvalSet(
        id="stand-in",
        items=(
            EvalItem(
                id="q1",
                question="How long do refunds take?",
                anchors=(GoldAnchor(quote="within 30 days", source_id="refunds.md"),),
            ),
        ),
    )

    results = Runner(corpus=corpus, headline="recall@3").run(
        matrix(chunker="recursive:128", index="bm25", embedder=None, reranker=[None, name], k=3),
        evalset,
        mode="factorial",
    )
    assert len(results.runs) == 2


def test_replacing_the_transport_leaves_everything_else_alone() -> None:
    """`transport` is a documented hook, so it has to survive `dataclasses.replace`."""
    original = TEIReranker(model="bge-reranker-base", batch_size=8)
    swapped = replace(original, transport=by_keyword("refund"))
    assert swapped.batch_size == 8
    assert swapped.model == original.model
