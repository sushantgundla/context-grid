"""Real embedding backends: litellm for hosted models, TEI for a local server.

Neither is called for real here. CI has no API key and no server, and a test suite that needs
either is a test suite nobody runs. What is tested is everything around the call -- prefixes,
batching, truncation, retries, and the messages -- because that is where these adapters can be
wrong in ways that lower every score by a few points without failing anything.
"""

from __future__ import annotations

import json
import urllib.error
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from contextgrid.embed import get_embedder
from contextgrid.embed.prefixes import Prefixes, for_model
from contextgrid.embed.remote import EmbedderError, LiteLLMEmbedder, TEIEmbedder


class FakeServer:
    """Stands in for TEI or for litellm, and records exactly what it was asked."""

    def __init__(self, *, dimensions: int = 4, fail_times: int = 0, error: str = "timeout") -> None:
        self.dimensions = dimensions
        self.seen: list[list[str]] = []
        self.fail_times = fail_times
        self.error = error

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise EmbedderError(self.error)
        self.seen.append(list(texts))
        return [[float(len(t)), 1.0, 0.0, 0.0][: self.dimensions] for t in texts]

    @property
    def batches(self) -> int:
        return len(self.seen)

    @property
    def all_texts(self) -> list[str]:
        return [text for batch in self.seen for text in batch]


def wire(embedder: Any, server: FakeServer) -> Any:
    """Rebuild the embedder with the fake server as its transport."""
    from dataclasses import replace

    return replace(embedder, transport=lambda batch: (server(batch), sum(len(t) for t in batch)))


@pytest.fixture
def tei() -> TEIEmbedder:
    return TEIEmbedder(model="bge-base-en-v1.5", dimensions=4, batch_size=2)


# ---------------------------------------------------------------------------
# prefixes -- the quietest way to get a comparison wrong
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "query", "document"),
    [
        ("e5-base-v2", "query: ", "passage: "),
        ("intfloat/multilingual-e5-large", "query: ", "passage: "),
        ("nomic-embed-text-v1.5", "search_query: ", "search_document: "),
        ("text-embedding-3-small", "", ""),
        ("something-nobody-has-heard-of", "", ""),
    ],
)
def test_prefixes_are_looked_up_from_the_model_name(model: str, query: str, document: str) -> None:
    """E5 was trained with `query:` and `passage:`. Embed both sides the same way and it still
    works -- the numbers just come out several points low, uniformly, with nothing to say why."""
    found = for_model(model)
    assert found.query == query
    assert found.document == document


def test_bge_gets_an_instruction_on_the_query_and_nothing_on_the_document() -> None:
    """The asymmetry is the point, and it is the part people get wrong."""
    found = for_model("BAAI/bge-base-en-v1.5")
    assert found.query.startswith("Represent this sentence")
    assert found.document == ""


def test_an_unknown_model_gets_no_prefix() -> None:
    """Adding a prefix a model was not trained with is as wrong as omitting one it was."""
    assert not for_model("mystery-model-v9").used


def test_the_prefix_actually_reaches_the_request() -> None:
    server = FakeServer()
    embedder = wire(TEIEmbedder(model="e5-base-v2", dimensions=4), server)

    embedder.embed_queries(["how long do refunds take"])
    embedder.embed_documents(["refunds take thirty days"])

    assert server.all_texts == [
        "query: how long do refunds take",
        "passage: refunds take thirty days",
    ]


def test_an_explicit_prefix_overrides_the_lookup() -> None:
    server = FakeServer()
    embedder = wire(TEIEmbedder(model="e5-base-v2", dimensions=4, query_prefix="ask: "), server)
    embedder.embed_queries(["question"])
    assert server.all_texts == ["ask: question"]


def test_an_explicit_empty_prefix_beats_the_lookup() -> None:
    """Somebody writing `query_prefix=""` is saying "this one needs none". Guessing over the
    top of that is worse than guessing in the first place."""
    server = FakeServer()
    embedder = wire(
        TEIEmbedder(model="e5-base-v2", dimensions=4, query_prefix="", document_prefix=""),
        server,
    )
    embedder.embed_queries(["question"])
    assert server.all_texts == ["question"]


def test_queries_and_documents_are_not_embedded_the_same_way() -> None:
    server = FakeServer()
    embedder = wire(TEIEmbedder(model="e5-base-v2", dimensions=4), server)
    embedder.embed_queries(["same text"])
    embedder.embed_documents(["same text"])
    assert server.all_texts[0] != server.all_texts[1]


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def test_inputs_are_split_into_batches(tei: TEIEmbedder) -> None:
    """Providers cap how many inputs one request may carry, and a corpus is thousands."""
    server = FakeServer()
    embedder = wire(tei, server)

    result = embedder.embed_documents([f"document {i}" for i in range(7)])

    assert server.batches == 4  # 2 + 2 + 2 + 1
    assert result.count == 7


def test_every_input_comes_back_in_order(tei: TEIEmbedder) -> None:
    """Vectors are matched to chunks by position. One row out of order attaches every score
    after it to the wrong text, and nothing would look wrong."""
    server = FakeServer()
    embedder = wire(tei, server)

    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    result = embedder.embed_documents(texts)

    # The fake puts the text's length in the first component and 1.0 in the second, so their
    # ratio survives normalising and the order can be read straight off the matrix.
    recovered = [round(float(row[0] / row[1])) for row in result.vectors]
    assert recovered == [len(t) for t in texts]


def test_a_short_count_is_refused(tei: TEIEmbedder) -> None:
    from dataclasses import replace

    short = replace(tei, transport=lambda batch: ([[1.0, 0.0, 0.0, 0.0]], 0))
    with pytest.raises(EmbedderError, match="must match"):
        short.embed_documents(["one", "two", "three"])


def test_no_texts_means_no_request(tei: TEIEmbedder) -> None:
    server = FakeServer()
    embedder = wire(tei, server)
    assert embedder.embed_documents([]).count == 0
    assert server.batches == 0


def test_batch_size_must_be_at_least_one() -> None:
    with pytest.raises(EmbedderError, match="batch_size"):
        TEIEmbedder(model="x", batch_size=0)


def test_a_model_name_is_required() -> None:
    with pytest.raises(EmbedderError, match="needs a model name"):
        TEIEmbedder()


# ---------------------------------------------------------------------------
# truncation and normalising
# ---------------------------------------------------------------------------


def test_over_long_text_is_cut_and_said_so() -> None:
    """The chunk that held the answer gets truncated, the answer was in the last paragraph, and
    nothing says so. That failure is invisible on every chart."""
    server = FakeServer()
    embedder = wire(TEIEmbedder(model="x", dimensions=4, max_tokens=8), server)

    result = embedder.embed_documents(["word " * 500])

    assert result.truncated == 1
    assert any("truncated" in w.message for w in result.warnings)


def test_vectors_come_back_unit_length(tei: TEIEmbedder) -> None:
    """So dot product and cosine agree, which every index here assumes."""
    vectors = wire(tei, FakeServer()).embed_documents(["one", "two"]).vectors
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_normalising_can_be_turned_off() -> None:
    embedder = wire(TEIEmbedder(model="x", dimensions=4, normalise_vectors=False), FakeServer())
    assert not embedder.normalised
    vectors = embedder.embed_documents(["a longer piece of text"]).vectors
    assert not np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_a_width_that_is_not_what_the_config_said_is_reported() -> None:
    """Otherwise the mismatch surfaces much later as a shape error inside an index, or not at
    all when a cached run from a different model gets reused."""
    embedder = wire(TEIEmbedder(model="x", dimensions=768), FakeServer(dimensions=4))
    result = embedder.embed_documents(["text"])
    assert any("768" in w.message and "4" in w.message for w in result.warnings)


def test_prepare_learns_nothing_from_the_corpus(tei: TEIEmbedder) -> None:
    """The model is already trained. A backend that quietly fitted on the corpus would be
    getting statistics the arms it is compared against do not have."""
    assert tei.prepare(["anything at all"]) is None


# ---------------------------------------------------------------------------
# failure
# ---------------------------------------------------------------------------


def test_a_transient_failure_is_retried(tei: TEIEmbedder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    server = FakeServer(fail_times=2, error="rate limit exceeded")
    assert wire(tei, server).embed_documents(["text"]).count == 1


def test_a_bad_key_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrying a wrong API key wastes a minute and changes nothing."""
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    server = FakeServer(fail_times=99, error="invalid api key")
    embedder = wire(TEIEmbedder(model="x", dimensions=4, retries=3), server)

    with pytest.raises(EmbedderError, match="api key"):
        embedder.embed_documents(["text"])
    assert server.fail_times == 98  # tried once, gave up


def test_a_failure_names_the_batch_it_happened_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """In a sweep of thousands of chunks, "it failed" is not actionable."""
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    embedder = wire(
        TEIEmbedder(model="bge-base-en-v1.5", dimensions=4, batch_size=2, retries=0),
        FakeServer(fail_times=99, error="unauthorized"),
    )
    with pytest.raises(EmbedderError, match=r"documents 0-1"):
        embedder.embed_documents(["a", "b", "c"])


# ---------------------------------------------------------------------------
# TEI over HTTP
# ---------------------------------------------------------------------------


def fake_urlopen(payload: object, *, status: int = 200) -> Any:
    class Response:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

    return lambda request, timeout=None: Response()


def test_tei_posts_the_texts_and_reads_the_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    def urlopen(request: Any, timeout: float | None = None) -> Any:
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data.decode("utf-8"))
        return fake_urlopen([[1.0, 0.0], [0.0, 1.0]])(request)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    embedder = TEIEmbedder(model="bge-base-en-v1.5", api_base="http://localhost:8080/")
    result = embedder.embed_documents(["one", "two"])

    assert sent["url"] == "http://localhost:8080/embed"
    assert sent["body"]["inputs"] == ["one", "two"]
    assert sent["body"]["truncate"] is True
    assert result.count == 2


def test_a_server_that_is_not_running_says_how_to_start_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The most likely failure by far, and the one worth spending a paragraph on."""

    def refuse(request: Any, timeout: float | None = None) -> Any:
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(EmbedderError, match="docker run"):
        TEIEmbedder(model="bge-base-en-v1.5", retries=0).embed_documents(["text"])


def test_an_http_error_carries_the_servers_own_message(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    def fail(request: Any, timeout: float | None = None) -> Any:
        raise urllib.error.HTTPError(
            "http://localhost:8080/embed",
            413,
            "Payload Too Large",
            {},  # type: ignore[arg-type]
            io.BytesIO(b'{"error":"batch too large"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(EmbedderError, match="batch too large"):
        TEIEmbedder(model="x", retries=0).embed_documents(["text"])


def test_something_that_is_not_a_tei_server_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"<html>hello</html>"

    monkeypatch.setattr(urllib.request, "urlopen", lambda r, timeout=None: Response())
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(EmbedderError, match="really"):
        TEIEmbedder(model="x", retries=0).embed_documents(["text"])


def test_tei_reports_no_token_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """TEI does not return usage, and a guessed number in a cost column is worse than an honest
    zero. A local model's cost is machine time, which the cost model takes from the clock."""
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen([[1.0, 0.0]]))
    assert TEIEmbedder(model="x").embed_documents(["text"]).input_tokens == 0


def test_the_server_can_be_asked_what_it_is_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    """`model` in the config is a label; this is the only way to find out whether the server
    behind it is running the weights you think it is."""
    monkeypatch.setattr(
        urllib.request, "urlopen", fake_urlopen({"model_id": "BAAI/bge-base-en-v1.5"})
    )
    assert TEIEmbedder(model="x").info()["model_id"] == "BAAI/bge-base-en-v1.5"


# ---------------------------------------------------------------------------
# litellm
# ---------------------------------------------------------------------------


def install_fake_litellm(monkeypatch: pytest.MonkeyPatch, **behaviour: Any) -> dict[str, Any]:
    """A stand-in litellm module, so the adapter can be tested with no key and no network."""
    import sys
    import types

    seen: dict[str, Any] = {}
    module = types.ModuleType("litellm")

    def embedding(**kwargs: Any) -> Any:
        seen.update(kwargs)
        if behaviour.get("raises"):
            raise behaviour["raises"]

        class Response:
            def __init__(self) -> None:
                self.data = [{"embedding": [1.0, 0.0]} for _ in kwargs["input"]]
                self.usage = type("Usage", (), {"prompt_tokens": 42})()

        return Response()

    module.embedding = embedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", module)
    return seen


def test_litellm_is_called_with_the_model_and_the_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = install_fake_litellm(monkeypatch)
    result = LiteLLMEmbedder(model="text-embedding-3-small").embed_documents(["one", "two"])

    assert seen["model"] == "text-embedding-3-small"
    assert seen["input"] == ["one", "two"]
    assert result.count == 2
    assert result.input_tokens == 42


def test_litellm_passes_an_api_base_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Which is how litellm reaches a local TEI or Infinity server too."""
    seen = install_fake_litellm(monkeypatch)
    LiteLLMEmbedder(model="openai/bge", api_base="http://localhost:8080").embed_documents(["x"])
    assert seen["api_base"] == "http://localhost:8080"


def test_the_key_comes_from_the_environment_not_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config file with a secret in it ends up in version control, and then in a screenshot."""
    seen = install_fake_litellm(monkeypatch)
    monkeypatch.setenv("CG_TEST_KEY", "sk-secret")

    LiteLLMEmbedder(model="m", api_key_env="CG_TEST_KEY").embed_documents(["x"])
    assert seen["api_key"] == "sk-secret"


def test_a_missing_key_says_which_variable_to_set(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_litellm(monkeypatch, raises=Exception("AuthenticationError: no api key provided"))
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(EmbedderError, match="OPENAI_API_KEY"):
        LiteLLMEmbedder(model="text-embedding-3-small", retries=0).embed_documents(["x"])


def test_an_unrecognised_model_explains_the_provider_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_litellm(monkeypatch, raises=Exception("LLM Provider NOT provided: not found"))
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(EmbedderError, match="provider/model"):
        LiteLLMEmbedder(model="embed-english-v3.0", retries=0).embed_documents(["x"])


def test_dimensions_are_requested_when_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = install_fake_litellm(monkeypatch)
    LiteLLMEmbedder(model="text-embedding-3-large", dimensions=256).embed_documents(["x"])
    assert seen["dimensions"] == 256


# ---------------------------------------------------------------------------
# reachable from a config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        "tei:bge-base-en-v1.5",
        "tei:e5-base-v2,api_base=http://elsewhere:9000",
        "litellm:text-embedding-3-small",
        "litellm:cohere/embed-english-v3.0,dimensions=1024",
    ],
)
def test_every_backend_is_reachable_from_one_config_line(spec: str) -> None:
    embedder = get_embedder(spec)
    assert embedder.model
    assert embedder.name in {"tei", "litellm"}


def test_the_model_is_part_of_what_identifies_the_embedder() -> None:
    """Two sweeps against servers running different models must not share a cache entry."""
    assert get_embedder("tei:bge-base-en-v1.5") != get_embedder("tei:e5-base-v2")


def test_prefixes_compare_equal_when_they_are_the_same() -> None:
    assert Prefixes(query="a: ") == Prefixes(query="a: ")


# ---------------------------------------------------------------------------
# costing a hosted model reached through a backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("tfidf", "tfidf"),
        ("tfidf:5000", "tfidf"),
        ("litellm:text-embedding-3-small", "text-embedding-3-small"),
        ("litellm:cohere/embed-english-v3.0,dimensions=1024", "embed-english-v3.0"),
        ("tei:bge-base-en-v1.5", "bge-base-en-v1.5"),
        ("tei:e5-base-v2,api_base=http://localhost:9000", "e5-base-v2"),
    ],
)
def test_the_price_is_looked_up_under_the_model_not_the_backend(spec: str, expected: str) -> None:
    """Splitting on the first colon would price every hosted model as the string "litellm",
    find no entry, and quietly cost the whole sweep at zero."""
    from contextgrid.cost.model import price_key

    assert price_key(spec) == expected


def test_a_hosted_model_through_litellm_is_actually_charged_for() -> None:
    from contextgrid.cost.model import CostModel

    pricing = CostModel().pricing_for("litellm:text-embedding-3-small")
    assert pricing.metered
    assert pricing.embed_per_million > 0


def test_a_local_server_is_free_per_token_without_complaining() -> None:
    """Machine time is the cost of a local model, and the cost model takes that from the clock.
    Warning about a missing price for something the user is running themselves is noise."""
    from contextgrid.cost.model import CostModel

    model = CostModel()
    assert not model.pricing_for("tei:some-model-nobody-published-a-price-for").metered
    assert not list(model.warnings)


def test_an_unpriced_hosted_model_still_says_so() -> None:
    from contextgrid.cost.model import CostModel

    model = CostModel()
    model.pricing_for("litellm:brand-new-model")
    assert any("no published price" in w.message for w in model.warnings)


def test_an_embedder_instance_can_be_priced() -> None:
    """The Python API takes objects where the config takes strings, and costing has to work
    for both or a sweep driven from Python silently reports zero."""
    from contextgrid.cost.model import CostModel

    assert CostModel().pricing_for(LiteLLMEmbedder(model="text-embedding-3-small")).metered


# ---------------------------------------------------------------------------
# axes take strings
# ---------------------------------------------------------------------------


def test_sweeping_over_an_instance_is_refused_with_the_string_to_use_instead() -> None:
    """Left through, it surfaces much later as "expected str" from inside a report formatter --
    long after the sweep ran, pointing at a place unrelated to the mistake."""
    from contextgrid.core.errors import ContextGridError
    from contextgrid.grid import matrix

    with pytest.raises(ContextGridError, match=r'embedder="tei"'):
        matrix(embedder=TEIEmbedder(model="bge-base-en-v1.5"))


def test_an_instance_hidden_in_a_list_is_caught_too() -> None:
    from contextgrid.core.errors import ContextGridError
    from contextgrid.grid import matrix

    with pytest.raises(ContextGridError, match="spec strings"):
        matrix(embedder=["tfidf", TEIEmbedder(model="x")])


def test_a_whole_sweep_runs_against_a_stand_in_server() -> None:
    """The end-to-end case: a real embedder, no server, no key, no network."""
    from contextgrid.core.documents import MediaType
    from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
    from contextgrid.corpus import Corpus
    from contextgrid.embed import EMBEDDERS
    from contextgrid.grid import Runner, matrix

    def toy(batch: Sequence[str]) -> tuple[list[list[float]], int]:
        return [[float(len(t)), float(t.lower().count("refund")), 1.0] for t in batch], 0

    name = "stand-in-tei"
    if name not in EMBEDDERS:
        EMBEDDERS.register(name, doc="offline stand-in for a TEI server")(
            lambda: TEIEmbedder(model="e5-base-v2", dimensions=3, transport=toy)
        )

    corpus = Corpus.from_texts(
        {
            "refunds.md": "# Refunds\n\nRefunds are issued within 30 days of purchase.\n",
            "shipping.md": "# Shipping\n\nStandard shipping takes 5 to 7 business days.\n",
        },
        media_type=MediaType.MARKDOWN,
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
        matrix(chunker="recursive:128", embedder=name, index="dense", k=3),
        evalset,
        mode="factorial",
    )
    assert results.best("recall@3") is not None
