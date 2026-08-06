"""Every axis, and whether what it advertises can actually be reached.

Every axis in this package already has a suite proving its *components* work: an embedder
embeds, an index searches, a retrieval strategy retrieves. None of those suites prove the thing
users actually do -- write a name in a config file and run it. That gap is exactly where real
bugs have hidden, all of them the same shape: something advertised on an axis that a config
file cannot actually reach or run.

Four found so far, before this suite existed:

* `transform: hyde` (and `multi-query`, `decompose`, `step-back`) needed a model the pipeline
  never passed through, so naming one from a config raised "needs a model" from a place the
  user had no way to influence.
* `budget_usd` was parsed, stored, written into the report, and never checked -- a config that
  asked to spend at most five dollars would spend whatever the matrix cost.
* `Runner(headline="recall@2")` reported 0.000 for every configuration, because 2 is not one of
  the default cut-offs and the metric was therefore never calculated.
* `usearch:b1` was registered as a valid `dtype` and raised `ValueError` on the first real
  `build()` call, because usearch's binary mode wants bit-packed input, not the float32 matrix
  every other arm on the index axis takes.

This suite tries to build and run **every name any registry or axis advertises**, from its spec
string, the way a config file would -- so the next one of these fails a test instead of
shipping.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any

import pytest

from contextgrid.chunk import CHUNKERS
from contextgrid.core.documents import MediaType, SourceFile
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.core.protocols import Parser
from contextgrid.core.registry import Registry
from contextgrid.corpus import Corpus
from contextgrid.embed import EMBEDDERS
from contextgrid.embed.remote import LiteLLMEmbedder, TEIEmbedder
from contextgrid.evalset.llm import RecordingLLM
from contextgrid.grid import Runner, matrix
from contextgrid.index import INDEXES, get_index
from contextgrid.index.base import Index
from contextgrid.index.quantize import Quantization
from contextgrid.ingest import INGESTERS, get_ingester
from contextgrid.parse import PARSERS, get_parser
from contextgrid.report.results import RunResult
from contextgrid.rerank import RERANKERS
from contextgrid.rerank.remote import LiteLLMReranker, TEIReranker
from contextgrid.retrieve import RETRIEVERS, AgenticRetrieval
from contextgrid.transform import MODEL_BACKED, available_transforms

# ---------------------------------------------------------------------------
# a tiny corpus and eval set every axis is measured against
# ---------------------------------------------------------------------------

CORPUS = Corpus.from_texts(
    {
        "refunds.md": "# Refunds\n\nRefunds are issued within 30 days of purchase.\n",
        "digital.md": "# Digital goods\n\nDigital goods are not refundable once downloaded.\n",
        "shipping.md": "# Shipping\n\nStandard shipping takes 5 to 7 business days.\n",
    },
    media_type=MediaType.MARKDOWN,
    name="reachability",
)

EVALSET = EvalSet(
    id="reachability",
    items=(
        EvalItem(
            id="q1",
            question="how long do refunds take?",
            anchors=(GoldAnchor(quote="within 30 days of purchase", source_id="refunds.md"),),
        ),
    ),
)

#: The parsers that only read PDF (pymupdf, pdfplumber, marker, pymupdf4llm) cannot see
#: `CORPUS` at all -- it is markdown. Built from the same generator the rest of the test suite
#: uses (`tests.pdf_fixtures`), so the bytes are the same ones already exercised elsewhere.
from tests.pdf_fixtures import contract_pdf  # noqa: E402

PDF_CORPUS = Corpus(
    files=(SourceFile(id="contract.pdf", media_type=MediaType.PDF, raw=contract_pdf()),),
    name="reachability-pdf",
)

PDF_EVALSET = EvalSet(
    id="reachability-pdf",
    items=(
        EvalItem(
            id="q1",
            question="how much notice must either party give to terminate?",
            anchors=(GoldAnchor(quote="thirty days written notice", source_id="contract.pdf"),),
        ),
    ),
)

#: Stands in for a real model everywhere one is needed: transforms, agentic retrieval, and the
#: LLM-backed ingestion strategies. `default` rather than `replies`, because how many times each
#: axis calls it depends on how many chunks or rounds that particular plugin uses, and the point
#: here is reachability, not counting calls.
SCRIPTED_LLM = RecordingLLM(
    default=json.dumps(["refunds are issued within 30 days of purchase", "the notice period"])
)


def _run(
    *,
    corpus: Corpus = CORPUS,
    evalset: EvalSet = EVALSET,
    llm: Any = None,
    budget_usd: float = 5.0,
    **axis: Any,
) -> RunResult:
    """Build one configuration from spec strings and run it end to end.

    `budget_usd` is set on every call, not just the model-backed ones, purely so a sweep over
    `agentic` or a generated ingestion strategy does not spam the "no spending limit" warning --
    it changes nothing about what gets asserted.
    """
    baseline: dict[str, Any] = {
        "chunker": "recursive:128",
        "index": "dense",
        "embedder": "tfidf",
        "k": 3,
    }
    baseline.update(axis)
    results = Runner(corpus=corpus, headline="recall@3", llm=llm).run(
        matrix(**baseline), evalset, mode="factorial", budget_usd=budget_usd
    )
    assert results.runs, f"matrix({baseline!r}) produced no runnable configuration at all"
    return results.runs[0]


def _reachable(run: RunResult) -> None:
    """The one invariant every axis value has to satisfy: it indexed something and searched it.

    Not "scored well" -- a tiny three-document corpus and an arbitrary chunker/embedder/index
    combination has no business winning on recall. Reachability is the claim under test, not
    quality.
    """
    assert run.chunk_count > 0, "indexed no chunks at all"
    assert run.run, "answered no queries at all"


def _register_stand_in(registry: Registry[Any], real_name: str, factory: Callable[[], Any]) -> str:
    """Register a transport-wired copy of a real plugin, under a private key.

    The grid only ever sees spec strings -- `Config.embedder` etc. are strings, not instances --
    so there is no way to hand a pre-built, transport-wired object to `matrix()` directly. This
    is the same trick `tests/unit/test_embed_remote.py` uses to run a whole sweep against a real
    embedder with no server and no key: register the wired instance under its own name, and
    sweep that name. The private key never appears in a test id; the parametrize list below
    always uses the real, advertised name, so a failure still names the real plugin.
    """
    private_name = f"reachability:{real_name}"
    if private_name not in registry:
        registry.register(private_name, doc=f"scripted stand-in for {real_name!r}; test-only")(
            factory
        )
    return private_name


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------

#: Parsers that load a layout model or are not installed in this environment. Proven
#: constructible and protocol-shaped below regardless; not run end to end, because doing so
#: would either download several hundred megabytes of weights on a CI box or fail on an absent
#: package -- and either belongs in a slow/integration suite, not here. Decided by reading
#: `src/contextgrid/parse/layout.py`: `docling` has no cached model directory in this
#: environment (`~/.cache/docling` does not exist) and `marker-pdf` is not installed at all.
HEAVY_OR_MISSING_PARSERS = frozenset({"docling", "marker"})


def test_parser_registry_is_not_empty() -> None:
    assert PARSERS.names()


@pytest.mark.parametrize("name", sorted(PARSERS.names()))
def test_every_parser_is_reachable(name: str) -> None:
    parser = get_parser(name)
    assert isinstance(parser, Parser), f"{name} does not satisfy the Parser protocol"
    assert parser.name

    if name in HEAVY_OR_MISSING_PARSERS:
        pytest.skip(
            f"{name}: needs a layout model (heavy download) or an uninstalled package; "
            "construction and protocol conformance are proven above, the parse itself is not"
        )

    # Routed by what the parser itself claims to read, not a hand-maintained list: pymupdf,
    # pdfplumber and pymupdf4llm only ever declare MediaType.PDF.
    corpus, evalset = (
        (CORPUS, EVALSET) if parser.supports(MediaType.MARKDOWN) else (PDF_CORPUS, PDF_EVALSET)
    )
    _reachable(_run(parser=name, corpus=corpus, evalset=evalset))


# ---------------------------------------------------------------------------
# chunkers
# ---------------------------------------------------------------------------


def test_chunker_registry_is_not_empty() -> None:
    assert CHUNKERS.names()


#: Every chunker's shorthand takes a token/word count except `semantic`, whose shorthand is a
#: *percentile* (0-100) of similarity drops -- not a size at all. Treating it like the others
#: would be `semantic:256`, which fails validation for a reason that has nothing to do with
#: reachability.
CHUNKER_SHORTHANDS: dict[str, str] = {"semantic": "80"}


@pytest.mark.parametrize("name", sorted(CHUNKERS.names()))
def test_every_chunker_is_reachable(name: str) -> None:
    # 256, not a smaller round number: `structural`'s shorthand is `max_size`, and its default
    # `min_size` is 64 -- a size below that collides with the chunker's *own* default rather
    # than testing reachability, which is a false failure this suite exists to not produce.
    value = CHUNKER_SHORTHANDS.get(name, "256")
    _reachable(_run(chunker=f"{name}:{value}"))


# ---------------------------------------------------------------------------
# embedders
# ---------------------------------------------------------------------------


def _toy_embed_transport(texts: Any) -> tuple[list[list[float]], int]:
    return [[float(len(t)), 1.0, 0.0] for t in texts], 0


#: `tei` and `litellm` reach a real server or a hosted API. `transport` replaces that call --
#: see `_register_stand_in` -- so these are proven reachable rather than skipped, per the same
#: rule the module docstring states: needing a key or a server is not a reason to skip when the
#: plugin already ships a way around it.
EMBEDDER_FACTORIES: dict[str, Callable[[], Any]] = {
    "tei": lambda: TEIEmbedder(model="reachability", dimensions=3, transport=_toy_embed_transport),
    "litellm": lambda: LiteLLMEmbedder(
        model="reachability", dimensions=3, transport=_toy_embed_transport
    ),
}


def test_embedder_registry_is_not_empty() -> None:
    assert EMBEDDERS.names()


@pytest.mark.parametrize("name", sorted(EMBEDDERS.names()))
def test_every_embedder_is_reachable(name: str) -> None:
    spec = name
    if name in EMBEDDER_FACTORIES:
        spec = _register_stand_in(EMBEDDERS, name, EMBEDDER_FACTORIES[name])
    _reachable(_run(embedder=spec))


# ---------------------------------------------------------------------------
# indexes -- including every kind/dtype/scheme a plugin takes as its shorthand
# ---------------------------------------------------------------------------


def _faiss_kinds() -> tuple[str, ...]:
    try:
        from contextgrid.index.ann import FaissIndex
    except ImportError:  # pragma: no cover - the [index] extra is not installed
        return ()
    return FaissIndex.KINDS


def _usearch_dtypes() -> tuple[str, ...]:
    try:
        from contextgrid.index.ann import USearchIndex
    except ImportError:  # pragma: no cover - the [index] extra is not installed
        return ()
    return USearchIndex.DTYPES


def _pgvector_kinds() -> tuple[str, ...]:
    try:
        from contextgrid.index.pgvector import PgVectorIndex
    except ImportError:  # pragma: no cover - the [pgvector] extra is not installed
        return ()
    return PgVectorIndex.KINDS


def _index_specs() -> list[str]:
    """One spec string per index *value* -- every registered index, expanded by whatever
    KINDS/DTYPES/SCHEMES-style axis it exposes as its own shorthand. Read from the plugins
    themselves rather than hand-copied, so a kind added or removed there is picked up here for
    free -- which is exactly how this suite is meant to have caught `usearch:b1`.
    """
    specs: list[str] = []
    for name in sorted(INDEXES.names()):
        if name == "quantized":
            # `Quantization` is the enum `QuantizedDenseIndex.scheme` validates against --
            # there is no `SCHEMES` ClassVar, so this is the actual source of truth for it.
            specs += [f"quantized:{scheme.value}" for scheme in Quantization]
        elif name == "faiss":
            specs += [f"faiss:{kind}" for kind in _faiss_kinds()] or ["faiss"]
        elif name == "usearch":
            specs += [f"usearch:{dtype}" for dtype in _usearch_dtypes()] or ["usearch"]
        elif name == "pgvector":
            specs += [f"pgvector:{kind}" for kind in _pgvector_kinds()] or ["pgvector"]
        elif name == "hybrid":
            # Validated inline in `HybridIndex.__post_init__` against `{"rrf", "weighted"}`;
            # there is no ClassVar to read it from, so it is named here from having read that.
            specs += ["hybrid:rrf", "hybrid:weighted"]
        else:
            specs.append(name)
    return specs


INDEX_SPECS = _index_specs()


def test_index_registry_is_not_empty() -> None:
    assert INDEXES.names()


def test_index_spec_enumeration_is_not_empty() -> None:
    """A bug in the enumeration above would make every parametrized test below pass on an
    empty list -- vacuously, and silently. This is the check that would catch that."""
    assert INDEX_SPECS
    assert len(INDEX_SPECS) >= len(INDEXES.names())


@pytest.mark.parametrize("spec", INDEX_SPECS, ids=INDEX_SPECS)
def test_every_index_is_reachable(spec: str) -> None:
    # Built from the spec string first, on its own -- proof the plugin can be constructed the
    # way a config file would construct it, independent of whether a full sweep can run it.
    index = get_index(spec)
    assert isinstance(index, Index), f"{spec} does not satisfy the Index protocol"

    if spec.startswith("pgvector:") and not os.environ.get("PGVECTOR_DSN"):
        pytest.skip(
            f"{spec}: needs a running Postgres with pgvector; set PGVECTOR_DSN. "
            "docker run -p 5432:5432 -e POSTGRES_PASSWORD=pg pgvector/pgvector:pg17"
        )

    # A fixed-width, corpus-independent embedder. `quantized:product`'s default 8 subspaces and
    # `faiss:ivfpq`'s subquantizers both need the embedding width to divide evenly, which a
    # 3-document corpus's TF-IDF vocabulary does not reliably do -- that would be a corpus-size
    # confound, not an index-reachability one, and 64 divides cleanly.
    _reachable(_run(index=spec, embedder="hash:64"))


# ---------------------------------------------------------------------------
# transforms -- including the model-backed ones that cannot live in the registry
# ---------------------------------------------------------------------------


def test_transform_enumeration_is_not_empty() -> None:
    assert available_transforms()
    assert set(MODEL_BACKED) <= set(available_transforms())


@pytest.mark.parametrize("name", sorted(available_transforms()))
def test_every_transform_is_reachable(name: str) -> None:
    llm = SCRIPTED_LLM if name in MODEL_BACKED else None
    _reachable(_run(transform=name, llm=llm))


# ---------------------------------------------------------------------------
# retrieval strategies
# ---------------------------------------------------------------------------


class _ScriptedPlanner:
    """A model that always has an opinion, so `agentic` never falls back to plain search."""

    def complete(self, prompt: str, *, max_tokens: int = 256) -> str:
        del prompt, max_tokens
        return json.dumps(["refunds", "purchase window"])


def _agentic_stand_in() -> AgenticRetrieval:
    """`AgenticRetrieval` gets its model from its own `_llm`, not from `Runner.llm` -- the grid
    never wires one through to the retrieval axis. Attaching a scripted planner directly, then
    registering the wired instance, is the same technique
    `tests/unit/test_retrieve_agentic.py::test_a_good_plan_beats_plain_search_on_two_part_questions`
    uses to run `agentic` end to end with no key and no network."""
    strategy = AgenticRetrieval()
    object.__setattr__(strategy, "_llm", _ScriptedPlanner())
    return strategy


RETRIEVAL_FACTORIES: dict[str, Callable[[], Any]] = {"agentic": _agentic_stand_in}


def test_retrieval_registry_is_not_empty() -> None:
    assert RETRIEVERS.names()


@pytest.mark.parametrize("name", sorted(RETRIEVERS.names()))
def test_every_retrieval_strategy_is_reachable(name: str) -> None:
    spec = name
    if name in RETRIEVAL_FACTORIES:
        spec = _register_stand_in(RETRIEVERS, name, RETRIEVAL_FACTORIES[name])
    _reachable(_run(retrieval=spec, budget_usd=5.0))


# ---------------------------------------------------------------------------
# rerankers
# ---------------------------------------------------------------------------


def _toy_rerank_transport(query: str, passages: Any) -> list[tuple[int, float]]:
    del query
    return [(i, float(len(passages) - i)) for i in range(len(passages))]


RERANKER_FACTORIES: dict[str, Callable[[], Any]] = {
    "tei-rerank": lambda: TEIReranker(model="reachability", transport=_toy_rerank_transport),
    "litellm-rerank": lambda: LiteLLMReranker(
        model="reachability", transport=_toy_rerank_transport
    ),
}


def test_reranker_registry_is_not_empty() -> None:
    assert RERANKERS.names()


@pytest.mark.parametrize("name", sorted(RERANKERS.names()))
def test_every_reranker_is_reachable(name: str) -> None:
    spec = name
    if name in RERANKER_FACTORIES:
        spec = _register_stand_in(RERANKERS, name, RERANKER_FACTORIES[name])
    _reachable(_run(reranker=spec))


# ---------------------------------------------------------------------------
# ingestion strategies
# ---------------------------------------------------------------------------


def test_ingestion_registry_is_not_empty() -> None:
    assert INGESTERS.names()


@pytest.mark.parametrize("name", sorted(INGESTERS.names()))
def test_every_ingestion_strategy_is_reachable(name: str) -> None:
    # `uses_model` is read off the built instance rather than hard-coded, so a future paid
    # strategy is classified correctly the moment it is registered, with no list to update here.
    strategy = get_ingester(name)
    llm = SCRIPTED_LLM if strategy.uses_model else None
    _reachable(_run(ingestion=name, llm=llm))


# ---------------------------------------------------------------------------
# the install instruction has to be the one that works
# ---------------------------------------------------------------------------


def _declared_extras(root: Any) -> dict[str, list[str]]:
    """The optional dependencies, read from pyproject.

    `tomllib` only arrived in Python 3.11 and this package supports 3.10, so the parse falls
    back to reading the one table it needs. Skipping on 3.10 would have been simpler and would
    have meant the guard covered four of the five Pythons in CI -- which is exactly the shape
    of hole it exists to close.
    """
    text = (root / "pyproject.toml").read_text(encoding="utf-8")

    try:
        import tomllib

        loaded: dict[str, list[str]] = tomllib.loads(text)["project"]["optional-dependencies"]
        return loaded
    except ModuleNotFoundError:
        pass

    extras: dict[str, list[str]] = {}
    inside = False
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            inside = stripped == "[project.optional-dependencies]"
            continue
        if not inside or not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("= ["):
            current = stripped.split("=", 1)[0].strip()
            extras[current] = []
        elif stripped == "]":
            current = None
        elif current is not None:
            extras[current].append(stripped.strip('",'))
    return extras


def test_every_extra_named_in_an_error_actually_exists() -> None:
    """An error naming an extra that does not exist -- or one that does not contain the package
    it promises -- is worse than no error: it costs an install and changes nothing.

    Three of these have been found. `unstructured` named `parse-ml`, which never contained it.
    `cl100k_base` named `embed`, which was not declared at all. And when `marker` moved to its
    own extra the registration was updated and the runtime `MissingExtraError` was not, so the
    message told you to install the extra marker had just been taken out of.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    declared = set(_declared_extras(root))

    pattern = re.compile(r'MissingExtraError\(\s*[^,]+,\s*"([a-z0-9-]+)"')
    named: dict[str, set[str]] = {}
    for source in (root / "src").rglob("*.py"):
        for extra in pattern.findall(source.read_text(encoding="utf-8")):
            named.setdefault(extra, set()).add(str(source.relative_to(root)))

    assert named, "found no MissingExtraError calls at all -- has the pattern drifted?"
    unknown = {extra: sorted(files) for extra, files in named.items() if extra not in declared}
    assert not unknown, f"errors name extras that do not exist: {unknown}"


def test_every_lazily_registered_package_is_in_the_extra_it_names() -> None:
    """The other half of the same promise. A registration can name a real extra that does not
    contain its package -- which is exactly how `unstructured` and `marker` went wrong."""
    from pathlib import Path

    from contextgrid.chunk import CHUNKERS
    from contextgrid.embed import EMBEDDERS
    from contextgrid.index import INDEXES
    from contextgrid.parse import PARSERS
    from contextgrid.rerank import RERANKERS
    from contextgrid.tokens import TOKENIZERS

    root = Path(__file__).resolve().parents[2]
    extras = _declared_extras(root)

    def base(requirement: str) -> str:
        return re.split(r"[<>=!\[;]", requirement, maxsplit=1)[0].strip().lower()

    wrong: list[str] = []
    for registry in (PARSERS, CHUNKERS, EMBEDDERS, INDEXES, RERANKERS, TOKENIZERS):
        for entry in registry:
            if not entry.extra or not entry.package:
                continue
            provided = {base(req) for req in extras.get(entry.extra, [])}
            if entry.package.lower() not in provided:
                wrong.append(
                    f"{entry.name!r} needs {entry.package!r} but [{entry.extra}] provides "
                    f"{sorted(provided)}"
                )

    assert not wrong, "\n".join(wrong)


# ---------------------------------------------------------------------------
# the two front doors have to reach the same things
# ---------------------------------------------------------------------------


def test_the_python_api_reaches_every_axis_the_config_file_does() -> None:
    """`Lab.grid()` had seven of the ten axes.

    Ingestion, retrieval and generation were reachable from a YAML file and not from Python, so
    half the plugins in the package were invisible to anybody using the library directly. The
    axis list grew three times and this signature did not follow, which is exactly the kind of
    drift nothing was watching for.
    """
    import inspect

    from contextgrid.grid.matrix import AXIS_ORDER
    from contextgrid.lab import Lab

    accepted = set(inspect.signature(Lab.grid).parameters)
    missing = [axis for axis in AXIS_ORDER if axis not in accepted]
    assert not missing, f"Lab.grid() cannot set: {missing}"


def test_the_python_api_reaches_every_run_setting_the_config_file_does() -> None:
    """The same drift on the other half. A `run:` key with no way to say it from Python is a
    feature half the users cannot use."""
    import inspect

    from contextgrid.config.schema import RunConfig
    from contextgrid.lab import Lab

    # `cache` is a Lab constructor argument, `resolution_*` belong to the scorer, and `k` is a
    # property of the matrix rather than of the run.
    elsewhere = {"cache", "resolution_policy", "resolution_threshold", "k"}
    wanted = set(RunConfig.KNOWN) - elsewhere

    reachable = (
        set(inspect.signature(Lab.run).parameters)
        | set(inspect.signature(Lab.__init__).parameters)
        | {"mode"}
    )
    missing = sorted(wanted - reachable)
    assert not missing, f"no way to set {missing} through Lab"


def test_a_model_given_to_the_lab_reaches_the_stages_that_need_one() -> None:
    """Four of the six transforms, `agentic` retrieval, four of the eight ingestion strategies
    and the `llm` generator cannot be built without a model. Before this, Python users could
    reach none of them."""
    from contextgrid.evalset.llm import LiteLLMChat
    from contextgrid.lab import Lab
    from contextgrid.pipeline import Config, build

    scripted = LiteLLMChat(
        model="scripted", transport=lambda prompt, limit: "a hypothetical answer"
    )
    lab = Lab({"a.md": "# Refunds\n\nRefunds are issued within 30 days.\n"}, model=scripted)

    assert lab.llm is scripted
    pipeline = build(
        Config(chunker="recursive:128", index="bm25", embedder=None, transform="hyde"),
        lab.corpus,
        llm=lab.llm,
    )
    assert pipeline.transform.name == "hyde"


def test_a_model_named_as_a_string_is_resolved() -> None:
    from contextgrid.lab import Lab

    lab = Lab({"a.md": "text"}, model="openai:gpt-4o-mini")
    assert lab.llm is not None
    assert lab.llm.name == "openai/gpt-4o-mini"


def test_the_labs_seed_reaches_the_results() -> None:
    """So a significance verdict from the Python API is reproducible from the same number the
    manifest records, rather than from a hidden zero."""
    from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
    from contextgrid.lab import Lab

    lab = Lab({"a.md": "# Refunds\n\nRefunds are issued within 30 days.\n"}, seed=42)
    lab.grid(chunker="recursive:128", index="bm25", embedder=None, k=3)

    evalset = EvalSet(
        id="seeded",
        items=(
            EvalItem(
                id="q1",
                question="refunds?",
                anchors=(GoldAnchor(quote="within 30 days", source_id="a.md"),),
            ),
        ),
    )
    assert lab.run(evalset, headline="recall@3").seed == 42
