"""One configuration, end to end.

A grid cell is this: parse, chunk, embed, index, search, score. Everything the grid does on
top -- expanding axes, reusing work, ranking results -- is bookkeeping around this loop.

Two things it is careful about.

**Ground truth is re-resolved per parse.** Two parsers produce different text, so the same
authored eval set has to be located again in each one. Skipping that is how a parser
comparison silently becomes nonsense.

**Every stage is timed and cached separately.** Sweeping rerankers should not re-embed
anything, and the only way to be sure it did not is to count.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any

from contextgrid.cache.store import Cache, CacheStats, cache_key, cached
from contextgrid.chunk import get_chunker
from contextgrid.core.documents import Chunk, ParsedDocument, SourceFile
from contextgrid.core.evalset import EvalSet, Qrels
from contextgrid.core.protocols import Chunker, Parser
from contextgrid.core.warnings import Severity, WarningCode, WarningLog
from contextgrid.corpus import Corpus
from contextgrid.embed import Embedder, get_embedder
from contextgrid.evalset.llm import LLM
from contextgrid.index.base import Index, Scored
from contextgrid.ingest import Ingested, IngestionContext, get_ingester
from contextgrid.parse import get_parser
from contextgrid.rerank import Reranker, get_reranker
from contextgrid.retrieve import (
    RetrievalStrategy,
    RetrievalTrace,
    SimpleRetrieval,
    get_retriever,
)
from contextgrid.score.anchor import AnchorResolver
from contextgrid.score.resolve import SpanResolver
from contextgrid.transform import NoTransform, QueryTransform, get_transform


@dataclass(frozen=True, slots=True)
class Config:
    """One point in the grid.

    Written as spec strings so a configuration is readable, serialisable and pasteable --
    `Config("markdown", "recursive:512,overlap=64", "tfidf", "dense")` is the whole thing.
    """

    parser: str = "markdown"
    chunker: str = "recursive:512"
    embedder: str | None = "tfidf"
    index: str = "dense"
    transform: str | None = None
    #: How the index is used, as opposed to what it is. `None` means plain search.
    retrieval: str | None = None
    reranker: str | None = None
    k: int = 10
    #: How many results the retriever hands the reranker. The parameter most reranking
    #: advice omits, and where most of the effect lives: over the top 10 a reranker can only
    #: reorder what was already found, over the top 100 it can rescue what ranked 47th.
    candidates: int = 50
    #: How a file becomes something the parser can read. `None` hands the bytes on unchanged.
    #:
    #: Last, not first, even though it runs first: `Config("markdown", "recursive:512", ...)`
    #: is public API and putting a new field ahead of `parser` would silently shift every
    #: positional argument anybody has already written.
    ingestion: str | None = None

    @property
    def label(self) -> str:
        """A short identifier that reads well in a leaderboard row."""
        parts = []
        if self.ingestion and self.ingestion != "direct":
            parts.append(f"{self.ingestion}>")
        parts += [self.parser, self.chunker]
        if self.embedder:
            parts.append(self.embedder)
        if self.transform:
            parts.append(f"+{self.transform}")
        if self.retrieval and self.retrieval != "simple":
            parts.append(f"~{self.retrieval}")
        parts.append(self.index)
        if self.reranker:
            parts.append(f"{self.reranker}@{self.candidates}")
        return " · ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ingestion": self.ingestion,
            "parser": self.parser,
            "chunker": self.chunker,
            "embedder": self.embedder,
            "index": self.index,
            "transform": self.transform,
            "retrieval": self.retrieval,
            "reranker": self.reranker,
            "k": self.k,
            "candidates": self.candidates,
        }

    def with_(self, **changes: Any) -> Config:
        return replace(self, **changes)


@dataclass(slots=True)
class Timings:
    """Wall-clock per stage, in milliseconds.

    Kept per stage rather than as one total, because "this config is slow" is not actionable
    and "this config spends 90% of its time in the reranker" is.
    """

    parse_ms: float = 0.0
    chunk_ms: float = 0.0
    embed_ms: float = 0.0
    index_ms: float = 0.0
    query_ms: list[float] = field(default_factory=list)

    @property
    def build_ms(self) -> float:
        return self.parse_ms + self.chunk_ms + self.embed_ms + self.index_ms

    def percentile(self, fraction: float) -> float:
        """Query latency at a percentile. p95 is the number users feel; the mean hides it."""
        if not self.query_ms:
            return 0.0
        ordered = sorted(self.query_ms)
        position = min(int(fraction * len(ordered)), len(ordered) - 1)
        return ordered[position]

    def as_dict(self) -> dict[str, float]:
        return {
            "parse_ms": self.parse_ms,
            "chunk_ms": self.chunk_ms,
            "embed_ms": self.embed_ms,
            "index_ms": self.index_ms,
            "build_ms": self.build_ms,
            "query_p50_ms": self.percentile(0.50),
            "query_p95_ms": self.percentile(0.95),
            "query_p99_ms": self.percentile(0.99),
        }


@dataclass(slots=True)
class BuiltPipeline:
    """A configuration that has read the corpus and is ready to answer queries."""

    config: Config
    parses: dict[str, ParsedDocument]
    chunks: list[Chunk]
    index: Index
    embedder: Embedder | None
    timings: Timings
    warnings: WarningLog
    index_bytes: int = 0
    embed_tokens: int = 0
    reranker: Reranker | None = None
    transform: QueryTransform = field(default_factory=NoTransform)
    retrieval: RetrievalStrategy = field(default_factory=SimpleRetrieval)
    #: What was indexed, what is retrievable, and the map between them.
    ingested: Ingested | None = None
    #: What the strategies did, accumulated across every query. Two configurations with the
    #: same recall and different model-call counts are a decision, not a tie.
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)

    def search(self, query: str, k: int | None = None) -> list[str]:
        """Chunk ids for one query, best first.

        With a reranker, the retriever is asked for `candidates` results and the reranker
        cuts them down to `k`. Without one, the retriever is asked for `k` directly -- so the
        no-reranker arm never pays for candidates it would have thrown away.

        *How* those results are gathered is the retrieval strategy's business: one search, a
        wider one, several fused, or a model deciding as it goes. The strategy never sees the
        index, so every strategy works with every store.
        """
        limit = k or self.config.k
        rewritten = self.transform.transform(query)
        depth = max(self.config.candidates, limit) if self.reranker else limit

        def searcher(text: str, wanted: int) -> Sequence[Scored]:
            # Asks for more than it needs: several indexed units can stand for the same
            # passage -- four generated questions for one chunk -- so a top-`wanted` of indexed
            # hits can collapse to far fewer distinct passages.
            depth = wanted * self._fan_out()
            hits = self.index.search(text, self._vector_for(text), depth)
            return self._to_retrievable(hits, wanted)

        ranked = self.retrieval.retrieve(query, rewritten.queries, searcher, depth, self.trace)

        if self.reranker is None:
            return [scored.chunk_id for scored in ranked[:limit]]

        by_id = self.chunk_by_id()
        candidates = [by_id[scored.chunk_id] for scored in ranked if scored.chunk_id in by_id]
        return [scored.chunk_id for scored in self.reranker.rerank(query, candidates, limit)]

    def _fan_out(self) -> int:
        """How many indexed units it takes, on average, to reach one distinct passage."""
        if self.ingested is None:
            return 1
        return max(1, min(8, round(self.ingested.expansion)))

    def _to_retrievable(self, hits: Sequence[Scored], wanted: int) -> list[Scored]:
        """Turn hits on indexed units into the passages they stand for.

        Deduplicated, keeping the best score, because two questions generated for the same
        chunk both matching is one passage found twice -- and counting it twice would fill the
        result list with a single passage while claiming to have found several.
        """
        if self.ingested is None:
            return list(hits[:wanted])

        best: dict[str, float] = {}
        for hit in hits:
            target = self.ingested.resolve(hit.chunk_id)
            if target not in best or hit.score > best[target]:
                best[target] = hit.score

        merged = self._merge_siblings(best)
        ranked = sorted(merged.items(), key=lambda pair: (-pair[1], pair[0]))
        return [Scored(chunk_id, score) for chunk_id, score in ranked[:wanted]]

    def scored_ids(self, returned: Sequence[str]) -> list[str]:
        """What the returned passages count as when scored, in order and without repeats.

        A presentation passage counts as the units it covers. Without this a strategy that
        hands a generator more context would be scored on ids the qrels have never heard of,
        and would appear to have found nothing at all.
        """
        if self.ingested is None:
            return list(returned)

        out: list[str] = []
        seen: set[str] = set()
        for chunk_id in returned:
            for scored in self.ingested.scored_ids(chunk_id):
                if scored not in seen:
                    seen.add(scored)
                    out.append(scored)
        return out

    def _merge_siblings(self, best: dict[str, float]) -> dict[str, float]:
        """Replace a run of sibling leaves with their parent, once enough of them have hit.

        The one place an ingestion strategy decides at query time. A single leaf matching means
        that line held the answer; most of a passage matching means the passage did, and
        returning it whole is better than returning three fragments of it.
        """
        if self.ingested is None:
            return best
        children = self.ingested.notes.get("children")
        if not isinstance(children, dict):
            return best

        raw = self.ingested.notes.get("threshold", 0.5)
        threshold = float(raw) if isinstance(raw, (int, float)) else 0.5
        merged = dict(best)

        for parent, leaves in children.items():
            hit = [leaf for leaf in leaves if leaf in merged]
            if leaves and len(hit) / len(leaves) >= threshold:
                merged[parent] = max(merged[leaf] for leaf in hit)
                for leaf in hit:
                    merged.pop(leaf, None)
        return merged

    def _vector_for(self, text: str) -> Any:
        if self.embedder is None or not self.index.needs_vectors:
            return None
        return self.embedder.embed_queries([text]).vectors[0]

    def run_queries(self, evalset: EvalSet, k: int | None = None) -> dict[str, list[str]]:
        """Answer every question, recording how long each one took."""
        run: dict[str, list[str]] = {}
        for item in evalset:
            started = time.perf_counter()
            run[item.id] = self.scored_ids(self.search(item.question, k))
            self.timings.query_ms.append((time.perf_counter() - started) * 1000)
        return run

    def chunk_by_id(self) -> dict[str, Chunk]:
        """Every chunk that can come back, including presentation passages.

        Reranking and generation both look chunks up here, and both should see the wider
        passage when a strategy chose to hand one over.
        """
        found = {chunk.id: chunk for chunk in self.chunks}
        if self.ingested is not None:
            found.update(self.ingested.presented_chunks)
        return found


def build(
    config: Config,
    corpus: Corpus,
    *,
    cache: Cache | None = None,
    stats: CacheStats | None = None,
    llm: LLM | None = None,
) -> BuiltPipeline:
    """Run a configuration's indexing side: parse, chunk, embed, index."""
    warnings = WarningLog()
    timings = Timings()
    parser = get_parser(config.parser)
    chunker = get_chunker(config.chunker)

    parses = _parse_all(parser, list(corpus), cache, stats, timings, warnings)
    chunks = _chunk_all(chunker, parses, cache, stats, timings)

    # Ingestion decides what is indexed and what a hit on it returns. For plain chunking the
    # two are the same list; every other strategy deliberately breaks that identity.
    started_ingest = time.perf_counter()
    ingested = get_ingester(config.ingestion).ingest(
        chunks, IngestionContext(parses=dict(parses), warnings=warnings, llm=llm)
    )
    timings.chunk_ms += (time.perf_counter() - started_ingest) * 1000

    if not chunks:
        warnings.add(
            WarningCode.EMPTY_CHUNK_SET,
            f"{config.label} produced no chunks at all, so every query will score zero. "
            "Either the parser found no text or the chunker rejected all of it",
            severity=Severity.INVALID,
            stage="chunk",
            subject=config.label,
        )

    embedder, vectors, embed_tokens = _embed_all(
        config, ingested.indexed, cache, stats, timings, warnings
    )

    started = time.perf_counter()
    index = _make_index(config)
    index.build(ingested.indexed, vectors)
    timings.index_ms = (time.perf_counter() - started) * 1000

    return BuiltPipeline(
        config=config,
        parses=parses,
        # Everything downstream -- reranking, generation, scoring -- works on what comes *back*,
        # which is not what went in unless the strategy is plain.
        chunks=ingested.retrievable,
        ingested=ingested,
        index=index,
        embedder=embedder,
        timings=timings,
        warnings=warnings,
        index_bytes=index.size_bytes(),
        embed_tokens=embed_tokens,
        reranker=get_reranker(config.reranker) if config.reranker else None,
        # The model has to reach here. Four of the five transforms -- HyDE, multi-query,
        # decompose, step-back -- cannot be built without one, so a config naming any of them
        # raised "needs a model" from a place the user had no way to influence. They were
        # unreachable from the config file, which is the primary interface.
        transform=get_transform(config.transform, llm),
        retrieval=get_retriever(config.retrieval),
    )


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def _parse_all(
    parser: Parser,
    corpus: Sequence[SourceFile],
    cache: Cache | None,
    stats: CacheStats | None,
    timings: Timings,
    warnings: WarningLog,
) -> dict[str, ParsedDocument]:
    started = time.perf_counter()
    parses: dict[str, ParsedDocument] = {}

    for source in corpus:
        if not parser.supports(source.media_type):
            warnings.add(
                WarningCode.PARSER_FALLBACK,
                f"{parser.name!r} does not read {source.media_type.value}, so {source.id!r} "
                "is not in this index at all. Nothing in it can be retrieved",
                severity=Severity.CAUTION,
                stage="parse",
                subject=source.id,
            )
            continue

        key = cache_key(
            "parse",
            parser.version,
            {"parser": parser.name, **params_of(parser)},
            [source.content_hash()],
        )
        parsed = cached(cache, key, "parse", lambda s=source: parser.parse(s), stats)
        parses[source.id] = parsed
        warnings.extend(parsed.warnings)

    timings.parse_ms = (time.perf_counter() - started) * 1000
    return parses


def _chunk_all(
    chunker: Chunker,
    parses: Mapping[str, ParsedDocument],
    cache: Cache | None,
    stats: CacheStats | None,
    timings: Timings,
) -> list[Chunk]:
    started = time.perf_counter()
    chunks: list[Chunk] = []

    for parsed in parses.values():
        key = cache_key(
            "chunk",
            chunker.version,
            {"chunker": chunker.name, **params_of(chunker)},
            [parsed.text_hash()],
        )
        produced = cached(cache, key, "chunk", lambda p=parsed: chunker.chunk(p), stats)
        chunks.extend(produced)

    timings.chunk_ms = (time.perf_counter() - started) * 1000
    return chunks


def _embed_all(
    config: Config,
    chunks: Sequence[Chunk],
    cache: Cache | None,
    stats: CacheStats | None,
    timings: Timings,
    warnings: WarningLog,
) -> tuple[Embedder | None, Any, int]:
    if config.embedder is None:
        return None, None, 0

    embedder = get_embedder(config.embedder)
    started = time.perf_counter()

    texts = [chunk.text for chunk in chunks]
    embedder.prepare(texts)

    key = cache_key(
        "embed",
        embedder.version,
        {"embedder": embedder.name, **params_of(embedder)},
        [texts_hash(texts)],
    )
    result = cached(cache, key, "embed", lambda: embedder.embed_documents(texts), stats)
    warnings.extend(result.warnings)

    timings.embed_ms = (time.perf_counter() - started) * 1000
    return embedder, result.vectors, result.input_tokens


def _make_index(config: Config) -> Index:
    from contextgrid.index import get_index

    return get_index(config.index)


def params_of(plugin: object) -> dict[str, Any]:
    """A plugin's configuration, for the cache key.

    Private attributes are skipped. A fitted TF-IDF vocabulary is derived from the corpus,
    which is already in the key via the text hash, and including it would make the key depend
    on the thing it is supposed to identify.
    """
    if not is_dataclass(plugin):
        return {}
    return {f.name: getattr(plugin, f.name) for f in fields(plugin) if not f.name.startswith("_")}


def texts_hash(texts: Sequence[str]) -> str:
    """A hash of an ordered list of texts. The chunk set's identity, for caching."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# scoring one configuration
# ---------------------------------------------------------------------------


def resolve_evalset(
    evalset: EvalSet,
    parses: Mapping[str, ParsedDocument],
    resolver: AnchorResolver | None = None,
) -> tuple[EvalSet, WarningLog]:
    """Locate the eval set's evidence inside this configuration's parse.

    Skipped when the eval set carries spans rather than anchors -- those were authored
    against a fixed text and are already correct for it.
    """
    if not any(item.anchors for item in evalset):
        return evalset, WarningLog()
    return (resolver or AnchorResolver()).resolve(evalset, parses)


def build_qrels(
    evalset: EvalSet, chunks: Sequence[Chunk], resolver: SpanResolver | None = None
) -> tuple[Qrels, WarningLog]:
    """Turn span-level gold into chunk-level judgements for this chunking."""
    span_resolver = resolver or SpanResolver()
    resolutions, log = span_resolver.resolve(evalset, chunks)
    qrels = {
        item_id: resolution.as_qrel()
        for item_id, resolution in resolutions.items()
        if resolution.labels
    }
    return qrels, log
