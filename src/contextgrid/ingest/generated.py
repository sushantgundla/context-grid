"""Ingestion strategies that pay a model at index time.

A different bargain from anything on the retrieval axis. These call a model **once per chunk
while building the index** and never again -- so on a corpus answering a thousand questions a
day the cost is amortised to nothing, and on one answering three it is the dominant expense.
That trade is invisible on a recall chart, which is why the cost columns exist.

All four write text that is not in the document, and index *that*:

* `contextual` prepends an explanation of where the chunk sits (Anthropic's result: 67% fewer
  retrieval failures, the strongest published number on this axis).
* `summary` indexes a summary of a whole document and returns the document.
* `hypothetical-questions` indexes the questions a chunk answers, on the reasoning that a
  question embeds closer to a question than a statement does.
* `propositions` indexes atomic facts, so a chunk covering six topics stops being a vector
  that means none of them.

**The written text is indexed and never returned.** The retrievable side is always the original
chunk, with its offsets intact, so gold evidence resolves exactly as it does for every other
arm. A strategy that returned LLM-written text would be scoring the model's paraphrase against
the document, which measures nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from contextgrid.core.documents import Chunk
from contextgrid.core.warnings import Severity, WarningCode
from contextgrid.ingest.base import Ingested, IngestionContext, IngestionError

CONTEXT_PROMPT = """\
Here is a document:
<document>
{document}
</document>

Here is a chunk taken from it:
<chunk>
{chunk}
</chunk>

Write one or two short sentences placing this chunk in the document, so that it can be found
by someone searching without the rest of the document in front of them. Name the section it
belongs to and what it is about. Answer with the sentences alone."""

QUESTIONS_PROMPT = """\
Here is a passage:
<passage>
{chunk}
</passage>

Write the {count} questions this passage answers most directly. Use the wording someone
searching would use, not the passage's own wording.

Reply with a JSON array of strings and nothing else."""

PROPOSITIONS_PROMPT = """\
Here is a passage:
<passage>
{chunk}
</passage>

Break it into atomic facts. Each fact must stand alone -- replace pronouns with what they refer
to -- and must be stated in the passage rather than inferred from it. At most {count}.

Reply with a JSON array of strings and nothing else."""

SUMMARY_PROMPT = """\
Here is a document:
<document>
{document}
</document>

Summarise what it covers in a short paragraph, naming the topics someone might search for.
Answer with the summary alone."""


@dataclass(frozen=True, slots=True)
class _GeneratedIngestion:
    """Shared body: call the model per unit, survive its failures, count what it cost."""

    model: str = "openai:gpt-4o-mini"
    max_document_chars: int = 12_000

    name: ClassVar[str] = "generated"
    version: ClassVar[str] = "1"
    uses_model: ClassVar[bool] = True

    def _llm(self, context: IngestionContext) -> Any:
        if context.llm is not None:
            return context.llm
        from contextgrid.evalset.llm import get_llm

        return get_llm(self.model)

    def _ask(
        self, llm: Any, prompt: str, context: IngestionContext, subject: str, *, limit: int = 256
    ) -> str:
        """One model call. A failure is recorded and skipped, never raised.

        Half an index is worse than a slow one, and a provider hiccup two thousand chunks into
        a build must not throw the first nineteen hundred away. The chunk falls back to being
        indexed as itself, and the warning says how many did.
        """
        try:
            return str(llm.complete(prompt, max_tokens=limit) or "").strip()
        except Exception as error:
            context.warnings.add(
                WarningCode.NON_DETERMINISTIC_STAGE,
                f"the {self.name} ingestion strategy could not enrich {subject!r}: {error}. "
                "That chunk was indexed as written, so this row mixes two strategies",
                severity=Severity.CAUTION,
                stage="ingest",
                subject=subject,
            )
            return ""

    def _document_text(self, chunks: Sequence[Chunk], doc_id: str) -> str:
        """The document a chunk came from, reassembled and trimmed.

        Trimmed because the whole point is a cheap call per chunk, and a hundred-page contract
        in every prompt is neither cheap nor better -- the model needs enough to place the
        chunk, not the entire text.
        """
        pieces = [chunk.text for chunk in chunks if chunk.doc_id == doc_id]
        joined = "\n".join(pieces)
        return joined[: self.max_document_chars]


@dataclass(frozen=True, slots=True)
class ContextualIngestion(_GeneratedIngestion):
    """Prepend an LLM-written explanation of where the chunk sits, then index that.

    Anthropic's contextual retrieval. The problem it solves is specific and common: a chunk
    reading "the notice period is thirty days" is a perfect answer that no search for
    "termination notice under the services agreement" will ever find, because the words that
    would connect the two are in a heading four chunks earlier.

    One call per chunk. The context is indexed and thrown away; the chunk is what comes back.
    """

    name: ClassVar[str] = "contextual"
    version: ClassVar[str] = "1"

    def ingest(self, chunks: Sequence[Chunk], context: IngestionContext) -> Ingested:
        llm = self._llm(context)
        indexed: list[Chunk] = []
        calls = 0
        enriched = 0

        for chunk in chunks:
            prompt = CONTEXT_PROMPT.format(
                document=self._document_text(chunks, chunk.doc_id), chunk=chunk.text
            )
            written = self._ask(llm, prompt, context, chunk.id, limit=128)
            calls += 1

            if not written:
                indexed.append(chunk)
                continue

            enriched += 1
            indexed.append(
                _rewritten(chunk, f"{written}\n\n{chunk.text}", suffix="ctx"),
            )

        return Ingested(
            indexed=indexed,
            retrievable=list(chunks),
            parent_of={
                written.id: original.id
                for written, original in zip(indexed, chunks, strict=True)
                if written.id != original.id
            },
            model_calls=calls,
            notes={"enriched": enriched, "of": len(chunks)},
        )


@dataclass(frozen=True, slots=True)
class HypotheticalQuestionsIngestion(_GeneratedIngestion):
    """Index the questions a chunk answers, and return the chunk.

    A question embeds closer to a question than a statement does, which is the asymmetry every
    other arm on this axis fights. Instead of rewriting the query to look like a document --
    what HyDE does at query time, per query, forever -- this rewrites the document to look like
    a query, once.

    Several vectors per chunk, so the index grows by `count` and so does the embedding bill.
    """

    count: int = 3

    name: ClassVar[str] = "hypothetical-questions"
    version: ClassVar[str] = "1"

    def __post_init__(self) -> None:
        if self.count < 1:
            raise IngestionError(f"count must be at least 1, got {self.count}")

    def ingest(self, chunks: Sequence[Chunk], context: IngestionContext) -> Ingested:
        return _index_generated_text(
            self,
            chunks,
            context,
            prompt=lambda chunk: QUESTIONS_PROMPT.format(chunk=chunk.text, count=self.count),
            limit=self.count,
            suffix="q",
        )


@dataclass(frozen=True, slots=True)
class PropositionsIngestion(_GeneratedIngestion):
    """Index atomic facts, and return the chunk they came from.

    A chunk covering six topics has a vector meaning roughly none of them. Splitting it into
    standalone facts -- pronouns resolved, each one true on its own -- gives six vectors that
    each mean one thing, all pointing back at the passage that supports them.

    The most expensive of the four in index size, and the one that most changes what "a chunk"
    means for retrieval.
    """

    count: int = 6

    name: ClassVar[str] = "propositions"
    version: ClassVar[str] = "1"

    def __post_init__(self) -> None:
        if self.count < 1:
            raise IngestionError(f"count must be at least 1, got {self.count}")

    def ingest(self, chunks: Sequence[Chunk], context: IngestionContext) -> Ingested:
        return _index_generated_text(
            self,
            chunks,
            context,
            prompt=lambda chunk: PROPOSITIONS_PROMPT.format(chunk=chunk.text, count=self.count),
            limit=self.count,
            suffix="p",
        )


@dataclass(frozen=True, slots=True)
class SummaryIngestion(_GeneratedIngestion):
    """Index a summary of the whole document, and return the whole document.

    The coarsest strategy on the axis and the cheapest of the paid ones -- one call per
    *document* rather than per chunk. It answers a different question from the rest: not "which
    passage answers this?" but "which document is this about?", which is what a corpus of many
    short documents actually needs.

    What comes back is the entire document, so it costs context window rather than model calls.
    """

    name: ClassVar[str] = "summary"
    version: ClassVar[str] = "1"

    def ingest(self, chunks: Sequence[Chunk], context: IngestionContext) -> Ingested:
        llm = self._llm(context)
        indexed: list[Chunk] = []
        retrievable: list[Chunk] = []
        parent_of: dict[str, str] = {}
        calls = 0

        for doc_id in dict.fromkeys(chunk.doc_id for chunk in chunks):
            whole = _whole_document(chunks, doc_id)
            retrievable.append(whole)

            summary = self._ask(
                llm,
                SUMMARY_PROMPT.format(document=self._document_text(chunks, doc_id)),
                context,
                doc_id,
                limit=256,
            )
            calls += 1

            # A failed summary falls back to indexing the document itself, which is worse at
            # matching and still findable -- better than a document that cannot be retrieved.
            written = _rewritten(whole, summary or whole.text, suffix="summary")
            indexed.append(written)
            parent_of[written.id] = whole.id

        return Ingested(
            indexed=indexed,
            retrievable=retrievable,
            parent_of=parent_of,
            model_calls=calls,
            notes={"documents": len(retrievable)},
        )


# ---------------------------------------------------------------------------
# shared machinery
# ---------------------------------------------------------------------------


def _index_generated_text(
    strategy: _GeneratedIngestion,
    chunks: Sequence[Chunk],
    context: IngestionContext,
    *,
    prompt: Any,
    limit: int,
    suffix: str,
) -> Ingested:
    """Ask the model for several strings per chunk, and index each as its own vector."""
    from contextgrid.retrieve.agentic import _parse_queries

    llm = strategy._llm(context)
    indexed: list[Chunk] = []
    parent_of: dict[str, str] = {}
    calls = 0
    generated = 0

    for chunk in chunks:
        reply = strategy._ask(llm, prompt(chunk), context, chunk.id, limit=64 * limit)
        calls += 1
        written = _parse_queries(reply, limit)

        if not written:
            # Nothing usable came back, so the chunk is indexed as itself. It stays findable,
            # and the row is honestly a mix of two strategies -- which `_ask` already warned
            # about when the call failed outright.
            indexed.append(chunk)
            continue

        generated += len(written)
        for position, text in enumerate(written):
            indexed.append(_rewritten(chunk, text, suffix=f"{suffix}{position}"))
            parent_of[f"{chunk.id}:{suffix}{position}"] = chunk.id

    return Ingested(
        indexed=indexed,
        retrievable=list(chunks),
        parent_of=parent_of,
        model_calls=calls,
        notes={"generated": generated, "per_chunk": generated / len(chunks) if chunks else 0.0},
    )


def _rewritten(chunk: Chunk, text: str, *, suffix: str) -> Chunk:
    """A chunk whose text is not what the document says.

    `offsets_exact=False` is the honest flag, and it is load-bearing: this chunk carries the
    span it derives from so it can be traced back, and its text is not a slice of that span.
    Anything scoring against character offsets has to know the difference.
    """
    return Chunk(
        id=f"{chunk.id}:{suffix}",
        span=chunk.span,
        text=text,
        meta={**chunk.meta, "generated_for": chunk.id},
        token_counts={},
        offsets_exact=False,
    )


def _whole_document(chunks: Sequence[Chunk], doc_id: str) -> Chunk:
    """Every chunk of one document as a single retrievable unit."""
    from contextgrid.ingest.structural import _merge

    pieces = sorted(
        (chunk for chunk in chunks if chunk.doc_id == doc_id), key=lambda c: c.span.start
    )
    return _merge(pieces, suffix="document")
