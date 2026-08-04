"""Generating an eval set from a corpus.

The single biggest barrier to measuring retrieval is that nobody has labelled data, and
writing a hundred questions by hand against your own documents is a day nobody spends. So
this generates a first draft -- and then, unlike everything else in the field, hands it to
the filters and the review queue rather than presenting it as ground truth.

Every generated question carries its evidence as a **quoted passage**, not a chunk id. That
is what makes the resulting eval set portable across parsers, and it is also a useful
constraint on the generator: it has to point at the text it used, which makes an invented
question detectable rather than merely suspected.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

from contextgrid.core.documents import Chunk
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.core.warnings import Severity, WarningCode, WarningLog
from contextgrid.evalset.llm import LLM, LLMError, parse_json_reply

_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")
_WORD = re.compile(r"\w+", re.UNICODE)

PROMPT = """\
You are helping build an evaluation set for a document retrieval system.

Below is one passage from a larger corpus. Write {count} question(s) that can be answered
ONLY by reading this specific passage.

Rules:
- Each question must be answerable from this passage alone, and NOT from general knowledge.
- Each question must stand on its own. A reader who has not seen the passage must know what
  it is asking about, so name things rather than saying "it" or "this".
- Quote the exact sentence or phrase from the passage that answers it. Copy it verbatim.
- Do not ask about the passage itself ("what does this section say"). Ask about its content.

Return JSON only, in this shape:
[{{"question": "...", "quote": "...", "answer": "..."}}]

Passage:
\"\"\"
{passage}
\"\"\"
"""


@runtime_checkable
class QuestionGenerator(Protocol):
    """Turns a chunk into candidate questions with quoted evidence."""

    @property
    def name(self) -> str: ...

    def generate(self, chunk: Chunk) -> list[EvalItem]: ...


@dataclass(slots=True)
class LLMQuestionGenerator:
    """Asks a model for questions answerable only from one passage.

    The quote requirement does most of the work. A model that has to point at the sentence
    it used produces answerable questions far more often than one asked only for questions --
    and when it invents a quote instead, the anchor fails to resolve and the filter catches
    it, which is a detectable failure rather than a silent one.
    """

    llm: LLM
    questions_per_chunk: int = 1
    max_tokens: int = 600

    name: ClassVar[str] = "llm"

    def generate(self, chunk: Chunk) -> list[EvalItem]:
        prompt = PROMPT.format(count=self.questions_per_chunk, passage=chunk.text)
        try:
            payload = parse_json_reply(self.llm.complete(prompt, max_tokens=self.max_tokens))
        except LLMError:
            return []

        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            return []

        items: list[EvalItem] = []
        for index, record in enumerate(payload):
            if not isinstance(record, dict):
                continue
            question = str(record.get("question", "")).strip()
            quote = str(record.get("quote", "")).strip()
            if not question or not quote:
                continue

            items.append(
                EvalItem(
                    id=f"{chunk.id}#{index}",
                    question=question,
                    anchors=(GoldAnchor(source_id=chunk.doc_id, quote=quote),),
                    answer=str(record.get("answer", "")).strip() or None,
                    meta={"generated_by": self.llm.name, "from_chunk": chunk.id},
                )
            )
        return items


@dataclass(slots=True)
class KeywordProbeGenerator:
    """Builds keyword probes from a passage's most distinctive terms. No model required.

    Deliberately not called a question generator, because these are not questions. They are
    bags of the rarest words in a passage, used as queries.

    That makes them useful for exactly one thing: proving a pipeline is wired up and
    measuring roughly nothing else. Real retrieval is hard because a user's phrasing does not
    match the document's, and a probe built *from* the document's own vocabulary skips that
    entire difficulty. Scores against these will be far higher than anything real.

    Use it to smoke-test. Do not publish a leaderboard built on it.
    """

    terms: int = 6
    seed: int = 0
    corpus_frequencies: dict[str, int] = field(default_factory=dict)

    name: ClassVar[str] = "keyword-probe"

    def fit(self, chunks: Sequence[Chunk]) -> None:
        """Learn how common each word is, so probes can favour the rare ones."""
        frequencies: dict[str, int] = {}
        for chunk in chunks:
            for word in set(_words(chunk.text)):
                frequencies[word] = frequencies.get(word, 0) + 1
        self.corpus_frequencies = frequencies

    def generate(self, chunk: Chunk) -> list[EvalItem]:
        sentence = _longest_sentence(chunk.text)
        if not sentence:
            return []

        distinctive = sorted(
            {word for word in _words(sentence) if len(word) > 3},
            key=lambda word: (self.corpus_frequencies.get(word, 1), word),
        )[: self.terms]
        if len(distinctive) < 2:
            return []

        return [
            EvalItem(
                id=f"{chunk.id}#probe",
                question=" ".join(distinctive),
                anchors=(GoldAnchor(source_id=chunk.doc_id, quote=sentence),),
                meta={"generated_by": self.name, "from_chunk": chunk.id, "probe": True},
            )
        ]


@dataclass(slots=True)
class Generation:
    """A draft eval set and everything worth knowing about how it was made."""

    evalset: EvalSet
    warnings: WarningLog = field(default_factory=WarningLog)
    chunks_sampled: int = 0
    chunks_skipped: int = 0

    @property
    def count(self) -> int:
        return len(self.evalset)


def generate(
    chunks: Sequence[Chunk],
    generator: QuestionGenerator,
    *,
    sample: int | None = 50,
    seed: int = 0,
    min_chunk_words: int = 25,
    evalset_id: str = "generated",
) -> Generation:
    """Draft an eval set from a corpus's chunks.

    Chunks are sampled rather than exhausted, and sampled *spread across documents* rather
    than uniformly: a corpus of one long document and nine short ones would otherwise
    produce an eval set almost entirely about the long one, and then report that every
    configuration is good at long documents.

    The result is a draft. It has not been filtered and no human has seen it, and the
    warnings say so.
    """
    log = WarningLog()
    usable = [chunk for chunk in chunks if len(_WORD.findall(chunk.text)) >= min_chunk_words]
    skipped = len(chunks) - len(usable)

    if not usable:
        log.add(
            WarningCode.SMALL_EVAL_SET,
            f"none of the {len(chunks)} chunks has {min_chunk_words} words, so there is "
            "nothing to write questions about. Try a larger chunk size",
            severity=Severity.INVALID,
            stage="evalset",
        )
        return Generation(EvalSet(id=evalset_id, items=()), log, 0, skipped)

    chosen = _spread_sample(usable, sample, seed) if sample else usable

    fit = getattr(generator, "fit", None)
    if callable(fit):
        # Corpus-statistical generators need to see everything before drafting anything.
        fit(usable)

    items: list[EvalItem] = []
    for chunk in chosen:
        items.extend(generator.generate(chunk))

    if not items:
        log.add(
            WarningCode.SMALL_EVAL_SET,
            f"the {generator.name!r} generator produced no questions from {len(chosen)} chunks",
            severity=Severity.INVALID,
            stage="evalset",
        )

    log.add(
        WarningCode.SMALL_EVAL_SET,
        f"{len(items)} questions drafted from {len(chosen)} chunks by {generator.name!r}. "
        "Nothing has filtered them and nobody has read them, so they are not ground truth "
        "yet -- run the filters, then the review queue",
        severity=Severity.INFO,
        stage="evalset",
    )

    return Generation(
        evalset=EvalSet(
            id=evalset_id,
            items=tuple(items),
            source=f"generated:{generator.name}",
            meta={"generator": generator.name, "seed": seed},
        ),
        warnings=log,
        chunks_sampled=len(chosen),
        chunks_skipped=skipped,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _spread_sample(chunks: Sequence[Chunk], size: int, seed: int) -> list[Chunk]:
    """Sample across documents rather than across chunks.

    Uniform sampling over chunks gives every document a share proportional to its length, so
    one long document dominates the eval set and the results describe that document.
    """
    if size >= len(chunks):
        return list(chunks)

    by_document: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.doc_id, []).append(chunk)

    rng = random.Random(seed)
    for group in by_document.values():
        rng.shuffle(group)

    chosen: list[Chunk] = []
    documents = sorted(by_document)
    position = 0
    while len(chosen) < size:
        added = False
        for doc_id in documents:
            group = by_document[doc_id]
            if position < len(group):
                chosen.append(group[position])
                added = True
                if len(chosen) == size:
                    break
        if not added:
            break
        position += 1

    return sorted(chosen, key=lambda chunk: (chunk.doc_id, chunk.char_start))


def _longest_sentence(text: str) -> str:
    sentences = [match.group(0).strip() for match in _SENTENCE.finditer(text)]
    return max(sentences, key=len, default="").strip()


def _words(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(text)]
