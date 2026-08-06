"""Query transformation: rewriting the question before searching with it.

Retrieval fails most often because the user's phrasing and the document's phrasing do not
match. Somebody asks "how long before I can walk away?" and the contract says "termination
for convenience on thirty days written notice"; they share almost no words. Every transform
here is an attempt to close that gap.

They all cost an LLM call per query, sometimes several, and that cost lands on **every
query forever** rather than once at index time. So the interesting question is never "does
HyDE help?" but "does HyDE help enough to justify a call on every query?" -- and the answer is
frequently no, which is a finding worth publishing rather than a disappointment.

Every transform therefore reports its calls and its tokens, and the cost panel attributes them
to the configuration. A transform whose gain is real and whose cost is ruinous should look
exactly like what it is.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

from contextgrid.core.registry import Registry
from contextgrid.evalset.llm import LLM, LLMError, parse_json_reply

_SENTENCE_SPLIT = re.compile(r"\s*\n+\s*|\s*;\s*")


@dataclass(frozen=True, slots=True)
class TransformedQuery:
    """One question, and the queries actually sent to the index.

    More than one query means more than one search, whose results are fused. That is a real
    cost in latency as well as in tokens, and both are recorded here rather than inferred.
    """

    original: str
    queries: tuple[str, ...]
    llm_calls: int = 0
    llm_tokens: int = 0

    @property
    def is_identity(self) -> bool:
        return self.queries == (self.original,)

    @property
    def fan_out(self) -> int:
        """How many searches this question now costs."""
        return len(self.queries)


@runtime_checkable
class QueryTransform(Protocol):
    """Rewrites a question into one or more search queries."""

    @property
    def name(self) -> str: ...

    def transform(self, query: str) -> TransformedQuery: ...


@dataclass(frozen=True, slots=True)
class NoTransform:
    """Search with the question as asked. The arm every transform has to beat.

    Not a placeholder: the whole value of this axis is that most transforms do not clear
    their own cost, and that cannot be shown without the untransformed baseline on the same
    chart with the same cost column.
    """

    name: ClassVar[str] = "none"

    def transform(self, query: str) -> TransformedQuery:
        return TransformedQuery(original=query, queries=(query,))


@dataclass(slots=True)
class HyDE:
    """Search with a hypothetical *answer* rather than the question.

    The idea is neat: a question and its answer share little vocabulary, but a plausible
    fake answer and the real one share a great deal. So the model invents an answer and that
    is what gets embedded.

    It works best where the model knows the domain and worst where it does not -- on a corpus
    of internal jargon it invents confident nonsense that matches nothing. That is exactly the
    case this tool exists to distinguish, and it cannot be predicted from the outside.
    """

    llm: LLM
    include_question: bool = True
    max_tokens: int = 200

    name: ClassVar[str] = "hyde"

    prompt: str = (
        "Write a short passage that would answer this question, as if quoting a document. "
        "Do not hedge, do not say you lack context, and do not mention the question. "
        "Two or three sentences.\n\nQuestion: {query}\n\nPassage:"
    )

    def transform(self, query: str) -> TransformedQuery:
        try:
            hypothetical = self.llm.complete(
                self.prompt.format(query=query), max_tokens=self.max_tokens
            ).strip()
        except LLMError:
            return TransformedQuery(original=query, queries=(query,))

        if not hypothetical:
            return TransformedQuery(original=query, queries=(query,))

        # Keeping the question alongside the invention hedges the failure mode: when the
        # model has invented nonsense, the real question is still in the fused results.
        queries = (query, hypothetical) if self.include_question else (hypothetical,)
        return TransformedQuery(
            original=query,
            queries=queries,
            llm_calls=1,
            llm_tokens=len(hypothetical.split()),
        )


@dataclass(slots=True)
class MultiQuery:
    """Ask the same thing several ways and fuse the results.

    The most reliable of these transforms and the least clever: it does not need the model to
    know anything, only to paraphrase. The gain is usually small and usually real, and it
    costs one call plus `n` searches per query.
    """

    llm: LLM
    variants: int = 3
    max_tokens: int = 250

    name: ClassVar[str] = "multi-query"

    prompt: str = (
        "Rewrite this question {n} different ways, each using different words for the same "
        'thing. Return JSON only: ["...", "...", "..."]\n\nQuestion: {query}'
    )

    def transform(self, query: str) -> TransformedQuery:
        try:
            payload = parse_json_reply(
                self.llm.complete(
                    self.prompt.format(n=self.variants, query=query),
                    max_tokens=self.max_tokens,
                )
            )
        except LLMError:
            return TransformedQuery(original=query, queries=(query,))

        rewrites = _strings(payload)[: self.variants]
        if not rewrites:
            return TransformedQuery(original=query, queries=(query,))

        return TransformedQuery(
            original=query,
            queries=(query, *rewrites),
            llm_calls=1,
            llm_tokens=sum(len(text.split()) for text in rewrites),
        )


@dataclass(slots=True)
class Decompose:
    """Break a question into the sub-questions it actually contains.

    The only one of these that addresses a structural failure rather than a vocabulary one.
    "Which vendor has the shortest notice period and what is their monthly fee?" cannot be
    answered by any single passage, so no amount of better embedding will retrieve it -- the
    question has to become two.

    On simple factoids it is pure overhead, which is why it belongs on an axis rather than
    switched on by default.
    """

    llm: LLM
    max_parts: int = 3
    max_tokens: int = 250

    name: ClassVar[str] = "decompose"

    prompt: str = (
        "Break this question into the smallest set of standalone questions that together "
        "answer it. If it is already a single question, return it unchanged. At most "
        '{n}. Return JSON only: ["...", "..."]\n\nQuestion: {query}'
    )

    def transform(self, query: str) -> TransformedQuery:
        try:
            payload = parse_json_reply(
                self.llm.complete(
                    self.prompt.format(n=self.max_parts, query=query),
                    max_tokens=self.max_tokens,
                )
            )
        except LLMError:
            return TransformedQuery(original=query, queries=(query,))

        parts = _strings(payload)[: self.max_parts]
        if not parts:
            return TransformedQuery(original=query, queries=(query,))

        return TransformedQuery(
            original=query,
            queries=tuple(dict.fromkeys(parts)),
            llm_calls=1,
            llm_tokens=sum(len(text.split()) for text in parts),
        )


@dataclass(slots=True)
class StepBack:
    """Ask the more general question alongside the specific one.

    "What is Northwind's notice period?" becomes "what do the termination clauses say?" as
    well. It helps when the specific answer sits inside a passage about the general topic,
    and hurts when the general query drags in every document that mentions termination --
    which on a corpus of near-neighbours is most of them.
    """

    llm: LLM
    max_tokens: int = 120

    name: ClassVar[str] = "step-back"

    prompt: str = (
        "Write one more general question whose answer would contain the answer to this one. "
        "Return the question and nothing else.\n\nQuestion: {query}\n\nGeneral question:"
    )

    def transform(self, query: str) -> TransformedQuery:
        try:
            broader = self.llm.complete(
                self.prompt.format(query=query), max_tokens=self.max_tokens
            ).strip()
        except LLMError:
            return TransformedQuery(original=query, queries=(query,))

        if not broader or broader == query:
            return TransformedQuery(original=query, queries=(query,))

        return TransformedQuery(
            original=query,
            queries=(query, broader),
            llm_calls=1,
            llm_tokens=len(broader.split()),
        )


@dataclass(slots=True)
class ExpandAcronyms:
    """Spell out acronyms and abbreviations. No model required.

    Unglamorous, free, and it moves BM25 more than most of the clever transforms above. A
    corpus that says "recovery point objective" cannot be found by a query that says "RPO",
    and no embedding fixes that on a term the model has never seen.
    """

    expansions: dict[str, str] = field(default_factory=dict)

    name: ClassVar[str] = "expand"

    def transform(self, query: str) -> TransformedQuery:
        if not self.expansions:
            return TransformedQuery(original=query, queries=(query,))

        expanded = query
        for short, long in self.expansions.items():
            expanded = re.sub(rf"\b{re.escape(short)}\b", f"{short} {long}", expanded, flags=re.I)

        if expanded == query:
            return TransformedQuery(original=query, queries=(query,))
        return TransformedQuery(original=query, queries=(expanded,))


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

TRANSFORMS: Registry[QueryTransform] = Registry(family="transform")

TRANSFORMS.register("none", doc="Search with the question as asked. The arm to beat.")(NoTransform)
TRANSFORMS.register(
    "expand", doc="Spell out acronyms. Free, and it moves BM25 more than most of the rest."
)(ExpandAcronyms)


#: Transforms that need a model, and cannot be registered because of it.
#:
#: Kept as a name list so `contextgrid plugins` and the config template can *say they exist*.
#: They were previously invisible: reachable if you already knew the name, and mentioned
#: nowhere the tool prints -- so the axis appeared to have two arms when it has six.
MODEL_BACKED: tuple[str, ...] = ("hyde", "multi-query", "decompose", "step-back")


def available_transforms() -> tuple[str, ...]:
    """Every transform, whether or not it needs a model."""
    return tuple(sorted({*TRANSFORMS.names(), *MODEL_BACKED}))


def get_transform(spec: str | QueryTransform | None, llm: LLM | None = None) -> QueryTransform:
    """Resolve a transform, supplying the model to the ones that need one.

    The model-backed transforms are not in the registry because they cannot be built from a
    spec string alone -- a transform with no model would silently become the identity, and a
    configuration that looks like it is testing HyDE while testing nothing is worse than an
    error.
    """
    if spec is None:
        return NoTransform()
    if not isinstance(spec, str):
        return spec

    name, _, tail = spec.partition(":")
    if name in TRANSFORMS:
        return TRANSFORMS.create(spec)

    if llm is None:
        raise LLMError(
            f"the {name!r} transform needs a model. Pass one to `get_transform`, or use "
            f"one of the model-free transforms: {', '.join(TRANSFORMS.names())}"
        )

    builders: dict[str, Callable[[], QueryTransform]] = {
        "hyde": lambda: HyDE(llm=llm),
        "multi-query": lambda: MultiQuery(llm=llm, variants=int(tail or 3)),
        "decompose": lambda: Decompose(llm=llm, max_parts=int(tail or 3)),
        "step-back": lambda: StepBack(llm=llm),
    }
    if name not in builders:
        raise LLMError(
            f"unknown transform {name!r}. Available: "
            f"{', '.join(sorted({*TRANSFORMS.names(), *builders}))}"
        )
    return builders[name]()


def _strings(payload: object) -> list[str]:
    """Pull a list of strings out of whatever shape the model returned."""
    if isinstance(payload, str):
        return [line.strip() for line in _SENTENCE_SPLIT.split(payload) if line.strip()]
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                payload = value
                break
    if not isinstance(payload, list):
        return []
    return [str(item).strip() for item in payload if str(item).strip()]


def describe_cost(transformed: Sequence[TransformedQuery]) -> str:
    """What a transform cost across an eval set, in the terms that decide whether to use it.

    Index-time cost is paid once. This is paid on every query, forever, which is why a
    transform has to earn considerably more than it appears to.
    """
    if not transformed:
        return "no queries transformed"

    calls = sum(t.llm_calls for t in transformed)
    searches = sum(t.fan_out for t in transformed)
    if not calls:
        return f"no model calls; {searches / len(transformed):.1f} searches per question"

    return (
        f"{calls / len(transformed):.1f} model calls and "
        f"{searches / len(transformed):.1f} searches per question, on every query forever"
    )
