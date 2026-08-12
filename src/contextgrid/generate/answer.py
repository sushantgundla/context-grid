"""Generation, and whether better retrieval actually produced a better answer.

Retrieval stays the default view of this tool for a reason: generation noise swamps retrieval
signal, and a sweep judged on answer quality mostly measures the generator. But retrieval is
not the goal, it is a means, and a tool that never checks whether its gains survive to the
answer is asking to be trusted about the one thing it did not measure.

So generation is a panel, and the chart that matters is the lift one: does +0.10 recall@5
become a better answer, or does the generator find it either way?

Three things are scored, and none of them needs a second model to judge:

**Groundedness** -- is the answer's content actually in the context, or invented?
**Citation accuracy** -- do the passages it cited actually support what it said?
**Abstention** -- when the evidence is absent, does it say so instead of guessing?

That last one is almost never measured, and a system that confidently answers questions its
corpus cannot support is worse than one that scores lower and declines.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

from contextgrid.assemble.context import AssembledContext
from contextgrid.core.documents import Chunk
from contextgrid.core.evalset import EvalItem
from contextgrid.evalset.llm import LLM

_WORD = re.compile(r"\w+", re.UNICODE)
_CITATION = re.compile(r"\[(\d+)\]")

#: Phrases a model uses when it declines. Deliberately broad: a false positive costs a
#: mislabelled abstention, a false negative hides the failure mode entirely.
_REFUSALS = (
    "i don't know",
    "i do not know",
    "not enough information",
    "insufficient information",
    "cannot be determined",
    "cannot answer",
    "does not say",
    "is not stated",
    "is not mentioned",
    "no information",
    "not specified",
    "unable to answer",
    "the context does not",
    "do not contain the answer",
    "does not contain the answer",
    "passages do not",
)

DEFAULT_PROMPT = """\
Answer the question using only the passages below.
Cite the passages you used by their number, like [1].
If the passages do not contain the answer, say so plainly rather than guessing.

Passages:
{context}

Question: {question}
Answer:"""


@dataclass(frozen=True, slots=True)
class Answer:
    """What a generator said, and what it cost."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    citations: tuple[int, ...] = ()

    @property
    def is_abstention(self) -> bool:
        """Whether the model declined rather than guessed.

        A correct refusal is a success, not a zero. Almost no eval measures it that way,
        which is why systems that confidently answer unanswerable questions keep shipping.
        """
        lowered = self.text.lower()
        return any(phrase in lowered for phrase in _REFUSALS)


@runtime_checkable
class Generator(Protocol):
    """Turns a question and its context into an answer."""

    @property
    def name(self) -> str: ...

    def answer(self, question: str, context: AssembledContext) -> Answer: ...


@dataclass(slots=True)
class LLMGenerator:
    """Answers with a model, using a prompt template that is itself a sweepable axis.

    Prompt changes routinely beat retrieval changes, which is an uncomfortable result and a
    valuable one -- it is worth knowing before spending a quarter on an embedding migration.
    """

    llm: LLM
    prompt: str = DEFAULT_PROMPT
    max_tokens: int = 400

    name: ClassVar[str] = "llm"

    def answer(self, question: str, context: AssembledContext) -> Answer:
        filled = self.prompt.format(context=context.text, question=question)
        text = self.llm.complete(filled, max_tokens=self.max_tokens).strip()
        return Answer(
            text=text,
            prompt_tokens=context.tokens,
            citations=tuple(sorted({int(n) for n in _CITATION.findall(text)})),
        )


@dataclass(frozen=True, slots=True)
class ExtractiveGenerator:
    """Returns the highest-ranked passage verbatim. No model required.

    Not a generator in any useful sense, and that is the point: it is the ceiling retrieval
    alone can reach. An answer-quality score against it separates "the retriever found the
    evidence" from "the generator did something useful with it", which is the distinction
    the lift chart exists to draw.
    """

    sentences: int = 2

    name: ClassVar[str] = "extractive"

    def answer(self, question: str, context: AssembledContext) -> Answer:
        del question
        if not context.chunks:
            return Answer(text="The passages do not contain the answer.")

        from contextgrid.chunk.sentence import sentence_ranges

        text = context.chunks[0].text
        ranges = sentence_ranges(text)[: self.sentences]
        extracted = " ".join(text[start:end] for start, end in ranges) or text
        return Answer(text=extracted, prompt_tokens=context.tokens, citations=(1,))


# ---------------------------------------------------------------------------
# scoring an answer
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AnswerScore:
    """How good one answer was, judged against the context and the gold evidence."""

    item_id: str
    groundedness: float = 0.0
    citation_accuracy: float | None = None
    evidence_overlap: float = 0.0
    abstained: bool = False
    should_have_abstained: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def abstention_correct(self) -> bool:
        """True when the model's decision to answer or decline matched the evidence.

        Scored as a success either way. A system that declines when the corpus cannot
        support an answer is behaving correctly, and marking that as a zero teaches exactly
        the wrong lesson.
        """
        return self.abstained == self.should_have_abstained


def score_answer(
    item: EvalItem,
    answer: Answer,
    context: AssembledContext,
    gold_chunks: Sequence[Chunk] = (),
) -> AnswerScore:
    """Judge an answer without a second model.

    Deliberately lexical. An LLM judge is more sensitive and introduces a second model whose
    biases nobody has measured, into a tool whose entire premise is that unmeasured
    assumptions are the problem. These are coarse and they are checkable.
    """
    score = AnswerScore(
        item_id=item.id,
        abstained=answer.is_abstention,
        # `is_resolved`, not `is_answerable`: abstention is about what this parse could
        # actually find. An item whose anchor exists but which this parser lost has no
        # evidence to answer from, so declining is the right behaviour and must score as
        # such. `is_answerable` is true for that item -- it carries an anchor -- and would
        # mark the abstention wrong.
        should_have_abstained=not item.is_resolved or not context.chunks,
    )

    answer_words = _words(answer.text)
    if not answer_words:
        score.warnings.append("the generator returned nothing")
        return score

    context_words = _words(context.text)
    if context_words:
        # Content words in the answer that are not in the context are either invention or
        # general knowledge. Both are reasons to trust the answer less.
        grounded = len(answer_words & context_words) / len(answer_words)
        score.groundedness = grounded
        if grounded < 0.5 and not score.abstained:
            score.warnings.append(
                f"only {grounded:.0%} of the answer's words appear in the context it was "
                "given, so most of it came from somewhere else"
            )

    if gold_chunks:
        gold_words = _words(" ".join(chunk.text for chunk in gold_chunks))
        if gold_words:
            score.evidence_overlap = len(answer_words & gold_words) / len(gold_words)

    if answer.citations:
        valid = [n for n in answer.citations if 1 <= n <= len(context.chunks)]
        score.citation_accuracy = len(valid) / len(answer.citations)
        if len(valid) < len(answer.citations):
            score.warnings.append(
                f"cited passage(s) that were not in the context: "
                f"{sorted(set(answer.citations) - set(valid))}"
            )

    return score


@dataclass(slots=True)
class GenerationReport:
    """Answer quality across an eval set, and whether retrieval gains reached the answer."""

    scores: list[AnswerScore] = field(default_factory=list)
    generator: str = "unknown"

    def mean(self, attribute: str) -> float:
        values = [
            getattr(score, attribute)
            for score in self.scores
            if getattr(score, attribute) is not None
        ]
        return sum(values) / len(values) if values else 0.0

    @property
    def abstention_accuracy(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s.abstention_correct) / len(self.scores)

    @property
    def confident_when_it_should_not_be(self) -> list[str]:
        """Questions the corpus could not answer, which the model answered anyway.

        The failure worth naming. A system that does this is worse than one scoring lower
        and declining, and no retrieval metric will ever show it.
        """
        return [s.item_id for s in self.scores if s.should_have_abstained and not s.abstained]

    def metrics(self) -> dict[str, float]:
        return {
            "groundedness": self.mean("groundedness"),
            "citation_accuracy": self.mean("citation_accuracy"),
            "evidence_overlap": self.mean("evidence_overlap"),
            "abstention_accuracy": self.abstention_accuracy,
        }

    def summary(self) -> str:
        if not self.scores:
            return "No answers were generated."

        metrics = self.metrics()
        lines = [
            f"{self.generator} answered {len(self.scores)} questions. "
            f"{metrics['groundedness']:.0%} of the average answer's words came from the "
            f"context it was given."
        ]

        overconfident = self.confident_when_it_should_not_be
        if overconfident:
            lines.append(
                f"{len(overconfident)} question(s) had no supporting evidence in the "
                "retrieved context and were answered anyway. That is worse than a lower "
                "score with a refusal, and no retrieval metric shows it."
            )
        else:
            lines.append(
                "It declined on every question whose evidence was missing, which is the "
                "correct behaviour and is almost never measured."
            )

        return " ".join(lines)


def lift(retrieval_score: float, answer_score: float, baseline_answer: float) -> str:
    """Whether a retrieval gain survived to the answer.

    The question the whole project implicitly promises to answer and which nothing in the
    field plots. A retrieval improvement that the generator would have compensated for is
    real and worth nothing.
    """
    gain = answer_score - baseline_answer
    if abs(gain) < 0.01:
        return (
            f"Retrieval scored {retrieval_score:.3f}, and answer quality is unchanged against "
            "the baseline. The generator was finding the answer either way, so this retrieval "
            "gain bought nothing."
        )
    if gain > 0:
        return (
            f"Retrieval scored {retrieval_score:.3f} and answer quality rose {gain:+.3f}. "
            "The retrieval gain survived to the answer."
        )
    return (
        f"Retrieval scored {retrieval_score:.3f} and answer quality *fell* {gain:+.3f}. "
        "Better retrieval that produces worse answers usually means more context, not better "
        "context -- check character precision before believing the retrieval number."
    )


def _words(text: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(text)}
