"""Generation metrics, through DeepEval.

Every dimension before this one is scored on whether the right passages came back. This one
asks the question the passages were retrieved *for*: is the answer any good, and is it actually
supported by what was retrieved?

They are different failures and they need separating. A configuration can retrieve perfectly and
generate a confident falsehood; it can retrieve badly and be saved by a model that says it does
not know. Reporting one number for both is how a retrieval problem gets misdiagnosed as a
prompting problem for a fortnight.

**DeepEval rather than our own prompts.** These metrics are prompt-and-parse, and writing four
prompts is not the hard part -- agreeing on what "faithful" means is, and DeepEval's definitions
are ones a reader can look up and argue with. Four names that mean something published beat four
that mean whatever this package decided.

**One model, one key, one budget.** DeepEval reaches for its own OpenAI configuration by
default, which would put a second, unpriced model call in the middle of a package whose whole
argument is that cost belongs on the chart. `_ContextGridJudge` hands it whatever the config
already chose, so `openai:gpt-4o-mini` in the YAML is the judge too, its calls are counted, and
`budget_usd` still means something.

The judge is never the model under test unless somebody asks for that. A model grading its own
answers scores them generously, and the effect is largest on exactly the answers worth doubting.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from contextgrid.core.errors import ContextGridError, MissingExtraError


class JudgeError(ContextGridError, RuntimeError):
    """A generation metric could not be computed."""


@dataclass(slots=True)
class JudgedAnswer:
    """What the judges said about one answer.

    Named apart from `generate.answer.AnswerScore`, which is a different thing entirely: that
    one records whether an extractive answer matched, this one records what a judge model
    thought of a generated one.
    """

    query_id: str
    scores: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    model_calls: int = 0
    failed: dict[str, str] = field(default_factory=dict)


#: The metrics worth having, and what each one catches that the others do not.
#:
#: Deliberately not all fifty DeepEval offers. These four are the ones that fail *differently*:
#: a wrong answer, an unsupported answer, an evasive answer, and a retrieval problem wearing a
#: generation problem's clothes.
METRICS: dict[str, str] = {
    # Is every claim in the answer supported by what was retrieved? The hallucination check,
    # and the one that needs no reference answer -- which makes it the only generation metric
    # usable on a corpus nobody has written answers for.
    "faithfulness": "FaithfulnessMetric",
    # Does the answer address the question, rather than being true and beside the point?
    "answer_relevancy": "AnswerRelevancyMetric",
    # Were the retrieved passages relevant to the question? A generation-time view of a
    # retrieval failure -- and the one that tells you which half to go and fix.
    "contextual_relevancy": "ContextualRelevancyMetric",
    # Did the retrieved passages contain what the reference answer needed? Needs a reference.
    "contextual_recall": "ContextualRecallMetric",
}

#: Metrics that cannot run without a written answer to compare against.
NEEDS_REFERENCE: frozenset[str] = frozenset({"contextual_recall"})


def _deepeval() -> Any:
    # Set before the import, because DeepEval reads this at module scope -- opting out
    # afterwards is too late. Turning on generation metrics used to start sending analytics to
    # PostHog with nothing in this package's docs or output saying so; the only clue was a
    # stray `[PostHog] analytics lane flush ran out of budget` line on stderr. Somebody
    # measuring a private corpus is entitled to know that, and more than entitled to have it
    # off by default.
    #
    # `setdefault`, so anybody who actively wants DeepEval's telemetry can still have it by
    # exporting `DEEPEVAL_TELEMETRY_OPT_OUT=NO` themselves.
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    try:
        import deepeval  # noqa: F401
        from deepeval import metrics
    except ImportError as exc:  # pragma: no cover - exercised by the extras test
        raise MissingExtraError("Generation scoring", "judge", package="deepeval") from exc
    return metrics


def build_judge(llm: Any) -> Any:
    """Wrap one of our models in the interface DeepEval expects.

    Built here rather than at import time because it subclasses a DeepEval type, and this
    package must import without DeepEval installed.
    """
    try:
        from deepeval.models.base_model import DeepEvalBaseLLM
    except ImportError as exc:  # pragma: no cover - exercised by the extras test
        raise MissingExtraError("Generation scoring", "judge", package="deepeval") from exc

    # The ignore list covers both toolchains deliberately. With DeepEval installed the base
    # class is real and mypy wants `no-untyped-call`; without it the class is `Any` and mypy
    # wants `misc`. `unused-ignore` stops each one complaining about the other's code.
    class _ContextGridJudge(DeepEvalBaseLLM):  # type: ignore[misc, no-untyped-call, unused-ignore]
        """Presents a contextgrid `LLM` as a DeepEval judge.

        Counts its own calls, because a judge that grades a thousand answers is a real expense
        and one this package refuses to leave off the chart.
        """

        def __init__(self) -> None:
            self.inner = llm
            self.calls = 0

        def load_model(self) -> Any:
            return self.inner

        def get_model_name(self) -> str:
            return str(getattr(self.inner, "name", "contextgrid-judge"))

        def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            del args, kwargs
            self.calls += 1
            return str(self.inner.complete(prompt, max_tokens=1024) or "")

        async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            # Our LLM protocol is synchronous by design -- one method, text in and text out.
            # DeepEval asks for an async path; running the sync one satisfies it without
            # inventing concurrency this package does not otherwise have.
            return self.generate(prompt, *args, **kwargs)

    return _ContextGridJudge()


@dataclass(slots=True)
class GenerationJudge:
    """Scores generated answers on the metrics that were asked for.

    `async_mode` is off. DeepEval defaults to running its judges concurrently, which is faster
    and makes the order of model calls non-deterministic -- and a sweep whose numbers move
    between identical runs is a sweep nobody can trust to compare anything.
    """

    llm: Any
    metrics: tuple[str, ...] = ("faithfulness", "answer_relevancy")
    threshold: float = 0.5

    name: ClassVar[str] = "deepeval"
    version: ClassVar[str] = "1"

    _judge: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        unknown = set(self.metrics) - set(METRICS)
        if unknown:
            raise JudgeError(
                f"unknown generation metric(s): {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(sorted(METRICS))}"
            )

    def judge(self) -> Any:
        if self._judge is None:
            self._judge = build_judge(self.llm)
        return self._judge

    def score(
        self,
        *,
        query_id: str,
        question: str,
        answer: str,
        contexts: Sequence[str],
        reference: str | None = None,
    ) -> JudgedAnswer:
        """Score one answer on every requested metric.

        A metric that raises is recorded and skipped rather than failing the run. A judge model
        that refuses one awkward question must not discard the other nine hundred answers it
        graded perfectly well.
        """
        metrics_module = _deepeval()
        from deepeval.test_case import LLMTestCase

        judge = self.judge()
        before = getattr(judge, "calls", 0)

        case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=list(contexts),
            expected_output=reference,
        )

        result = JudgedAnswer(query_id=query_id)
        for name in self.metrics:
            if name in NEEDS_REFERENCE and not reference:
                result.failed[name] = "needs a reference answer, and this question has none"
                continue

            try:
                metric = getattr(metrics_module, METRICS[name])(
                    threshold=self.threshold,
                    model=judge,
                    async_mode=False,
                    include_reason=True,
                )
                metric.measure(case)
            except Exception as error:
                result.failed[name] = str(error)[:200]
                continue

            score = getattr(metric, "score", None)
            if score is None:
                result.failed[name] = "the metric returned no score"
                continue
            result.scores[name] = float(score)
            reason = getattr(metric, "reason", None)
            if reason:
                result.reasons[name] = str(reason)

        result.model_calls = getattr(judge, "calls", 0) - before
        return result


def available_generation_metrics() -> tuple[str, ...]:
    return tuple(sorted(METRICS))
