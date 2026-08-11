"""Generation, and whether retrieval gains survived to the answer."""

from __future__ import annotations

from collections.abc import Callable

from contextgrid.core.registry import Registry
from contextgrid.evalset.llm import LLM, LLMError
from contextgrid.generate.answer import (
    DEFAULT_PROMPT,
    Answer,
    AnswerScore,
    ExtractiveGenerator,
    GenerationReport,
    Generator,
    LLMGenerator,
    lift,
    score_answer,
)
from contextgrid.generate.judge import METRICS as GENERATION_METRICS
from contextgrid.generate.judge import (
    GenerationJudge,
    JudgedAnswer,
    JudgeError,
    available_generation_metrics,
)

GENERATORS: Registry[Generator] = Registry(family="generator")

GENERATORS.register(
    "extractive",
    shorthand="sentences",
    doc="Return the top passage verbatim. The ceiling retrieval alone can reach.",
)(ExtractiveGenerator)


#: The one generator that needs a model, and cannot be registered because of it.
#:
#: Kept as a name list so `contextgrid plugins` and the config template can *say it exists*
#: even though building it needs an `LLM` a spec string alone cannot supply.
MODEL_BACKED: tuple[str, ...] = ("llm",)


def available_generators() -> tuple[str, ...]:
    """Every generator, whether or not it needs a model."""
    return tuple(sorted({*GENERATORS.names(), *MODEL_BACKED}))


def get_generator(spec: str | Generator | None, llm: LLM | None = None) -> Generator | None:
    """Resolve a generator, supplying the model to the one that needs it.

    `None` means no generation at all -- the axis is simply switched off, exactly as it was
    before this axis existed: no assembly, no answer, no cost. That is different from every
    other axis's `None`, which still builds an identity plugin (`NoTransform`, `NoReranker`);
    generation has nothing to be the identity *of*.

    `llm` is not in the registry for the reason `hyde` and the rest of `transform.MODEL_BACKED`
    are not: a generator built with no model would silently have nothing to generate with, and
    a configuration that looks like it is testing an LLM generator while testing nothing is
    worse than an error.
    """
    if spec is None:
        return None
    if not isinstance(spec, str):
        return spec

    name, _, _tail = spec.partition(":")
    if name in GENERATORS:
        return GENERATORS.create(spec)

    if llm is None:
        raise LLMError(
            f"the {name!r} generator needs a model. Set `run.model` in your config, or use "
            f"one of the model-free generators: {', '.join(GENERATORS.names())}"
        )

    builders: dict[str, Callable[[], Generator]] = {"llm": lambda: LLMGenerator(llm=llm)}
    if name not in builders:
        raise LLMError(
            f"unknown generator {name!r}. Available: "
            f"{', '.join(sorted({*GENERATORS.names(), *builders}))}"
        )
    return builders[name]()


# `JudgedAnswer` is deliberately not `AnswerScore`: that one records whether an extractive
# answer matched its reference, this one records what a judge model thought of a generated one.
__all__ = [
    "DEFAULT_PROMPT",
    "GENERATION_METRICS",
    "GENERATORS",
    "MODEL_BACKED",
    "Answer",
    "AnswerScore",
    "ExtractiveGenerator",
    "GenerationJudge",
    "GenerationReport",
    "Generator",
    "JudgeError",
    "JudgedAnswer",
    "LLMGenerator",
    "available_generation_metrics",
    "available_generators",
    "get_generator",
    "lift",
    "score_answer",
]
