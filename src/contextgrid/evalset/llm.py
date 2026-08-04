"""The narrowest possible LLM interface.

One method. Everything this package asks a model to do -- write a question, classify one,
answer one without context -- is a string in and a string out, and anything richer would be
an abstraction over provider SDKs that this project has no business maintaining.

Providers are registered lazily, so `import contextgrid` never imports an SDK and a missing
one produces an install instruction rather than a traceback.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from contextgrid.core.errors import ContextGridError, MissingExtraError
from contextgrid.core.registry import Registry


class LLMError(ContextGridError, RuntimeError):
    """A model call failed, or returned something unusable."""


@runtime_checkable
class LLM(Protocol):
    """Text in, text out."""

    @property
    def name(self) -> str: ...

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str: ...


LLMS: Registry[LLM] = Registry(family="llm")


@dataclass(slots=True)
class RecordingLLM:
    """A model that returns scripted replies and remembers what it was asked.

    Not a mock hidden in the tests: generation and filtering both need a model, and being
    able to exercise them without a network or an API key is what keeps that code covered
    rather than hoped about.
    """

    replies: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    default: str = ""
    name: str = "recording"

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        del max_tokens
        self.prompts.append(prompt)
        if self.replies:
            return self.replies.pop(0)
        return self.default


@dataclass(frozen=True, slots=True)
class OpenAIChat:
    """OpenAI chat completions. Bring your own key."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    api_key: str | None = None

    @property
    def name(self) -> str:
        return self.model

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise MissingExtraError("The OpenAI provider", "llm", package="openai") from exc

        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError(
                "no OpenAI key. Pass `api_key=` or set OPENAI_API_KEY. Keys are never "
                "written to disk or into a run manifest."
            )

        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


@dataclass(frozen=True, slots=True)
class AnthropicChat:
    """Anthropic messages. Bring your own key."""

    model: str = "claude-sonnet-5"
    temperature: float = 0.0
    api_key: str | None = None

    @property
    def name(self) -> str:
        return self.model

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise MissingExtraError("The Anthropic provider", "llm", package="anthropic") from exc

        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError(
                "no Anthropic key. Pass `api_key=` or set ANTHROPIC_API_KEY. Keys are never "
                "written to disk or into a run manifest."
            )

        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text"))


LLMS.register("openai", shorthand="model", doc="OpenAI chat completions (bring your own key).")(
    OpenAIChat
)
LLMS.register("anthropic", shorthand="model", doc="Anthropic messages (bring your own key).")(
    AnthropicChat
)


def get_llm(spec: str | LLM) -> LLM:
    """Resolve a model from a spec like `openai:gpt-4o-mini`, or pass an instance through."""
    return LLMS.create(spec) if isinstance(spec, str) else spec


# ---------------------------------------------------------------------------
# getting structured data back out
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_reply(reply: str) -> Any:
    """Pull JSON out of a model's answer, fenced or not.

    Models wrap JSON in code fences, prefix it with "Here is the JSON:", and occasionally do
    both. Insisting on clean output would mean discarding usable replies, so this is
    forgiving about the wrapper and strict about the content.
    """
    text = reply.strip()
    if not text:
        raise LLMError("the model returned nothing")

    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost bracketed region, which handles a prose preamble.
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise LLMError(f"could not find JSON in the model's reply: {reply[:200]!r}")


def answerer_from(llm: LLM) -> Callable[[str], str]:
    """A closed-book answerer, for the general-knowledge filter.

    Explicitly told to guess rather than decline, because a model that says "I don't know"
    would let every question through and quietly disable the filter.
    """

    def answer(question: str) -> str:
        return llm.complete(
            "Answer this question from your own knowledge, in one short sentence. "
            "Do not say that you lack context -- give your best guess.\n\n"
            f"Question: {question}\nAnswer:",
            max_tokens=100,
        )

    return answer


def batched(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    """Split a sequence into fixed-size batches."""
    if size < 1:
        raise ValueError(f"batch size must be at least 1, got {size}")
    return [items[start : start + size] for start in range(0, len(items), size)]
