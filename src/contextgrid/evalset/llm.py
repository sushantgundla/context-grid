"""The narrowest possible LLM interface.

One method. Everything this package asks a model to do -- write a question, classify one,
answer one without context -- is a string in and a string out, and anything richer would be
an abstraction over provider SDKs that this project has no business maintaining.

Providers are registered lazily, so `import contextgrid` never imports an SDK and a missing
one produces an install instruction rather than a traceback.
"""

from __future__ import annotations

import json
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
class LiteLLMChat:
    """Any chat model, through litellm.

    One adapter instead of one per provider. `openai/gpt-4o-mini`, `anthropic/claude-sonnet-5`,
    `gemini/gemini-2.0-flash`, `ollama/llama3` and a hundred others are the same call with a
    different name, and litellm carries the per-provider quirks so this package does not have
    to grow an adapter every time somebody wants a model it has not heard of.

    The key comes from the environment. `api_key` exists for the caller who is managing keys
    themselves, and is never written to disk or into a run manifest.
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    api_key: str | None = None
    api_base: str | None = None
    timeout: float = 120.0
    #: Replaces the network call: prompt in, reply out. Set it to exercise anything that calls
    #: a model without a key or a network.
    transport: Callable[[str, int], str] | None = None

    @property
    def name(self) -> str:
        return self.model

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        if self.transport is not None:
            return self.transport(prompt, max_tokens)

        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise MissingExtraError("Model calls", "llm", package="litellm") from exc

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "timeout": self.timeout,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        try:
            response = litellm.completion(**kwargs)
        except Exception as error:
            raise LLMError(_explain(error, self.model)) from error

        try:
            return str(response.choices[0].message.content or "")
        except (AttributeError, IndexError, KeyError) as error:
            raise LLMError(
                f"{self.model} returned a reply this adapter could not read: {error}"
            ) from error


def _explain(error: Exception, model: str) -> str:
    """A provider exception, turned into something a person can act on."""
    text = str(error)
    lowered = text.lower()

    if "api" in lowered and "key" in lowered:
        return (
            f"no usable API key for {model}. Set the provider's key in the environment -- "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY and so on. Keys are never "
            f"written to disk or into a run manifest. ({text})"
        )
    if "not found" in lowered or "does not exist" in lowered:
        return (
            f"{model} was not recognised. litellm expects `provider/model` for anything that "
            f"is not OpenAI -- for example `anthropic/claude-sonnet-5`. ({text})"
        )
    if "rate limit" in lowered or "429" in lowered:
        return f"{model} is rate limiting. Slow the run down or use a smaller model. ({text})"
    if "connection" in lowered or "refused" in lowered:
        return f"could not reach the endpoint for {model}. Is it running? ({text})"
    return text


def _prefixed(prefix: str, default: str) -> Callable[..., LiteLLMChat]:
    """A provider-named factory, so `openai:gpt-4o-mini` keeps working.

    These were separate hand-written clients. They are now one adapter under three names --
    configs people have already written stay valid, and there is one code path to keep right
    rather than three. A bare `openai` still resolves, because an axis value that needs a
    parameter to be usable at all is a bad axis value.
    """

    def build(model: str = default, **kwargs: Any) -> LiteLLMChat:
        qualified = model if "/" in model else f"{prefix}/{model}"
        return LiteLLMChat(model=qualified, **kwargs)

    return build


LLMS.register(
    "litellm", shorthand="model", doc="Any chat model through litellm. Bring your own key."
)(LiteLLMChat)
LLMS.register("openai", shorthand="model", doc="OpenAI chat, through litellm.")(
    _prefixed("openai", "gpt-4o-mini")
)
LLMS.register("anthropic", shorthand="model", doc="Anthropic messages, through litellm.")(
    _prefixed("anthropic", "claude-sonnet-5")
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
