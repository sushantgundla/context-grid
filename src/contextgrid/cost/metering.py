"""Counting what a model actually consumed, so generation can be priced.

The cost model could price an embedder from the moment it existed, because embedding is a
count of tokens somebody already had to compute. Generation was different: `LLM.complete()`
returns a bare string, so nothing downstream knew how long the prompt was or how much came
back, and a configuration calling `gpt-4o-mini` fifteen times was costed at exactly zero.

That was not a rounding error, it was three separate lies at once. The leaderboard printed
`$/1k queries = 0.0000` for a configuration talking to OpenAI. The summary paragraph said
"it runs locally at no cost per query" about the same run. And `budget_usd` -- the one guard
against a sweep quietly spending real money -- compared its limit against a total that could
never grow, so any positive budget was unlimited. Only `budget_usd: 0.0` did anything, by
stopping before the first configuration.

`MeteredLLM` wraps any `LLM` and counts both sides of every call. It is a proxy rather than a
protocol change: `complete()` still returns a string, so every existing implementation, every
test double and every `transport=` hook keeps working untouched.

**Prompt tokens are counted exactly; completion tokens are counted from the text that came
back.** Neither is the provider's own usage number, and the difference is real -- a provider
counts its own template and tool scaffolding, which we cannot see. The count here is a floor,
and `metered` stays true only when an exact tokenizer was available to produce it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contextgrid.core.protocols import Tokenizer


@dataclass(slots=True)
class Usage:
    """What passed through a model, in tokens.

    Prompt and completion are kept apart because they are not priced the same -- output runs
    three to five times the price of input on most providers, so a single "tokens" number
    would misprice anything with short questions and long answers.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    #: False once anything was counted with an approximate tokenizer. A cost built from
    #: approximate counts is a guess, and the report says so rather than printing a dollar
    #: figure that looks measured.
    exact: bool = True

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls
        self.exact = self.exact and other.exact

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "calls": self.calls,
            "exact": self.exact,
        }


@dataclass(slots=True)
class MeteredLLM:
    """An `LLM` that remembers what went through it.

    Wraps rather than replaces, so anything already satisfying the protocol can be metered
    without being modified -- including the scripted doubles the tests use, which is what lets
    the metering itself be tested with no network and no key.
    """

    inner: object
    tokenizer: Tokenizer | None = None
    usage: Usage = field(default_factory=Usage)

    @property
    def name(self) -> str:
        return str(getattr(self.inner, "name", "llm"))

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        reply = self.inner.complete(prompt, max_tokens=max_tokens)  # type: ignore[attr-defined]
        self.usage.add(
            Usage(
                prompt_tokens=self._count(prompt),
                completion_tokens=self._count(reply),
                calls=1,
                exact=self.tokenizer is not None and bool(self.tokenizer.exact),
            )
        )
        return str(reply)

    def _count(self, text: str) -> int:
        """Tokens in one string, exactly where possible.

        Falls back to a crude characters-over-four rule with `exact` set false, rather than
        refusing to count. A cost flagged as approximate is more use than no cost at all, and
        the alternative -- reporting zero -- is what this module exists to stop.
        """
        if not text:
            return 0
        if self.tokenizer is not None:
            return int(self.tokenizer.count(text))
        return max(1, len(text) // 4)


def exact_tokenizer_or_none(encoding: str = "cl100k_base") -> Tokenizer | None:
    """The byte-pair tokenizer if `tiktoken` is installed, otherwise nothing.

    Never raises. Metering is a side-effect of running a sweep, and a missing optional
    dependency must not fail a run that would otherwise succeed -- it downgrades the cost to
    approximate and says so.
    """
    try:
        from contextgrid.tokenizers_tiktoken import TiktokenTokenizer

        tokenizer = TiktokenTokenizer(encoding)
        tokenizer.count("warm")  # Force the vocabulary load here, not mid-sweep.
    except Exception:
        return None
    return tokenizer
