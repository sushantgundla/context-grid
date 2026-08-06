"""Agentic retrieval: a model decides what to search for, and when to stop.

The strategy the field talks about most and measures least. Every other arm on this axis
decides its searches in advance -- one query, a wider one, the question split on conjunctions.
An agent reads what came back and decides what to look for next, so the number of searches is
not knowable before the run.

That is exactly what makes it worth putting on a grid next to the free strategies. "Agentic RAG
improves retrieval" is a claim about a trade: recall against latency and dollars. Nobody can
check it on their own corpus without building both sides, and nothing builds both sides.

**The ranking comes from what the agent searched for, not from what it says.** The agent's
contribution is deciding the queries; the results are fused by rank across those searches. A
model asked to name its chosen chunk ids invents them, and an invented id is either a crash or
-- much worse -- a silent mismatch that scores as a miss. So the agent drives the searching and
the index still decides what matches.

Two backends, one behaviour: **agno** when it is installed, and a plain loop over this
package's own `LLM` protocol when it is not. The loop exists because a strategy that cannot run
without a heavy optional dependency is a strategy most people will never measure, and because
the comparison it enables is the point of the axis.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from contextgrid.index.base import Scored
from contextgrid.retrieve.base import Lookup, RetrievalTrace, Searcher, _no_lookup, fuse
from contextgrid.retrieve.strategies import RetrievalError

if TYPE_CHECKING:
    pass

#: What the model is told to do. Deliberately short: a long prompt buys agreement with the
#: prompt rather than better searching, and every token is paid for on every query.
PLAN_PROMPT = """\
You are helping search a document collection to answer a question.

Question: {question}

{seen}
Write the search queries most likely to find passages that answer it. Prefer specific wording
from the question over paraphrase. If the question has several parts, write one query per part.

Reply with a JSON array of strings and nothing else. At most {limit} queries."""

_SEEN_PREFIX = """\
Searches already run, and the first line of what each returned:
{summary}
Write only queries that would find something the searches above missed. If they already cover
the question, reply with an empty array."""


@dataclass(frozen=True, slots=True)
class AgenticRetrieval:
    """A model plans the searches, optionally in more than one round.

    `rounds=1` is one planning call per question: the model reads the question and writes the
    queries. `rounds=2` and above let it see what came back and search again for what is
    missing, which is where the interesting behaviour is and where the cost doubles.

    `max_queries` caps how many searches one round may produce, because a model asked for
    "the queries" will happily write nine, and nine searches per question across a sweep is
    how an afternoon becomes a week.
    """

    model: str = "openai:gpt-4o-mini"
    rounds: int = 1
    max_queries: int = 4
    backend: str = "auto"

    name: ClassVar[str] = "agentic"
    version: ClassVar[str] = "1"
    uses_model: ClassVar[bool] = True

    BACKENDS: ClassVar[tuple[str, ...]] = ("auto", "agno", "llm")

    _llm: Any = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise RetrievalError(f"agentic rounds must be at least 1, got {self.rounds}")
        if self.max_queries < 1:
            raise RetrievalError(f"max_queries must be at least 1, got {self.max_queries}")
        if self.backend not in self.BACKENDS:
            raise RetrievalError(
                f"unknown agentic backend {self.backend!r}. Choose one of: "
                f"{', '.join(self.BACKENDS)}"
            )

    # -- the model -----------------------------------------------------------

    def planner(self) -> Any:
        """The thing that turns a prompt into text. Built once, reused for every query."""
        if self._llm is not None:
            return self._llm

        built = self._build_agno() if self.backend in {"auto", "agno"} else None
        if built is None:
            if self.backend == "agno":
                raise RetrievalError(
                    "the agno backend needs agno. Install it with: "
                    "pip install 'context-grid[agent]'"
                )
            from contextgrid.evalset.llm import get_llm

            built = get_llm(self.model)

        object.__setattr__(self, "_llm", built)
        return built

    def _build_agno(self) -> Any:
        """An agno Agent, when agno is installed.

        Both layers reach models through litellm, so `openai:gpt-4o-mini` here and
        `litellm:text-embedding-3-small` on the embedder axis mean the same provider story --
        one key, one place it comes from.
        """
        try:
            from agno.agent import Agent
            from agno.models.litellm import LiteLLM
        except ImportError:
            return None

        provider, _, model = self.model.partition(":")
        qualified = model or provider
        if provider in {"openai", "anthropic"} and "/" not in qualified:
            qualified = f"{provider}/{qualified}"

        return _AgnoPlanner(
            Agent(
                model=LiteLLM(id=qualified),
                instructions=(
                    "Write search queries. Reply with a JSON array of strings and nothing else."
                ),
                markdown=False,
            )
        )

    # -- the strategy --------------------------------------------------------

    def retrieve(
        self,
        query: str,
        queries: Sequence[str],
        searcher: Searcher,
        k: int,
        trace: RetrievalTrace,
        lookup: Lookup = _no_lookup,
    ) -> list[Scored]:
        del queries, lookup  # the model plans from the question as asked
        planner = self.planner()

        results: list[Sequence[Scored]] = []
        searched: list[str] = []
        rounds_used = 0

        for round_number in range(self.rounds):
            planned = self._plan(planner, query, searched, results, trace)
            if not planned:
                # An empty plan on a later round means the model thinks it has enough. That is
                # a real signal and stopping is cheaper than one more round of nothing.
                break

            rounds_used = round_number + 1
            for text in planned:
                trace.record_search(text)
                searched.append(text)
                results.append(searcher(text, k))

        if not results:
            # The model failed or refused. Falling back to the question as asked is the only
            # honest option: returning nothing would score as a retrieval failure when what
            # failed was the planner.
            trace.record_search(query)
            trace.notes["fell_back"] = True
            results.append(searcher(query, k))

        trace.notes["rounds"] = rounds_used
        return fuse(results, k)

    def _plan(
        self,
        planner: Any,
        question: str,
        searched: Sequence[str],
        results: Sequence[Sequence[Scored]],
        trace: RetrievalTrace,
    ) -> list[str]:
        """Ask the model for the next batch of queries."""
        seen = ""
        if searched:
            lines = [
                f"- {text!r} -> {len(found)} result(s)"
                for text, found in zip(searched, results, strict=False)
            ]
            seen = _SEEN_PREFIX.format(summary="\n".join(lines)) + "\n\n"

        prompt = PLAN_PROMPT.format(question=question, seen=seen, limit=self.max_queries)

        try:
            reply = planner.complete(prompt, max_tokens=256)
        except Exception as error:
            # A planner that fails must not fail the sweep -- the fallback search below still
            # produces a result, and the trace says the planner was the part that broke.
            trace.notes["planner_error"] = str(error)[:200]
            return []
        finally:
            trace.record_model_call()

        return _parse_queries(reply, self.max_queries)


@dataclass(slots=True)
class _AgnoPlanner:
    """Presents an agno Agent through the one-method interface everything else here uses."""

    agent: Any

    def complete(self, prompt: str, *, max_tokens: int = 256) -> str:
        del max_tokens
        response = self.agent.run(prompt)
        content = getattr(response, "content", response)
        return str(content or "")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
#: A numbered or bulleted line. Anything else is prose, not a plan.
_LIST_ITEM = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+\S")


def _parse_queries(reply: str, limit: int) -> list[str]:
    """Read a list of queries out of whatever the model actually returned.

    Models wrap JSON in code fences, prefix it with "Here are the queries:", and sometimes give
    up on JSON and write a numbered list. Insisting on clean output would throw away usable
    plans and make the strategy look worse than it is -- so this is forgiving about the wrapper
    and strict about the result: strings only, stripped, deduplicated, capped.
    """
    text = reply.strip()
    if not text:
        return []

    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                parsed = None

    if parsed is None:
        # Last resort: a numbered or bulleted list, which is what a model does when it forgets
        # the format. Better than discarding a perfectly good plan over punctuation.
        #
        # Only when the lines actually *look* like a list, though. Without that check a refusal
        # -- "I'm sorry, I can't help with that." -- becomes a search query, and the strategy
        # quietly searches for the apology instead of falling back to the question.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not any(_LIST_ITEM.match(line) for line in lines):
            return []
        parsed = [
            line.lstrip("-*0123456789. ").strip(' "') for line in lines if _LIST_ITEM.match(line)
        ]

    if not isinstance(parsed, list):
        return []

    seen: set[str] = set()
    queries: list[str] = []
    for item in parsed:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        queries.append(cleaned)
        if len(queries) == limit:
            break
    return queries
