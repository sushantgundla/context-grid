"""Cross-encoder rerankers, local or hosted.

A cross-encoder reads the query and the passage *together*, so it can judge whether a passage
answers this question rather than whether it sits nearby in a vector space. That is why it
helps, and why it costs a model call per candidate rather than per query.

The same two backends as the embedders, for the same reasons:

* **TEI** exposes `/rerank` from the very server that serves `/embed`. Start one process, get
  both axes. No key, no network, no extra dependency -- it is reached over `urllib`.
* **litellm** reaches hosted rerankers (Cohere, Jina, Voyage, AWS) through one name.

Two things this module insists on, both of which decide whether the reranker axis means
anything at all.

**`candidates` is the parameter that matters.** Over the top 10 a reranker can only reorder
what was already found; over the top 100 it can rescue what ranked 47th. Most reranking advice
omits the number entirely, which is why "use a reranker" is such unreliable advice. Here it is
an axis, so the depth curve is something you measure rather than assume.

**Every candidate gets scored, or the run fails.** A backend that quietly returns fewer results
than it was given -- because a passage was too long, or a batch was capped -- silently drops
documents from the ranking. That looks like the reranker deciding they were bad, and it is not.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from contextgrid.core.documents import Chunk
from contextgrid.core.errors import ContextGridError
from contextgrid.index.base import Scored


class RerankerError(ContextGridError, RuntimeError):
    """A reranking backend could not be reached, or refused the request."""


#: One scored candidate as a backend returns it: its position in the list it was given, and
#: its relevance. Position rather than id, because that is what every rerank API speaks.
Ranking = list[tuple[int, float]]


@dataclass(frozen=True, slots=True)
class _RemoteReranker:
    """Shared body: batching, verification, retries, and turning positions back into ids."""

    model: str = ""
    api_base: str | None = None
    api_key_env: str | None = None
    timeout: float = 60.0
    retries: int = 2
    #: How many candidates go in one request. Cross-encoders are quadratic in nothing but
    #: linear in candidates, and a server will refuse a batch that is too large.
    batch_size: int = 64
    max_chars: int | None = 8000

    #: Replaces the network call: given a query and its passages, returns `(position, score)`
    #: for each. Set it to run a whole sweep with no server and no key.
    transport: Callable[[str, Sequence[str]], Ranking] | None = None

    name: ClassVar[str] = "remote-rerank"
    version: ClassVar[str] = "1"

    def __post_init__(self) -> None:
        if not self.model:
            raise RerankerError(
                f"{self.name} needs a model name, e.g. `reranker: {self.name}:bge-reranker-base`"
            )
        if self.batch_size < 1:
            raise RerankerError(f"batch_size must be at least 1, got {self.batch_size}")

    def rerank(self, query: str, candidates: Sequence[Chunk], k: int) -> list[Scored]:
        if not candidates:
            return []

        passages = [self._trim(chunk.text) for chunk in candidates]

        scored: list[tuple[int, float]] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            ranking = self._score_with_retries(query, batch, start)
            self._check(ranking, len(batch), start)
            scored.extend((start + position, score) for position, score in ranking)

        # Stable on ties: `-score` first, then the incoming position, so two passages the model
        # scores identically keep the retriever's order rather than an arbitrary one. Without
        # it a rerun can reorder ties and a diff shows a change that did not happen.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return [Scored(candidates[position].id, float(score)) for position, score in scored[:k]]

    def _trim(self, text: str) -> str:
        if self.max_chars is None or len(text) <= self.max_chars:
            return text
        return text[: self.max_chars]

    def _check(self, ranking: Ranking, expected: int, offset: int) -> None:
        """A backend that returns fewer scores than passages has dropped documents.

        On the leaderboard that is indistinguishable from the reranker judging them irrelevant,
        which is a completely different claim.
        """
        if len(ranking) != expected:
            raise RerankerError(
                f"{self.name} scored {len(ranking)} of {expected} candidates in the batch "
                f"starting at {offset}. Every candidate must come back, or the ones that did "
                "not look like the model rejected them."
            )
        for position, _ in ranking:
            if not 0 <= position < expected:
                raise RerankerError(
                    f"{self.name} returned index {position} for a batch of {expected}. "
                    "Scores would be attached to the wrong passages."
                )

    def _score_with_retries(self, query: str, passages: Sequence[str], offset: int) -> Ranking:
        delay = 1.0
        for attempt in range(self.retries + 1):
            try:
                if self.transport is not None:
                    return self.transport(query, passages)
                return self._call(query, passages)
            except RerankerError as error:
                if attempt >= self.retries or not _is_transient(error):
                    raise RerankerError(
                        f"{self.name} failed reranking candidates "
                        f"{offset}-{offset + len(passages) - 1} with {self.model}: {error}"
                    ) from error
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")  # pragma: no cover

    def _call(self, query: str, passages: Sequence[str]) -> Ranking:
        raise NotImplementedError

    def _resolved_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


def _is_transient(error: Exception) -> bool:
    text = str(error).lower()
    if any(word in text for word in ("api key", "unauthorized", "not found", "invalid model")):
        return False
    return any(
        word in text
        for word in (
            "timeout",
            "timed out",
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
            "connection",
            "temporarily",
        )
    )


# ---------------------------------------------------------------------------
# TEI
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TEIReranker(_RemoteReranker):
    """A cross-encoder served by text-embeddings-inference.

        docker run -p 8081:80 ghcr.io/huggingface/text-embeddings-inference:cpu-latest \\
            --model-id BAAI/bge-reranker-base

    then `reranker: tei-rerank:bge-reranker-base,api_base=http://localhost:8081`.

    Note the *separate port*: one TEI process serves one model, so reranking and embedding need
    two servers. Worth saying because the natural assumption is otherwise, and the failure is a
    confusing 400 rather than anything that names the real problem.
    """

    api_base: str | None = "http://localhost:8081"

    name: ClassVar[str] = "tei-rerank"

    def _call(self, query: str, passages: Sequence[str]) -> Ranking:
        url = f"{(self.api_base or '').rstrip('/')}/rerank"
        payload = json.dumps(
            {"query": query, "texts": list(passages), "truncate": True, "raw_scores": False}
        ).encode("utf-8")

        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        key = self._resolved_key()
        if key:
            request.add_header("Authorization", f"Bearer {key}")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:300]
            raise RerankerError(f"TEI at {url} returned {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RerankerError(
                f"could not reach a TEI reranker at {self.api_base}. A TEI process serves one "
                "model, so this needs its own server, separate from the embedding one:\n"
                "  docker run -p 8081:80 "
                "ghcr.io/huggingface/text-embeddings-inference:cpu-latest \\\n"
                f"      --model-id BAAI/{self.model}\n"
                f"({error.reason})"
            ) from error
        except json.JSONDecodeError as error:
            raise RerankerError(
                f"TEI at {url} returned something that is not JSON. Is {self.api_base} really "
                "a TEI server, and is it serving a reranking model rather than an embedding one?"
            ) from error

        if not isinstance(body, list):
            raise RerankerError(f"TEI at {url} returned {type(body).__name__}, expected a list")
        try:
            return [(int(item["index"]), float(item["score"])) for item in body]
        except (KeyError, TypeError, ValueError) as error:
            raise RerankerError(
                f"TEI at {url} returned rows without an index and a score. A model that is not "
                "a reranker will do this -- check what the server is serving."
            ) from error


# ---------------------------------------------------------------------------
# litellm
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LiteLLMReranker(_RemoteReranker):
    """A hosted reranker, through litellm. Cohere, Jina, Voyage, AWS.

    `reranker: litellm-rerank:cohere/rerank-english-v3.0`. The key comes from the environment,
    the same as everywhere else here.
    """

    name: ClassVar[str] = "litellm-rerank"

    def _call(self, query: str, passages: Sequence[str]) -> Ranking:
        try:
            import litellm
        except ImportError as error:  # pragma: no cover - exercised by the extras test
            raise RerankerError(
                "hosted rerankers need litellm. Install it with: pip install 'context-grid[llm]'"
            ) from error

        kwargs: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": list(passages),
            "top_n": len(passages),  # everything back, so nothing is silently dropped
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        key = self._resolved_key()
        if key:
            kwargs["api_key"] = key

        try:
            response = litellm.rerank(**kwargs)
        except Exception as error:
            raise RerankerError(_explain(error, self.model)) from error

        rows = getattr(response, "results", None)
        if rows is None and isinstance(response, dict):
            rows = response.get("results")
        if not rows:
            raise RerankerError(f"{self.model} returned no results for {len(passages)} passages")

        try:
            return [
                (int(row["index"]), float(row["relevance_score"]))
                if isinstance(row, dict)
                else (int(row.index), float(row.relevance_score))
                for row in rows
            ]
        except (KeyError, AttributeError, TypeError, ValueError) as error:
            raise RerankerError(
                f"could not read {self.model}'s results: expected an index and a relevance "
                f"score on every row ({error})"
            ) from error


def _explain(error: Exception, model: str) -> str:
    text = str(error)
    lowered = text.lower()
    if "api" in lowered and "key" in lowered:
        return (
            f"no usable API key for {model}. Set the provider's key in the environment -- "
            f"COHERE_API_KEY, JINA_API_KEY, VOYAGE_API_KEY. Never in the config file. ({text})"
        )
    if "not found" in lowered or "does not exist" in lowered:
        return (
            f"{model} was not recognised. litellm expects `provider/model` for rerankers, for "
            f"example `cohere/rerank-english-v3.0`. ({text})"
        )
    return text
