"""Real embedding models, hosted or self-hosted.

Two backends, one shared body, because the difference between them is where the HTTP request
goes and nothing else:

* **litellm** reaches every hosted provider through one interface -- OpenAI, Cohere, Voyage,
  Gemini, Bedrock, Azure -- so bringing your own key means naming a model, not writing an
  adapter. `embedder: litellm:text-embedding-3-small`.
* **TEI** is HuggingFace's text-embeddings-inference server: start it once, point at it, and
  any open model runs locally with no key and no network.
  `embedder: tei:bge-base-en-v1.5,api_base=http://localhost:8080`.

The TEI backend deliberately uses `urllib` from the standard library rather than an HTTP
client. A running TEI server plus a bare `pip install context-grid` is then enough for real
embeddings -- no extra package at all.

Three things this module is careful about, each of which quietly ruins a comparison when it
is got wrong:

**Prefixes.** Handled in `prefixes.py`, applied here. Embedding queries and documents the same
way when the model was trained otherwise costs several points, invisibly, and costs them
unevenly across the arms of a sweep.

**Batching.** Providers cap how many inputs one request may carry, and a corpus is thousands.
Batches are sized here, and a failure names the batch it happened in rather than the whole run.

**Failure.** A missing API key, a server that is not running and a rate limit are three
different problems with three different fixes, so they get three different messages. The most
expensive failure in a long sweep is the one you cannot act on.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from contextgrid.core.errors import ContextGridError
from contextgrid.core.warnings import Severity, WarningCode, WarningLog
from contextgrid.embed.base import EmbeddingResult, Vectors, normalise, truncate
from contextgrid.embed.prefixes import Prefixes, for_model


class EmbedderError(ContextGridError, RuntimeError):
    """An embedding backend could not be reached, or refused the request."""


@dataclass(frozen=True, slots=True)
class _RemoteEmbedder:
    """Shared body: prefixes, batching, truncation, normalising, token accounting."""

    model: str = ""
    dimensions: int = 0
    batch_size: int = 32
    max_tokens: int | None = 512
    api_base: str | None = None
    api_key_env: str | None = None
    timeout: float = 60.0
    retries: int = 2
    query_prefix: str | None = None
    document_prefix: str | None = None
    normalise_vectors: bool = True

    #: Replaces the network call. Given one batch of texts, returns its vectors and the token
    #: count. Set it to stand a whole sweep up with no server and no key -- which is the
    #: difference between being able to test a pipeline built on this package and not.
    transport: Callable[[Sequence[str]], tuple[list[list[float]], int]] | None = None

    version: ClassVar[str] = "1"
    name: ClassVar[str] = "remote"

    _prefixes: Prefixes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.model:
            raise EmbedderError(
                f"{self.name} needs a model name, e.g. `embedder: {self.name}:bge-base-en-v1.5`"
            )
        if self.batch_size < 1:
            raise EmbedderError(f"batch_size must be at least 1, got {self.batch_size}")

        # An explicit prefix always wins, including an explicit empty string -- somebody who
        # writes `query_prefix=""` is saying "this model needs none", and guessing over the
        # top of that would be worse than guessing in the first place.
        looked_up = for_model(self.model)
        object.__setattr__(
            self,
            "_prefixes",
            Prefixes(
                query=self.query_prefix if self.query_prefix is not None else looked_up.query,
                document=(
                    self.document_prefix if self.document_prefix is not None else looked_up.document
                ),
            ),
        )

    # -- protocol ------------------------------------------------------------

    @property
    def normalised(self) -> bool:
        return self.normalise_vectors

    def prepare(self, documents: Sequence[str]) -> None:
        """Nothing to learn from the corpus. The model is already trained."""
        return None

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        return self._embed(texts, self._prefixes.document, "documents")

    def embed_queries(self, texts: Sequence[str]) -> EmbeddingResult:
        return self._embed(texts, self._prefixes.query, "queries")

    # -- the work ------------------------------------------------------------

    def _embed(self, texts: Sequence[str], prefix: str, side: str) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=np.zeros((0, self.dimensions), dtype=np.float32))

        cut, log, truncated = truncate(texts, self.max_tokens, model=self.name)
        self._warn_about_prefixes(log)
        prefixed = [prefix + text for text in cut] if prefix else list(cut)

        rows: list[list[float]] = []
        tokens = 0
        for start in range(0, len(prefixed), self.batch_size):
            batch = prefixed[start : start + self.batch_size]
            vectors, batch_tokens = self._request_with_retries(batch, start, side)
            rows.extend(vectors)
            tokens += batch_tokens

        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise EmbedderError(
                f"{self.name} returned {matrix.shape[0] if matrix.ndim else 0} vectors for "
                f"{len(texts)} inputs. The two must match or every score after this is "
                "attached to the wrong text."
            )

        self._check_dimensions(matrix, log)
        return EmbeddingResult(
            vectors=normalise(matrix) if self.normalise_vectors else matrix,
            warnings=log,
            input_tokens=tokens,
            truncated=truncated,
        )

    def _warn_about_prefixes(self, log: WarningLog) -> None:
        """Say so when nothing is known about whether this model wants prefixes.

        E5 was trained with `query:` and `passage:`, BGE with an instruction on the query alone.
        Get it wrong and nothing fails -- the numbers come out several points low, uniformly,
        and *unevenly across the arms of a sweep*, which turns a model comparison into a
        comparison of one model against a handicapped version of another.

        Silence is the right default for a name nobody recognises, since inventing a prefix is
        as wrong as omitting one. Silence with no warning is not.
        """
        if self._prefixes.used or self.query_prefix is not None:
            return
        if for_model(self.model).used:  # pragma: no cover - unreachable while the rule matched
            return

        log.add(
            WarningCode.MISSING_QUERY_PREFIX,
            f"nothing is known about whether {self.model!r} wants query and document prefixes, "
            "so none were added. If it was trained with them, every score for this arm is "
            "several points low. Set `query_prefix=` and `document_prefix=` explicitly, or "
            'silence this with `query_prefix=""`',
            severity=Severity.CAUTION,
            stage="embed",
            subject=self.model,
        )

    def _check_dimensions(self, matrix: Vectors, log: WarningLog) -> None:
        """A model whose width is not what the config said is a config that is wrong.

        Left unsaid, the mismatch surfaces much later as a shape error deep in an index, or --
        worse -- not at all, when a cached run from a different model is silently reused.
        """
        actual = int(matrix.shape[1])
        if self.dimensions and actual != self.dimensions:
            log.add(
                WarningCode.UNNORMALISED_VECTORS,
                f"{self.name} was configured for {self.dimensions} dimensions but {self.model} "
                f"returned {actual}. Using {actual}; set `dimensions={actual}` to silence this",
                severity=Severity.CAUTION,
                stage="embed",
                subject=self.model,
                expected=self.dimensions,
                actual=actual,
            )

    def _request_with_retries(
        self, batch: Sequence[str], offset: int, side: str
    ) -> tuple[list[list[float]], int]:
        """One batch, retried on transient failures only.

        Retrying a bad API key wastes a minute and changes nothing; retrying a rate limit or a
        dropped connection is usually all that is needed. `_is_transient` draws that line.
        """
        delay = 1.0
        for attempt in range(self.retries + 1):
            try:
                return self._request(batch)
            except EmbedderError as error:
                if attempt >= self.retries or not _is_transient(error):
                    raise self._with_context(error, offset, len(batch), side) from error
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")  # pragma: no cover

    def _with_context(
        self, error: EmbedderError, offset: int, size: int, side: str
    ) -> EmbedderError:
        return EmbedderError(
            f"{self.name} failed embedding {side} {offset}-{offset + size - 1} "
            f"with {self.model}: {error}"
        )

    def _request(self, batch: Sequence[str]) -> tuple[list[list[float]], int]:
        if self.transport is not None:
            return self.transport(batch)
        return self._call(batch)

    def _call(self, batch: Sequence[str]) -> tuple[list[list[float]], int]:
        raise NotImplementedError

    def _resolved_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


def _is_transient(error: Exception) -> bool:
    """Whether retrying could plausibly help."""
    text = str(error).lower()
    permanent = ("api key", "unauthorized", "authentication", "not found", "invalid model")
    if any(word in text for word in permanent):
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
# litellm: every hosted provider
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LiteLLMEmbedder(_RemoteEmbedder):
    """Any hosted embedding model, through litellm.

    `embedder: litellm:text-embedding-3-small` or
    `litellm:cohere/embed-english-v3.0,dimensions=1024`.

    The key comes from the environment, never from the config file -- a config with a secret
    in it ends up in version control and then in a screenshot. litellm reads the standard
    provider variables (`OPENAI_API_KEY`, `COHERE_API_KEY`, and so on) by itself.
    """

    name: ClassVar[str] = "litellm"

    def _call(self, batch: Sequence[str]) -> tuple[list[list[float]], int]:
        try:
            import litellm
        except ImportError as error:  # pragma: no cover - exercised by the extras test
            raise EmbedderError(
                "litellm embedders need litellm. Install it with: pip install 'context-grid[llm]'"
            ) from error

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": list(batch),
            "timeout": self.timeout,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.dimensions:
            # Only some models accept it; litellm drops it for the ones that do not.
            kwargs["dimensions"] = self.dimensions
        key = self._resolved_key()
        if key:
            kwargs["api_key"] = key

        try:
            response = litellm.embedding(**kwargs)
        except Exception as error:
            raise EmbedderError(_explain(error, self.model)) from error

        data = getattr(response, "data", None) or []
        vectors = [list(item["embedding"]) for item in data]
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        return vectors, tokens


def _explain(error: Exception, model: str) -> str:
    """Turn a provider's exception into something a person can act on."""
    text = str(error)
    lowered = text.lower()

    if "api" in lowered and "key" in lowered:
        return (
            f"no usable API key for {model}. Set the provider's key in the environment -- "
            "OPENAI_API_KEY, COHERE_API_KEY, VOYAGE_API_KEY and so on. Never put it in the "
            f"config file. ({text})"
        )
    if "not found" in lowered or "does not exist" in lowered:
        return (
            f"{model} was not recognised. litellm expects `provider/model` for anything that "
            f"is not OpenAI, for example `cohere/embed-english-v3.0`. ({text})"
        )
    if "connection" in lowered or "refused" in lowered:
        return f"could not reach the endpoint for {model}. Is it running and reachable? ({text})"
    return text


# ---------------------------------------------------------------------------
# TEI: a local server, no key, no network
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TEIEmbedder(_RemoteEmbedder):
    """HuggingFace text-embeddings-inference, over plain HTTP.

    Start the server once:

        docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:cpu-latest \\
            --model-id BAAI/bge-base-en-v1.5

    then `embedder: tei:bge-base-en-v1.5,api_base=http://localhost:8080`.

    `model` here is a label -- the server decides which weights it serves -- but it is not
    cosmetic: it selects the prefixes and it goes into the cache key, so two sweeps against
    servers running different models cannot be confused for one another.

    Deliberately built on `urllib` rather than an HTTP client, so a running server plus a bare
    `pip install context-grid` is enough. No extra dependency at all.
    """

    api_base: str | None = "http://localhost:8080"

    name: ClassVar[str] = "tei"

    def _call(self, batch: Sequence[str]) -> tuple[list[list[float]], int]:
        url = f"{(self.api_base or '').rstrip('/')}/embed"
        payload = json.dumps(
            {
                "inputs": list(batch),
                # The server truncates rather than rejecting. It has the real tokenizer; the
                # character estimate upstream cannot be exact, and a hard failure halfway
                # through a corpus over one long chunk helps nobody.
                "truncate": True,
                "normalize": self.normalise_vectors,
            }
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
            raise EmbedderError(f"TEI at {url} returned {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise EmbedderError(
                f"could not reach a TEI server at {self.api_base}. Start one with:\n"
                "  docker run -p 8080:80 "
                "ghcr.io/huggingface/text-embeddings-inference:cpu-latest \\\n"
                f"      --model-id BAAI/{self.model}\n"
                f"({error.reason})"
            ) from error
        except json.JSONDecodeError as error:
            raise EmbedderError(
                f"TEI at {url} returned something that is not JSON. Is {self.api_base} really "
                "a TEI server?"
            ) from error

        if not isinstance(body, list):
            raise EmbedderError(
                f"TEI at {url} returned {type(body).__name__}, expected a list of vectors"
            )
        # TEI does not report token usage on /embed. Counting it here would mean running the
        # model's tokenizer ourselves, and a guessed number in a cost column is worse than an
        # honest zero -- a local model's cost is machine time, which the cost model prices from
        # the clock.
        return [list(vector) for vector in body], 0

    def info(self) -> dict[str, Any]:
        """What the server is actually serving.

        Worth calling before a sweep: `model` in the config is a label, and this is the only
        way to find out whether the server behind it is running the weights you think.
        """
        url = f"{(self.api_base or '').rstrip('/')}/info"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                loaded: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                return loaded
        except (urllib.error.URLError, json.JSONDecodeError) as error:
            raise EmbedderError(f"could not read TEI server info from {url}: {error}") from error
