"""Structured warnings.

Retrieval comparisons go wrong quietly. A chunk gets truncated because it exceeded a
model's context, a parser reflows a column and loses its character offsets, an approximate
index returns 92% of what exact search would have found -- and none of it shows up in the
final number. The user reads a leaderboard that looks fine and draws a confident, wrong
conclusion.

So warnings here are data, not log lines. Every result object carries a `WarningLog`, it
survives serialisation, and anything rendering results is expected to show it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WarningCode(str, Enum):
    """Machine-readable warning kinds.

    Grouped by the stage that raises them. String-valued so they serialise as themselves.
    """

    # -- parsing -------------------------------------------------------------
    APPROXIMATE_OFFSETS = "approximate_offsets"
    OCR_APPLIED = "ocr_applied"
    EMPTY_TEXT_LAYER = "empty_text_layer"
    PARSER_FALLBACK = "parser_fallback"

    # -- chunking ------------------------------------------------------------
    CHUNK_EXCEEDS_MODEL_CONTEXT = "chunk_exceeds_model_context"
    CHUNK_TEXT_REWRITTEN = "chunk_text_rewritten"
    EMPTY_CHUNK_SET = "empty_chunk_set"

    # -- embedding -----------------------------------------------------------
    INPUT_TRUNCATED = "input_truncated"
    MISSING_QUERY_PREFIX = "missing_query_prefix"
    UNNORMALISED_VECTORS = "unnormalised_vectors"

    # -- indexing and search -------------------------------------------------
    ANN_RECALL_LOSS = "ann_recall_loss"
    QUANTIZATION_APPLIED = "quantization_applied"

    # -- eval sets and scoring -----------------------------------------------
    GOLD_SPAN_UNREACHABLE = "gold_span_unreachable"
    SPLIT_GOLD_SPAN = "split_gold_span"
    APPROXIMATE_RESOLUTION = "approximate_resolution"
    SMALL_EVAL_SET = "small_eval_set"

    # -- runs ----------------------------------------------------------------
    BUDGET_REACHED = "budget_reached"
    CACHE_MISS_STORM = "cache_miss_storm"
    NON_DETERMINISTIC_STAGE = "non_deterministic_stage"


class Severity(str, Enum):
    """How much a warning should change what you believe.

    INFO      something happened that you should know about; results stand.
    CAUTION   results are usable but a comparison may be slightly unfair.
    INVALID   a comparison built on this is not sound. Do not publish it.
    """

    INFO = "info"
    CAUTION = "caution"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class GridWarning:
    """One thing that happened which could change how a result should be read."""

    code: WarningCode
    message: str
    severity: Severity = Severity.CAUTION
    stage: str | None = None
    subject: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        where = f" [{self.stage}]" if self.stage else ""
        what = f" ({self.subject})" if self.subject else ""
        return f"{self.severity.value.upper()}{where}{what}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "severity": self.severity.value,
            "stage": self.stage,
            "subject": self.subject,
            "detail": dict(self.detail),
        }


@dataclass(slots=True)
class WarningLog:
    """An ordered, mergeable collection of warnings.

    Mutable by design: stages append to it as a run progresses, and a run's log is the
    concatenation of its stages' logs.
    """

    entries: list[GridWarning] = field(default_factory=list)

    def add(
        self,
        code: WarningCode,
        message: str,
        *,
        severity: Severity = Severity.CAUTION,
        stage: str | None = None,
        subject: str | None = None,
        **detail: Any,
    ) -> GridWarning:
        """Record a warning and return it."""
        warning = GridWarning(
            code=code,
            message=message,
            severity=severity,
            stage=stage,
            subject=subject,
            detail=detail,
        )
        self.entries.append(warning)
        return warning

    def extend(self, other: Iterable[GridWarning]) -> None:
        self.entries.extend(other)

    def merge(self, other: WarningLog) -> WarningLog:
        """Return a new log holding this log's entries followed by the other's."""
        return WarningLog(entries=[*self.entries, *other.entries])

    def of_code(self, *codes: WarningCode) -> list[GridWarning]:
        wanted = set(codes)
        return [w for w in self.entries if w.code in wanted]

    def at_least(self, severity: Severity) -> list[GridWarning]:
        """Warnings at or above a severity."""
        order = {Severity.INFO: 0, Severity.CAUTION: 1, Severity.INVALID: 2}
        floor = order[severity]
        return [w for w in self.entries if order[w.severity] >= floor]

    @property
    def invalidating(self) -> list[GridWarning]:
        """Warnings that mean a comparison built on this result is not sound."""
        return [w for w in self.entries if w.severity is Severity.INVALID]

    @property
    def is_sound(self) -> bool:
        """True when nothing recorded here invalidates a comparison."""
        return not self.invalidating

    def counts(self) -> dict[str, int]:
        return dict(Counter(w.code.value for w in self.entries))

    def summary(self) -> str:
        if not self.entries:
            return "no warnings"
        parts = [f"{code} x{n}" for code, n in sorted(self.counts().items())]
        return ", ".join(parts)

    def to_list(self) -> list[dict[str, Any]]:
        return [w.to_dict() for w in self.entries]

    def __iter__(self) -> Iterator[GridWarning]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)
