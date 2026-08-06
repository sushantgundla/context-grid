"""The contract for a metric plugin, and the registry of them.

Every other axis in this package -- chunker, retriever, reranker, ingestion strategy -- is a
plugin behind a `Registry`. Metrics were the exception: the six in `score/metrics.py` were
plain functions, reached through a private dict, and the only way to add a seventh was to
compute it after the fact and staple it onto `RunResult.metrics` -- which never touches the
per-question scoring `Runner.run_one` actually does. There was no principled reason for the
gap, so this closes it: a metric is `name`, `version`, and one method, exactly like a
`Chunker` is `name`, `version` and `chunk()`.

**Why the registry lives here and not in `score/__init__.py`.** Every other family's registry
(`CHUNKERS`, `RETRIEVERS`, `RERANKERS`, ...) is built in that family's `__init__.py`, importing
the concrete plugin classes and registering them there -- nothing inside `chunk/fixed.py` ever
needs to look another chunker up by name. Metrics are different: `evaluate()` and `per_query()`
in `metrics.py` have to resolve a metric *by name*, including one they know nothing about at
import time, so they need the registry object itself, not just the ability to populate it.
Putting `METRICS` in `__init__.py` the way the other families do would make `metrics.py` import
its own package's `__init__`, which runs `metrics.py` in the first place -- a real cycle, not a
stylistic one. Living here, in the module `metrics.py` already has no reason not to import,
avoids it. `score/__init__.py` still does the registering, the same as every other family.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from contextgrid.core.registry import Registry


@runtime_checkable
class Metric(Protocol):
    """Scores one query: relevance judgements in, a ranked list in, one float out.

    `judgements` maps chunk id to grade, where `grade > 0` means relevant and the grade
    itself is only meaningful to metrics that use it (`ndcg_at_k` does; `recall_at_k`
    doesn't). `ranked` is the retriever's ordered chunk ids for this query, and `k` is the
    cut-off to score at -- cut-offs are part of a metric's *name* everywhere this package
    reports one (`recall@5`, not `recall` with `k` reported separately), so `evaluate()` and
    `per_query()` call `evaluate()` once per `k` and build the name themselves rather than a
    metric choosing its own cut-offs.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def evaluate(self, judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
        """One query's score. Never raises for an empty or missing `judgements`/`ranked` --
        see the built-ins in `metrics.py` for the shape that convention expects."""
        ...


#: Every registered metric, built-in and custom. `score/__init__.py` registers the six
#: built-ins into this at import time, the same way `chunk/__init__.py` populates `CHUNKERS`.
METRICS: Registry[Metric] = Registry(family="metric")
