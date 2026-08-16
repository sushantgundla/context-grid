"""Scoring: span resolution, metrics and significance testing.

Metrics are a plugin family like every other axis -- see `score/base.py` for why the
registry lives there rather than here, and `docs/internals/extending.md` for a metric
written and registered end to end.
"""

from __future__ import annotations

from contextgrid.score.base import METRICS, Metric
from contextgrid.score.metrics import (
    HitRateMetric,
    MAPMetric,
    MRRMetric,
    NDCGMetric,
    PrecisionMetric,
    RecallMetric,
    available_metrics,
)

METRICS.register("recall", doc="Fraction of relevant chunks that appear in the top k.")(
    RecallMetric
)
METRICS.register("precision", doc="Fraction of the top k that is relevant.")(PrecisionMetric)
METRICS.register("hit_rate", doc="1.0 when anything relevant is in the top k, else 0.0.")(
    HitRateMetric
)
METRICS.register("mrr", doc="1 / the position of the first relevant result.")(MRRMetric)
METRICS.register("map", doc="Mean average precision, trec_eval's convention.")(MAPMetric)
METRICS.register("ndcg", doc="Graded nDCG against the best possible ordering.")(NDCGMetric)


def get_metric(spec: str) -> Metric:
    """Resolve a metric from a spec string, e.g. `recall`. See `Registry.create`."""
    return METRICS.create(spec)


__all__ = [
    "METRICS",
    "HitRateMetric",
    "MAPMetric",
    "MRRMetric",
    "Metric",
    "NDCGMetric",
    "PrecisionMetric",
    "RecallMetric",
    # Re-exported from `metrics.py` so that everything needed to write and check a metric is
    # reachable from one place. Somebody who found `METRICS` and `get_metric` here would
    # reasonably expect to find "what is registered?" beside them, rather than one package down.
    "available_metrics",
    "get_metric",
]
