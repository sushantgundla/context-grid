"""The experiment file: one YAML that describes and runs the whole thing."""

from __future__ import annotations

from contextgrid.config.loader import (
    build_corpus,
    build_evalset,
    build_runner,
    load,
    loads,
    run,
    write_report,
)
from contextgrid.config.schema import (
    ConfigError,
    ExperimentConfig,
    GridConfig,
    ReportConfig,
    RunConfig,
)
from contextgrid.config.template import render

__all__ = [
    "ConfigError",
    "ExperimentConfig",
    "GridConfig",
    "ReportConfig",
    "RunConfig",
    "build_corpus",
    "build_evalset",
    "build_runner",
    "load",
    "loads",
    "render",
    "run",
    "write_report",
]
