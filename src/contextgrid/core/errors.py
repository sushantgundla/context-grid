"""Exception types for context-grid.

Every error carries enough context to act on without reading a traceback.
"""

from __future__ import annotations


class ContextGridError(Exception):
    """Base class for everything this package raises."""


class SpanError(ContextGridError, ValueError):
    """A span is malformed, or two spans were compared in a way that makes no sense."""


class DocumentError(ContextGridError, ValueError):
    """A document or a reference into one is inconsistent."""


class EvalSetError(ContextGridError, ValueError):
    """An eval set, an item, or a gold span is malformed."""


class ResolutionError(ContextGridError, ValueError):
    """Gold spans could not be resolved to chunks."""


class MissingExtraError(ContextGridError, ImportError):
    """An optional dependency is needed and is not installed.

    Raised instead of a bare ImportError so the message names the exact install command
    rather than leaving the user to guess which extra a module belongs to.
    """

    def __init__(self, feature: str, extra: str, package: str | None = None) -> None:
        self.feature = feature
        self.extra = extra
        self.package = package
        hint = f'pip install "context-grid[{extra}]"'
        detail = f" (needs {package})" if package else ""
        super().__init__(f"{feature} requires the '{extra}' extra{detail}. Install it with: {hint}")
