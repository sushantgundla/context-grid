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

    def __init__(
        self,
        feature: str,
        extra: str,
        package: str | None = None,
        *,
        detail: str | None = None,
    ) -> None:
        self.feature = feature
        self.extra = extra
        self.package = package
        self.detail = detail
        hint = f'pip install "context-grid[{extra}]"'
        needs = f" (needs {package})" if package else ""
        # `feature` is the subject of the sentence this finishes, so it has to be a noun phrase
        # -- "The docling parser", not a sentence of its own. Anything explanatory goes in
        # `detail`, which lands after the install hint where a full sentence reads correctly.
        # A sentence in the subject slot used to render "...needs network once requires the
        # 'embed' extra", which is not English.
        # The hint ends on a quote, not a full stop, so `detail` supplies the sentence break.
        tail = f". {detail}" if detail else ""
        super().__init__(
            f"{feature} requires the '{extra}' extra{needs}. Install it with: {hint}{tail}"
        )
