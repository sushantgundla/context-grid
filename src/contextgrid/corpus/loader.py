"""Loading a corpus off disk, out of memory, or out of a dictionary."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contextgrid.core.documents import MediaType, SourceFile
from contextgrid.core.errors import ContextGridError

#: Directories that are never corpus content, however the glob was written.
_SKIP_DIRECTORIES = frozenset(
    {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".ruff_cache"}
)

DEFAULT_PATTERNS: tuple[str, ...] = (
    "*.txt",
    "*.md",
    "*.markdown",
    "*.mdx",
    "*.html",
    "*.htm",
    "*.pdf",
    "*.docx",
    "*.pptx",
    "*.xlsx",
)


class CorpusError(ContextGridError, ValueError):
    """A corpus could not be loaded, or is not usable as given."""


@dataclass(frozen=True, slots=True)
class Corpus:
    """A named set of source files, before anything has been extracted from them.

    Deliberately dumb: it holds bytes and identity, nothing more. Everything about
    *content* -- length, structure, table density -- depends on which parser read it, and
    belongs to the fingerprint rather than here.
    """

    files: tuple[SourceFile, ...]
    name: str = "corpus"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for source in self.files:
            if source.id in seen:
                raise CorpusError(f"duplicate source id {source.id!r} in corpus {self.name!r}")
            seen.add(source.id)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_dir(
        cls,
        path: str | Path,
        *,
        patterns: Sequence[str] = DEFAULT_PATTERNS,
        recursive: bool = True,
        max_files: int | None = None,
        name: str | None = None,
    ) -> Corpus:
        """Read every matching file under a directory.

        Ids are paths relative to the directory, so they stay readable in a leaderboard and
        stable if the corpus moves.
        """
        root = Path(path).expanduser()
        if not root.is_dir():
            raise CorpusError(f"{root} is not a directory")

        matched: set[Path] = set()
        for pattern in patterns:
            glob = f"**/{pattern}" if recursive else pattern
            matched.update(p for p in root.glob(glob) if p.is_file())

        ordered = sorted(p for p in matched if _is_candidate(p, root))
        if not ordered:
            raise CorpusError(
                f"no files under {root} matched {list(patterns)}."
                f"{_why_nothing_matched(root, matched, recursive=recursive)} "
                "(Widening the list is Python-API only: "
                "`Corpus.from_dir(path, patterns=[...])`. There is no `patterns:` config key.)"
            )
        if max_files is not None:
            ordered = ordered[:max_files]

        files = tuple(
            SourceFile(
                id=str(item.relative_to(root)),
                media_type=MediaType.from_suffix(item.suffix),
                path=str(item),
                raw=_read_bytes(item),
            )
            for item in ordered
        )
        return cls(files=files, name=name or root.name)

    @classmethod
    def from_files(cls, paths: Iterable[str | Path], *, name: str = "corpus") -> Corpus:
        """Read an explicit list of files. Ids are the file names."""
        files: list[SourceFile] = []
        for raw_path in paths:
            item = Path(raw_path).expanduser()
            if not item.is_file():
                raise CorpusError(f"{item} is not a file")
            files.append(
                SourceFile(
                    id=item.name,
                    media_type=MediaType.from_suffix(item.suffix),
                    path=str(item),
                    raw=_read_bytes(item),
                )
            )
        return cls(files=tuple(files), name=name)

    @classmethod
    def from_texts(
        cls,
        texts: Mapping[str, str],
        *,
        media_type: MediaType = MediaType.TEXT,
        name: str = "corpus",
    ) -> Corpus:
        """Build from a mapping of id to text. The ten-second path to a first result."""
        return cls(
            files=tuple(
                SourceFile(id=key, media_type=media_type, raw=value.encode("utf-8"))
                for key, value in texts.items()
            ),
            name=name,
        )

    # -- access --------------------------------------------------------------

    def get(self, source_id: str) -> SourceFile | None:
        for source in self.files:
            if source.id == source_id:
                return source
        return None

    def require(self, source_id: str) -> SourceFile:
        found = self.get(source_id)
        if found is None:
            raise CorpusError(
                f"no source {source_id!r} in corpus {self.name!r}. "
                f"Available: {', '.join(sorted(s.id for s in self.files)) or 'none'}"
            )
        return found

    def of_type(self, *media_types: MediaType) -> tuple[SourceFile, ...]:
        wanted = set(media_types)
        return tuple(s for s in self.files if s.media_type in wanted)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.files)

    @property
    def total_bytes(self) -> int:
        return sum(s.size_bytes or 0 for s in self.files)

    def content_hash(self) -> str:
        """A hash of the whole corpus, independent of the order files were listed in.

        Part of the run manifest: two runs over the same documents must agree on this, and
        two runs over different documents must not.
        """
        digest = hashlib.sha256()
        for file_hash in sorted(s.content_hash() for s in self.files):
            digest.update(file_hash.encode("ascii"))
        return digest.hexdigest()

    def __iter__(self) -> Iterator[SourceFile]:
        return iter(self.files)

    def __len__(self) -> int:
        return len(self.files)


_ADVICE = (
    "Point the corpus at a directory holding files with those extensions, or "
    "rename the files to one of them."
)

_HIDDEN_RULE = (
    "Entries inside the corpus whose name starts with a dot, and build directories "
    f"({', '.join(sorted(_SKIP_DIRECTORIES))}), are always skipped -- the corpus "
    "directory you named is never skipped, only what sits below it."
)


def _read_bytes(path: Path) -> bytes:
    """Read one corpus file, or say what stopped it in an exception people can catch.

    `read_bytes` raises `PermissionError`, `IsADirectoryError` and the rest of `OSError`.
    None of those are `ContextGridError`, so the `except cg.CorpusError` that `/reference/errors`
    tells users to write never fired and a full traceback came out of the documented public
    API instead. The CLI has always caught this cleanly; only the Python path leaked.
    """
    try:
        return path.read_bytes()
    except OSError as error:
        raise CorpusError(
            f"{path} could not be read: {error.strerror or error}. Every file the corpus "
            "patterns match has to be readable by the user running contextgrid -- fix its "
            "permissions, or narrow the corpus so it is not matched: "
            "`Corpus.from_dir(path, patterns=[...])`."
        ) from error


def _why_nothing_matched(root: Path, matched: set[Path], *, recursive: bool) -> str:
    """Say why the corpus came out empty, checking each claim before making it.

    Error path only, so the extra walk costs nothing anybody waits for. An empty directory,
    a directory full of `.parquet`, and a directory whose files are all hidden are three
    different mistakes and deserve three different sentences.
    """
    glob = "**/*" if recursive else "*"
    present = [p for p in root.glob(glob) if p.is_file()]
    if not present:
        return f" The directory holds no files at all. {_ADVICE}"

    if matched:
        count = len(matched)
        verb = "was" if count == 1 else "were"
        noun = "file" if count == 1 else "files"
        return (
            f" {count} {noun} matched the patterns but {verb} skipped as hidden. "
            f"{_HIDDEN_RULE} Move them out of the hidden directory to load them."
        )

    visible = [p for p in present if _is_candidate(p, root)]
    total = len(present)
    noun = "file" if total == 1 else "files"
    if not visible:
        return (
            f" It holds {total} {noun}, none matching the patterns, and every one hidden. "
            f"{_HIDDEN_RULE} {_ADVICE}"
        )

    suffixes = sorted({p.suffix for p in visible if p.suffix})
    if not suffixes:
        return f" It holds {total} {noun}, none with a file extension. {_ADVICE}"
    listed = ", ".join(suffixes[:5])
    more = "" if len(suffixes) <= 5 else f" and {len(suffixes) - 5} more"
    return f" It holds {total} {noun}: {listed}{more}. {_ADVICE}"


def _is_candidate(path: Path, root: Path) -> bool:
    """Is this a file the corpus should load?

    Hidden and build directories are skipped, but only *below* the root. The root itself is
    whatever the user named -- a corpus under `.data/` or `~/.local/share/...` is a real
    corpus, not something to filter away.
    """
    if not path.is_file():
        return False
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    if any(part.startswith(".") for part in parts):
        return False
    return not any(part in _SKIP_DIRECTORIES for part in parts)
