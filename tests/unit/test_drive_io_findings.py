"""Five ways bytes reached the pipeline wrong, all of them quietly.

Found driving the released 0.9.2 as a new user would. What they have in common is that
nothing failed: a BOM deleted a heading, a Latin-1 file turned into replacement characters
and got the parser blamed for it, 4KB of `/dev/urandom` indexed as ten chunks of text, a
`PermissionError` walked out of the public API untranslated, and two processes sharing a
cache directory destroyed each other's writes. Four of the five produced a plausible number
rather than an error, which is the failure this project exists to avoid.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

import contextgrid as cg
from contextgrid.cache.store import DiskCache
from contextgrid.core.documents import MediaType, SourceFile
from contextgrid.core.warnings import Severity, WarningCode
from contextgrid.parse.decode import looks_binary
from contextgrid.parse.text import MarkdownParser, TextParser

BODY = "# Returns\n\nText here.\n\n## Refund Window\n\nMore text.\n"
UTF8_BOM = b"\xef\xbb\xbf"


def _source(raw: bytes, *, name: str = "a.md") -> SourceFile:
    return SourceFile(id=name, media_type=MediaType.MARKDOWN, raw=raw)


def _headings(parsed: cg.ParsedDocument) -> list[str]:
    return [block.text for block in parsed.blocks if block.is_heading]


# ---------------------------------------------------------------------------
# 1. a UTF-8 BOM used to delete the first heading
# ---------------------------------------------------------------------------


def test_a_bom_does_not_hide_the_first_heading() -> None:
    """`\\ufeff# Returns` is not an ATX heading, so the whole document shifted one level."""
    plain = MarkdownParser().parse(_source(BODY.encode("utf-8")))
    bommed = MarkdownParser().parse(_source(UTF8_BOM + BODY.encode("utf-8")))

    assert _headings(plain) == ["# Returns", "## Refund Window"]
    assert _headings(bommed) == _headings(plain)


def test_a_bom_is_not_in_the_parsed_text_or_its_offsets() -> None:
    """The BOM is an encoding artefact, not content -- offsets must not count it."""
    parsed = MarkdownParser().parse(_source(UTF8_BOM + BODY.encode("utf-8")))

    assert not parsed.text.startswith("﻿")
    assert parsed.text == BODY
    assert parsed.verify_blocks() == []
    assert parsed.blocks[0].span.start == 0


def test_a_bom_does_not_change_the_heading_path() -> None:
    """The bug reached every heading-aware chunker and metric through this."""
    position = BODY.index("More text.")
    plain = MarkdownParser().parse(_source(BODY.encode("utf-8")))
    bommed = MarkdownParser().parse(_source(UTF8_BOM + BODY.encode("utf-8")))

    assert plain.heading_path_at(position) == ("# Returns", "## Refund Window")
    assert bommed.heading_path_at(position) == plain.heading_path_at(position)


def test_the_plain_text_parser_strips_the_bom_too() -> None:
    parsed = TextParser().parse(_source(UTF8_BOM + BODY.encode("utf-8")))

    assert parsed.text == BODY
    assert list(parsed.warnings) == []


def test_a_bom_on_its_own_is_still_an_empty_file() -> None:
    """Stripping the BOM must not turn "no text" into "some text"."""
    parsed = MarkdownParser().parse(_source(UTF8_BOM))

    assert parsed.text == ""
    assert [w.code for w in parsed.warnings] == [WarningCode.EMPTY_TEXT_LAYER]


# ---------------------------------------------------------------------------
# 2. non-UTF-8 bytes were mangled, and the parser got the blame
# ---------------------------------------------------------------------------


def test_a_latin1_file_is_reported_as_not_utf8() -> None:
    """The user saw `anchor_not_found` naming the parser. The file was the problem."""
    raw = "Café orders are final".encode("latin-1")
    parsed = MarkdownParser().parse(_source(raw, name="latin1.md"))

    codes = [w.code for w in parsed.warnings]
    assert codes == [WarningCode.PARSER_FALLBACK]
    message = parsed.warnings.entries[0].message
    assert "latin1.md" in message
    assert "not UTF-8" in message
    assert "invalid continuation byte" in message
    assert "byte 3" in message


def test_a_latin1_file_does_not_reach_the_index_mangled() -> None:
    """`Caf�` embedded and scored is a plausible wrong number. Nothing is better."""
    raw = "Café orders are final".encode("latin-1")
    parsed = MarkdownParser().parse(_source(raw, name="latin1.md"))

    assert "�" not in parsed.text
    assert parsed.text == ""
    assert parsed.blocks == ()


def test_the_not_utf8_warning_says_how_to_fix_it() -> None:
    raw = "Café orders are final".encode("latin-1")
    parsed = MarkdownParser().parse(_source(raw, name="latin1.md"))

    assert "UTF-8" in parsed.warnings.entries[0].message
    assert parsed.warnings.entries[0].severity is Severity.CAUTION
    assert parsed.warnings.entries[0].stage == "parse"
    assert parsed.warnings.entries[0].subject == "latin1.md"


def test_a_utf8_file_with_accents_is_left_alone() -> None:
    """The check must not fire on the ordinary case it is protecting."""
    parsed = MarkdownParser().parse(_source("Café orders are final".encode()))

    assert parsed.text == "Café orders are final"
    assert list(parsed.warnings) == []


# ---------------------------------------------------------------------------
# 3. a binary file named `.md` was indexed as text
# ---------------------------------------------------------------------------


def test_random_bytes_named_md_are_not_indexed_as_text() -> None:
    """4096 bytes of `/dev/urandom` used to produce ten chunks and no warning."""
    parsed = MarkdownParser().parse(_source(b"\x00\x01\x02\xff\xfe" * 800, name="b.md"))

    assert parsed.blocks == ()
    assert parsed.text == ""
    assert [w.code for w in parsed.warnings] == [WarningCode.PARSER_FALLBACK]


def test_a_binary_file_is_named_as_binary_not_as_a_typo() -> None:
    """Telling somebody to re-save a renamed `.pkl` as UTF-8 is useless advice."""
    parsed = MarkdownParser().parse(_source(os.urandom(4096), name="b.md"))

    message = parsed.warnings.entries[0].message
    assert "b.md" in message
    assert "binary" in message


@pytest.mark.parametrize(
    ("raw", "binary"),
    [
        (b"", False),
        (b"Caf\xe9 orders are final", False),
        (b"a text file\x00with a NUL in it", True),
        (bytes(range(256)) * 4, True),
    ],
)
def test_binary_is_told_apart_from_the_wrong_encoding(raw: bytes, binary: bool) -> None:
    """The two failures need opposite advice, so the sniff has to separate them."""
    assert looks_binary(raw) is binary


def test_an_unreadable_file_does_not_also_claim_to_be_empty() -> None:
    """One fact per file. `EMPTY_TEXT_LAYER` here would send the user looking for OCR."""
    parsed = TextParser().parse(_source(os.urandom(4096), name="b.md"))

    assert [w.code for w in parsed.warnings] == [WarningCode.PARSER_FALLBACK]


def test_a_binary_file_reaches_the_pipeline_warnings(tmp_path: Path) -> None:
    """The documented promise: the file is skipped and `pipeline.warnings` says which."""
    (tmp_path / "good.md").write_text("# Returns\n\nRefunds take five days.\n")
    (tmp_path / "b.md").write_bytes(b"\x00\x9c\xfe" * 1400)

    corpus = cg.Corpus.from_dir(tmp_path)
    pipeline = cg.build(
        cg.Config(parser="markdown", chunker="recursive:256", embedder="tfidf", index="dense"),
        corpus,
    )

    skipped = pipeline.warnings.of_code(WarningCode.PARSER_FALLBACK)
    assert [w.subject for w in skipped] == ["b.md"]
    assert {chunk.doc_id for chunk in pipeline.chunks} == {"good.md"}


def test_a_corpus_of_only_binary_files_says_why(tmp_path: Path) -> None:
    """`require_parsed_text` used to blame the parser for a file that is not text."""
    (tmp_path / "b.md").write_bytes(b"\x00\x9c\xfe" * 1400)

    with pytest.raises(cg.CorpusError) as caught:
        cg.build(
            cg.Config(parser="markdown", chunker="recursive:256", embedder="tfidf", index="dense"),
            cg.Corpus.from_dir(tmp_path),
        )

    assert "not UTF-8" in str(caught.value)


# ---------------------------------------------------------------------------
# 4. `Corpus.from_dir` leaked a raw `PermissionError`
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root reads a mode-000 file, so the leak cannot be reproduced",
)
def test_an_unreadable_file_raises_a_documented_error(tmp_path: Path) -> None:
    """`/reference/errors` tells users which exception to catch. This was not one of them."""
    (tmp_path / "ok.md").write_text("# Fine\n")
    unreadable = tmp_path / "n.md"
    unreadable.write_text("# Secret\n")
    unreadable.chmod(0o000)

    try:
        with pytest.raises(cg.CorpusError) as caught:
            cg.Corpus.from_dir(tmp_path)
    finally:
        unreadable.chmod(0o644)

    assert isinstance(caught.value, cg.ContextGridError)
    assert "n.md" in str(caught.value)
    assert "Permission denied" in str(caught.value)
    assert isinstance(caught.value.__cause__, PermissionError)


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root reads a mode-000 file, so the leak cannot be reproduced",
)
def test_from_files_translates_the_same_failure(tmp_path: Path) -> None:
    unreadable = tmp_path / "n.md"
    unreadable.write_text("# Secret\n")
    unreadable.chmod(0o000)

    try:
        with pytest.raises(cg.CorpusError) as caught:
            cg.Corpus.from_files([unreadable])
    finally:
        unreadable.chmod(0o644)

    assert "n.md" in str(caught.value)
    assert isinstance(caught.value.__cause__, PermissionError)


# ---------------------------------------------------------------------------
# 5. two runs sharing a cache directory destroyed each other's writes
# ---------------------------------------------------------------------------


def test_concurrent_writes_to_one_key_do_not_fail(tmp_path: Path) -> None:
    """Both writers used `<key>.tmp`, so the loser's rename found nothing and a sweep died."""
    cache = DiskCache(tmp_path)
    key = "a" * 64
    failures: list[BaseException] = []
    ready = threading.Barrier(8)

    def write() -> None:
        ready.wait()
        for _ in range(200):
            try:
                cache.put(key, {"value": 1})
            except BaseException as error:
                # Catching everything is the point: the bug surfaced as FileNotFoundError,
                # and a narrower catch would only prove that one spelling of it is gone.
                failures.append(error)

    workers = [threading.Thread(target=write) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert failures == []
    assert cache.get(key) == {"value": 1}


def test_each_write_uses_its_own_temporary_file(tmp_path: Path) -> None:
    """The fix, stated directly: no two writers can ever name the same temporary file."""
    cache = DiskCache(tmp_path)
    key = "b" * 64
    seen: list[str] = []
    real_dump = pickle.dump

    def spy(value: object, handle: object, *args: object, **kwargs: object) -> None:
        seen.append(Path(handle.name).name)  # type: ignore[attr-defined]
        real_dump(value, handle, *args, **kwargs)  # type: ignore[arg-type]

    import contextgrid.cache.store as store

    original = store.pickle.dump
    store.pickle.dump = spy  # type: ignore[assignment]
    try:
        cache.put(key, 1)
        cache.put(key, 2)
    finally:
        store.pickle.dump = original  # type: ignore[assignment]

    assert len(set(seen)) == 2
    assert all(str(os.getpid()) in name for name in seen)


def test_a_failed_write_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    """Unique names are only safe if they are also cleaned up."""
    cache = DiskCache(tmp_path)

    class Unpicklable:
        def __reduce__(self) -> tuple[object, ...]:
            raise TypeError("nope")

    with pytest.raises(TypeError):
        cache.put("c" * 64, Unpicklable())

    assert list(tmp_path.rglob("*.tmp")) == []


def test_the_cache_still_reads_back_what_it_wrote(tmp_path: Path) -> None:
    """The atomic-replace guarantee the temporary file exists for, unchanged."""
    DiskCache(tmp_path).put("d" * 64, {"value": 42})

    assert DiskCache(tmp_path).get("d" * 64) == {"value": 42}
    assert list(tmp_path.rglob("*.tmp")) == []
    assert len(DiskCache(tmp_path)) == 1


# ---------------------------------------------------------------------------
# 6. a `DiskCache` handed to a `Lab` was thrown away for being empty
# ---------------------------------------------------------------------------

#: One sweep through a `Lab` with a `DiskCache`, printed as two numbers a test can read.
#: A separate process is the whole point: inside one process a `MemoryCache` behaves exactly
#: like a working `DiskCache`, so an in-process test passes against the bug.
_SWEEP = """
import sys
from contextgrid.cache.store import DiskCache
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.lab import Lab

cache = DiskCache(root=sys.argv[1])
lab = Lab(
    {
        "contract.md": "# Notice\\n\\nEither party may terminate on thirty days notice.\\n",
        "api.md": "# Keys\\n\\nThe X-Api-Key header carries the API key.\\n",
    },
    cache=cache,
)
evalset = EvalSet(
    id="es",
    items=(
        EvalItem(
            id="q1",
            question="How much notice to terminate?",
            anchors=(GoldAnchor(source_id="contract.md", quote="thirty days"),),
        ),
    ),
)
lab.grid(chunker="sentence:1", index=["dense", "bm25"], k=3)
results = lab.run(evalset, mode="factorial")
print("SUMMARY:", results.cache_summary)
print("ON_DISK:", len(cache))
"""


@dataclass(frozen=True)
class Sweep:
    """What one process's sweep reused, and what it left on disk afterwards."""

    hits: int
    lookups: int
    on_disk: int
    summary: str


def _sweep(root: Path, tmp_path: Path) -> Sweep:
    """Run one sweep in a brand new interpreter and report what it saw."""
    script = tmp_path / "sweep.py"
    script.write_text(_SWEEP)
    finished = subprocess.run(
        [sys.executable, str(script), str(root)],
        capture_output=True,
        text=True,
        check=True,
    )
    printed = dict(
        line.split(": ", 1)
        for line in finished.stdout.splitlines()
        if line.startswith(("SUMMARY:", "ON_DISK:"))
    )
    summary = printed["SUMMARY"]
    hits, _, lookups = summary.split(" ", 3)[:3]
    return Sweep(int(hits), int(lookups), int(printed["ON_DISK"]), summary)


def test_a_lab_writes_to_the_disk_cache_it_was_given(tmp_path: Path) -> None:
    """`cache or MemoryCache()` swapped out any cache that was empty, which is all of them."""
    root = tmp_path / "cg-cache"

    first = _sweep(root, tmp_path)

    assert first.on_disk > 0, f"nothing reached the cache directory: {first.summary}"
    assert list(root.rglob("*.pkl"))


def test_a_second_process_reuses_the_first_process_work(tmp_path: Path) -> None:
    """The reason a `DiskCache` exists. Both drives measured 0% reuse between runs.

    Cross-process on purpose. Inside one process the `MemoryCache` that got substituted
    reuses work perfectly well, so a same-process test reports a healthy hit rate and passes
    against the bug -- which is roughly how this survived to a release.
    """
    root = tmp_path / "cg-cache"

    first = _sweep(root, tmp_path)
    second = _sweep(root, tmp_path)

    assert first.hits < first.lookups, f"first run should miss something: {first.summary}"
    assert second.hits == second.lookups, f"second run should be all hits: {second.summary}"


def test_the_lab_keeps_the_exact_cache_it_was_handed(tmp_path: Path) -> None:
    """Said directly, so the next reader does not have to run two processes to see it."""
    disk = DiskCache(root=tmp_path)

    assert len(disk) == 0
    assert cg.Lab({"a.md": "# A\n\nsome text\n"}, cache=disk).cache is disk


def test_no_cache_still_means_a_memory_cache() -> None:
    """Only `None` may be replaced. That is the whole fix."""
    assert isinstance(cg.Lab({"a.md": "# A\n\nsome text\n"}).cache, cg.MemoryCache)
