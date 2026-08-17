"""Turning a source file's bytes into the text a parser reads.

Decoding looks like a detail and is not. Everything downstream -- block offsets, heading
paths, chunk spans, the anchors that tie ground truth to a parse -- is measured in characters
of *this* string, so a wrong string is a wrong comparison that still produces a leaderboard.

Two things used to go wrong here, both silently:

* A UTF-8 byte order mark survived into the text, so the first line began `\\ufeff# Returns`,
  which is not an ATX heading. The document's first heading vanished from every
  `heading_path`, and Windows editors write a BOM by default.
* Bytes that are not UTF-8 were decoded with `errors="replace"`. `Café` became `Caf\\ufffd`,
  got embedded, retrieved and scored, and the only thing the user ever saw was
  `anchor_not_found` naming the parser -- so they changed parsers and never suspected the
  encoding. Random bytes named `.md` went the same way and indexed as ten chunks of
  replacement characters.

So a file that is not UTF-8 is not text, and is treated the way any unreadable file is
treated: skipped, with a `PARSER_FALLBACK` warning naming it. That is the same code, wording
and consequence as a parser declining a media type it does not read -- the file is not in the
index and nothing in it can be retrieved -- and it is honest here for the same reason. Losing
one document loudly beats indexing it wrong quietly.
"""

from __future__ import annotations

from contextgrid.core.documents import SourceFile
from contextgrid.core.errors import DocumentError
from contextgrid.core.warnings import Severity, WarningCode, WarningLog

#: The UTF-8 byte order mark. Not content, and not a character any offset should count.
UTF8_BOM = b"\xef\xbb\xbf"

#: How much of the file to look at when deciding whether it is binary. A header is enough,
#: and this runs on the failure path only, after a strict decode has already refused.
_SNIFF_BYTES = 8192

#: Above this share of non-text bytes in the sample, call it binary rather than mis-encoded.
#: Latin-1 prose sits near zero -- a handful of accented bytes in a page of ASCII. Compressed
#: or random bytes sit near two thirds, because only a quarter of byte values are printable.
_BINARY_SHARE = 0.30

#: Byte values that appear in ordinary text: printable ASCII plus the usual control codes a
#: text file really does contain.
_TEXT_BYTES = frozenset(range(0x20, 0x7F)) | frozenset(b"\t\n\r\f\v")


def decode_source(source: SourceFile) -> tuple[str, WarningLog]:
    """The text of a source file, plus anything worth knowing about how it decoded.

    Returns an empty string and a `PARSER_FALLBACK` warning when the bytes are not UTF-8.
    Callers should treat that as "this file produced no blocks", exactly as they would treat
    a file their parser declined -- and must not add `EMPTY_TEXT_LAYER` on top of it, which
    would send the user looking for a missing OCR pass instead of a wrong encoding.
    """
    warnings = WarningLog()
    if source.raw is None:
        raise DocumentError(
            f"source file {source.id!r} has no bytes loaded. Read the file before parsing it."
        )

    raw = source.raw.removeprefix(UTF8_BOM)
    try:
        return raw.decode("utf-8"), warnings
    except UnicodeDecodeError as error:
        warnings.add(
            WarningCode.PARSER_FALLBACK,
            f"{source.id!r} is not UTF-8 text -- {error.reason} at byte {error.start} -- so "
            "it is not in this index at all. Nothing in it can be retrieved. "
            f"{_advice(raw, source.id)}",
            severity=Severity.CAUTION,
            stage="parse",
            subject=source.id,
            reason=error.reason,
            byte_offset=error.start,
        )
        return "", warnings


def _advice(raw: bytes, source_id: str) -> str:
    """What to actually do about it, which is a different answer for each of the two cases.

    "Re-save it as UTF-8" is useless advice about a `.pkl` somebody renamed to `.md`, and
    "this is a binary file" is wrong about a perfectly good Latin-1 page.
    """
    if looks_binary(raw):
        return (
            "It looks like a binary file with a text extension -- most of its bytes are not "
            "text at all. Check what it really is, and drop it from the corpus or give it "
            "the right extension so a parser that reads that format can be chosen."
        )
    return (
        "It looks like text in another encoding, most likely Latin-1 or Windows-1252. "
        f"Convert it and run again: iconv -f windows-1252 -t utf-8 {source_id}"
    )


def looks_binary(raw: bytes) -> bool:
    """Is this a binary file rather than text in the wrong encoding?

    A NUL byte settles it -- no text encoding this tool reads produces one. Otherwise count
    how much of the sample is outside ordinary text bytes. Deliberately only consulted after
    a strict UTF-8 decode has already failed, so valid UTF-8 is never called binary on the
    strength of a few high bytes.
    """
    sample = raw[:_SNIFF_BYTES]
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    odd = sum(1 for byte in sample if byte not in _TEXT_BYTES)
    return odd / len(sample) > _BINARY_SHARE
