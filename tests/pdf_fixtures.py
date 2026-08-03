"""Generated PDF fixtures.

Built rather than committed, so the bytes are reproducible and the test data is readable as
code. They are deliberately small and deliberately awkward: a heading in a larger font, a
paragraph that wraps across lines, a bordered table, and a page with no text layer at all.
"""

from __future__ import annotations

import functools

import pytest

pymupdf = pytest.importorskip("pymupdf", reason="the 'parse' extra is not installed")

HEADING = "Master Services Agreement"
SECTION = "2. Termination"
BODY_LINES = (
    "Either party may terminate this agreement for convenience",
    "by giving thirty days written notice. Notice must be",
    "delivered to the address in Schedule A.",
)
TABLE_ROWS = (
    ("Service", "Monthly fee", "Setup fee"),
    ("Standard", "1200", "500"),
    ("Premium", "3400", "500"),
)


@functools.cache
def contract_pdf() -> bytes:
    """A one-page contract: a large heading, a smaller one, body text and a bordered table."""
    document = pymupdf.open()
    page = document.new_page()

    page.insert_text((72, 90), HEADING, fontsize=18)
    page.insert_text((72, 130), SECTION, fontsize=14)
    for index, line in enumerate(BODY_LINES):
        page.insert_text((72, 160 + index * 16), line, fontsize=11)

    _draw_table(page, top=230, left=72, rows=TABLE_ROWS)

    data: bytes = document.tobytes()
    document.close()
    return data


@functools.cache
def prose_pdf() -> bytes:
    """Two pages of plain body text at one font size, so nothing looks like a heading."""
    document = pymupdf.open()
    for page_number in range(2):
        page = document.new_page()
        for index in range(12):
            page.insert_text(
                (72, 90 + index * 16),
                f"Page {page_number + 1} line {index + 1}: the notice period is thirty days.",
                fontsize=11,
            )
    data: bytes = document.tobytes()
    document.close()
    return data


@functools.cache
def scanned_pdf() -> bytes:
    """A page with no text layer, as a scan would arrive. Nothing here is retrievable."""
    document = pymupdf.open()
    page = document.new_page()
    page.draw_rect(pymupdf.Rect(72, 72, 300, 200), color=(0, 0, 0), width=1)
    data: bytes = document.tobytes()
    document.close()
    return data


@functools.cache
def mixed_pdf() -> bytes:
    """One page with text, one without. The common shape of a real scanned-in appendix."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 90), "Schedule A", fontsize=16)
    page.insert_text((72, 120), "The address for notices is 1 High Street.", fontsize=11)
    document.new_page()  # deliberately blank
    data: bytes = document.tobytes()
    document.close()
    return data


def _draw_table(
    page: object, *, top: float, left: float, rows: tuple[tuple[str, ...], ...]
) -> None:
    """Draw a bordered grid with text in it, so a table-aware parser can find it."""
    column_width = 120.0
    row_height = 22.0

    for row_index, row in enumerate(rows):
        y = top + row_index * row_height
        for column_index, cell in enumerate(row):
            x = left + column_index * column_width
            rect = pymupdf.Rect(x, y, x + column_width, y + row_height)
            page.draw_rect(rect, color=(0, 0, 0), width=0.75)  # type: ignore[attr-defined]
            page.insert_text((x + 5, y + 15), cell, fontsize=10)  # type: ignore[attr-defined]
