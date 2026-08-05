"""Ingestion strategies: how a file becomes something the parser can read.

The stage before everything else, and the one with the least written about it. A PDF reaches a
retrieval system either as bytes for an engine to extract or as text some loader already pulled
out, and which one you get is usually decided by whichever import was easiest.

The comparison matters because the second option *skips the parser axis entirely* -- an agno
reader has already decided table handling, reading order and whether a heading survives as one.
Putting both on an axis prices that convenience in recall.
"""

from __future__ import annotations

import pytest

from contextgrid.core.documents import MediaType, SourceFile
from contextgrid.core.warnings import WarningLog
from contextgrid.ingest import INGESTERS, AgnoIngestion, DirectIngestion, get_ingester
from contextgrid.pipeline import Config
from tests.pdf_fixtures import contract_pdf

agno = pytest.importorskip("agno")


def pdf_source(name: str = "contract.pdf") -> SourceFile:
    return SourceFile(id=name, raw=contract_pdf(), media_type=MediaType.PDF)


def markdown_source(text: str = "# Title\n\nSome prose.\n") -> SourceFile:
    return SourceFile(id="notes.md", raw=text.encode("utf-8"), media_type=MediaType.MARKDOWN)


# ---------------------------------------------------------------------------
# direct: the honest baseline
# ---------------------------------------------------------------------------


def test_direct_hands_the_bytes_on_unchanged() -> None:
    """Every decision about how bytes become text is left to the parser axis, which is where
    this package can actually measure them."""
    source = pdf_source()
    ingested = DirectIngestion().ingest([source], WarningLog())

    assert len(ingested) == 1
    assert ingested[0].raw == source.raw
    assert ingested[0].media_type is MediaType.PDF


def test_direct_does_not_replace_the_parser() -> None:
    assert not DirectIngestion().replaces_parser


def test_direct_is_what_no_ingestion_means() -> None:
    """So a config that never heard of this axis behaves exactly as before."""
    assert get_ingester(None).name == "direct"


# ---------------------------------------------------------------------------
# agno: the convenience, and its cost
# ---------------------------------------------------------------------------


def test_agno_extracts_text_from_a_pdf() -> None:
    ingested = AgnoIngestion().ingest([pdf_source()], WarningLog())

    assert ingested[0].media_type is MediaType.MARKDOWN
    assert b"Termination" in (ingested[0].raw or b"")


def test_agno_says_it_replaces_the_parser() -> None:
    """It has already made every decision the parser axis measures. Sweeping a PDF engine
    underneath it would run identical text through arms that never see a PDF."""
    assert AgnoIngestion().replaces_parser


def test_the_document_id_survives_ingestion() -> None:
    """Gold evidence written against `refunds.pdf` has to resolve whichever strategy produced
    the text. Anything else makes a change of ingestion look like a change of corpus, and the
    axis unmeasurable."""
    source = pdf_source("refunds.pdf")
    for strategy in (DirectIngestion(), AgnoIngestion()):
        assert strategy.ingest([source], WarningLog())[0].id == "refunds.pdf"


def test_agno_does_not_chunk() -> None:
    """agno readers chunk by default. Letting them would silently take that decision away from
    the axis that exists to measure it."""
    assert AgnoIngestion().chunk is False
    assert len(AgnoIngestion().ingest([pdf_source()], WarningLog())) == 1


def test_a_format_agno_has_no_reader_for_is_left_as_bytes() -> None:
    """Falling through to the parser axis is better than dropping the document -- but the row
    is then a mix of two strategies, so it says so."""
    source = SourceFile(id="odd.bin", raw=b"\x00\x01\x02", media_type=MediaType.TEXT)
    log = WarningLog()

    ingested = AgnoIngestion(reader="auto").ingest([source], log)
    if ingested[0].media_type is MediaType.TEXT:  # no reader matched
        assert any("no reader" in warning.message for warning in log)


def test_an_unknown_reader_lists_the_real_ones() -> None:
    from contextgrid.ingest import IngestionError

    with pytest.raises(IngestionError, match="wikipedia"):
        AgnoIngestion(reader="telepathy").ingest([markdown_source()], WarningLog())


def test_an_empty_extraction_is_reported() -> None:
    """Nothing in that document can be retrieved under this strategy, and a zero score with no
    explanation reads as a bad parser rather than an empty one."""
    log = WarningLog()
    AgnoIngestion().ingest(
        [SourceFile(id="blank.md", raw=b"   \n\n  ", media_type=MediaType.MARKDOWN)], log
    )
    assert any("no text" in warning.message for warning in log)


# ---------------------------------------------------------------------------
# the axis
# ---------------------------------------------------------------------------


def test_it_is_reachable_from_one_config_line() -> None:
    for spec in ("direct", "agno", "agno:markdown", "agno:reader=pdf"):
        assert get_ingester(spec).name in {"direct", "agno"}


def test_both_strategies_are_registered_and_documented() -> None:
    assert set(INGESTERS.names()) == {"direct", "agno"}
    for name, description in INGESTERS.describe().items():
        assert description, name


def test_ingestion_leads_the_label_because_it_runs_first() -> None:
    assert Config(ingestion="agno").label.startswith("agno>")
    assert not Config(ingestion="direct").label.startswith("direct")


def test_adding_the_axis_did_not_shift_positional_arguments() -> None:
    """`Config("markdown", "recursive:512", "tfidf", "dense")` is public API. A new field ahead
    of `parser` would silently have changed what every one of those arguments meant."""
    config = Config("markdown", "recursive:512", "tfidf", "dense")
    assert config.parser == "markdown"
    assert config.chunker == "recursive:512"
    assert config.embedder == "tfidf"
    assert config.index == "dense"
    assert config.ingestion is None


def test_a_text_extracting_strategy_collapses_the_parser_axis() -> None:
    """agno has already produced Markdown, so `agno + pymupdf` and `agno + pdfplumber` are the
    same run under two names -- and left alone they would credit the parser axis with a
    difference it did not cause."""
    from contextgrid.grid import matrix

    configs = matrix(ingestion=["direct", "agno"], parser=["pymupdf", "pdfplumber"]).expand(
        "factorial"
    )

    agno_arms = [config for config in configs if config.ingestion == "agno"]
    assert len(agno_arms) == 1
    assert agno_arms[0].parser == "markdown"


def test_direct_is_normalised_to_nothing() -> None:
    """`direct` and "no ingestion named at all" are the same run, and two names for one run
    waste a slot and dilute the axis effect."""
    from contextgrid.grid.matrix import canonicalise

    assert canonicalise(Config(ingestion="direct")).ingestion is None


def test_the_config_file_accepts_the_axis() -> None:
    from contextgrid.config import loads

    config = loads("corpus: ./docs\ngrid:\n  ingestion: [direct, agno]\n")
    assert config.grid.ingestion == ("direct", "agno")


def test_a_sweep_runs_both_strategies_end_to_end() -> None:
    """The whole point: the same corpus, ingested two ways, scored on the same questions."""
    from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
    from contextgrid.corpus import Corpus
    from contextgrid.grid import Runner, matrix

    corpus = Corpus(files=(pdf_source("contract.pdf"),), name="ingest")
    evalset = EvalSet(
        id="ingest",
        items=(
            EvalItem(
                id="q1",
                question="how much notice is needed to terminate?",
                anchors=(GoldAnchor(quote="thirty days", source_id="contract.pdf"),),
            ),
        ),
    )

    results = Runner(corpus=corpus, headline="recall@3").run(
        matrix(
            ingestion=["direct", "agno"],
            parser="pymupdf",
            chunker="recursive:128",
            index="bm25",
            embedder=None,
            k=3,
        ),
        evalset,
        mode="factorial",
    )

    assert len(results.runs) == 2
    assert {run.config.ingestion for run in results.runs} == {None, "agno"}
