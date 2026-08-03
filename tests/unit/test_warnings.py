"""Unit tests for the structured warnings channel.

Warnings are the mechanism that stops a silently unfair comparison being read as a fair one,
so the severity ladder and the soundness check need to behave exactly as advertised.
"""

from __future__ import annotations

from contextgrid import GridWarning, MissingExtraError, Severity, WarningCode, WarningLog


def test_a_fresh_log_is_empty_and_sound() -> None:
    log = WarningLog()
    assert not log
    assert len(log) == 0
    assert log.is_sound
    assert log.summary() == "no warnings"


def test_add_returns_the_warning_it_recorded() -> None:
    log = WarningLog()
    warning = log.add(
        WarningCode.INPUT_TRUNCATED,
        "chunk exceeded the model context",
        stage="embed",
        subject="c17",
        limit=512,
        actual=730,
    )
    assert warning.code is WarningCode.INPUT_TRUNCATED
    assert warning.stage == "embed"
    assert warning.detail == {"limit": 512, "actual": 730}
    assert list(log) == [warning]


def test_severity_defaults_to_caution() -> None:
    log = WarningLog()
    warning = log.add(WarningCode.OCR_APPLIED, "ocr ran on 3 pages")
    assert warning.severity is Severity.CAUTION


def test_invalidating_warnings_make_a_log_unsound() -> None:
    log = WarningLog()
    log.add(WarningCode.OCR_APPLIED, "fine", severity=Severity.INFO)
    assert log.is_sound

    log.add(
        WarningCode.APPROXIMATE_OFFSETS,
        "offsets are guesses, this comparison is not valid",
        severity=Severity.INVALID,
    )
    assert not log.is_sound
    assert len(log.invalidating) == 1


def test_at_least_filters_up_the_severity_ladder() -> None:
    log = WarningLog()
    log.add(WarningCode.OCR_APPLIED, "info", severity=Severity.INFO)
    log.add(WarningCode.ANN_RECALL_LOSS, "caution", severity=Severity.CAUTION)
    log.add(WarningCode.APPROXIMATE_OFFSETS, "invalid", severity=Severity.INVALID)

    assert len(log.at_least(Severity.INFO)) == 3
    assert len(log.at_least(Severity.CAUTION)) == 2
    assert len(log.at_least(Severity.INVALID)) == 1


def test_of_code_selects_by_kind() -> None:
    log = WarningLog()
    log.add(WarningCode.INPUT_TRUNCATED, "a")
    log.add(WarningCode.INPUT_TRUNCATED, "b")
    log.add(WarningCode.ANN_RECALL_LOSS, "c")

    assert len(log.of_code(WarningCode.INPUT_TRUNCATED)) == 2
    assert len(log.of_code(WarningCode.INPUT_TRUNCATED, WarningCode.ANN_RECALL_LOSS)) == 3
    assert log.of_code(WarningCode.BUDGET_REACHED) == []


def test_extend_and_merge() -> None:
    first = WarningLog()
    first.add(WarningCode.OCR_APPLIED, "a")
    second = WarningLog()
    second.add(WarningCode.ANN_RECALL_LOSS, "b")

    merged = first.merge(second)
    assert len(merged) == 2
    assert len(first) == 1  # merge does not mutate

    first.extend(second)
    assert len(first) == 2


def test_counts_and_summary() -> None:
    log = WarningLog()
    log.add(WarningCode.INPUT_TRUNCATED, "a")
    log.add(WarningCode.INPUT_TRUNCATED, "b")
    log.add(WarningCode.OCR_APPLIED, "c")

    assert log.counts() == {"input_truncated": 2, "ocr_applied": 1}
    assert log.summary() == "input_truncated x2, ocr_applied x1"


def test_warning_renders_readably() -> None:
    warning = GridWarning(
        code=WarningCode.INPUT_TRUNCATED,
        message="chunk was cut at 512 tokens",
        severity=Severity.CAUTION,
        stage="embed",
        subject="c17",
    )
    assert str(warning) == "CAUTION [embed] (c17): chunk was cut at 512 tokens"


def test_warning_renders_without_stage_or_subject() -> None:
    warning = GridWarning(code=WarningCode.BUDGET_REACHED, message="stopped at $2.00")
    assert str(warning) == "CAUTION: stopped at $2.00"


def test_warnings_serialise() -> None:
    log = WarningLog()
    log.add(WarningCode.ANN_RECALL_LOSS, "lost 8% recall", stage="index", ef_search=64)

    payload = log.to_list()
    assert payload == [
        {
            "code": "ann_recall_loss",
            "message": "lost 8% recall",
            "severity": "caution",
            "stage": "index",
            "subject": None,
            "detail": {"ef_search": 64},
        }
    ]


def test_codes_serialise_as_their_own_strings() -> None:
    assert WarningCode.INPUT_TRUNCATED.value == "input_truncated"
    assert Severity.INVALID.value == "invalid"


# ---------------------------------------------------------------------------
# MissingExtraError
# ---------------------------------------------------------------------------


def test_missing_extra_names_the_install_command() -> None:
    error = MissingExtraError("The docling parser", "parse-ml", package="docling")
    message = str(error)
    assert 'pip install "context-grid[parse-ml]"' in message
    assert "docling" in message
    assert error.extra == "parse-ml"


def test_missing_extra_without_a_named_package() -> None:
    error = MissingExtraError("Semantic chunking", "chunk")
    assert 'pip install "context-grid[chunk]"' in str(error)
    assert error.package is None


def test_missing_extra_is_an_import_error() -> None:
    """So `except ImportError` in user code still catches it."""
    assert isinstance(MissingExtraError("x", "y"), ImportError)
