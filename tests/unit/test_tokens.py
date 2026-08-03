"""Unit tests for tokenizers.

The property that matters most is that `token_spans` tiles the text without overlapping,
because chunkers cut at those boundaries and claim the result is a literal slice.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from contextgrid.core.errors import MissingExtraError
from contextgrid.tokens import TOKENIZERS, CharacterTokenizer, RegexTokenizer, get_tokenizer

TOKENIZER_CASES = [RegexTokenizer(), CharacterTokenizer()]
IDS = [t.name for t in TOKENIZER_CASES]

SAMPLE = "The notice period is thirty days. Either party may terminate."


@pytest.fixture(params=TOKENIZER_CASES, ids=IDS)
def tokenizer(request: pytest.FixtureRequest) -> RegexTokenizer | CharacterTokenizer:
    return request.param  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------


def test_count_matches_the_number_of_spans(tokenizer: RegexTokenizer) -> None:
    assert tokenizer.count(SAMPLE) == len(tokenizer.token_spans(SAMPLE))


def test_spans_are_in_order_and_do_not_overlap(tokenizer: RegexTokenizer) -> None:
    spans = tokenizer.token_spans(SAMPLE)
    for (_, end), (next_start, _) in pairwise(spans):
        assert end <= next_start


def test_spans_stay_inside_the_text(tokenizer: RegexTokenizer) -> None:
    for start, end in tokenizer.token_spans(SAMPLE):
        assert 0 <= start < end <= len(SAMPLE)


def test_spans_cover_every_non_whitespace_character(tokenizer: RegexTokenizer) -> None:
    """A chunker cutting at token boundaries must not be able to drop content between them."""
    covered: set[int] = set()
    for start, end in tokenizer.token_spans(SAMPLE):
        covered.update(range(start, end))
    missing = [i for i, c in enumerate(SAMPLE) if not c.isspace() and i not in covered]
    assert not missing


def test_empty_text_has_no_tokens(tokenizer: RegexTokenizer) -> None:
    assert tokenizer.token_spans("") == []
    assert tokenizer.count("") == 0


def test_declares_whether_it_is_exact(tokenizer: RegexTokenizer) -> None:
    assert isinstance(tokenizer.exact, bool)


# ---------------------------------------------------------------------------
# the specific tokenizers
# ---------------------------------------------------------------------------


def test_regex_splits_words_and_punctuation() -> None:
    spans = RegexTokenizer().token_spans("Hello, world!")
    assert ["Hello, world!"[s:e] for s, e in spans] == ["Hello", ",", "world", "!"]


def test_regex_is_honest_about_being_approximate() -> None:
    """It is fine for chunking and wrong for costing, and the flag is what enforces that."""
    assert RegexTokenizer().exact is False


def test_character_tokenizer_is_exact_and_counts_characters() -> None:
    tokenizer = CharacterTokenizer()
    assert tokenizer.exact is True
    assert tokenizer.count("abc") == 3
    assert tokenizer.token_spans("abc") == [(0, 1), (1, 2), (2, 3)]


def test_character_tokenizer_counts_whitespace_too() -> None:
    assert CharacterTokenizer().count("a b") == 3


def test_the_two_disagree_which_is_the_whole_point() -> None:
    """ "512 tokens" means different text under different tokenizers, and a comparison that
    does not name its tokenizer is not reproducible."""
    assert RegexTokenizer().count(SAMPLE) != CharacterTokenizer().count(SAMPLE)


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def test_registry_knows_the_built_ins() -> None:
    assert "regex" in TOKENIZERS
    assert "character" in TOKENIZERS


def test_get_tokenizer_from_a_name() -> None:
    assert get_tokenizer("character").name == "character"


def test_get_tokenizer_defaults_to_regex() -> None:
    assert get_tokenizer(None).name == "regex"


def test_get_tokenizer_passes_an_instance_through() -> None:
    instance = CharacterTokenizer()
    assert get_tokenizer(instance) is instance


def test_a_model_tokenizer_needs_its_extra() -> None:
    with pytest.raises(MissingExtraError, match="tiktoken"):
        TOKENIZERS.create("cl100k_base")
