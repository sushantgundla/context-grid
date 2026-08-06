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


def test_a_tokenizer_whose_package_is_absent_names_its_extra() -> None:
    """Tested against a synthetic registration rather than a real one.

    It used to assert that `cl100k_base` raised -- which it did, because it was registered
    against a module nobody had written and an extra that did not exist. The test passed for
    the wrong reason, and passing kept the bug invisible. `cl100k_base` works now.
    """
    from contextgrid.core.protocols import Tokenizer
    from contextgrid.core.registry import Registry

    registry: Registry[Tokenizer] = Registry(family="tokenizer")
    registry.register_lazy(
        "imaginary",
        module="contextgrid.this_tokenizer_does_not_exist",
        attr="imaginary",
        extra="embed",
        package="something-unpublished",
    )

    with pytest.raises(MissingExtraError, match="something-unpublished"):
        registry.create("imaginary")


def test_every_registered_tokenizer_can_actually_be_built() -> None:
    """The guarantee the previous test quietly gave up on."""
    for name in TOKENIZERS.names():
        tokenizer = get_tokenizer(name)
        assert tokenizer.count("the refund window is thirty days") > 0


# ---------------------------------------------------------------------------
# exact tokenizers
# ---------------------------------------------------------------------------

tiktoken = pytest.importorskip("tiktoken")

EXACT = ["cl100k_base", "o200k_base"]

HARD_TEXTS = [
    "plain ascii text",
    "The refund window is 30 days — naïve résumé 日本語.",
    "ありがとうございます",
    "🙂🙃 emoji " * 5,
    "",
    "   \n\n  ",
]


@pytest.mark.parametrize("name", EXACT)
def test_an_exact_tokenizer_says_it_is_exact(name: str) -> None:
    """The cost model refuses to price with an approximate one, so this flag is load-bearing.

    Both of these were registered against a module nobody had written, so asking for one raised
    an install instruction naming an extra that also did not exist -- an error telling you to
    do something that would not have helped.
    """
    tokenizer = get_tokenizer(name)
    assert tokenizer.exact
    assert tokenizer.name == name


@pytest.mark.parametrize("name", EXACT)
@pytest.mark.parametrize("text", HARD_TEXTS)
def test_spans_are_ordered_and_never_overlap(name: str, text: str) -> None:
    """A chunker cuts on these. Overlapping or out-of-order ranges put a chunk's text somewhere
    it does not belong, and every gold span inside it moves with it."""
    spans = get_tokenizer(name).token_spans(text)

    for (_, end), (start, _) in pairwise(spans):
        assert end <= start
    for start, end in spans:
        assert 0 <= start < end <= len(text)


@pytest.mark.parametrize("name", EXACT)
@pytest.mark.parametrize("text", HARD_TEXTS)
def test_the_spans_reconstruct_the_text(name: str, text: str) -> None:
    """No character may be lost between tokens. A gap is text no chunk can ever contain."""
    spans = get_tokenizer(name).token_spans(text)
    assert "".join(text[start:end] for start, end in spans) == text


@pytest.mark.parametrize("name", EXACT)
def test_a_token_boundary_inside_a_character_does_not_break_the_offsets(name: str) -> None:
    """tiktoken splits on bytes, and a CJK character is three of them. A boundary landing
    mid-character has no character offset of its own -- the first attempt at this fell back to
    the end of the text and produced spans that ran backwards."""
    text = "日本語のテキストと English mixed together 中文"
    spans = get_tokenizer(name).token_spans(text)

    assert "".join(text[start:end] for start, end in spans) == text
    assert spans == sorted(spans)


@pytest.mark.parametrize("name", EXACT)
def test_it_counts_more_tokens_than_word_splitting_does(name: str) -> None:
    """Roughly a third more on English prose, which is the entire reason the regex tokenizer is
    barred from the cost model."""
    text = "The quick brown fox jumps over the lazy dog, repeatedly and enthusiastically. " * 4
    assert get_tokenizer(name).count(text) > get_tokenizer("regex").count(text)


@pytest.mark.parametrize("name", EXACT)
def test_counting_agrees_with_tiktoken_itself(name: str) -> None:
    """Cross-checked against the library rather than trusted, the same way the retrieval
    metrics are cross-checked against ranx."""
    text = "Refunds are issued within 30 days of purchase."
    assert get_tokenizer(name).count(text) == len(tiktoken.get_encoding(name).encode(text))


def test_an_empty_string_has_no_tokens() -> None:
    assert get_tokenizer("cl100k_base").count("") == 0
    assert get_tokenizer("cl100k_base").token_spans("") == []


def test_a_real_tokenizer_can_drive_a_chunker() -> None:
    """The point of `token_spans`: a chunker has to cut at a token boundary *and* report the
    character offsets it cut at."""
    from contextgrid.chunk import get_chunker
    from contextgrid.core.documents import MediaType, SourceFile
    from contextgrid.parse import get_parser

    text = "# Notes\n\n" + ("The refund window is thirty days. " * 40)
    parsed = get_parser("markdown").parse(
        SourceFile(id="d", raw=text.encode("utf-8"), media_type=MediaType.MARKDOWN)
    )
    chunks = get_chunker("recursive:64,tokenizer=cl100k_base").chunk(parsed)

    assert len(chunks) > 1
    for chunk in chunks:
        assert parsed.text[chunk.span.start : chunk.span.end] == chunk.text
        assert chunk.token_counts["cl100k_base"] > 0
