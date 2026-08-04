"""Query and document prefixes, per model family.

The single most common way a homegrown retrieval evaluation is quietly wrong. E5 was trained
with `query:` and `passage:` in front of the text; BGE wants an instruction on the query and
nothing on the document; most OpenAI models want neither. Embed both sides identically with an
E5 model and it still works -- the numbers just come out several points lower than they should,
uniformly, with nothing on screen to say why.

Worse for this package specifically: the effect is not uniform *across arms*. A sweep that gets
the prefixes wrong for one model and right for another is not comparing the models, it is
comparing one model against a handicapped version of the other. So the prefixes are looked up
from the model name by default, and can always be set explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Prefixes:
    """What to put in front of a query and in front of a document."""

    query: str = ""
    document: str = ""

    @property
    def used(self) -> bool:
        return bool(self.query or self.document)


NONE = Prefixes()

#: Matched against the lowercased model name, longest pattern first, so `e5-mistral` does not
#: pick up the plain `e5` rule when it needs its own.
_RULES: tuple[tuple[str, Prefixes], ...] = (
    # E5 and its descendants. Symmetric-looking but not symmetric.
    ("multilingual-e5", Prefixes(query="query: ", document="passage: ")),
    ("e5-mistral", Prefixes(query="Instruct: Given a query, retrieve relevant passages\nQuery: ")),
    ("e5-", Prefixes(query="query: ", document="passage: ")),
    # BGE English v1.5: instruction on the query, nothing on the passage. v2/m3 need neither.
    (
        "bge-large-en-v1.5",
        Prefixes(query="Represent this sentence for searching relevant passages: "),
    ),
    (
        "bge-base-en-v1.5",
        Prefixes(query="Represent this sentence for searching relevant passages: "),
    ),
    (
        "bge-small-en-v1.5",
        Prefixes(query="Represent this sentence for searching relevant passages: "),
    ),
    # GTE, Nomic, Jina, Snowflake: task prefixes on both sides.
    ("nomic-embed", Prefixes(query="search_query: ", document="search_document: ")),
    ("gte-multilingual", Prefixes()),
)


def for_model(model: str) -> Prefixes:
    """The prefixes this model was trained with, or none if it needs none.

    Unknown models get nothing, which is right: adding a prefix a model was not trained with
    is as wrong as omitting one it was, and silence is the safer default for a name we do not
    recognise.
    """
    lowered = model.lower()
    for pattern, prefixes in _RULES:
        if pattern in lowered:
            return prefixes
    return NONE


def known_families() -> list[str]:
    """The name patterns that carry prefixes, for error messages and documentation."""
    return [pattern for pattern, _ in _RULES]
