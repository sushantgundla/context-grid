"""The demo corpus: three documents the questions point at, and enough distractors to make
retrieval an actual test.

The first version of this demo used the three documents alone. Almost every configuration
scored 1.000, because a corpus of 4,900 characters produces about twenty chunks and asking
for the top five retrieves a quarter of the index. The eval set's own quality score said as
much -- it could only detect differences above 0.40 -- and a saturated leaderboard would have
undermined the entire page.

So the corpus carries distractors: thirty more documents of the same *shape and vocabulary*
as the real ones. They talk about notice periods, rate limits, retention windows and support
tiers, with different parties and different numbers. That is what makes retrieval hard in
practice -- not unrelated text, which any method separates, but near-neighbours that share
every word except the one that matters.

They are synthetic, and the page says so.
"""

from __future__ import annotations

VENDORS = [
    ("Northwind Logistics", "NL"),
    ("Bramble Analytics", "BA"),
    ("Corvid Systems", "CS"),
    ("Dunmore Freight", "DF"),
    ("Eastgate Media", "EM"),
    ("Fenwick Legal", "FL"),
    ("Glasswing Labs", "GL"),
    ("Harrow Instruments", "HI"),
    ("Ironbridge Capital", "IC"),
    ("Juniper Retail", "JR"),
]

SERVICES = [
    ("Ledger", "ledger", "invoices"),
    ("Beacon", "beacon", "alerts"),
    ("Quarry", "quarry", "datasets"),
    ("Tessellate", "tessellate", "layouts"),
    ("Windlass", "windlass", "jobs"),
]

DOMAINS = [
    ("Business continuity", "continuity", "recovery point objective"),
    ("Change management", "change", "change advisory board"),
    ("Physical security", "physical", "badge access"),
    ("Acceptable use", "acceptable-use", "personal devices"),
    ("Supplier code of conduct", "supplier-conduct", "labour standards"),
]


def _contract(name: str, code: str, index: int) -> tuple[str, str]:
    """A contract with the same clauses and different numbers."""
    notice = 15 + (index * 5) % 60
    cure = 7 + (index * 3) % 21
    renewal = 30 + (index * 10) % 90
    standard = 800 + index * 137
    premium = 2400 + index * 211
    liability = 6 + index % 18
    confidential = 2 + index % 6
    interest = 1 + index % 4

    return (
        f"{code.lower()}-services-agreement.md",
        f"""\
# Services Agreement — {name}

## 1. Term and renewal

This agreement runs for twelve months from the Effective Date and renews for successive
twelve month terms unless either party gives notice of non-renewal at least {renewal} days
before the end of the then-current term.

## 2. Termination

### 2.1 For convenience

Either party may terminate this agreement for convenience on {notice} days written notice
to the other party.

### 2.2 For cause

A party may terminate immediately if the other commits a material breach and fails to
remedy it within {cure} days of written notice.

## 3. Fees

| Service tier | Monthly fee | Setup fee |
|---|---|---|
| Standard | ${standard:,} | $250 |
| Premium | ${premium:,} | $750 |

Invoices are payable within thirty days. Late payment attracts interest at
{interest} percent per month.

## 4. Liability and confidentiality

Aggregate liability under this agreement is capped at the fees paid in the preceding
{liability} months. Confidential Information must be kept secret for {confidential} years
from disclosure.
""",
    )


def _api(display: str, slug: str, noun: str, index: int) -> tuple[str, str]:
    """An API reference with the same sections and different limits."""
    limit = 100 + index * 250
    burst = limit * 2
    retention = 7 + (index * 11) % 60
    rotation = 30 + (index * 7) % 120

    return (
        f"{slug}-api.md",
        f"""\
# {display} API reference

The {display} API manages {noun}. All endpoints return JSON under /v1.

## Authentication

Send your key in the `{display}-Key` header. Requests without a valid key return 401.
Rotated keys stay valid for {rotation} minutes after rotation.

## Rate limits

The default limit is {limit} requests per minute per key. Exceeding it returns 429 with a
`Retry-After` header. Burst traffic up to {burst} requests per minute is tolerated briefly.

## Endpoints

### POST /v1/{slug}

Creates a record. The body must be JSON and must include a `name`.

### DELETE /v1/{slug}/{{id}}

Deletes a record. Deletion is soft for {retention} days, after which the record is purged.
""",
    )


def _policy(display: str, slug: str, subject: str, index: int) -> tuple[str, str]:
    """A policy with the same headings and different windows."""
    report = 1 + index % 8
    retain = 30 + (index * 15) % 300
    review = ["monthly", "quarterly", "twice yearly", "annually"][index % 4]
    notify = 24 + (index % 4) * 24

    return (
        f"{slug}-policy.md",
        f"""\
# {display} policy

## Scope

This policy covers {subject} across all production environments and applies to employees
and contractors alike.

## Controls

Access is reviewed {review} and revoked within one business day of departure. Records
relating to {subject} are retained for {retain} days.

## Incidents

Suspected incidents must be reported within {report} hours of discovery. Affected parties
are notified within {notify} hours of confirmation.

## Review

This policy is reviewed {review} by the owning team and approved by the security lead.
""",
    )


def distractors() -> dict[str, str]:
    """Thirty documents that share the real ones' vocabulary and none of their answers.

    The point is near-neighbours. A corpus of unrelated text is easy -- any method separates
    a contract from an API reference. What is hard, and what happens in every real corpus, is
    twenty documents that all say "notice period" and mean different numbers.
    """
    documents: dict[str, str] = {}

    for index, (name, code) in enumerate(VENDORS):
        filename, text = _contract(name, code, index)
        documents[filename] = text

    for index, (display, slug, noun) in enumerate(SERVICES):
        filename, text = _api(display, slug, noun, index)
        documents[filename] = text

    for index, (display, slug, subject) in enumerate(DOMAINS):
        filename, text = _policy(display, slug, subject, index)
        documents[filename] = text

    # A second pass of contracts under different counterparties, so the distractor set is
    # large enough that the top five is a real choice rather than most of the index.
    for index, (name, code) in enumerate(VENDORS):
        filename, text = _contract(f"{name} (Renewal)", f"{code}R", index + 7)
        documents[filename] = text

    return documents


def distractor_questions() -> list[tuple[str, str, str, str, str]]:
    """Questions against the distractors, with the exact numbers as their evidence.

    Twenty-four hand-written questions can only detect differences above 0.40, which the
    eval set's own quality score says plainly -- and a demo whose winner is indistinguishable
    from the runner-up is a weak demo. Generating questions against documents whose numbers
    are known produces a set large enough to settle something.

    Every one is answerable from exactly one document, and every distractor says something
    similar with a different number -- so getting them right requires retrieving the right
    document, not just the right topic.
    """
    questions: list[tuple[str, str, str, str, str]] = []

    for index, (name, code) in enumerate(VENDORS):
        filename = f"{code.lower()}-services-agreement.md"
        notice = 15 + (index * 5) % 60
        cure = 7 + (index * 3) % 21
        standard = 800 + index * 137
        questions += [
            (
                f"d{index}a",
                f"How much notice must {name} be given to terminate for convenience?",
                filename,
                f"on {notice} days written notice",
                "numeric",
            ),
            (
                f"d{index}b",
                f"How long does {name} have to remedy a material breach?",
                filename,
                f"remedy it within {cure} days",
                "numeric",
            ),
            (
                f"d{index}c",
                f"What is the Standard tier monthly fee for {name}?",
                filename,
                f"| Standard | ${standard:,} |",
                "tabular",
            ),
        ]

    for index, (display, slug, _noun) in enumerate(SERVICES):
        filename = f"{slug}-api.md"
        limit = 100 + index * 250
        retention = 7 + (index * 11) % 60
        questions += [
            (
                f"s{index}a",
                f"What is the rate limit on the {display} API?",
                filename,
                f"The default limit is {limit} requests per minute",
                "numeric",
            ),
            (
                f"s{index}b",
                f"How long is a deleted {display} record recoverable?",
                filename,
                f"Deletion is soft for {retention} days",
                "numeric",
            ),
        ]

    for index, (display, slug, subject) in enumerate(DOMAINS):
        filename = f"{slug}-policy.md"
        report = 1 + index % 8
        retain = 30 + (index * 15) % 300
        questions += [
            (
                f"p{index}a",
                f"How quickly must a {display.lower()} incident be reported?",
                filename,
                f"reported within {report} hours of discovery",
                "numeric",
            ),
            (
                f"p{index}b",
                f"How long are {subject} records retained?",
                filename,
                f"retained for {retain} days",
                "numeric",
            ),
        ]

    return questions


def describe() -> str:
    """One line for the page, so nobody mistakes the distractors for real documents."""
    return (
        f"{len(distractors())} synthetic near-neighbour documents accompany the three real "
        "ones. They use the same vocabulary -- notice periods, rate limits, retention "
        "windows -- with different parties and different numbers, because near-neighbours "
        "are what makes retrieval hard in practice."
    )
