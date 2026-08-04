"""Building, filtering, reviewing and importing eval sets.

The heart of the product. Everything else measures; this is what it measures against.
"""

from __future__ import annotations

from contextgrid.evalset.classify import (
    Classifier,
    classify_question,
    type_distribution,
)
from contextgrid.evalset.filters import (
    DanglingReferenceFilter,
    DuplicateFilter,
    Filter,
    FilterChain,
    FilterResult,
    GeneralKnowledgeFilter,
    NonDiscriminatingFilter,
    Rejection,
    ShortQuestionFilter,
    UnresolvedEvidenceFilter,
    default_filters,
)
from contextgrid.evalset.generate import (
    Generation,
    KeywordProbeGenerator,
    LLMQuestionGenerator,
    QuestionGenerator,
    generate,
)
from contextgrid.evalset.io import (
    read_beir,
    read_csv,
    read_jsonl,
    read_legalbench_rag,
    write_csv,
    write_jsonl,
)
from contextgrid.evalset.llm import (
    LLM,
    LLMS,
    LiteLLMChat,
    LLMError,
    RecordingLLM,
    answerer_from,
    get_llm,
    parse_json_reply,
)
from contextgrid.evalset.quality import (
    EvalSetQuality,
    assess,
    minimum_detectable_difference,
)
from contextgrid.evalset.review import (
    Decision,
    ReviewQueue,
    Verdict,
    pending,
    review_summary,
)

__all__ = [
    "LLM",
    "LLMS",
    "Classifier",
    "DanglingReferenceFilter",
    "Decision",
    "DuplicateFilter",
    "EvalSetQuality",
    "Filter",
    "FilterChain",
    "FilterResult",
    "GeneralKnowledgeFilter",
    "Generation",
    "KeywordProbeGenerator",
    "LLMError",
    "LLMQuestionGenerator",
    "LiteLLMChat",
    "NonDiscriminatingFilter",
    "QuestionGenerator",
    "RecordingLLM",
    "Rejection",
    "ReviewQueue",
    "ShortQuestionFilter",
    "UnresolvedEvidenceFilter",
    "Verdict",
    "answerer_from",
    "assess",
    "classify_question",
    "default_filters",
    "generate",
    "get_llm",
    "minimum_detectable_difference",
    "parse_json_reply",
    "pending",
    "read_beir",
    "read_csv",
    "read_jsonl",
    "read_legalbench_rag",
    "review_summary",
    "type_distribution",
    "write_csv",
    "write_jsonl",
]
