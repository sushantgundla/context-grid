"""Model calls, through litellm.

Three hand-written provider clients became one adapter. What is tested here is that the config
surface did not change under people, that failures say what to do, and that litellm's price
table now backs the ten hand-maintained entries the cost model carries.

No model is called for real -- CI has no key, and a suite that needs one is a suite nobody runs.
"""

from __future__ import annotations

import sys
import types
from typing import Any, ClassVar

import pytest

from contextgrid.cost.model import PRICES, CostModel
from contextgrid.evalset import LLMS, get_llm
from contextgrid.evalset.llm import LiteLLMChat, LLMError


def install_fake_litellm(monkeypatch: pytest.MonkeyPatch, **behaviour: Any) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    module = types.ModuleType("litellm")

    def completion(**kwargs: Any) -> Any:
        seen.update(kwargs)
        if behaviour.get("raises"):
            raise behaviour["raises"]

        class Message:
            content = behaviour.get("reply", "the answer")

        class Choice:
            message = Message()

        class Response:
            choices: ClassVar[list[Any]] = [Choice()]

        return Response()

    module.completion = completion  # type: ignore[attr-defined]
    module.model_cost = behaviour.get("model_cost", {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", module)
    return seen


# ---------------------------------------------------------------------------
# the config surface people already wrote
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("openai", "openai/gpt-4o-mini"),
        ("openai:gpt-4o", "openai/gpt-4o"),
        ("anthropic", "anthropic/claude-sonnet-5"),
        ("anthropic:claude-opus-5", "anthropic/claude-opus-5"),
        ("litellm:gemini/gemini-2.0-flash", "gemini/gemini-2.0-flash"),
        ("litellm:ollama/llama3", "ollama/llama3"),
    ],
)
def test_every_spec_still_resolves(spec: str, expected: str) -> None:
    """`openai:gpt-4o-mini` and `anthropic:...` were separate hand-written clients. They are
    now one adapter under three names, so configs written against the old ones stay valid."""
    assert get_llm(spec).name == expected


def test_a_bare_provider_name_works() -> None:
    """An axis value that needs a parameter before it is usable at all is a bad axis value."""
    assert get_llm("openai").name.startswith("openai/")


def test_an_already_qualified_model_is_not_double_prefixed() -> None:
    assert get_llm("openai:openai/gpt-4o").name == "openai/gpt-4o"


def test_the_registry_lists_all_three() -> None:
    assert set(LLMS.names()) == {"litellm", "openai", "anthropic"}
    for name, description in LLMS.describe().items():
        assert description, name


# ---------------------------------------------------------------------------
# the call
# ---------------------------------------------------------------------------


def test_the_prompt_reaches_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = install_fake_litellm(monkeypatch)
    reply = LiteLLMChat(model="openai/gpt-4o-mini").complete("what is the refund window?")

    assert seen["messages"] == [{"role": "user", "content": "what is the refund window?"}]
    assert seen["model"] == "openai/gpt-4o-mini"
    assert reply == "the answer"


def test_max_tokens_and_temperature_are_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = install_fake_litellm(monkeypatch)
    LiteLLMChat(model="m", temperature=0.7).complete("prompt", max_tokens=64)

    assert seen["max_tokens"] == 64
    assert seen["temperature"] == 0.7


def test_temperature_defaults_to_zero() -> None:
    """Question generation and filtering both want the same answer twice. A default that
    varies makes an eval set that cannot be reproduced."""
    assert LiteLLMChat().temperature == 0.0


def test_an_api_base_is_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Which is how a local ollama or vLLM server is reached."""
    seen = install_fake_litellm(monkeypatch)
    LiteLLMChat(model="ollama/llama3", api_base="http://localhost:11434").complete("hi")
    assert seen["api_base"] == "http://localhost:11434"


def test_no_api_key_is_sent_when_none_was_given(monkeypatch: pytest.MonkeyPatch) -> None:
    """litellm reads the provider's environment variable itself. Passing an empty key would
    override that with nothing."""
    seen = install_fake_litellm(monkeypatch)
    LiteLLMChat(model="m").complete("hi")
    assert "api_key" not in seen


def test_an_empty_reply_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that declines to answer is a result, and the caller decides what it means."""
    install_fake_litellm(monkeypatch, reply=None)
    assert LiteLLMChat(model="m").complete("hi") == ""


def test_the_transport_hook_replaces_the_call_entirely() -> None:
    """So anything that calls a model -- question generation, filtering, query transforms --
    can be exercised without a key or a network."""
    seen: list[tuple[str, int]] = []
    model = LiteLLMChat(
        model="m", transport=lambda prompt, limit: (seen.append((prompt, limit)), "scripted")[1]
    )
    assert model.complete("question?", max_tokens=32) == "scripted"
    assert seen == [("question?", 32)]


# ---------------------------------------------------------------------------
# failure
# ---------------------------------------------------------------------------


def test_a_missing_key_names_the_variables_to_set(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_litellm(monkeypatch, raises=Exception("AuthenticationError: no api key"))
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        LiteLLMChat(model="anthropic/claude-sonnet-5").complete("hi")


def test_a_missing_key_says_keys_are_never_written_down(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_litellm(monkeypatch, raises=Exception("AuthenticationError: no api key"))
    with pytest.raises(LLMError, match="never written to disk"):
        LiteLLMChat(model="m").complete("hi")


def test_an_unqualified_model_explains_the_provider_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_litellm(monkeypatch, raises=Exception("LLM Provider NOT provided: not found"))
    with pytest.raises(LLMError, match="provider/model"):
        LiteLLMChat(model="claude-sonnet-5").complete("hi")


def test_rate_limiting_suggests_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_litellm(monkeypatch, raises=Exception("429 rate limit exceeded"))
    with pytest.raises(LLMError, match="smaller model"):
        LiteLLMChat(model="m").complete("hi")


def test_an_unreachable_endpoint_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_litellm(monkeypatch, raises=Exception("Connection refused"))
    with pytest.raises(LLMError, match="Is it running"):
        LiteLLMChat(model="ollama/llama3").complete("hi")


def test_an_unreadable_reply_says_which_model(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("litellm")
    module.completion = lambda **kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", module)

    with pytest.raises(LLMError, match="could not read"):
        LiteLLMChat(model="odd-model").complete("hi")


# ---------------------------------------------------------------------------
# pricing, which comes free with the dependency
# ---------------------------------------------------------------------------


def test_the_hand_written_table_still_wins() -> None:
    """A cost comparison has to be reproducible. Where this package has an opinion, a table
    that moves under you between runs must not override it."""
    assert CostModel().pricing_for("text-embedding-3-small").embed_per_million == (
        PRICES["text-embedding-3-small"].embed_per_million
    )


def test_a_model_the_table_never_heard_of_is_priced_by_litellm() -> None:
    """Ten hand-maintained entries against the whole field means most real models would be
    costed at zero -- and a silent zero reads as "free" rather than "unknown"."""
    pricing = CostModel().pricing_for("anthropic/claude-sonnet-5")
    assert pricing.metered
    assert pricing.generate_input_per_million > 0
    assert pricing.generate_output_per_million > pricing.generate_input_per_million


def test_output_tokens_are_priced_separately() -> None:
    """Every provider charges more for output than input, often several times more. Collapsing
    them into one number makes a verbose model look as cheap as a terse one."""
    pricing = CostModel().pricing_for("openai/gpt-4o-mini")
    assert pricing.generate_output_per_million > pricing.generate_input_per_million


def test_a_hosted_embedding_model_is_priced_as_an_embedder() -> None:
    pricing = CostModel().pricing_for("litellm:voyage-3-large")
    assert pricing.embed_per_million > 0
    assert pricing.generate_input_per_million == 0


def test_a_model_nobody_has_a_price_for_still_warns() -> None:
    """The fallback must not swallow the warning, or an unpriced model silently costs zero."""
    model = CostModel()
    model.pricing_for("a-model-that-does-not-exist-anywhere")
    assert any("no published price" in warning.message for warning in model.warnings)


def test_a_local_model_is_not_looked_up_at_all() -> None:
    model = CostModel()
    assert not model.pricing_for("tei:bge-base-en-v1.5").metered
    assert not list(model.warnings)


def test_pricing_works_without_litellm_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """litellm is an optional extra. The cost model has to keep working on a bare install."""
    monkeypatch.setitem(sys.modules, "litellm", None)
    model = CostModel()
    assert model.pricing_for("text-embedding-3-small").embed_per_million > 0
    assert not model.pricing_for("something-unknown").metered
