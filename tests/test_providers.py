"""Any model, over the standard library.

"Works with your model" has to mean more than "works once you install our four
SDKs", so every network adapter here is plain HTTP and every one of them is
tested against a fake server rather than a mocked SDK.
"""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest

from llm_judge import AnthropicModel, ChatModel, ModelError, OllamaModel, from_spec


class FakeHTTP:
    """Stands in for urlopen: records the requests, returns canned answers."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def __call__(self, request: Any, timeout: float = 0) -> Any:
        self.requests.append(
            (request.full_url, dict(request.headers), json.loads(request.data.decode()))
        )
        answer = self.answers.pop(0) if self.answers else {}
        if isinstance(answer, Exception):
            raise answer
        return _Response(json.dumps(answer).encode())


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self, limit: int | None = None) -> bytes:
        return self._body[:limit] if limit else self._body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def chat_answer(text: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": text}}]}


def install(monkeypatch: pytest.MonkeyPatch, http: FakeHTTP) -> None:
    monkeypatch.setattr("urllib.request.urlopen", http)
    monkeypatch.setattr("llm_judge.providers.BACKOFF_SECONDS", 0.0)


def test_a_chat_server_gets_the_system_prompt_as_its_own_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    http = FakeHTTP(chat_answer("pass"))
    install(monkeypatch, http)
    assert ChatModel("gpt-4.1-mini").complete("judge this", system="be strict") == "pass"
    url, headers, body = http.requests[0]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer k"
    assert body["messages"] == [
        {"role": "system", "content": "be strict"},
        {"role": "user", "content": "judge this"},
    ]


def test_a_local_server_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty key variable means the server wants none, which is how every
    self-hosted vLLM and LM Studio runs."""
    http = FakeHTTP(chat_answer("fail"))
    install(monkeypatch, http)
    model = ChatModel("qwen2.5-72b", base_url="http://localhost:8000/v1", api_key_env="")
    assert model.complete("p") == "fail"
    assert "Authorization" not in http.requests[0][1]


def test_a_missing_key_says_which_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    install(monkeypatch, FakeHTTP())
    with pytest.raises(ModelError, match="GROQ_API_KEY"):
        from_spec("groq:llama-3.3-70b").complete("p")


def test_content_blocks_are_joined(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some servers answer with blocks where OpenAI answers with a string."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    blocks = {"choices": [{"message": {"content": [{"text": "bl"}, {"text": "ocked"}]}}]}
    install(monkeypatch, FakeHTTP(blocks))
    assert ChatModel("m").complete("p") == "blocked"


def test_a_rate_limit_is_retried_and_a_refusal_is_not(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    too_many = urllib.error.HTTPError("u", 429, "slow down", {}, io.BytesIO(b"rate limited"))  # type: ignore[arg-type]
    http = FakeHTTP(too_many, chat_answer("pass"))
    install(monkeypatch, http)
    assert ChatModel("m").complete("p") == "pass"
    assert len(http.requests) == 2

    forbidden = urllib.error.HTTPError("u", 403, "no", {}, io.BytesIO(b"key revoked"))  # type: ignore[arg-type]
    install(monkeypatch, FakeHTTP(forbidden))
    with pytest.raises(ModelError, match="403"):
        ChatModel("m").complete("p")


def test_it_gives_up_and_says_what_it_saw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    errors = [urllib.error.URLError("connection refused") for _ in range(3)]
    install(monkeypatch, FakeHTTP(*errors))
    with pytest.raises(ModelError, match="gave up after 3"):
        ChatModel("m").complete("p")


def test_an_unexpected_shape_is_an_error_not_a_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """A judge that reads an error page as a verdict is worse than one that stops."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    install(monkeypatch, FakeHTTP({"error": {"message": "model not found"}}))
    with pytest.raises(ModelError, match="unexpected answer shape"):
        ChatModel("m").complete("p")


def test_ollama_posts_to_its_own_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    http = FakeHTTP({"message": {"content": "pass"}})
    install(monkeypatch, http)
    assert OllamaModel("qwen2.5:14b").complete("p", temperature=0.2) == "pass"
    url, _, body = http.requests[0]
    assert url == "http://localhost:11434/api/chat"
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0.2


def test_anthropic_speaks_the_messages_api_without_an_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    http = FakeHTTP({"content": [{"type": "text", "text": "blocked"}]})
    install(monkeypatch, http)
    assert AnthropicModel().complete("p", system="s") == "blocked"
    url, headers, body = http.requests[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["X-api-key"] == "k"
    assert body["system"] == "s"


@pytest.mark.parametrize(
    ("spec", "expected_url"),
    [
        ("openai:gpt-4.1-mini", "https://api.openai.com/v1"),
        ("openrouter:qwen/qwen3-max", "https://openrouter.ai/api/v1"),
        ("groq:llama-3.3-70b", "https://api.groq.com/openai/v1"),
        ("gemini:gemini-2.5-flash", "https://generativelanguage.googleapis.com/v1beta/openai"),
        ("ollama:llama3.1", "http://localhost:11434"),
        ("anthropic:claude-sonnet-5", "https://api.anthropic.com/v1"),
    ],
)
def test_the_presets_point_at_the_right_service(spec: str, expected_url: str) -> None:
    assert from_spec(spec).base_url == expected_url


def test_a_model_of_your_own_needs_a_base_url() -> None:
    with pytest.raises(ModelError, match="needs --base-url"):
        from_spec("chat:my-model")
    model = from_spec("chat:my-model", base_url="https://internal.example/v1")
    assert isinstance(model, ChatModel)
    assert model.api_key_env == "LLM_JUDGE_API_KEY"


def test_an_unknown_provider_lists_the_known_ones() -> None:
    with pytest.raises(ModelError, match="known: anthropic, ollama"):
        from_spec("cohere:command")


def test_a_spec_without_a_model_name_is_refused() -> None:
    with pytest.raises(ModelError, match="provider:name"):
        from_spec("openai")


def test_a_base_url_that_is_not_http_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A base URL is configuration, and file:// in configuration is either a
    mistake or an attack. Refused before the request is built."""
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "k")
    model = from_spec("chat:m", base_url="file:///etc")
    with pytest.raises(ModelError, match="only http and https"):
        model.complete("p")
