"""Models a judge can run on: hosted, local, or none at all.

The core needs nothing installed. Everything here that talks to a network does
it over the standard library, so "works with your model" does not mean "works
once you install our four SDKs".
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Retried: the peer said "not now". Anything else is a real answer, including
# a refusal, and retrying it just spends money twice.
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
BACKOFF_SECONDS = 2.0
# A verdict is a word. Anything past this is not an answer to this question.
MAX_ANSWER_BYTES = 8 * 1024 * 1024


class ModelError(RuntimeError):
    """The model could not be reached, or answered with something unusable."""


@dataclass
class ScriptedModel:
    """Answers from a table. For tests, and for a demo whose numbers are real."""

    answers: dict[str, str]
    default: str = ""
    calls: list[str] = field(default_factory=list)

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0) -> str:
        self.calls.append(prompt)
        for needle, answer in self.answers.items():
            if needle in prompt:
                return answer
        if self.default:
            return self.default
        raise KeyError("no scripted answer for this prompt")


@dataclass
class RuleModel:
    """A judge with no model behind it at all: a rule over the prompt text.

    It exists to be beaten. Measuring an LLM judge against people is only half
    the question; the other half is whether it beats `if "error" in output`.
    """

    decide: Callable[[str], str]

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0) -> str:
        return self.decide(prompt)


def keyword_rule(patterns: dict[str, str], default: str) -> Callable[[str], str]:
    """First pattern that matches wins, in the order given."""
    compiled = [(re.compile(pattern, re.I), verdict) for pattern, verdict in patterns.items()]

    def decide(prompt: str) -> str:
        for pattern, verdict in compiled:
            if pattern.search(prompt):
                return verdict
        return default

    return decide


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    """One POST, retried on the statuses that mean "ask again"."""
    if not url.startswith(("http://", "https://")):
        # Before anything is built: a base URL is configuration, and file://
        # or data:// in configuration is either a mistake or an attack.
        raise ModelError(f"refusing to POST to {url!r}: only http and https")
    body = json.dumps(payload).encode("utf-8")
    last = ""
    for attempt in range(retries):
        request = urllib.request.Request(  # noqa: S310 - scheme checked above
            url, data=body, headers={"Content-Type": "application/json", **headers}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                # Capped: a misconfigured endpoint answering with a video is
                # not a reason for the run to die on memory.
                raw: bytes = response.read(MAX_ANSWER_BYTES)
            parsed: dict[str, Any] = json.loads(raw.decode("utf-8"))
            return parsed
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last = f"HTTP {exc.code}: {detail}"
            if exc.code not in RETRY_STATUS:
                raise ModelError(last) from None
        except (urllib.error.URLError, TimeoutError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        except json.JSONDecodeError as exc:
            raise ModelError(f"the answer was not JSON: {exc}") from None
        if attempt < retries - 1:
            time.sleep(BACKOFF_SECONDS * (2**attempt))
    raise ModelError(f"gave up after {retries} attempt(s): {last}")


def _key(env_var: str) -> str:
    """Read the key from the environment, never from an argument.

    A key passed in is a key that ends up in a config file, a shell history and
    a traceback.
    """
    if not env_var:
        return ""
    key = os.environ.get(env_var, "")
    if not key:
        raise ModelError(f"{env_var} is not set")
    return key


@dataclass
class ChatModel:
    """Any server that speaks the OpenAI chat-completions shape.

    Which is most of them: OpenAI, OpenRouter, Together, Groq, Fireworks,
    DeepInfra, Mistral, vLLM, LM Studio, llama.cpp's server, Ollama's
    compatibility endpoint. Point `base_url` at it and name the model.
    """

    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 1024
    timeout: float = 60.0
    retries: int = 3
    headers: dict[str, str] = field(default_factory=dict)
    # Anything the server wants that nothing else does: reasoning_effort,
    # provider routing, safety settings.
    extra_body: dict[str, Any] = field(default_factory=dict)

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            **self.extra_body,
        }
        headers = dict(self.headers)
        key = _key(self.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data = post_json(
            f"{self.base_url.rstrip('/')}/chat/completions",
            payload,
            headers=headers,
            timeout=self.timeout,
            retries=self.retries,
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ModelError(f"unexpected answer shape: {json.dumps(data)[:300]}") from None
        if isinstance(content, list):
            # Some servers answer with content blocks rather than a string.
            return "".join(str(part.get("text", "")) for part in content)
        return str(content or "")


@dataclass
class OllamaModel:
    """A model running on this machine, through Ollama's own endpoint.

    No key, no account, no per-token cost - which makes repeats, and therefore
    a self-consistency check, free.
    """

    model: str
    base_url: str = "http://localhost:11434"
    timeout: float = 300.0
    retries: int = 2
    options: dict[str, Any] = field(default_factory=dict)

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        data = post_json(
            f"{self.base_url.rstrip('/')}/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, **self.options},
            },
            headers={},
            timeout=self.timeout,
            retries=self.retries,
        )
        try:
            return str(data["message"]["content"])
        except (KeyError, TypeError):
            raise ModelError(f"unexpected answer shape: {json.dumps(data)[:300]}") from None


@dataclass
class AnthropicModel:
    """Claude, over the Messages API. No SDK required."""

    model: str = "claude-sonnet-5"
    base_url: str = "https://api.anthropic.com/v1"
    api_key_env: str = "ANTHROPIC_API_KEY"
    version: str = "2023-06-01"
    max_tokens: int = 1024
    timeout: float = 60.0
    retries: int = 3

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        data = post_json(
            f"{self.base_url.rstrip('/')}/messages",
            payload,
            headers={
                "x-api-key": _key(self.api_key_env),
                "anthropic-version": self.version,
            },
            timeout=self.timeout,
            retries=self.retries,
        )
        try:
            blocks = data["content"]
        except (KeyError, TypeError):
            raise ModelError(f"unexpected answer shape: {json.dumps(data)[:300]}") from None
        return "".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text")


# Where the common services live, so `--model groq:llama-3.3-70b` is enough.
# Anything not listed is still reachable: give a base URL and a key variable.
PRESETS: dict[str, tuple[str, str]] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "deepinfra": ("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
    ),
    "local": ("http://localhost:8000/v1", ""),
}


def from_spec(
    spec: str, *, base_url: str = "", api_key_env: str | None = None
) -> ChatModel | OllamaModel | AnthropicModel:
    """Build a model from `provider:name`.

    `anthropic:claude-sonnet-5`, `openai:gpt-4.1-mini`, `groq:llama-3.3-70b`,
    `ollama:qwen2.5:14b`, or `chat:<name>` with your own `base_url` for
    anything that speaks the same shape.
    """
    provider, _, name = spec.partition(":")
    if not name:
        raise ModelError(f"model {spec!r} must be provider:name, e.g. openai:gpt-4.1-mini")
    if provider == "anthropic":
        return AnthropicModel(
            model=name,
            base_url=base_url or AnthropicModel.base_url,
            api_key_env=api_key_env or AnthropicModel.api_key_env,
        )
    if provider == "ollama":
        return OllamaModel(model=name, base_url=base_url or OllamaModel.base_url)
    if provider in PRESETS:
        preset_url, preset_env = PRESETS[provider]
        return ChatModel(
            model=name,
            base_url=base_url or preset_url,
            api_key_env=preset_env if api_key_env is None else api_key_env,
        )
    if provider == "chat":
        if not base_url:
            raise ModelError("chat:<name> needs --base-url")
        return ChatModel(
            model=name,
            base_url=base_url,
            api_key_env="LLM_JUDGE_API_KEY" if api_key_env is None else api_key_env,
        )
    known = ", ".join(["anthropic", "ollama", "chat", *sorted(PRESETS)])
    raise ModelError(f"unknown provider {provider!r}; known: {known}")
