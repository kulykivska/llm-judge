"""Models a judge can run on. The core needs none of these installed."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field


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


@dataclass
class AnthropicModel:
    """Claude, through the official SDK. `pip install llm-judge[anthropic]`.

    The API key comes from the environment; this class never takes one as an
    argument, so it cannot end up in a config file or a traceback.
    """

    model: str = "claude-sonnet-5"
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        try:
            import anthropic  # noqa: PLC0415 - optional extra, imported on use
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "AnthropicModel needs the anthropic package: pip install 'llm-judge[anthropic]'"
            ) from exc
        self._client = anthropic.Anthropic()

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0) -> str:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text")
