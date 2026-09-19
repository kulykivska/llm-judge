"""Measure an LLM judge before you trust it.

A judge you have not measured is an opinion you have automated: it decides what
passes, it never gets reviewed, and the first sign it drifted is a release that
should not have shipped.
"""

from __future__ import annotations

from .core import Case, Judgement, JudgeSpec, Label, Model, load_cases, load_labels
from .metrics import Agreement, ClassScore, agreement, by_slice, cohen_kappa, flakiness, majority
from .providers import (
    PRESETS,
    AnthropicModel,
    ChatModel,
    ModelError,
    OllamaModel,
    RuleModel,
    ScriptedModel,
    from_spec,
    keyword_rule,
)
from .report import kappa_reading, markdown
from .runner import Cache, failures, run

__version__ = "0.1.0"

__all__ = [
    "PRESETS",
    "Agreement",
    "AnthropicModel",
    "Cache",
    "Case",
    "ChatModel",
    "ClassScore",
    "JudgeSpec",
    "Judgement",
    "Label",
    "Model",
    "ModelError",
    "OllamaModel",
    "RuleModel",
    "ScriptedModel",
    "__version__",
    "agreement",
    "by_slice",
    "cohen_kappa",
    "failures",
    "flakiness",
    "from_spec",
    "kappa_reading",
    "keyword_rule",
    "load_cases",
    "load_labels",
    "majority",
    "markdown",
    "run",
]
