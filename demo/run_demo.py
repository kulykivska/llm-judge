"""The demo runs a rule, not a model, so its numbers are real and free.

The rule is the judge most teams actually ship first: look for the word error,
otherwise compare the two strings. It scores well enough to look fine, and the
report says exactly where it is not.

    python demo/run_demo.py

Swap RuleModel for AnthropicModel and the same files measure a real judge.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from llm_judge import (
    JudgeSpec,
    RuleModel,
    agreement,
    by_slice,
    flakiness,
    keyword_rule,
    load_cases,
    load_labels,
    markdown,
    run,
)
from llm_judge.core import write_jsonl
from llm_judge.metrics import results_json

HERE = Path(__file__).parent

NAIVE = keyword_rule(
    {
        r"error|died|crash|not signed|503": "fail",
        r"Expected: (.*)\nActual: \1": "pass",
    },
    default="fail",
)


def main() -> None:
    spec = JudgeSpec.from_toml(HERE / "judge.toml")
    cases = load_cases(HERE / "cases.jsonl")
    labels = load_labels(HERE / "labels.jsonl")

    judgements = run(spec, cases, RuleModel(NAIVE), repeats=1, workers=1)
    write_jsonl(HERE / "judgements.jsonl", [asdict(j) for j in judgements])

    overall = agreement(judgements, labels)
    slices = by_slice(judgements, labels, {c.id: c.tags for c in cases}, key="surface")
    flaky = flakiness(judgements)

    (HERE / "results.json").write_text(
        json.dumps(results_json(overall, slices, flaky), indent=2) + "\n", encoding="utf-8"
    )
    report = markdown(spec.name, spec.version, overall, slices, flaky, labels)
    (HERE / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
