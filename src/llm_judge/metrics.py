"""How well the judge agrees with the people, and with itself.

Accuracy alone flatters a judge on a skewed set: one that answers "pass" to
everything scores 0.9 where nine cases in ten pass. Cohen's kappa is here
because it takes that free credit away.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .core import Judgement, Label


@dataclass(frozen=True)
class ClassScore:
    verdict: str
    support: int
    precision: float
    recall: float

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 0.0 if total == 0 else 2 * self.precision * self.recall / total


@dataclass
class Agreement:
    judged: int
    matched: int
    accuracy: float
    kappa: float
    per_class: list[ClassScore] = field(default_factory=list)
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    disagreements: list[tuple[str, str, str]] = field(default_factory=list)
    unjudged: list[str] = field(default_factory=list)
    unlabelled: list[str] = field(default_factory=list)


def majority(judgements: list[Judgement]) -> dict[str, str]:
    """One verdict per case. With repeats, the one the judge gave most often.

    Ties go to the first answer, because a judge that cannot decide has already
    told you what you need to know - `flakiness` is where that shows up.
    """
    by_case: dict[str, list[str]] = {}
    for judgement in judgements:
        if judgement.verdict:
            by_case.setdefault(judgement.case_id, []).append(judgement.verdict)
    return {
        case_id: Counter(verdicts).most_common(1)[0][0] for case_id, verdicts in by_case.items()
    }


# One answer per case cannot disagree with itself.
MIN_REPEATS = 2


def flakiness(judgements: list[Judgement]) -> dict[str, float]:
    """Per case, the share of repeats that disagreed with the case's own
    majority. A judge that contradicts itself cannot be measured against
    anything else until that is fixed."""
    by_case: dict[str, list[str]] = {}
    for judgement in judgements:
        if judgement.verdict:
            by_case.setdefault(judgement.case_id, []).append(judgement.verdict)
    out: dict[str, float] = {}
    for case_id, verdicts in by_case.items():
        if len(verdicts) < MIN_REPEATS:
            continue
        top = Counter(verdicts).most_common(1)[0][1]
        out[case_id] = round((len(verdicts) - top) / len(verdicts), 4)
    return out


def agreement(judgements: list[Judgement], labels: dict[str, Label]) -> Agreement:
    chosen = majority(judgements)
    paired = [(case_id, verdict) for case_id, verdict in chosen.items() if case_id in labels]
    confusion: dict[tuple[str, str], int] = {}
    matched = 0
    disagreements: list[tuple[str, str, str]] = []
    for case_id, verdict in sorted(paired):
        truth = labels[case_id].verdict
        confusion[(truth, verdict)] = confusion.get((truth, verdict), 0) + 1
        if truth == verdict:
            matched += 1
        else:
            disagreements.append((case_id, truth, verdict))
    judged = len(paired)
    accuracy = matched / judged if judged else 0.0
    return Agreement(
        judged=judged,
        matched=matched,
        accuracy=round(accuracy, 4),
        kappa=round(cohen_kappa(confusion), 4),
        per_class=_per_class(confusion),
        confusion=confusion,
        disagreements=disagreements,
        unjudged=sorted(set(labels) - set(chosen)),
        unlabelled=sorted(set(chosen) - set(labels)),
    )


def cohen_kappa(confusion: dict[tuple[str, str], int]) -> float:
    """Agreement past what two raters would reach by guessing the same mix."""
    total = sum(confusion.values())
    if total == 0:
        return 0.0
    observed = sum(count for (truth, guess), count in confusion.items() if truth == guess) / total
    truths: Counter[str] = Counter()
    guesses: Counter[str] = Counter()
    for (truth, guess), count in confusion.items():
        truths[truth] += count
        guesses[guess] += count
    expected = sum(truths[v] * guesses[v] for v in set(truths) | set(guesses)) / (total * total)
    if expected == 1:
        # Both raters used one label for everything: agreement carries no
        # information, and the usual formula divides by zero.
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def _per_class(confusion: dict[tuple[str, str], int]) -> list[ClassScore]:
    verdicts = sorted({v for pair in confusion for v in pair})
    scores: list[ClassScore] = []
    for verdict in verdicts:
        hits = confusion.get((verdict, verdict), 0)
        guessed = sum(c for (_, guess), c in confusion.items() if guess == verdict)
        support = sum(c for (truth, _), c in confusion.items() if truth == verdict)
        scores.append(
            ClassScore(
                verdict=verdict,
                support=support,
                precision=round(hits / guessed, 4) if guessed else 0.0,
                recall=round(hits / support, 4) if support else 0.0,
            )
        )
    return scores


def by_slice(
    judgements: list[Judgement],
    labels: dict[str, Label],
    tags: dict[str, dict[str, str]],
    *,
    key: str,
) -> dict[str, Agreement]:
    """The same measurement, split by one tag. An average that hides a slice
    where the judge is useless is the reason slices exist."""
    groups: dict[str, list[Judgement]] = {}
    for judgement in judgements:
        value = tags.get(judgement.case_id, {}).get(key)
        if value is not None:
            groups.setdefault(value, []).append(judgement)
    return {name: agreement(rows, labels) for name, rows in sorted(groups.items())}


def results_json(
    overall: Agreement, slices: dict[str, Agreement], flaky: dict[str, float]
) -> dict[str, dict[str, Any]]:
    """The shape baseline-guard reads: {slice: {metric: value}}.

    So the gate that guards a model can guard the judge as well, with the same
    file, the same budgets and the same command.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, score in slices.items():
        out[name] = _metrics(score, flaky)
    out["overall"] = _metrics(overall, flaky)
    return out


def _metrics(score: Agreement, flaky: dict[str, float]) -> dict[str, Any]:
    measured = [flaky[c] for c in flaky] if flaky else []
    metrics: dict[str, Any] = {
        "accuracy": score.accuracy,
        "kappa": score.kappa,
        "judged": score.judged,
    }
    if measured:
        metrics["flaky_share"] = round(sum(1 for f in measured if f > 0) / len(measured), 4)
    return metrics
