"""The report a person reads, and the one CI reads."""

from __future__ import annotations

from .core import Label
from .metrics import Agreement

# Landis & Koch, the usual reading of kappa. Printed so a number nobody
# remembers the scale for arrives with its meaning attached.
MAX_NAMED = 10

KAPPA_BANDS = (
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.01, "slight"),
    (-1.0, "none"),
)


def kappa_reading(kappa: float) -> str:
    for threshold, name in KAPPA_BANDS:
        if kappa >= threshold:
            return name
    return "none"


def markdown(  # noqa: PLR0917 - a report takes what it reports on
    name: str,
    version: str,
    overall: Agreement,
    slices: dict[str, Agreement],
    flaky: dict[str, float],
    labels: dict[str, Label],
    *,
    max_disagreements: int = 20,
) -> str:
    lines = [
        f"# {name} {version}",
        "",
        f"Judged {overall.judged} labelled case(s): "
        f"{overall.matched} agreed, {len(overall.disagreements)} did not.",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| agreement with people | {overall.accuracy:.1%} |",
        f"| Cohen's kappa | {overall.kappa:.3f} ({kappa_reading(overall.kappa)}) |",
    ]
    if flaky:
        unstable = sum(1 for value in flaky.values() if value > 0)
        lines.append(f"| cases the judge answered differently on a repeat | {unstable} |")
    lines += [
        "",
        "## Per verdict",
        "",
        "| verdict | support | precision | recall | f1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for score in overall.per_class:
        lines.append(
            f"| {score.verdict} | {score.support} | {score.precision:.2f} | "
            f"{score.recall:.2f} | {score.f1:.2f} |"
        )
    if slices:
        lines += [
            "",
            "## Per slice",
            "",
            "| slice | judged | agreement | kappa |",
            "| --- | --- | --- | --- |",
        ]
        for slice_name, sliced in slices.items():
            lines.append(
                f"| {slice_name} | {sliced.judged} | {sliced.accuracy:.1%} | {sliced.kappa:.3f} |"
            )
    if overall.disagreements:
        lines += [
            "",
            "## Where it disagreed",
            "",
            "The list to read first: every one is either a judge to fix or a label to fix.",
            "",
            "| case | person | judge | note |",
            "| --- | --- | --- | --- |",
        ]
        for case_id, truth, guess in overall.disagreements[:max_disagreements]:
            note = labels[case_id].note if case_id in labels else ""
            lines.append(f"| {case_id} | {truth} | {guess} | {note} |")
        if len(overall.disagreements) > max_disagreements:
            lines.append(f"| ... | | | {len(overall.disagreements) - max_disagreements} more |")
    if overall.unjudged:
        lines += [
            "",
            f"**{len(overall.unjudged)} labelled case(s) got no usable verdict**: "
            + ", ".join(overall.unjudged[:MAX_NAMED])
            + ("..." if len(overall.unjudged) > MAX_NAMED else ""),
            "",
            "A run that drops cases reports the ones it kept, which flatters it.",
        ]
    return "\n".join(lines) + "\n"
