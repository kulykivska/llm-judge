"""The things a judge run is made of: cases, labels, a spec, and judgements."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Case:
    """One thing to judge. `inputs` fills the prompt; `tags` cut the results."""

    id: str
    inputs: dict[str, Any]
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Label:
    """What a person said the verdict is. The only ground truth there is."""

    case_id: str
    verdict: str
    note: str = ""


@dataclass(frozen=True)
class Judgement:
    """What the judge said, once."""

    case_id: str
    verdict: str
    raw: str
    repeat: int = 0
    error: str = ""


@dataclass(frozen=True)
class JudgeSpec:
    """A judge is a prompt, a closed set of verdicts, and a version.

    The version is not decoration: it is what makes two runs comparable, and
    what a gate compares against. Change the prompt, change the version.
    """

    name: str
    version: str
    prompt: str
    verdicts: tuple[str, ...]
    system: str = ""
    temperature: float = 0.0
    fallback: str = ""

    @classmethod
    def from_toml(cls, path: Path) -> JudgeSpec:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        judge = data.get("judge", data)
        missing = {"name", "version", "prompt", "verdicts"} - set(judge)
        if missing:
            raise ValueError(f"{path}: judge is missing {sorted(missing)}")
        verdicts = tuple(str(v) for v in judge["verdicts"])
        fallback = str(judge.get("fallback", ""))
        if fallback and fallback not in verdicts:
            raise ValueError(f"{path}: fallback {fallback!r} is not one of {list(verdicts)}")
        return cls(
            name=str(judge["name"]),
            version=str(judge["version"]),
            prompt=str(judge["prompt"]),
            verdicts=verdicts,
            system=str(judge.get("system", "")),
            temperature=float(judge.get("temperature", 0.0)),
            fallback=fallback,
        )

    def render(self, case: Case) -> str:
        """Fill the template. A missing field is an error, not an empty string:
        a judge silently reading half a case is the worst kind of wrong."""
        try:
            return self.prompt.format(**case.inputs)
        except KeyError as exc:
            raise KeyError(f"case {case.id}: prompt needs {exc.args[0]!r}") from None

    def parse(self, text: str) -> str:
        """Find the verdict in the model's answer.

        Models pad answers with reasoning and punctuation, so this looks for a
        declared verdict rather than demanding the whole reply be one word.
        """
        cleaned = text.strip()
        for line in reversed(cleaned.splitlines()):
            stripped = line.strip().strip("*_`.:# ")
            for verdict in self.verdicts:
                if stripped.lower() == verdict.lower():
                    return verdict
                if stripped.lower().startswith(f"verdict: {verdict.lower()}"):
                    return verdict
        lowered = cleaned.lower()
        hits = [v for v in self.verdicts if v.lower() in lowered]
        if len(hits) == 1:
            return hits[0]
        if self.fallback:
            return self.fallback
        raise ValueError(f"no verdict in: {cleaned[:200]!r}")


class Model(Protocol):
    """Anything that turns a prompt into text. Adapters live in providers.py."""

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0) -> str: ...


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from None
            yield row


def load_cases(path: Path) -> list[Case]:
    cases = [
        Case(id=str(row["id"]), inputs=row.get("inputs", {}), tags=row.get("tags", {}))
        for row in read_jsonl(path)
    ]
    _refuse_duplicates(path, [c.id for c in cases], "case")
    return cases


def load_labels(path: Path) -> dict[str, Label]:
    labels = [
        Label(case_id=str(row["case_id"]), verdict=str(row["verdict"]), note=row.get("note", ""))
        for row in read_jsonl(path)
    ]
    _refuse_duplicates(path, [x.case_id for x in labels], "label")
    return {x.case_id: x for x in labels}


def load_judgements(path: Path) -> list[Judgement]:
    return [
        Judgement(
            case_id=str(row["case_id"]),
            verdict=str(row.get("verdict", "")),
            raw=str(row.get("raw", "")),
            repeat=int(row.get("repeat", 0)),
            error=str(row.get("error", "")),
        )
        for row in read_jsonl(path)
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _refuse_duplicates(path: Path, ids: list[str], what: str) -> None:
    seen: set[str] = set()
    for value in ids:
        if value in seen:
            # Two rows for one id means one of them is silently ignored, and
            # which one depends on file order.
            raise ValueError(f"{path}: {what} {value!r} appears twice")
        seen.add(value)
