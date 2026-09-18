"""Run a judge over cases: in parallel, cached, and never silently.

A judgement that failed is recorded as a failure rather than dropped. A run
that quietly judged 88 of 100 cases and reported the 88 is how a judge appears
to improve.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .core import Case, Judgement, JudgeSpec, Model


@dataclass
class Cache:
    """Keyed by judge version and prompt, so a re-run costs nothing and a
    changed prompt costs everything - which is the honest way round."""

    path: Path | None

    def __post_init__(self) -> None:
        self._entries: dict[str, str] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._entries[row["key"]] = row["raw"]

    def get(self, key: str) -> str | None:
        return self._entries.get(key)

    def put(self, key: str, raw: str) -> None:
        if self.path is None or key in self._entries:
            return
        self._entries[key] = raw
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "raw": raw}, ensure_ascii=False) + "\n")


def cache_key(spec: JudgeSpec, prompt: str, repeat: int) -> str:
    material = f"{spec.name}\x00{spec.version}\x00{spec.temperature}\x00{repeat}\x00{prompt}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def run(
    spec: JudgeSpec,
    cases: list[Case],
    model: Model,
    *,
    repeats: int = 1,
    workers: int = 4,
    cache: Cache | None = None,
) -> list[Judgement]:
    """Judge every case `repeats` times. Repeats are what make flakiness
    visible: one answer per case cannot tell a confident judge from a coin."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    store = cache or Cache(None)
    work = [(case, repeat) for case in cases for repeat in range(repeats)]

    def judge_one(item: tuple[Case, int]) -> Judgement:
        case, repeat = item
        prompt = spec.render(case)
        key = cache_key(spec, prompt, repeat)
        raw = store.get(key)
        if raw is None:
            try:
                raw = model.complete(prompt, system=spec.system, temperature=spec.temperature)
            except Exception as exc:  # a failure is recorded, never dropped
                error = f"{type(exc).__name__}: {exc}"
                return Judgement(case.id, "", "", repeat=repeat, error=error)
            store.put(key, raw)
        try:
            verdict = spec.parse(raw)
        except ValueError as exc:
            return Judgement(case.id, "", raw, repeat=repeat, error=str(exc))
        return Judgement(case.id, verdict, raw, repeat=repeat)

    if workers <= 1:
        return [judge_one(item) for item in work]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(judge_one, work))


def failures(judgements: list[Judgement]) -> list[Judgement]:
    return [j for j in judgements if j.error]
