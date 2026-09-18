"""The `llm-judge` command: run a judge, then measure it."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .core import JudgeSpec, load_cases, load_judgements, load_labels, write_jsonl
from .metrics import agreement, by_slice, flakiness, results_json
from .report import markdown
from .runner import Cache, failures, run

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_UNUSABLE = 2


def _model(name: str):  # type: ignore[no-untyped-def]
    if name.startswith("anthropic:"):
        # Imported here so the core never needs the SDK installed.
        from .providers import AnthropicModel  # noqa: PLC0415

        return AnthropicModel(model=name.split(":", 1)[1] or "claude-sonnet-5")
    raise SystemExit(
        f"unknown model {name!r}: use anthropic:<model>, or call llm_judge.run() with your own"
    )


def _run(args: argparse.Namespace) -> int:
    spec = JudgeSpec.from_toml(Path(args.judge))
    cases = load_cases(Path(args.cases))
    cache = Cache(Path(args.cache)) if args.cache else Cache(None)
    judgements = run(
        spec,
        cases,
        _model(args.model),
        repeats=args.repeats,
        workers=args.workers,
        cache=cache,
    )
    write_jsonl(Path(args.out), [asdict(j) for j in judgements])
    broken = failures(judgements)
    print(f"{len(judgements)} judgement(s) -> {args.out}")
    if broken:
        # Named here because a judged set with holes in it is not a judged set.
        print(f"{len(broken)} failed, including: {broken[0].error}", file=sys.stderr)
        for judgement in broken[:5]:
            print(f"  {judgement.case_id}: {judgement.error}", file=sys.stderr)
    return EXIT_FAILED if broken else EXIT_OK


def _eval(args: argparse.Namespace) -> int:
    judgements = load_judgements(Path(args.judgements))
    labels = load_labels(Path(args.labels))
    tags = {c.id: c.tags for c in load_cases(Path(args.cases))} if args.cases else {}
    overall = agreement(judgements, labels)
    flaky = flakiness(judgements)
    slices = by_slice(judgements, labels, tags, key=args.slice_by) if args.slice_by else {}

    if args.results:
        Path(args.results).write_text(
            json.dumps(results_json(overall, slices, flaky), indent=2) + "\n", encoding="utf-8"
        )
    text = markdown(args.name, args.judge_version, overall, slices, flaky, labels)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    else:
        print(text)

    if overall.judged == 0:
        print("no case has both a judgement and a label", file=sys.stderr)
        return EXIT_UNUSABLE
    passed = overall.accuracy >= args.min_agreement and overall.kappa >= args.min_kappa
    if not passed:
        print(
            f"below the floor: agreement {overall.accuracy:.1%} (min {args.min_agreement:.0%}), "
            f"kappa {overall.kappa:.3f} (min {args.min_kappa:.2f})",
            file=sys.stderr,
        )
    return EXIT_OK if passed else EXIT_FAILED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-judge", description=__doc__)
    parser.add_argument("--version", action="version", version=f"llm-judge {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    runner = commands.add_parser("run", help="judge every case and write the verdicts")
    runner.add_argument("--judge", required=True, help="judge spec (TOML)")
    runner.add_argument("--cases", required=True, help="cases (JSONL)")
    runner.add_argument("--out", required=True, help="where to write judgements (JSONL)")
    runner.add_argument("--model", default="anthropic:claude-sonnet-5")
    runner.add_argument(
        "--repeats", type=int, default=1, help="answers per case; 3 shows flakiness"
    )
    runner.add_argument("--workers", type=int, default=4)
    runner.add_argument("--cache", help="reuse identical prompts from this file")
    runner.set_defaults(func=_run)

    evaluate = commands.add_parser("eval", help="measure the judge against human labels")
    evaluate.add_argument("--judgements", required=True)
    evaluate.add_argument("--labels", required=True, help="human labels (JSONL)")
    evaluate.add_argument("--cases", help="cases, for their tags")
    evaluate.add_argument("--slice-by", help="tag to split the results by")
    evaluate.add_argument("--results", help="write {slice: {metric: value}} JSON (baseline-guard)")
    evaluate.add_argument("--report", help="write the Markdown report here instead of stdout")
    evaluate.add_argument("--name", default="judge")
    evaluate.add_argument("--judge-version", default="")
    evaluate.add_argument("--min-agreement", type=float, default=0.0)
    evaluate.add_argument("--min-kappa", type=float, default=0.0)
    evaluate.set_defaults(func=_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result: int = args.func(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"{exc}", file=sys.stderr)
        return EXIT_UNUSABLE
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
