"""What the library must get right, starting with the arithmetic.

A metrics library whose numbers are not checked against worked examples is the
same problem as an unmeasured judge, one level down.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_judge import (
    Case,
    Judgement,
    JudgeSpec,
    Label,
    RuleModel,
    ScriptedModel,
    agreement,
    by_slice,
    cohen_kappa,
    failures,
    flakiness,
    kappa_reading,
    keyword_rule,
    load_cases,
    load_labels,
    majority,
    markdown,
    run,
)
from llm_judge.cli import main
from llm_judge.core import write_jsonl
from llm_judge.metrics import results_json
from llm_judge.runner import Cache, cache_key

SPEC = JudgeSpec(
    name="t",
    version="v1",
    prompt="Step: {step}\nActual: {actual}",
    verdicts=("pass", "fail", "blocked"),
)


def case(case_id: str, **inputs: str) -> Case:
    return Case(id=case_id, inputs={"step": "s", "actual": "a", **inputs})


# --- the spec -------------------------------------------------------------


def test_a_missing_prompt_field_names_the_case_and_the_field():
    """A judge silently reading half a case is the worst kind of wrong."""
    with pytest.raises(KeyError, match=r"c1.*expected"):
        JudgeSpec(name="t", version="v1", prompt="{expected}", verdicts=("pass",)).render(
            case("c1")
        )


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("pass", "pass"),
        ("  FAIL  ", "fail"),
        ("The step did what it said.\n\npass", "pass"),
        ("Reasoning here.\nVerdict: blocked", "blocked"),
        ("**fail**", "fail"),
        ("The environment was down, so blocked.", "blocked"),
    ],
)
def test_the_verdict_is_found_in_a_padded_answer(answer: str, expected: str):
    assert SPEC.parse(answer) == expected


def test_an_answer_with_no_verdict_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="no verdict"):
        SPEC.parse("I am not sure what to say about this one.")


def test_two_verdicts_in_prose_are_refused():
    """"pass or fail" is not a verdict, and picking one would invent data."""
    with pytest.raises(ValueError):
        SPEC.parse("this is either a pass or a fail")


def test_a_fallback_answers_where_a_strict_judge_would_raise():
    spec = JudgeSpec(
        name="t", version="v1", prompt="{step}", verdicts=("pass", "fail"), fallback="fail"
    )
    assert spec.parse("no idea") == "fail"


def test_a_fallback_outside_the_verdicts_is_refused(tmp_path: Path):
    path = tmp_path / "judge.toml"
    path.write_text(
        'name = "t"\nversion = "v1"\nprompt = "{step}"\nverdicts = ["pass"]\nfallback = "maybe"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fallback"):
        JudgeSpec.from_toml(path)


# --- running --------------------------------------------------------------


def test_a_model_that_throws_is_recorded_not_dropped():
    """A run that quietly judged 88 of 100 cases reports the 88, and the judge
    appears to improve."""

    class Broken:
        def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0) -> str:
            raise TimeoutError("upstream gone")

    judgements = run(SPEC, [case("c1")], Broken(), workers=1)
    assert [j.verdict for j in judgements] == [""]
    assert failures(judgements)[0].error == "TimeoutError: upstream gone"


def test_an_unparseable_answer_keeps_the_raw_text():
    model = ScriptedModel({}, default="the judge waffled")
    judgement = run(SPEC, [case("c1")], model, workers=1)[0]
    assert judgement.verdict == ""
    assert judgement.raw == "the judge waffled"
    assert "no verdict" in judgement.error


def test_repeats_ask_the_same_case_more_than_once():
    model = ScriptedModel({}, default="pass")
    judgements = run(SPEC, [case("c1")], model, repeats=3, workers=1)
    assert [j.repeat for j in judgements] == [0, 1, 2]
    assert len(model.calls) == 3


def test_the_cache_pays_for_a_prompt_once(tmp_path: Path):
    model = ScriptedModel({}, default="pass")
    cache = Cache(tmp_path / "cache.jsonl")
    run(SPEC, [case("c1")], model, workers=1, cache=cache)
    run(SPEC, [case("c1")], model, workers=1, cache=Cache(tmp_path / "cache.jsonl"))
    assert len(model.calls) == 1


def test_a_new_judge_version_is_a_new_cache_key():
    """Otherwise a changed prompt is graded on the old answers."""
    other = JudgeSpec(name="t", version="v2", prompt=SPEC.prompt, verdicts=SPEC.verdicts)
    assert cache_key(SPEC, "p", 0) != cache_key(other, "p", 0)


# --- metrics --------------------------------------------------------------


def test_kappa_is_zero_when_a_judge_only_matches_by_chance():
    """Half pass, half fail on both sides, agreeing half the time."""
    confusion = {
        ("pass", "pass"): 25,
        ("pass", "fail"): 25,
        ("fail", "pass"): 25,
        ("fail", "fail"): 25,
    }
    assert cohen_kappa(confusion) == 0.0


def test_kappa_is_one_on_perfect_agreement():
    assert cohen_kappa({("pass", "pass"): 10, ("fail", "fail"): 10}) == 1.0


def test_accuracy_flatters_a_judge_that_always_says_pass():
    """Nine cases in ten pass, so 'pass' scores 90% and knows nothing. This is
    the whole reason kappa is in the report."""
    labels = {f"c{i}": Label(f"c{i}", "pass" if i else "fail") for i in range(10)}
    judgements = [Judgement(f"c{i}", "pass", "pass") for i in range(10)]
    score = agreement(judgements, labels)
    assert score.accuracy == 0.9
    assert score.kappa == 0.0


def test_one_rater_using_one_label_is_not_perfect_agreement():
    assert cohen_kappa({("pass", "pass"): 5, ("fail", "pass"): 5}) == 0.0


def test_disagreements_name_both_sides():
    labels = {"c1": Label("c1", "blocked", note="the device was full")}
    score = agreement([Judgement("c1", "fail", "fail")], labels)
    assert score.disagreements == [("c1", "blocked", "fail")]


def test_a_labelled_case_with_no_verdict_is_reported_not_ignored():
    labels = {"c1": Label("c1", "pass"), "c2": Label("c2", "pass")}
    score = agreement([Judgement("c1", "pass", "pass")], labels)
    assert score.unjudged == ["c2"]
    assert score.judged == 1


def test_repeats_collapse_to_the_answer_given_most_often():
    judgements = [
        Judgement("c1", "pass", "", repeat=0),
        Judgement("c1", "fail", "", repeat=1),
        Judgement("c1", "fail", "", repeat=2),
    ]
    assert majority(judgements) == {"c1": "fail"}
    assert flakiness(judgements) == {"c1": 0.3333}


def test_a_judge_that_never_contradicts_itself_has_no_flakiness():
    judgements = [Judgement("c1", "pass", "", repeat=r) for r in range(3)]
    assert flakiness(judgements) == {"c1": 0.0}


def test_slices_split_the_same_measurement():
    labels = {"a": Label("a", "pass"), "b": Label("b", "fail")}
    judgements = [Judgement("a", "pass", ""), Judgement("b", "pass", "")]
    tags = {"a": {"surface": "web"}, "b": {"surface": "mobile"}}
    scores = by_slice(judgements, labels, tags, key="surface")
    assert scores["web"].accuracy == 1.0
    assert scores["mobile"].accuracy == 0.0


def test_results_json_is_what_baseline_guard_reads():
    labels = {"a": Label("a", "pass")}
    judgements = [Judgement("a", "pass", "")]
    out = results_json(agreement(judgements, labels), {}, {})
    assert out["overall"] == {"accuracy": 1.0, "kappa": 1.0, "judged": 1}


@pytest.mark.parametrize(
    ("kappa", "reading"),
    [(0.9, "almost perfect"), (0.7, "substantial"), (0.5, "moderate"), (0.0, "none")],
)
def test_kappa_arrives_with_its_meaning(kappa: float, reading: str):
    assert kappa_reading(kappa) == reading


# --- files and the command -----------------------------------------------


def test_a_duplicate_case_id_is_refused(tmp_path: Path):
    """One of the two rows would be silently ignored, depending on file order."""
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "c1", "inputs": {}}\n{"id": "c1", "inputs": {}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="appears twice"):
        load_cases(path)


def test_a_broken_line_names_its_line_number(tmp_path: Path):
    path = tmp_path / "labels.jsonl"
    path.write_text('{"case_id": "c1", "verdict": "pass"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        load_labels(path)


def test_the_report_names_the_class_the_judge_never_gets_right():
    labels = {
        "a": Label("a", "blocked"),
        "b": Label("b", "blocked"),
        "c": Label("c", "pass"),
    }
    judgements = [
        Judgement("a", "fail", ""),
        Judgement("b", "fail", ""),
        Judgement("c", "pass", ""),
    ]
    text = markdown("j", "v1", agreement(judgements, labels), {}, {}, labels)
    assert "| blocked | 2 | 0.00 | 0.00 | 0.00 |" in text


def test_eval_writes_results_and_gates_on_the_floor(tmp_path: Path):
    write_jsonl(
        tmp_path / "judgements.jsonl",
        [
            {"case_id": "a", "verdict": "pass", "raw": "pass"},
            {"case_id": "b", "verdict": "pass", "raw": "pass"},
        ],
    )
    write_jsonl(
        tmp_path / "labels.jsonl",
        [{"case_id": "a", "verdict": "pass"}, {"case_id": "b", "verdict": "fail"}],
    )
    argv = [
        "eval",
        "--judgements",
        str(tmp_path / "judgements.jsonl"),
        "--labels",
        str(tmp_path / "labels.jsonl"),
        "--results",
        str(tmp_path / "results.json"),
        "--report",
        str(tmp_path / "report.md"),
    ]
    assert main(argv) == 0
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["overall"]["accuracy"] == 0.5
    assert main([*argv, "--min-agreement", "0.8"]) == 1


def test_eval_says_so_when_nothing_lines_up(tmp_path: Path):
    write_jsonl(tmp_path / "judgements.jsonl", [{"case_id": "a", "verdict": "pass", "raw": "p"}])
    write_jsonl(tmp_path / "labels.jsonl", [{"case_id": "z", "verdict": "pass"}])
    assert (
        main(
            [
                "eval",
                "--judgements",
                str(tmp_path / "judgements.jsonl"),
                "--labels",
                str(tmp_path / "labels.jsonl"),
            ]
        )
        == 2
    )


def test_the_rule_model_needs_no_network():
    decide = keyword_rule({r"error": "fail"}, default="pass")
    judgements = run(SPEC, [case("c1", actual="an error appeared")], RuleModel(decide), workers=1)
    assert judgements[0].verdict == "fail"


def test_the_demo_numbers_are_reproducible():
    """The README quotes these. If the demo changes, the README is wrong."""
    demo = Path(__file__).resolve().parents[1] / "demo"
    spec = JudgeSpec.from_toml(demo / "judge.toml")
    cases = load_cases(demo / "cases.jsonl")
    labels = load_labels(demo / "labels.jsonl")
    import runpy  # noqa: PLC0415

    module = runpy.run_path(str(demo / "run_demo.py"))
    judgements = run(spec, cases, RuleModel(module["NAIVE"]), workers=1)
    score = agreement(judgements, labels)
    assert (score.judged, score.matched) == (20, 14)
    assert score.accuracy == 0.7
    assert score.kappa == 0.476
    blocked = next(s for s in score.per_class if s.verdict == "blocked")
    assert (blocked.support, blocked.recall) == (3, 0.0)
