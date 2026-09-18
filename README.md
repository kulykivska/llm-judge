# llm-judge

[![ci](https://github.com/kulykivska/llm-judge/actions/workflows/ci.yml/badge.svg)](https://github.com/kulykivska/llm-judge/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Measure an LLM judge against human labels before you trust it.

A judge you have not measured is an opinion you have automated. It decides what
passes, nobody reviews it, and the first sign it drifted is a release that
should not have shipped.

```bash
pip install llm-judge
```

No dependencies. The Anthropic adapter is an optional extra; any object with a
`complete()` method works.

## What it answers

- **Does it agree with people?** Agreement, and Cohen's kappa, which takes away
  the credit a judge gets for guessing the common answer.
- **Where does it disagree?** Every case, with the human note beside it. Each
  one is either a judge to fix or a label to fix.
- **Which verdict is it bad at?** Per-class precision and recall. An average of
  70% hides a class it never gets right.
- **Does it agree with itself?** Ask each case three times and count the
  contradictions. A judge that flips cannot be measured against anything else
  until that is fixed.
- **Did it get worse?** The results file is the shape
  [baseline-guard](https://github.com/kulykivska/baseline-guard) reads, so the
  gate that guards a model guards the judge too.

## The demo, with its real numbers

`demo/` judges twenty QA steps. The judge under test is a rule, not a model -
the one most teams ship first: look for the word *error*, otherwise compare the
strings. It runs offline, so the numbers below are reproducible with
`python demo/run_demo.py`, and a test asserts them.

```
| metric | value |
| --- | --- |
| agreement with people | 70.0% |
| Cohen's kappa | 0.476 (moderate) |

| verdict | support | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| blocked | 3 | 0.00 | 0.00 | 0.00 |
| fail    | 9 | 0.64 | 0.78 | 0.70 |
| pass    | 8 | 0.78 | 0.88 | 0.82 |

| slice  | judged | agreement | kappa |
| ---    | ---    | ---       | ---   |
| mobile | 8      | 62.5%     | 0.400 |
| web    | 12     | 75.0%     | 0.532 |
```

70% sounds usable. The rest of the report is why it is not:

- **`blocked`: zero precision, zero recall.** Every environment failure - a
  simulator out of disk, storage down, an unsigned build - was reported as a
  product bug. Three engineers get sent to debug features that were never run.
  The aggregate cannot see this; the per-class table is where it appears.
- **`web-02`: a passing test called a failure.** The expected message was
  *"Inline error: this email is already registered"*. The judge saw *error* and
  said fail. The word appeared in the requirement, not in the outcome.
- **`mob-07`: a failure called a pass.** Expected *"The call connects"*, actual
  *"The call connects, audio is one-way"*. The match was a prefix, and a prefix
  is not an outcome.

Three failure modes, none visible in the headline number, all of them in a
report that takes one command.

## Use it

Cases and labels are JSONL. A case carries whatever the prompt needs, plus tags
to slice by:

```json
{"id": "web-03", "inputs": {"step": "Add two items", "expected": "Badge shows 2", "actual": "Badge shows 0"}, "tags": {"surface": "web"}}
{"case_id": "web-03", "verdict": "fail", "note": "cart count wrong; no error text anywhere"}
```

The judge is a TOML file, with a version that changes when the prompt does:

```toml
[judge]
name = "qa-step-verdict"
version = "v3"
verdicts = ["pass", "fail", "blocked"]
temperature = 0.0
system = "You judge one automated test step..."
prompt = """
Step: {step}
Expected: {expected}
Actual: {actual}

Think in one sentence, then give the verdict on the last line.
"""
```

```bash
llm-judge run  --judge judge.toml --cases cases.jsonl --out judgements.jsonl \
               --model anthropic:claude-sonnet-5 --repeats 3 --cache .judge-cache.jsonl
llm-judge eval --judgements judgements.jsonl --labels labels.jsonl --cases cases.jsonl \
               --slice-by surface --results results.json --report report.md \
               --min-agreement 0.85 --min-kappa 0.7
```

`eval` exits non-zero below the floor, so a prompt change that costs agreement
does not merge. The cache is keyed by judge version and prompt: re-running is
free, and changing the prompt pays in full, which is the honest way round.

## In Python

```python
from llm_judge import JudgeSpec, agreement, load_cases, load_labels, run

spec = JudgeSpec.from_toml(Path("judge.toml"))
judgements = run(spec, load_cases(Path("cases.jsonl")), my_model, repeats=3)
score = agreement(judgements, load_labels(Path("labels.jsonl")))
print(score.accuracy, score.kappa, score.disagreements)
```

`my_model` is anything with
`complete(prompt, *, system="", temperature=0.0) -> str`.

## Gate it like a model

```bash
llm-judge eval ... --results results.json
baseline-guard check results.json --baseline judge-baseline.json --spec demo/spec.toml
```

A prompt that gains two points overall and loses ten on mobile is a regression.
An aggregate gate ships it.

## What it does not do

- It does not label your data. Human labels are the only ground truth here, and
  a judge graded against another judge measures nothing.
- It does not rank models against each other - that is a leaderboard, and a
  leaderboard is a different tool.
- It does not hide failures. A case whose answer could not be parsed is
  reported, not dropped: a run that quietly judged 88 of 100 cases and reported
  the 88 is how a judge appears to improve.

## License

MIT.
