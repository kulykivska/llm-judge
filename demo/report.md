# qa-step-verdict v3

Judged 20 labelled case(s): 14 agreed, 6 did not.

| metric | value |
| --- | --- |
| agreement with people | 70.0% |
| Cohen's kappa | 0.476 (moderate) |

## Per verdict

| verdict | support | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| blocked | 3 | 0.00 | 0.00 | 0.00 |
| fail | 9 | 0.64 | 0.78 | 0.70 |
| pass | 8 | 0.78 | 0.88 | 0.82 |

## Per slice

| slice | judged | agreement | kappa |
| --- | --- | --- | --- |
| mobile | 8 | 62.5% | 0.400 |
| web | 12 | 75.0% | 0.532 |

## Where it disagreed

The list to read first: every one is either a judge to fix or a label to fix.

| case | person | judge | note |
| --- | --- | --- | --- |
| mob-04 | blocked | fail | simulator disk, environment problem |
| mob-07 | fail | pass | connected but unusable |
| mob-08 | blocked | fail | signing, not the feature |
| web-02 | pass | fail |  |
| web-06 | blocked | fail | storage was down for the whole run, not the app's fault |
| web-07 | fail | pass | partial translation is still a fail |
