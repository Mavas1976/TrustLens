# Trust Score Explained

This page is the canonical reference for how TrustLens turns diagnostics into a deployment recommendation.

## Why a Composite Score Exists

Raw metrics are useful, but teams still need one final decision signal for release gates and model comparison.
TrustLens computes a composite Trust Score from calibration, failure behavior, bias risk, and representation quality.

The score is not meant to replace detailed metrics. It is meant to summarize them for decision making.

## What the Score Is Not

The Trust Score is a heuristic summary of diagnostic evidence. It is not a
probability of failure, a guarantee of safe behaviour, a certification or a
regulatory assessment. Use it to rank candidates and to gate releases, and use
the sub-scores and metric pages to decide what to fix.

This page describes **methodology 2.1** (`TrustScoreResult.score_version`),
introduced after the 2026-09 audit. The design decisions and their rationale
are recorded in [ADR-001](adr/ADR-001-trust-score-methodology.md).

## Inputs Used

**Classification:**
- `calibration`: ECE (the score), overconfidence error (a blocker); Brier and MCE are reported only
- `failure`: error-detection AUROC, error rate, accuracy and the majority-class baseline
- `bias`: subgroup accuracy gaps and equalized-odds TPR/FPR gaps, only when `sensitive_features` are supplied
- `representation`: silhouette-based separability, only with `embeddings`

**Regression:**
- `accuracy`: skill score (R²), docked for heavy-tailed errors
- `interval_calibration`: single-level PICP or multi-level ICE
- `uncertainty_informativeness`: sharpness proxy or error-variance correlation

## Scoring Workflow (classification)

1. **Sub-scores** for every dimension that was actually assessed (0–100).
2. **Weighted mean** over those dimensions (`base_score`).
3. **Blockers**: grade D, score capped at 39.
4. **Caps**: grade C, score capped at 59.

Every signal is counted once. There are no additive penalties; the
`penalties_applied` field stays empty and is kept only for backward
compatibility. The grade always matches the score band.

### Sub-score formulas

| Dimension | Formula |
|---|---|
| Calibration | `100 × clip(1 − ECE / 0.25, 0, 1)` |
| Failure | `100 × (1 − clip(error_rate / 0.20, 0, 1) × (1 − clip(2 × AUROC − 1, 0, 1)))`; AUROC is the error-detection AUROC of top-label confidence. Undetectable errors weigh by how often they occur, so a report without errors scores 100 and one error in a thousand costs at most half a point |
| Bias | `100 × clip(1 − max_gap / 0.30, 0, 1)`, where `max_gap` is the largest defined subgroup accuracy gap or equalized-odds TPR/FPR gap |
| Representation | `100 × clip(0.5 + 0.5 × silhouette, 0, 1)` |

Fairness gaps ignore undefined rates (a group without positives has no TPR)
and groups with fewer than 30 samples, which are reported as `low_support`.
Class imbalance is a property of the data: it is reported, not scored.

### Default weights

| Classification | Weight | Regression | Weight |
|---|---|---|---|
| Calibration | 0.35 | Interval calibration | 0.40 |
| Failure | 0.30 | Accuracy (skill) | 0.30 |
| Bias | 0.25 | Uncertainty informativeness | 0.30 |
| Representation | 0.10 | | |

Weights are renormalised over the dimensions that were scored. Custom weights
must use these keys and be non-negative.

### Blockers (grade D, score ≤ 39)

**Classification:**
- No predictive skill: accuracy does not beat the majority-class baseline *and* confidence carries little information about errors (error-detection AUROC below 0.6, or no probabilities). Not applied when `y_true` has a single class. If confidence does separate errors (for example a calibrated rare-event model whose scores never cross 0.5), the result is capped at C with a "review the decision threshold" note instead.
- Overconfidence: overconfidence error (the part of top-label ECE where confidence exceeds accuracy) above 0.10, on at least 100 samples. Underconfidence lowers the calibration sub-score but does not block.
- Severe fairness violation: a fairness gap above 0.15.

Blockers do not switch on abruptly. Before each threshold a **ceiling ramp**
lowers the maximum score linearly, so the score is continuous as a signal
crosses its threshold:

| Signal | Ceiling starts (100) | Ceiling reaches 39 (= blocker) |
|---|---|---|
| Overconfidence error | 0.05 | 0.10 |
| Largest fairness gap | 0.10 | 0.15 |

Below 100 samples the overconfidence estimate is too noisy to block on (about
20% false alarms at n = 30 for a perfectly calibrated model), so its ceiling
stops at 59 (grade C) instead of blocking.

**Regression:**
- Negative skill (worse than predicting the mean).
- Severe interval miscoverage (coverage more than 0.10 below nominal).

### Caps and ceilings (grade C, score ≤ 59)

- **Incomplete assessment** (`is_partial`): calibration or failure was not assessed, for example without `y_prob` or with a `modules=` subset.
- **Weak dimension**: an assessed sub-score below 40 lowers the ceiling linearly from 100 (at 40) to 59 (at 30 and below).

Top-label measures (multiclass ECE, overconfidence error, error-detection
AUROC) judge the probabilities' own prediction: a sample counts as correct
when `argmax(y_prob)` equals `y_true`, with confidence `max(y_prob)`. Accuracy,
the error rate and the no-skill baseline describe the reported `y_pred`. A
warning is logged when `y_pred` differs from `argmax(y_prob)` for more than 1%
of samples, which also catches probability columns in the wrong order. Binary
calibration is scored with the positive-class ECE, the standard binary
definition; ECE uses 10 equal-width bins on [0, 1].

Results saved before methodology 2.0 lack the error-detection AUROC and the
overconfidence error. Scoring them warns, falls back to the normalised
confidence gap for the failure sub-score, cannot apply the overconfidence
blocker, and add `-legacy-input` to `score_version` (e.g. `2.1-legacy-input`).
Re-run `analyze()` for a full score.

When no dimension can be scored at all (no probabilities, no sensitive
features, no embeddings) and no blocker applies, the grade is **N/A** and the
deployment verdict is `INSUFFICIENT_EVIDENCE`.

## Grade Interpretation

Grades are assigned to the reported (rounded, integer) score, so a raw
weighted score of 79.5 rounds to 80 and receives an A.

| Score | Grade | Meaning |
|---|---|---|
| 80–100 | A | High trust, no critical issues detected |
| 60–79 | B | Good trust, minor issues to address |
| 40–59 | C | Moderate trust, investigate before deployment |
| 0–39 | D | Low trust, do not deploy |
| — | N/A | Not assessed, insufficient evidence |

## Changes from methodology 1.x (v0.5.0)

Reference models scored with v0.5.0 and with the current methodology (same
data splits and seeds). `tests/reference/test_published_table.py` recomputes
every value in the current-methodology column from this page, so the table
cannot drift from the code:

| Model | v0.5.0 | current (2.1) |
|---|---|---|
| breast_cancer · LogReg | 68/D blocked | 93/A |
| breast_cancer · RandomForest | 70/B | 87/A |
| breast_cancer · GaussianNB | 58/D blocked | 88/A |
| iris · LogReg | 57/C | 78/B |
| iris · RandomForest | 61/B | 81/A |
| iris · GaussianNB | 52/D blocked | 81/A |
| wine · LogReg | 81/A | 88/A |
| wine · RandomForest | 41/D blocked | 75/B |
| wine · GaussianNB | 58/D blocked | 96/A |
| breast_cancer · LogReg, y_pred only | 20/D blocked | N/A |
| breast_cancer · swapped probabilities (confidently wrong) | 0/D blocked | 0/D blocked |
| breast_cancer · coin-flip model | 25/D blocked | 27/D blocked |
| diabetes · Ridge (regression) | 42/C | 42/C |

The main causes: the 1.x failure sub-score could not exceed 60 for binary
models, ECE above 0.10 blocked even underconfident models, and signals were
counted up to three times (sub-score, penalty and blocker). Methodology 2.1
additionally weights undetectable errors by how often they occur and replaces
the jumps at blocker thresholds with continuous ceilings.

## How to Use This in Practice

- Use the score for ranking and release gating, together with `blockers`, `caps_applied` and `is_partial`.
- Use sub-scores to identify which dimension needs work; the weakest dimension is named in explanations and in `compare()`.
- Use full metric pages for root-cause analysis.

## Related Reading

- [Features and Modules](features.md)
- [Known Limitations](known_limitations.md)
- [ADR-001: Trust Score methodology](adr/ADR-001-trust-score-methodology.md)
- [Metric: Calibration](metrics/calibration.md)
- [Metric: Failure](metrics/failure.md)
- [Metric: Bias](metrics/bias.md)
- [Metric: Regression](metrics/regression.md)
