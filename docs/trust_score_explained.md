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

This page describes **methodology 2.0** (`TrustScoreResult.score_version`),
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
| Failure | `100 × (0.8 × clip(2 × AUROC − 1, 0, 1) + 0.2 × accuracy)`; AUROC is the error-detection AUROC of top-label confidence, and a report without errors uses 1 for the detection term |
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
- Overconfidence: overconfidence error (the part of top-label ECE where confidence exceeds accuracy) above 0.10, on at least 100 samples. Below 100 samples the estimate is too noisy (about 20% false alarms at n = 30 for a perfectly calibrated model), so it caps at C instead. Underconfidence lowers the calibration sub-score but does not block.
- Severe fairness violation: a fairness gap above 0.15.

**Regression:**
- Negative skill (worse than predicting the mean).
- Severe interval miscoverage (coverage more than 0.10 below nominal).

### Caps (grade C, score ≤ 59)

- **Incomplete assessment** (`is_partial`): calibration or failure was not assessed, for example without `y_prob` or with a `modules=` subset.
- **Weak dimension**: any assessed sub-score below 40.

Top-label measures (multiclass ECE, overconfidence error, error-detection
AUROC) all treat a prediction as correct when `y_pred == y_true` and use
`max(y_prob)` as its confidence. A warning is logged when `y_pred` differs from
`argmax(y_prob)` for more than 1% of samples. Binary calibration is scored with
the positive-class ECE, the standard binary definition. It is numerically close
to, but not identical with, the top-label ECE used for multiclass.

Results saved before methodology 2.0 lack the error-detection AUROC and the
overconfidence error. Scoring them warns, falls back to the normalised
confidence gap for the failure sub-score, cannot apply the overconfidence
blocker, and sets `score_version` to `2.0-legacy-input`. Re-run `analyze()` for
a full 2.0 score.

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

Reference models scored with v0.5.0 and with methodology 2.0 (same data
splits and seeds; `tests/reference/` pins the expected bands):

| Model | v0.5.0 | 2.0 |
|---|---|---|
| breast_cancer · LogReg | 68/D blocked | 88/A |
| breast_cancer · RandomForest | 70/B | 86/A |
| breast_cancer · GaussianNB | 58/D blocked | 85/A |
| iris · LogReg | 57/C | 77/B |
| iris · RandomForest | 61/B | 79/B |
| iris · GaussianNB | 52/D blocked | 78/B |
| wine · LogReg | 81/A | 88/A |
| wine · RandomForest | 41/D blocked | 75/B |
| wine · GaussianNB | 58/D blocked | 96/A |
| breast_cancer · LogReg, y_pred only | 20/D blocked | N/A |
| breast_cancer · swapped probabilities (confidently wrong) | 0/D blocked | 0/D blocked |
| breast_cancer · coin-flip model | 25/D blocked | 32/D blocked |
| diabetes · Ridge (regression) | 42/C | 42/C |

The main causes: the 1.x failure sub-score could not exceed 60 for binary
models, ECE above 0.10 blocked even underconfident models, and signals were
counted up to three times (sub-score, penalty and blocker).

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
