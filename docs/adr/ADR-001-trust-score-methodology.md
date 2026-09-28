# ADR-001: Trust Score methodology contract

- **Status:** Accepted. Interim rules (v0.5.x) superseded by methodology 2.0 (§4)
- **Date:** 2026-09-28
- **Context source:** `audit/2026-09-trustlens-audit.md` (issues TL-01 to TL-31)

## Context

The individual metrics behind the Trust Score (Brier, ECE, MCE, confidence gap,
TPR/FPR, silhouette, PICP, ICE, CRPS) match independent reference
implementations. The composition of those metrics into one 0–100 score and a
deployment verdict does not. The 2026-09 audit showed that under normal use:

- accurate, well-calibrated models were blocked. Examples are logistic
  regression on breast_cancer and a 100%-accurate random forest on wine (TL-01);
- a dimension that was not assessed was scored as 0 (TL-02);
- a subset of modules, or a typo in `modules=`, produced a full-looking verdict (TL-03);
- the regression score depended on the unit of the target (TL-04);
- undefined fairness rates and single-sample groups triggered fairness blocks (TL-06).

The 500+ existing tests did not catch this because none of them asserted
properties of the score itself.

## Decision

### 1. The score is a heuristic indicator, not a certification

The Trust Score summarises diagnostic evidence. It is not a probability of
failure, a guarantee of safety or a regulatory assessment. Documentation and
verdict texts must not claim more.

### 2. Invariants are the contract

`tests/invariants/` and `tests/reference/` define behaviour that every
methodology version must satisfy:

- A model that is always right with high confidence gets grade A and is never blocked.
- A perfectly calibrated, accurate model is never blocked.
- A dimension that was not assessed is never scored. It is reported in
  `missing_dimensions`, and the assessment is marked `is_partial`.
- A partial assessment can never earn a passing grade (A or B).
- Unknown module names raise an error.
- Rescaling a regression target and its predictions together does not change the score.
- Fairness gaps are computed only from defined rates (TPR needs positives, FPR
  needs negatives) and from groups with enough support.
- Worse calibration with the same predictions never raises the score.
- Text derived from user data is escaped in every HTML output.

A change that moves a reference model to another grade band must update the
expectation in `tests/reference/`, this ADR and the CHANGELOG in the same pull
request.

### 3. Interim rules for v0.5.x (patch level)

| Topic | Rule |
|---|---|
| Missing dimensions | A module result with `status` `skipped`, or a failure result without confidence metrics (`degraded`), is excluded. Its weight is redistributed. |
| Partial assessments | If a core dimension (`calibration` or `failure`) is missing, the result is `is_partial=True` and the grade is capped at **C**. The verdict names the missing dimensions. `compare()` never recommends a partial or grade-D report. |
| Module selection | `modules=` accepts only `calibration`, `failure`, `bias` and `representation`. Anything else raises `ValueError`. |
| Failure sub-score | The confidence gap is normalised by its attainable maximum `1 − 1/K` for K classes. A report with no errors has a gap score of 1.0. |
| Fairness gaps | TPR and FPR are `None` when undefined. Groups below `min_group_size` (30 in the pipeline) are reported with `low_support: true` and excluded from the gap. With fewer than two eligible groups the gap is `None` and the violation is `insufficient_data`. |
| Regression | Scoring uses unrounded error statistics. Rounding is applied only for display. |
| Task detection | An integer-valued target with more than 20 distinct values and a high distinct-value ratio is treated as regression. A fitted scikit-learn regressor always routes to regression. |

### 4. Methodology 2.0 (`score_version = "2.0"`)

Decided after the interim rules above. The formulas are in
[Trust Score Explained](../trust_score_explained.md) and are pinned by
`tests/invariants/test_formula_contract.py`.

| Topic | Decision | Issue |
|---|---|---|
| Calibration dimension | Scored from ECE (`100 × clip(1 − ECE/0.25)`). Brier is reported but no longer scored, because the multiclass Brier range grows with K and mixes accuracy into calibration. | TL-05 |
| Failure dimension | Scored from the error-detection AUROC of top-label confidence plus 20% accuracy. AUROC does not shrink with K or with accuracy, unlike the mean confidence gap. | TL-01 |
| One mechanism per signal | No additive penalties. Each metric counts once, in its sub-score. `penalties_applied` stays empty for compatibility. The regression weak-correlation penalty is removed for the same reason. | TL-09 |
| Blockers | Override the weighted score (AH-35): no predictive skill (accuracy not above the majority baseline while the error-detection AUROC is below 0.6 or unknown; never for a single-class `y_true`), overconfidence error > 0.10 on at least 100 samples, a fairness gap > 0.15, and the two existing regression blockers. Underconfidence does not block. Where the evidence is weaker (confidence still ranks errors, or n < 100), the same condition caps at C instead. | TL-09, review F1–F3 |
| Legacy input | Results without the 2.0 inputs warn and are scored with `score_version = "2.0-legacy-input"`. | review F4 |
| Correctness definition | Top-label measures (ECE, overconfidence, error-detection AUROC) use `argmax(y_prob) == y_true` with confidence `max(y_prob)`, as TL-14 requires; accuracy and the no-skill baseline use `y_pred`. A warning is logged when they differ for more than 1% of samples. (Superseded the review-F5 choice of `y_pred`, which independent verification found violates TL-14 and left the warning dead for integer labels.) | TL-14, GA-03 |
| Score and grade | Always consistent. Blocked results are capped at 39 (D), capped results at 59 (C). `base_score` keeps the uncapped weighted score. | TL-02, TL-03 |
| Caps | Partial assessment, or any assessed sub-score below 40. | TL-03 |
| Nothing assessable | Grade `N/A`, deployment verdict `INSUFFICIENT_EVIDENCE`, unless a blocker applies. | TL-02 |
| Bias dimension | Scored only from fairness gaps when sensitive features are supplied. Class imbalance is reported, not scored. | TL-15 |
| Versioning | `score_version` is stored in `TrustScoreResult`, report metadata and saved score files. v0.5.0 remains installable from PyPI for side-by-side comparison. | — |
| Weights | Unknown keys and negative values raise `ValueError`. | TL-28 |
| `compare()` | Takes display names, returns a structured result, excludes partial, blocked and grade-D reports, and warns when eligible reports were scored on different dimensions. | TL-11 |

The reference model `wine · RandomForest` is 100% accurate but underconfident
(ECE ≈ 0.12, overconfidence error 0). Its expected band was widened from {A} to
{A, B}. B is the intended result: calibration is the flagged dimension, not a
blocker.

### 5. Open questions

- The package version is still 0.5.0 while the scores follow methodology 2.0.
  `score_version` identifies the methodology. Bump the package version when
  releasing (a maintainer decision).
- The example notebooks (`examples/*.ipynb`) still describe penalties from 1.x
  in their stored outputs. They need re-running under 2.0.

- Regression point-only reports score on accuracy alone (RFC #145) and are not
  marked partial. Revisit this if users read such scores as complete.
- The thresholds (0.25 ECE ramp, 0.30 gap ramp, 0.10 and 0.15 blockers, 40
  weak-dimension cap) are judgment calls anchored on the reference models, not
  statistically calibrated. Re-run the model-zoo benchmark notebook under 2.0
  before publishing new research claims.

## Consequences

- Scores change for affected models: first with the interim fixes, and again
  with methodology 2.0. See the before/after table in Trust Score Explained. The
  CHANGELOG lists each change with its issue ID.
- Consumers that relied on `tpr == 0.0` for groups without positives must
  handle `None`.
- Stored reports from earlier versions keep their stored scores. Recomputing
  them with v0.5.1 can give different values.
