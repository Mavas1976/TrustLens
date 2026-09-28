# ADR-001: Trust Score methodology contract

- **Status:** Accepted (interim rules for v0.5.x); Trust Score v2 redesign pending
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

### 4. Deferred to Trust Score v2 (minor release, versioned)

- **Calibration dimension (TL-05).** Base it on ECE, or on the reliability
  component of Brier, instead of raw multiclass Brier.
- **Failure dimension (TL-01, full fix).** Replace the mean confidence gap with
  a ranking measure, such as the AUROC of confidence for correct versus wrong
  predictions.
- **Single mechanism per signal (TL-09).** No sub-score plus penalty plus
  blocker for the same metric, and blockers aligned with the end of penalty
  ramps.
- **Bias dimension (TL-15).** Score it only when sensitive features are
  supplied. Class imbalance becomes a warning.
- **Versioning.** Add `score_version` to `TrustScoreResult` and to saved
  reports, and publish a migration table for the reference models.
- **Regression point-only reports.** Decide whether these count as partial.
  They currently score on accuracy alone by design (RFC #145).

## Consequences

- Scores change for affected models in v0.5.1. The CHANGELOG lists each change
  with its issue ID.
- Consumers that relied on `tpr == 0.0` for groups without positives must
  handle `None`.
- Stored reports from earlier versions keep their stored scores. Recomputing
  them with v0.5.1 can give different values.
