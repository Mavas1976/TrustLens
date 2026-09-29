# ADR-001: Trust Score methodology contract

- **Status:** Accepted. Interim rules (v0.5.x) superseded by methodology 2.0 (§4), refined as 2.1 (§4b)
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
- Worse calibration with the same predictions never raises the score. For
  regression intervals this holds for widening a level or shifting it away
  from nominal coverage; narrowing a level trades sharpness against
  calibration (the resolution/reliability trade-off). Since methodology 2.3
  the calibration weight has no free zone, so narrowing into over-confidence
  gains at most a few points (§4d).
  This invariant is about the scoring mechanisms (ceilings, blockers, the
  informativeness rule). The underlying metrics are not monotone in every
  change of input: an extra error can lower ECE for an under-confident model,
  shrink a fairness gap by levelling down, or raise the error-detection AUROC,
  and the failure sub-score's error weight saturates at a 20% error rate.
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

### 4b. Methodology 2.1 (`score_version = "2.1"`)

Independent verification of 2.0 found two new defects. Both are fixed
without changing weights, thresholds or dimensions.

| Topic | Decision | Issue |
|---|---|---|
| Failure sub-score | `100 × (1 − clip(error_rate/0.20) × (1 − detection))`. Undetectable errors weigh by how often they occur. In 2.0 a single error in 1,000 at constant confidence set detection to 0 and dropped a 98/A model to 59/C. The error rate at full weight (0.20) is a judgment call. | GA-01 |
| Blocker continuity | Ceiling ramps lead up to the overconfidence (0.05 → 0.10) and fairness (0.10 → 0.15) blockers. The maximum score falls linearly to 39 at the threshold, so the score no longer jumps from B/A to D (for example 88 → 39 at an overconfidence error of 0.1001 in 2.0). This implements "blocker threshold = end of the ramp" (TL-09). | GA-02 |
| Weak dimension | Replaced the hard cap at a sub-score of 40 with a ceiling from 100 (at 40) to 59 (at 30). | GA-02 |
| Published table | `tests/reference/test_published_table.py` recomputes every current-methodology value in Trust Score Explained, so the documentation cannot drift from the code. | GA-10 |

### 4c. Methodology 2.2 (`score_version = "2.2"`)

A second independent verification found that two blockers were still cliffs
and that `compare()` could rank incomparable reports. Weights, dimensions and
the ramped thresholds of 2.1 are unchanged; no reference model changes score.

| Topic | Decision | Issue |
|---|---|---|
| No-skill blocker | Ceiling on the normalised skill `(accuracy − baseline) / (1 − baseline)`: 100 at 0.10, falling to its end point at 0. The end point is 39 when the error-detection AUROC is at most 0.6 (a blocker below 0.6 or without probabilities) and rises linearly to 59 at AUROC 0.7, replacing the separate "review the decision threshold" cap. In 2.1 one extra correct positive in 2,000 moved a model from 39/D to 88/A. | NF-01 |
| Regression blockers | Ceilings from 100 to 39 as the skill (R²) falls from 0.10 to 0 and as the coverage shortfall grows from 0.05 to 0.10. In 2.1 coverage −0.0998 gave 68/B and −0.1016 gave 39/D. | NF-03 |
| Failed equalized odds | Fairness is not assessed (no score from the subgroup gap alone) and the report is partial, listing `bias (equalized odds failed)`. The score can still exceed that of a successful run with a large gap, but a partial report is capped at C and is never recommended by `compare()`. | NF-06 |
| `compare()` | No recommendation when eligible reports were scored on different dimensions (for example one with fairness, one without). | R-011 |
| Rare classes | The no-skill ramp spans at least 10 correctly predicted non-majority samples (`max(0.10, 10 / n_non_majority)`, deliberately not capped at 1: with fewer than 10 non-majority samples even a perfect model cannot fully lift the ceiling). With 10 positives in 2,000 one correct positive moved a model from 39/D to 99/A. Results without `n_samples` take the count from the class counts. | NF3-01, NF4-02, NF4-03 |
| Regression informativeness | Levels enter the sharpness proxy with a weight that falls from 1 at the calibration tolerance (0.05) to 0 at twice it, and the sub-score is `max(100 × sharpness_evidence, correlation score)`, where `sharpness_evidence = max_i w_i × clip(1 − width ratio_i, 0, 1)` is monotone in every level (a weighted mean rose when a wide level was widened further and dropped out, NF6-01) and the correlation score only when predicted variance was supplied (else 0). A level crossing the tolerance moved the score 51 → 68. The earlier rules "the proxy always wins over the correlation" and "fall back to the correlation when no level is usable" together made worse intervals raise the score (32 → 44, later up to +18 with a linear blend), so the proxy-first rule was dropped: the stronger evidence counts, and miscalibration is scored once, in interval calibration. Unusable intervals from a single-level mapping now score 0 like multi-level ones instead of dropping the dimension. The calibration weight is an admissibility condition for the sharpness evidence, not a second penalty for miscoverage. | NF3-02, NF4-01, NF5-01, NF5-03, NF6-01 |
| Perfect models, few minority samples | A model without errors counts as full error detection (end point 59), as in the failure sub-score, so one more error never raises the no-skill ceiling; the limit for a perfect model is `59 + 41 × n_non_majority / 10` below ten non-majority samples. Results without `n_samples` or class counts keep the 0.10 ramp. | NF5-02, NF5-05 |
| Fairness not assessable | Sensitive features supplied but no two groups of 30 or more samples: the report is partial, like a failed equalized-odds computation. | NF3-06 |
| Sample-count steps | The 30-sample, 100-sample (overconfidence) and 30-per-group rules stay deliberate steps and are documented as such; a single-class `y_true` logs a warning. | NF3-05 |
| Continuity tests | Continuity is tested as a Lipschitz bound (score change per signal change), because measured signals such as coverage move in discrete steps on a finite sample. | NF-03 |

**Deviation from the phase-0 plan (R-032).** The plan asked for an xfail test
per critical and high issue before its fix. The fixes for TL-09, TL-10 and
TL-11 landed before those tests existed, so xfail-first can no longer be shown
for them. Instead, every issue has a regression test that fails on the
pre-fix code; for the 2.2 changes this was checked by running the new tests
against the 2.1 code and against targeted mutants.

### 4d. Methodology 2.3 (`score_version = "2.3"`)

Decision by the repository owner on the open question NF7-01 (option b).

| Topic | Decision | Issue |
|---|---|---|
| Calibration weight of an interval level | `w = clip(1 − |emp − tau| / 0.10, 0, 1)`: 1 at nominal coverage, 0 at a coverage error of 0.10. In 2.2 the weight stayed 1 up to 0.05, so narrowing a level into slight over-confidence was free and could raise the score (64 → 72 in the verifier's repro; +11 at most in a 51-configuration grid). With 2.3 the same repro stays at 64–65 and then falls, and the grid's largest gain is +4, in each case at a coverage error of 0.025 or less, where the level is sharper at almost the same calibration. | NF7-01 |
| Unchanged | The 0.05 tolerance still defines `n_calibrated_levels` and the verdict; weights of the dimensions, the ceilings and the classification score are unchanged, and no reference model changes score. | |

Considered and rejected: accepting the trade-off as documented (leaves an
exception to the calibration invariant), and weighting sharpness by each
level's own coverage error (a larger redesign of a mechanism that had just
been stabilised).

### 5. Open questions

- The package version is still 0.5.0 while the scores follow methodology 2.0.
  `score_version` identifies the methodology. Bump the package version when
  releasing (a maintainer decision).
- The example notebooks are executed in CI; their stored outputs were refreshed
  under 2.2 with `python scripts/run_notebooks.py --write`.

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
