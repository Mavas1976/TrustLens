# Changelog

All notable changes to TrustLens are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Hugging Face Text Classification Pipeline Support**: Added native backend resolution for `transformers.TextClassificationPipeline`. The pipeline extracts probabilities (using `top_k=None`) and aligns labels cleanly against the model's `id2label` mapping, falling back dynamically when no config is present. Integrated comprehensively into the backend registry alongside extensive tests covering output parsing, edge cases, and unsupported pipeline rejection. Thanks @AaronProbha18
- **Maximum Calibration Error (MCE)**: Added `maximum_calibration_error()` to the calibration module — the worst-case per-bin confidence/accuracy gap that ECE averages away (Naeini et al., 2015). Signature mirrors `expected_calibration_error()` (including the `uniform`/`quantile` binning strategies), and both metrics now compute their per-bin gaps through one shared binning core so they cannot diverge (per the consistency requirements discussed in #134); ECE results are unchanged. Wired into the analysis pipeline for both the binary and the multiclass top-label paths (`results["calibration"]["mce"]`), surfaced in `TrustReport.show()` alongside ECE, and documented in `docs/metrics/calibration.md`. (closes #134) Thanks @Whatsonyourmind
- **Conformal Prediction Diagnostics (metrics)**: Added `trustlens/metrics/conformal.py` with conditional-coverage diagnostics for split-conformal prediction sets: `to_membership_matrix` (normalises an `(n, K)` binary matrix or ragged label lists to a boolean set matrix), `marginal_coverage`, `class_conditional_coverage` (per-class + worst-class gap), `size_stratified_coverage` (Angelopoulos & Bates SSC, with a `min_stratum` small-stratum guard), `set_size_summary` (avg size, singleton rate, size efficiency, empty rate), and `conformal_diagnostics` (single orchestrated report; gap/violation fields are `None`, never a silent `0`, when nominal coverage is unavailable). Method-agnostic — evaluates sets from any source (LAC/APS/RAPS/Mondrian) without a conformal-library dependency. A metrics-only first step per the #157 RFC; `analyze()` activation via a new `y_pred_sets` input and the report panel follow in the next PR, and Trust Score integration is deferred to a later phase. (refs #157) Thanks @Whatsonyourmind
- **Conformal Prediction Diagnostics (pipeline & report)**: Wired the conformal metrics into the analysis pipeline behind a new optional `analyze(..., y_pred_sets=...)` input (an `(n, K)` membership matrix or ragged label lists). When supplied, the diagnostics are emitted as `results["calibration"]["conformal"]` and rendered as a dedicated **Conformal Prediction** panel under Calibration Analysis in `TrustReport.show()`; `confidence_level` doubles as the conformal nominal target. The sets coexist with `y_prob` (ECE/Brier from probabilities, conformal coverage from the sets) and the block is even emitted when probabilities are withheld; absent `y_pred_sets`, nothing changes. The panel labels the two opposite-signed gap fields explicitly (`coverage_gap` over/under, `worst_class_gap` shortfall) so they can't be misread, and malformed sets (length mismatch, all-empty) degrade to a visible skipped block rather than aborting the analysis. Strictly diagnostic-only — the Trust Score is unchanged (Phase-2 scoring stays deferred). (refs #157) Thanks @Whatsonyourmind
- **Radar Comparison Visualization**: Added `plot_radar_comparison()` for visually comparing multiple models across TrustLens dimensions (e.g., Calibration, Failure, Bias, Representation) using a publication-quality radar (spider) chart. Built on the centralized visualization styling system with themed colors, scoped styling, input validation, and support for saving figures. (closes #121, implemented in #156) Thanks @devanprigent
- **Multi-level Interval Calibration (ICE) & Calibration-Conditioned Sharpness**: Extended the regression Trust Score's uncertainty dimensions per RFC #155. `analyze()` (`prediction_intervals`) now accepts a `{level: (lower, upper)}` mapping alongside the existing single `(lower, upper)` tuple, routing to the new `multilevel_interval_coverage()` which reports the **Interval Calibration Error (ICE)** — the mean coverage gap across nominal levels — and a **calibration-conditioned sharpness proxy** (interval width measured only among the levels that pass calibration, normalized against the climatology spread; the CRPS-Resolution analog). Interval Calibration scores from ICE when multi-level data is present (single-level PICP fallback); Uncertainty Informativeness prefers the sharpness proxy (error-variance correlation fallback), with graceful degradation and weight redistribution preserved. Raw CRPS/CRPSS is intentionally deferred to a report diagnostic so calibration is not double-counted. Fully backward compatible. (closes #155) (refs #82) Thanks @Whatsonyourmind

### Fixed
- **Trust Score no longer blocks accurate, calibrated models (TL-01)**: the Failure sub-score normalises the confidence gap by its attainable maximum `1 − 1/K`, and an error-free model gets the full gap score. Scaled LogisticRegression on breast_cancer (98.8% accuracy) was 68/D "Blocked"; it now passes. Failure results record `n_classes`.
- **Unassessed dimensions are no longer scored as 0 (TL-02)**: a skipped calibration (no `y_prob`) or a failure analysis without confidence metrics is excluded and its weight redistributed.
- **Partial runs cannot pass (TL-03)**: unknown names in `modules=` raise `ValueError`. When calibration or failure is not assessed, `TrustScoreResult.is_partial` is `True`, `missing_dimensions` lists them, the grade is capped at C and report metadata carries `partial: true`. `compare()` never recommends a partial, blocked or grade-D report (TL-11).
- **Regression Trust Score is unit-independent (TL-04)**: `error_distribution()` returns unrounded values, so a model worse than the mean no longer scores 100/A on small-unit targets.
- **Fairness gaps use defined rates and adequate groups only (TL-06)**: `equalized_odds()` reports `tpr`/`fpr` as `None` when undefined instead of `0.0`. `subgroup_performance()` and `equalized_odds()` take `min_group_size` (30 in the pipeline); low-support groups are flagged and excluded from gaps, and a gap without two eligible values is `None` with violation `insufficient_data`.
- **Task auto-detection (TL-07)**: fitted scikit-learn-style regressors and integer-valued targets with many spread-out values (counts, prices) route to regression; `y_prob` implies classification.
- **HTML reports escape user-derived text (TL-08)**: feature names, verdicts and narrative text are escaped in `TrustReport._repr_html_` and `TrustScoreResult._repr_html_`.
- `tests/backends/test_xgboost_logic.py` is skipped when xgboost is not installed (TL-25).
- **Input validation (TL-12)**: `analyze()` validates every input once (`trustlens.core.inputs`): lists and pandas objects are converted, mismatched lengths raise an error naming the argument, probability rows must sum to 1, 1-D binary probabilities are expanded, differing pandas indexes trigger a warning, `sensitive_features` may be a DataFrame and missing values form a `"<missing>"` group.
- **Labels (TL-13)**: with `y_pred` + `y_prob`, class labels come from `model.classes_` or are inferred when unambiguous; otherwise a clear "pass class_labels" error replaces an `IndexError`. Conformal diagnostics use labels encoded to the prediction-set columns (coverage was silently wrong for labels 1..K).
- **Plots and saving (TL-20, TL-27)**: every plot saves through `visualization.style.save_figure`, which creates missing directories; `plot_bias(mode="all")` names the underlying errors. `TrustReport.save()` accepts `os.PathLike`, rejects unsupported file suffixes instead of creating a directory named `report.png`, and gains `overwrite=False` protection.
- **Quiet runs (TL-26)**: `verbose=False` prints nothing; the unused tqdm bar is removed; `quick_analyze()` prints its demo banner only for demo data.
- **Metric input checks (TL-28)**: ECE, MCE and the overconfidence error reject empty input, non-finite or out-of-range probabilities and non-binary labels.
- **Pattern detection (TL-17)**: skipped modules no longer read as a perfect 0.0 in report patterns and insights.
- **Independent-verification fixes**:
  - Top-label ECE, overconfidence error and error-detection AUROC judge `argmax(y_prob)`, and the argmax warning also works for integer labels, catching swapped probability columns (GA-03).
  - NaN/Inf in `y_true`/`y_pred` raise (GA-07); NaN in a numeric sensitive feature forms the `"<missing>"` group (GB-04).
  - Saved JSON is strict (NaN/Inf become null) and carries `score_version` everywhere (GB-16, GA-09).
  - Equalized odds runs for any two labels, using `class_labels[1]` as positive (GA-05).
  - Ragged prediction sets written in class labels are encoded (GA-08).
  - Excluded low-support groups are logged and named in the report (GA-04).
  - Ambiguous integer targets (many contiguous values, or few distinct values per sample) raise and ask for `task=`; fractional targets are regression (GA-06).
  - `modules="calibration"` is accepted as one module; fewer than 30 samples caps at C; the all-correct "overconfident" insight is gone (GA-11).
  - Missing metrics are "not assessed", never scored from defaults (GB-05).
  - A failed equalized-odds computation caps at C instead of improving the bias score (GB-06).
  - CKA is translation-invariant (GB-10).
  - Brier and weights reject non-finite values (GB-12).
  - `quick_analyze(model)` without data raises (GB-17).
  - `save()` refuses any unknown suffix unless the path is an existing directory (GB-18).
  - Characterization tolerance covers the supported scikit-learn range (GB-19).
  - The docs build on a clean checkout (GB-02).
  - The wheel ships only `trustlens` (GB-03).
- **Second verification fixes**: `pd.NA`/`NaT` in a sensitive feature form the `"<missing>"` group instead of crashing (NF-02); integer labels 0..K / 1..K with a few classes absent from the sample no longer auto-route to regression, they ask for `task=` (NF-05); `quick_analyze(None, X, y)` raises instead of silently analysing the demo dataset (NF-08); tests now pin the weak-dimension ramp, the low-support warning and escaping in the Trust Score HTML card (NF-04).
- **Metric fixes (TL-16, TL-29, TL-31)**: `crps_decomposition` includes Hersbach's outlier segments (an observation far outside the intervals no longer contributes almost nothing); `embedding_separability` excludes self-pairs from the within-class distance; `centered_kernel_alignment` is scale-invariant again; `brier_score` docstring example corrected (0.048); CRPS grid-bias direction corrected in the docs; methodology weights are rounded instead of truncated.

### Documentation
- The docs build with `sphinx -W` (the `[docs]` extra gains `sphinxcontrib-mermaid`; the conformal page is in the toctree). The API reference covers `quick_analyze`, `compare`, `compute_trust_score`, `regression_trust_score` and the results contract. README drops the hard-coded test-count and coverage badges; ROADMAP and SECURITY.md are corrected (TL-21, TL-22).
- Stale "methodology 2.0" wording updated to 2.x; the example notebooks' stored outputs were refreshed under 2.2 (`scripts/run_notebooks.py --write`), and the regression showcase prints sub-scores and limits instead of the always-empty penalties, and the model-zoo notebook reports blockers/caps instead of penalties, leaves unassessed sub-scores empty instead of 0 and states conclusions that match its automated verdicts (NF-07, NF3-03). ADR-001 §4c records methodology 2.2 and the xfail-first deviation (R-032).
- ROADMAP no longer marks tqdm progress bars, subgroup ECE, Keras/TensorFlow support, a video series or Colab badges as done; the API reference opens with an autosummary table of `trustlens.__all__` (GB-14).
- Added `audit/2026-09-trustlens-audit.md` (31 findings with reproduction scripts) and ADR-001 *Trust Score methodology contract* (`docs/adr/`).

### Changed
- **Trust Score methodology 2.0 (`score_version = "2.0"`)**. The classification score is now the weighted mean of the assessed sub-scores, followed by blockers and caps. Each signal is counted once; the regression score keeps its dimensions but loses the second weak-correlation penalty (TL-09). See `docs/trust_score_explained.md` (formulas plus a v0.5.0 → 2.0 table for reference models) and ADR-001.
  - Calibration is scored from ECE (`100 × clip(1 − ECE/0.25)`); multiclass Brier is reported but no longer scored (TL-05).
  - Failure is scored from the new error-detection AUROC (`trustlens.metrics.failure.error_detection_auroc`) plus 20% accuracy (TL-01).
  - Bias is scored only when `sensitive_features` are supplied, from the largest fairness gap; class imbalance is reported, not scored (TL-15).
  - Blockers (grade D, score ≤ 39): no predictive skill (accuracy not above the majority baseline), overconfidence error > 0.10 (new `trustlens.metrics.calibration.overconfidence_error`), fairness gap > 0.15. Underconfidence no longer blocks.
  - Caps (grade C, score ≤ 59): partial assessment or any sub-score below 40. The grade now always matches the score band; `base_score` keeps the uncapped score.
  - New `TrustScoreResult` fields: `blockers`, `caps_applied`, `score_version`. Grade `N/A` with deployment verdict `INSUFFICIENT_EVIDENCE` when nothing could be scored. `penalties_applied` is always empty (deprecated).
  - Custom weights with unknown keys or negative values raise `ValueError` (TL-28).
  - `compare()` accepts `names=`, returns a structured result (`recommended`, `ranking`, `excluded`, `warnings`) and ranks by score with each candidate's weakest dimension instead of penalty burden (TL-11).
  - Grade A verdict reads "High Trust - no critical issues detected" instead of "production-ready"; README claims toned down (TL-23).
  - Independent review hardening: the no-skill blocker needs uninformative confidence (error-detection AUROC < 0.6) and two classes in `y_true`, otherwise it caps at C; the overconfidence blocker needs at least 100 samples, otherwise it caps at C; results saved before 2.0 warn and carry `score_version="2.0-legacy-input"`; top-label measures (ECE, overconfidence, error-detection AUROC) judge `argmax(y_prob)` with confidence `max(y_prob)`, and a warning is logged when `y_pred` differs from the argmax for more than 1% of samples (also for integer labels, catching swapped probability columns); zero total weight raises; `compare()` rejects duplicate names and warns on ties; report narratives for N/A and skipped modules no longer contradict the verdict.
- **Trust Score methodology 2.1 (`score_version = "2.1"`)**, after independent verification:
  - The failure sub-score weighs undetectable errors by their frequency: `100 × (1 − clip(error_rate/0.20) × (1 − detection))`. One error in 1,000 no longer drops a 98/A model to 59/C (GA-01).
  - Ceiling ramps lead continuously into the overconfidence (0.05 → 0.10) and fairness-gap (0.10 → 0.15) blockers, and the weak-dimension cap becomes a ramp (40 → 30). Crossing a threshold no longer makes the score jump from A/B to D (GA-02).
  - `caps_applied` lists only the limits that actually lowered the score.
  - `tests/reference/test_published_table.py` recomputes the published before/after table from the docs.
- **Trust Score methodology 2.2 (`score_version = "2.2"`)**, after a second independent verification. No reference model changes score:
  - The no-skill blocker is approached by a ceiling on the normalised skill `(accuracy − baseline)/(1 − baseline)` (100 at 0.10), whose end point rises from 39 to 59 as the error-detection AUROC goes from 0.6 to 0.7. One more correct prediction no longer moves a model from 39/D to 88/A (NF-01).
  - The regression blockers get ceilings: skill (R²) 0.10 → 0 and coverage shortfall 0.05 → 0.10 lower the ceiling to 39. Regression results now report binding limits in `caps_applied` (NF-03).
  - A failed equalized-odds computation leaves fairness unassessed and the report partial (`missing_dimensions` includes `bias (equalized odds failed)`) instead of scoring the subgroup gap alone (NF-06).
  - `compare()` recommends no model when the eligible reports were scored on different dimensions (R-011).
  - Third verification: the no-skill ramp spans at least 10 correctly predicted non-majority samples (also for results without `n_samples`), so a rare-class model no longer jumps from 39/D to 99/A with one correct positive; with fewer than 10 such samples even a perfect model cannot fully lift the ceiling (NF3-01, NF4-02, NF4-03); regression interval levels enter the sharpness proxy with a calibration weight (1 at the 0.05 tolerance, 0 at 0.10) and informativeness is `max(weight × sharpness score, correlation score)`, removing a 51 → 68 jump and cases where worse intervals raised the score; unusable single-level mappings score 0 like multi-level ones (NF3-02, NF4-01, NF5-01, NF5-03, new `sharpness_weight` field). **Behaviour change:** with both intervals and `predicted_variance`, a stronger correlation now outweighs the sharpness proxy (previously the proxy always won); a perfect model counts as full error detection, so one more error never raises the score (NF5-02); fairness requested but without two groups of 30+ samples makes the report partial (NF3-06); a single-class `y_true` logs a warning and sample-count steps are documented (NF3-05); `None` in an object `y_true`/`y_pred` raises a clear error (NF3-07); the no-skill AUROC constants are pinned by tests (NF3-04).
- **Score changes from the fixes above**: Trust Scores change for models affected by TL-01, TL-02, TL-04 and TL-06. Stored reports keep their stored scores; recomputing them can give different values. Characterization baselines were regenerated (only the failure sub-score and derived values moved).
- **Behaviour change**: callers that relied on `tpr == 0.0` / `fpr == 0.0` for groups without positives / negatives must handle `None`.
- **Architecture (TL-17, TL-18, TL-30)**: `TrustReport` is split into mixins under `trustlens/_report/` (public API unchanged; `report.py` shrinks from 2,170 to about 330 lines); `trustlens/results_schema.py` documents the results dict as TypedDicts with a `check_results_contract()` checker used in tests; brand colours move to `trustlens._palette`, so `import trustlens` no longer loads matplotlib; framework detection matches the top-level package exactly; a mypy strictness ratchet covers the new modules.
- **CI (TL-24)**: actions pinned to commit SHAs with Dependabot; pip-audit ignores live in `.github/pip-audit-ignore.txt` with reason and expiry, checked by `scripts/check_audit_ignores.py`; the security job's unquoted `mistune>=3.2.1` (a shell redirect) is fixed; new CI jobs build the docs with `-W` and run all examples from an empty directory; mypy runs with one configuration everywhere.
- **Maintainability after independent verification**:
  - All HTML output escapes through one helper, `trustlens._report.html.escape_text`; the TrustScoreResult HTML card moved out of the scoring module, and colours live in `trustlens._palette` (no matplotlib import needed for HTML) (GB-09).
  - The report modules and `trustlens.core.inputs` are type-checked with strict mypy flags (GB-13).
  - pip-audit ignore entries carry OSV-reviewed, code-specific reasons; the expiry checker also accepts GHSA identifiers (GB-07).
  - Docstring examples run in CI (`pytest trustlens --doctest-modules`); the ECE, PICP and `analyze()` examples are executable with checked outputs, illustrative snippets are marked `+SKIP` (GB-11).
  - A Notebooks workflow executes `examples/*.ipynb` weekly and on notebook changes (`scripts/run_notebooks.py`, new `[notebooks]` extra). The demo notebook no longer calls `Figure.show()` on a closed figure and the model-zoo notebook creates its `output/` folder (GB-15).
- **Test safety net**: `tests/invariants/` (score properties) and `tests/reference/` (deterministic sklearn reference models with expected grade bands) guard the methodology; remaining known defects are tracked as strict xfails tagged with their issue id.
- **Unusable-uncertainty scoring for the regression Trust Score**: When multi-level prediction intervals are supplied but *no* level passes the calibration gate (and no error-variance correlation fallback exists), the Uncertainty Informativeness dimension is now scored a truthful `0.0` — "the supplied uncertainty delivered zero usable resolution" — instead of being dropped and having its weight redistributed onto the other dimensions. A new `TrustScoreResult.informativeness_status` field (`"present"` / `"unusable_uncertainty"` / `"absent"`, `None` for classification) lets downstream consumers distinguish "0.0 because the intervals were all miscalibrated" from "dropped because none were supplied." Scoped to the multi-level path (`n_levels >= 2`); the single-level PICP path keeps the existing redistribute behavior. (refs #155, #161) Thanks @Whatsonyourmind

### Improvements

---

## [v0.5.0] - 2026-06-27

### Added
- **Regression Metrics**: Added `trustlens/metrics/regression.py` with `error_distribution` (MedAE, 90th-percentile error, max, MAE, RMSE + histogram data), `prediction_interval_coverage` (PICP vs. nominal confidence, with graceful skip when intervals are absent), and `error_variance_correlation` (Pearson/Spearman between predicted uncertainty and actual error). A metrics-only first step toward regression support; `analyze()` auto-dispatch and visualization to follow. (refs #82) Thanks @Whatsonyourmind
- **Regression Analysis Pipeline & TrustReport Integration:** Added automatic regression-task detection and dispatch within `trustlens.analyze()`. Continuous targets are now routed through a dedicated regression evaluation pipeline with support for regression report rendering, serialization, export, and integration with `TrustReport`. (#142) (refs #82) Thanks @Whatsonyourmind
- **Regression Visualizations:** Added residual-analysis and error-distribution visualizations for regression workloads, enabling residual inspection, heteroscedasticity detection, and error-pattern analysis directly from regression reports. (#143) (refs #82) Thanks @Whatsonyourmind
- **Regression Trust Score:** Added a regression-specific Trust Score framework based on three trust dimensions:
  - Accuracy / Skill
  - Interval Calibration (PICP)
  - Uncertainty Informativeness
Includes A–D grading, deployment verdicts, blocker conditions for negative skill and severe interval miscoverage, weight redistribution when uncertainty signals are unavailable, regression/classification task separation via `task_type`, and full `TrustReport` integration. (#147) (refs #82) Thanks @Whatsonyourmind
- **Regression Trust Score Reproducibility**: Persisted `target_variance` (`Var(y)`) in regression analysis results, allowing `regression_trust_score()` to be recomputed from stored reports without requiring the original `y_true` array. Added backward-compatible fallback logic, mismatch warnings when persisted variance disagrees with supplied targets, and improved portability of serialized regression artifacts. (closes #150, implemented in #151) Thanks @Whatsonyourmind
- **Model Zoo Benchmark**: Introduced a comprehensive scientific validation notebook (`examples/trustlens_model_zoo_benchmark.ipynb`) that systematically evaluates TrustLens across 6 model architectures and multiple data corruption scenarios with statistical aggregation.
- **Centralized Visualization Styling**: Introduced an internal `trustlens/visualization/style.py` as the single source of truth for color palettes, semantic colors (severity, deployment verdict, grade, direction), typography, grid, and figure defaults. Added an `apply_style()` context manager that scopes `matplotlib.rcParams` mutations to a `with` block, preventing global state leakage when TrustLens is used inside notebooks or larger ML pipelines. Existing plotting modules are being migrated to the centralized system without changing visual output (so far: `calibration_plots.py`, `failure_plots.py`, `bias_plots.py`, `representation_plots.py`, `fairness.py`, `summary_plot.py`). (refs #57) Thanks @komoike-oss28-ui
- **Deployment Recommendation Explanations**: Added `TrustReport.deployment_explanation` and `TrustReport.deployment_summary` to provide structured deployment verdict explanations, identify primary risks, and surface actionable recommendations based on Trust Score penalties and sub-scores.
- **Deployment Recommendation UX**: Deployment recommendations are now surfaced directly in `TrustReport.show()`, text exports, and HTML report views. Users now receive deployment verdicts, primary risk identification, and actionable recommendations without needing to access `deployment_summary` manually.
- **Native LightGBM & CatBoost Backends**: Added automatic backend detection and prediction resolution for the LightGBM (LGBMClassifier, Booster) and CatBoost (CatBoostClassifier) models, including probability extraction, classification validation, and integration with the standard PredictionBundle pipeline. Thanks @vaishnavidesai09

### Fixed
- Fixed `reliability_curve(strategy="quantile")` for collapsed quantile bin edges, returning a valid single-bin curve for zero-variance probability distributions instead of raising `ValueError`. (fixes #144)
- **XGBoost Booster String Label Mapping**: Fixed an issue where raw `xgboost.Booster` models in multiclass classification returned ordinal prediction indices instead of semantic class labels when `y_true` contained string labels. TrustLens now correctly maps probability-column indices back to user-provided class labels, preventing downstream metric distortion and label-type mismatches. Added validation to ensure `class_labels` match the probability matrix shape and introduced regression tests covering raw Booster multiclass workflows. (fixes #117) Thanks @nanookclaw
- Fixed incorrect `top_mistake_indices` in `misclassification_summary()` to return **global dataset indices** instead of local filtered subset positions, improving downstream EDA and debugging workflows for high-confidence model errors. (PR #104) Thanks @dicnunz 🙌
- Fixed `Security Audit` CI failures caused by newly published upstream dependency vulnerabilities by updating `pip-audit` handling and ignore rules for unresolved ecosystem CVEs/PYSEC advisories. (PR #105)
- Fixed the visual narrative of the **Accuracy vs Trust (“Decoupling”)** analysis to more clearly communicate the relationship between predictive performance and trustworthiness in the benchmark notebook. (PR #100)

### Documentation
- **Core Architecture Documentation**: Improved module-level and class-level docstrings for `api.py`, `report.py`, `trust_score.py`, and `pipeline.py`.
- **Backend Architecture Documentation**: Documented the internal backend architecture in `trustlens/backends/`, detailing the `PredictionBundle` lifecycle, resolver architecture, probability extraction, and label mapping strategies.
- **Metrics Documentation**: Enhanced public docstrings for major metrics (`brier_score`, `expected_calibration_error`, `confidence_gap`, `equalized_odds`, `embedding_separability`) with clear explanations of what they measure, why they matter, their limitations, and how to interpret them.
- **Visualization Architecture Documentation**: Documented the centralized visualization architecture in `trustlens/visualization/style.py`, providing rules for maintaining visual parity and using semantic colors.
- **Developer Experience**: Updated `CONTRIBUTING.md` to formally outline documentation expectations, including a mandate for NumPy-style docstrings and a high-level architecture reference for new contributors.
- **Research & Validation Layer**: Added a comprehensive, research-grade documentation section (`docs/research/`) featuring empirical benchmark results, scientific trust score validation, robustness under distribution shift, metric limitations, and explicitly outlined failure modes.
- **Methodology & Threats to Validity**: Introduced a brutally honest `methodology.md` page detailing benchmark experimental setup and transparently acknowledging limitations such as reliance on synthetic datasets and binary classification constraints.
- **Why TrustLens**: Added a `why_trustlens.md` page to directly compare TrustLens against traditional metrics (like Accuracy and ROC-AUC) using tangible failure case studies.
- Generated publication-quality (300 DPI) visual assets demonstrating TrustLens's behavior under noise, calibration degradation, and severe class imbalance, inheriting the project's centralized visual styling.
- Added a complete, copy-paste runnable example to the analyze() docstring that demonstrates: Dataset creation using make_classification, Train/test split, Training a RandomForestClassifier, Predicting probabilities, Running analyze(), Displaying results with report.show(). Thanks @q404365631
- Added hosted TrustLens documentation website integration across the repository, including README links, package metadata (`pyproject.toml`), and documentation navigation improvements. (PR #101)

### Improvements
- Improved validation feedback in `brier_score()` with clearer and more beginner-friendly error messages for invalid input shape mismatches, making debugging easier for users. (PR #106) Thanks @JavadTe 🙌
- Improved macOS CI reliability by resolving `xgboost.core.XGBoostError` related to missing `libomp.dylib` discovery during GitHub Actions execution. (PR #96)
- Improved cross-platform CI stability and macOS workflow reliability. (PR #97)

### Changed
- Added explicit `__all__` exports to metrics modules (`calibration`, `failure`, `bias`, `representation`) to improve API clarity and consistency. Thanks @JavadTe 🙌

### Maintenance
- Added a temporary CI trigger workflow to validate and debug macOS GitHub Actions behavior during infrastructure stabilization. (PR #98, later superseded and closed)

---

## [v0.4.0] - 2026-05-15

### Major Architectural Milestone: Framework-Agnostic Core
This release marks the transition of TrustLens from a scikit-learn-specific library to a framework-agnostic trustworthiness platform.

### Added
- **Prediction Resolver Architecture**: A new plugin-based backend system for resolving predictions across different ML frameworks.
- **XGBoost Support**: Native support for `XGBClassifier` and raw `Booster` objects (including DMatrix conversion and objective-based task blocking).
- **Manual Override Mode**: Full support for `model=None` workflows where users provide `y_pred` and `y_prob` manually.
- **Degraded Mode Transparency**: Explicit metadata tracking for missing components (`degraded_mode`, `missing_components`) when probabilistic data is unavailable.
- **Hardened Prediction Contract**: Strict validation for non-finite values (NaN/Inf) and probability range enforcement with EPS tolerance.
- **Unified JSON Artifact Export**: `TrustReport.save("report.json")` now produces a single, self-contained JSON artifact containing results, metadata, and trust scores.

### Improved
- **Architectural Decoupling**: Fully refactored `trustlens/api.py` and `trustlens/core/pipeline.py` to be framework-agnostic.
- **Lazy Loading**: Framework-specific dependencies (like XGBoost) are now loaded lazily, ensuring a minimal footprint for scikit-learn users.
- **CI/CD Pipeline**: Added Python 3.13 support, security auditing (`pip-audit`), and automated build validation.
- **Documentation**: Fully synchronized Sphinx documentation, including new internal RFCs for backend developers.

### Fixed
- **Multiclass Brier Score**: Fixed a core metric bug where calibration analysis assumed binary probabilities for all models. TrustLens now correctly computes the Multiclass Brier Score (Mean Squared Error across all classes) for N-class problems.
- Fixed numerical instability in calibration metrics via automatic probability clipping.
- Improved classifier detection to support custom mock objects and non-BaseEstimator wrappers.
- Fixed Sphinx build warnings related to non-consecutive header levels and broken links.
- Corrected relative links in `docs/index.md` and `docs/EXPERIMENTAL.md`.
- Fixed CI failures by updating mypy configuration for Python 3.9 EOL compliance.
- Resolved type-shadowing and incompatible assignment errors in `GradCAM`.
- Hardened prediction resolver registry with explicit type hints and improved error handling.
- Fully propagated `Optional[y_prob]` support through the core pipeline, plugin architecture, and visualization dashboard, ensuring robust handling of non-probabilistic models.

### Compatibility / Migration
- **No breaking changes** for existing scikit-learn users.
- `analyze()` remains backward compatible; existing workflows will continue to work unchanged while benefiting from improved internal validation.

## [0.3.0] — 2026-05-06

### Added
- 2D embedding visualization (`plot_embedding_2d`) with automatic UMAP → t-SNE → PCA fallback, class-colored scatter plot, silhouette score annotation, and configurable subsampling (`n_max`). Integrated into `report.plot()` auto-dispatch. Thanks @WeiGuang-2099
- `embedding_separability` metric computing silhouette score, within/between-class distances, and separability ratio. Thanks @WeiGuang-2099
- 14 tests covering representation metrics, CKA, and 2D embedding visualization. Thanks @WeiGuang-2099
- Model comparison API (`trustlens.compare`) for head-to-head multi-model evaluation and recommendation.
- Pattern detection system (e.g., "Calibration Drift", "Confidently Wrong") to surface high-level semantic risks.
- Initial `equalized_odds()` fairness metric with per-group TPR/FPR analysis (closes #17). Thanks @komoike-oss28-ui
- Ranked score explanation layer to justify Trust Score deductions.
- `equalized_odds()`: added input validation, configurable violation thresholds (`severe_threshold`, `moderate_threshold`), and concrete docstring examples (closes #41) Thanks @komoike-oss28-ui
- Fairness visualization module (`trustlens/visualization/fairness.py`) with `plot_subgroup_performance()`, `plot_equalized_odds()`, and `plot_fairness_gap()` (closes #52) Thanks @komoike-oss28-ui
- Upgraded `TrustReport.plot_bias()` with multi-mode diagnostic support:
  - New `mode` parameter: `"summary"` (default), `"all"`, `"subgroup"`, `"equalized_odds"`, and `"gap"`.
  - Added deterministic return contracts (Returns `Figure` or `dict[str, Figure | None]`).
  - Implemented backend-safe `plt.show()` and automated `save_path` suffixing for batch plotting.
  - Hardened validation for bias data structures and added memory hygiene documentation.
- Added bias analysis demo with subgroup diagnostics (`examples/bias_analysis_demo.py`). Thanks @sidharth-vijayan
- Added SECURITY.md. Thanks @MustansirNisar
- Added unit tests for multi-feature fairness visualizations covering all-features-processed guarantee, output key matching, and figure smoke tests (`tests/test_fairness_visualization_multi.py`). Thanks @komoike-oss28-ui
- `_plot_multi_helper()` — internal helper that eliminates duplication across `*_multi` wrappers and enforces deterministic (sorted) feature iteration.
- `_safe_name()` — filename sanitizer for feature names containing spaces or special characters (e.g., `"income level"` → `income_level`).
- `_BIAS_PLOT_TYPES` — internal registry for deterministic plot-type dispatch ordering in `_plot_bias()`.
- `tests/conftest.py` — centralized Agg backend configuration for the test suite.
- `tests/test_plot_module_multi_feature.py` — 23 integration tests covering nested figure outputs, filename sanitization, orchestrated saving, and edge cases.
- `TrustReport.plot_bias()` now accepts an opt-in `multi_feature: bool = False` parameter for per-feature visualization output. With `multi_feature=True`, single modes (`"subgroup"`, `"equalized_odds"`, `"gap"`) return `dict[str, Figure]` keyed by feature name, and `mode="all"` returns a nested `dict[str, dict[str, Figure]]` keyed by mode then feature. The structure is fixed by the `(mode, multi_feature)` combination, missing components are represented by empty dicts (never `None`), and feature ordering is deterministic (`sorted(feature_names)`). Default behavior (`multi_feature=False`) is unchanged. `tests/test_plot_bias_multi_feature.py` adds 18 tests covering the four return-shape cells, partial-data handling, deterministic ordering, and invalid-mode interaction (closes #74). Thanks @komoike-oss28-ui

### Improved
- Final Trust Score logic now includes a base score, penalty breakdown, and decisive deployment verdicts.
- Standardized canonical terminology to "confidence-weighted errors".
- Enhanced failure diagnostics with confidence concentration insights (range analysis).
- Bias reporting now includes explicit margin calculations relative to the 0.10 threshold.
- Comparison engine includes causal reasoning (e.g., linking selection to lower penalty burdens).
- Integrated fairness metrics into the main `analyze()` pipeline with safe fallback handling and margin reporting.
- Unified validation error message format in `equalized_odds()` for consistency. Thanks @komoike-oss28-ui
- Enhanced `_violation_level()` docstring with parameter descriptions and threshold details. Thanks @komoike-oss28-ui
- Fairness visualization now supports multiple sensitive features via `plot_subgroup_performance_multi()`, `plot_equalized_odds_multi()`, and `plot_fairness_gap_multi()`, which return per-feature figures as `{feature_name: Figure}`. Fixed `_plot_bias()` to no longer silently drop features after the first (closes #56). Thanks @komoike-oss28-ui
- Enhanced bias module usability with visual diagnostics for easier interpretation.
- Refactored `_plot_bias()` into a pure figure-generation function (no file I/O or side effects). All saving and figure closing is now centralized in `plot_module()`.
- `plot_module()` now handles nested `dict[str, dict[str, Figure]]` outputs for multi-feature bias data, with standardized filenames (`bias_<type>_<feature>.png`).
- Updated `docs/metrics/bias.md`, `docs/features.md`, and `README.md` with multi-feature visualization documentation, usage examples, and file output reference.

### Fixed
- Removed all `matplotlib.use("Agg")` calls from library modules (6 visualization files, `report.py`, `gradcam.py`). This was silently overriding the user's matplotlib backend at import time, breaking interactive use in Jupyter and GUI environments.

### Stability
- Maintained full backward compatibility with the `analyze()` API.
- All 219 tests passing.


---

## [0.2.0] — 2026-04-24

### Added
- Extended CI test matrix to include Python 3.13 (closes #29). Thanks @CrepuscularIRIS
- Standardized GitHub contribution infrastructure:
  - Pull Request template with integrated checklists.
  - Structured YAML Issue templates for Bug Reports and Feature Requests.
  - Dedicated `good-first-issue` template and `config.yml` for triage.
- Overhauled `CONTRIBUTING.md` with a command-driven "First Contribution Guide" and difficulty labeling system.
- Comprehensive test suite in `tests/test_utils.py` covering edge cases for all utility functions.
- `report.save()` now supports direct export to single `.json` and `.txt` files.
- Human-readable text report generation without ANSI colors.
- `docs/EXPERIMENTAL.md` — contributor-facing guide for experimental module governance.


### Improved
- Enhanced `utils.py` with robust input validation and NumPy-aware numeric type checking.
- Added progress messages in `analyze()` for better runtime visibility. Thanks @jayssSmm
- Codebase stabilization: isolated experimental modules (`explainability/`, `metrics/faithfulness.py`) from the production pipeline with clear `# NOTE:` headers and documentation.
- Cleaned public API surface — `__init__.py` docstring now reflects only production-ready capabilities.
- Updated README architecture tree to distinguish stable vs experimental modules.
- Replaced misleading `pyproject.toml` keyword `"explainability"` with `"model trust"`.
- Renamed `examples/cnn_vs_vit_trustlens.py` → `examples/model_comparison.py` to match actual content (sklearn models, not deep learning).
- Added actionable Pipeline Module Registry guard in `api.py` to prevent accidental re-exposure of experimental code.

### Fixed
- Prevented crashes in `describe_array` for empty inputs.
- Corrected bin count computation in `reliability_curve()` to use exact binning logic. Thanks @WeiGuang-2099

---

## [0.1.2] — 2026-04-16

### Fixed
- Stabilized Matplotlib plotting backends for headless environments
- Resolved NumPy division-by-zero warnings in histograms
- Fixed trailing whitespace and end-of-file linting violations

### Improved
- Standardized `pyproject.toml` and documentation
- Enhanced small-dataset reliability warnings
- Robust CI/CD pipeline integration across Python versions

---

## [0.1.1] — 2026-04-16

### Fixed
- Resolved NumPy runtime warnings in histogram normalization
- Fixed Matplotlib non-interactive backend warning (`FigureCanvasAgg` warning suppressed via backend-aware `plt.show()` guard)
- Improved plotting stability with controlled rendering and `plt.close()` cleanup

### Improved
- Cleaner console output in headless and CI environments
- Small dataset warning added for `n < 30` samples
- `show: bool = True` parameter added to all visualization functions for optional interactive display

---

## [0.1.0] — 2026-04-16

- `trustlens.quick_analyze()` — zero-friction, branded entry point with auto-loading demo data
- `trustlens.analyze()` — primary analysis API with module dispatch
- `TrustReport` result container with rich `_repr_html_` for Jupyter, plus `show()`, `plot()`, `save()`
- **Calibration module**: `brier_score`, `expected_calibration_error`, `reliability_curve`
- **Failure module**: `misclassification_summary`, `confidence_gap`
- **Bias module**: `class_imbalance_report`, `subgroup_performance`
- **Representation module**: `embedding_separability`, `centered_kernel_alignment`
- **Explainability**: `GradCAM` class with hook-based PyTorch implementation
- **Faithfulness**: `pixel_deletion_test`, `pixel_insertion_test` with AUPC metric
- **Visualization**: Professional base64-rendered Jupyter dashboards and premium Matplotlib visualizations
- **UX**: `tqdm` progress tracking for long-running batch analysis
- **Plugin system**: `BasePlugin` ABC + `PluginRegistry` singleton
- Full test suite: `test_calibration`, `test_failure`, `test_bias`, `test_representation`, `test_api`, `test_plugins`
- Examples: `trustlens_demo.ipynb` (Colab-ready), `quickstart.py`, `calibration_deep_dive.py`
- GitHub Actions CI workflow (linting, testing, and formatting)
- Complete documentation: README (with logo), CONTRIBUTING, ROADMAP, this CHANGELOG

[Unreleased]: https://github.com/Khanz9664/TrustLens/compare/v0.5.0...HEAD
[v0.5.0]: https://github.com/Khanz9664/TrustLens/compare/v0.4.0...v0.5.0
[v0.4.0]: https://github.com/Khanz9664/TrustLens/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Khanz9664/TrustLens/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Khanz9664/TrustLens/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/Khanz9664/TrustLens/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Khanz9664/TrustLens/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Khanz9664/TrustLens/releases/tag/v0.1.0
