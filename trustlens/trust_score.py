"""
trustlens.trust_score.
======================
The TrustLens Trust Score — a single 0–100 composite measure of model
trustworthiness.

Responsibilities
----------------
* Aggregate metrics from various modules (calibration, failure, bias, representation).
* Weight the assessed dimensions and apply blockers, caps and ceilings.
* Determine the model's deployment verdict and letter grade.

Relationship to other components
--------------------------------
Used primarily by `TrustReport` to instantly summarize the complex `results`
dictionary into an actionable metric.

Why a single score?
-------------------
Practitioners face "metric overload": ECE, Brier Score, silhouette scores,
confidence gaps — great individually but hard to act on as a whole.

The Trust Score distils the TrustLens analysis into one number and grade. It
is a heuristic summary of diagnostic evidence, not a probability of failure,
a certification or a regulatory assessment (ADR-001).

 * **80–100 (A)** — High trust. No critical issues detected.
 * **60–79 (B)**  — Good. Minor issues to address.
 * **40–59 (C)**  — Moderate. Investigate flagged dimensions.
 * **0–39 (D)**   — Low. Serious issues; do not deploy.
 * **N/A**        — No dimension could be scored (insufficient evidence).

Formula (methodology 2.2)
-------------------------
1. Score every dimension that was assessed (0–100):

 * CalibrationScore    = 100 × clip(1 − ECE / 0.25, 0, 1)
 * FailureScore        = 100 × (1 − clip(error_rate / 0.20, 0, 1)
                                 × (1 − clip(2 × AUROC − 1, 0, 1)))
   AUROC = error-detection AUROC of top-label confidence (correct vs wrong).
   Undetectable errors weigh by how often they occur.
 * BiasScore           = 100 × clip(1 − max_gap / 0.30, 0, 1), only when
   sensitive features were supplied; max_gap is the largest defined subgroup
   accuracy gap or equalized-odds TPR/FPR gap over groups with enough support.
 * RepresentationScore = 100 × clip(0.5 + 0.5 × silhouette, 0, 1), only with
   embeddings.

2. Weighted mean over the assessed dimensions. Default weights calibration 0.35,
   failure 0.30, bias 0.25, representation 0.10, renormalised over the
   dimensions present. This is ``base_score``.

3. Blockers (grade D, score capped at 39): no predictive skill (accuracy not
   above the majority-class baseline while the error-detection AUROC is below
   0.6 or unavailable); overconfidence error > 0.10 on at least 100 samples; a
   fairness gap > 0.15. Ceiling ramps lead up to each blocker, so the score is
   continuous in the signal: the maximum score falls linearly from 100 to the
   blocker's 39 as overconfidence goes 0.05 → 0.10, the gap 0.10 → 0.15, and
   the normalised skill (accuracy − baseline) / (1 − baseline) 0.10 → 0. The
   skill ramp spans at least 10 correctly predicted non-majority samples, and
   its end point rises from 39 to 59 as the AUROC goes 0.6 → 0.7.

4. Caps (grade C, score capped at 59): an incomplete assessment
   (``is_partial``: calibration or failure not assessed, or fairness requested
   but not assessable) and fewer than 30 samples. A sub-score below 40 lowers
   the ceiling from 100 to 59 (reached at 30). Sample-count rules (30 samples,
   100 samples for the overconfidence blocker, 30 per fairness group) are
   deliberate steps, not ramps.

Every signal is counted once. There are no additive penalties
(``penalties_applied`` is always empty since 2.0). The grade always matches the
score band.

References
----------
* Brier (1950), Guo et al. (2017) — calibration
* Hardt et al. (2016) — fairness
* Rousseeuw (1987) — silhouette
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCORE_VERSION = "2.2"

# Grade reported when no dimension could be scored (insufficient evidence).
NOT_ASSESSED_GRADE = "N/A"

_DEFAULT_WEIGHTS: dict[str, float] = {
    "calibration": 0.35,
    "failure": 0.30,
    "bias": 0.25,
    "representation": 0.10,
}

_GRADE_THRESHOLDS = [
    (80, "A", "High Trust - no critical issues detected"),
    (60, "B", "Good Trust - minor issues to address"),
    (40, "C", "Moderate Trust - investigate flagged dimensions"),
    (0, "D", "Low Trust - serious issues, do not deploy"),
]

# Dimensions without which a classification verdict is incomplete (ADR-001).
_CORE_DIMENSIONS = ("calibration", "failure")

# Sub-score ramps: the metric value at which a sub-score reaches 0.
_CAL_ECE_AT_ZERO = 0.25
_BIAS_GAP_AT_ZERO = 0.30

# Blockers (overrides; ADR-001 §3). Each fires on a signal that the weighted
# score could otherwise average away.
_BLOCK_OVERCONFIDENCE = 0.10  # overconfidence error (top-label)
_BLOCK_FAIRNESS_GAP = 0.15  # equals the "severe" level of the fairness metrics
# No-skill blocker only when confidence also carries little information about
# errors; otherwise the decision threshold, not the model, is the problem.
_NO_SKILL_AUROC = 0.6
# No-skill ceiling (methodology 2.2, NF-01): normalised skill
# (accuracy - baseline) / (1 - baseline) at which the ceiling is lifted, and the
# error-detection AUROC at which a model without decision skill is limited to
# grade C instead of D. Between the ends the ceiling moves linearly, so one
# extra correct prediction cannot jump a model from D to A.
_NO_SKILL_RAMP_END = 0.10
_NO_SKILL_AUROC_FULL = 0.70
# With few minority samples 0.10 of skill is a single prediction, so the ramp
# spans at least this many correctly predicted non-majority samples (NF3-01).
_NO_SKILL_RAMP_MIN_SAMPLES = 10
# Below this many samples the overconfidence error is too noisy to block on
# (P(OCE > 0.10) ≈ 0.2 at n = 30 for a perfectly calibrated model).
_MIN_SAMPLES_OVERCONFIDENCE = 100
# Fewer samples than this cannot earn a passing grade (GA-11).
_MIN_SAMPLES_FOR_GRADE = 30

# Failure: error rate at which undetectable errors weigh fully (methodology 2.1).
_FAILURE_ERROR_RATE_AT_FULL_WEIGHT = 0.20

# Ceiling ramps (methodology 2.1, GA-02): the maximum score falls linearly from
# 100 at the ramp start to 39 at the blocker threshold, so crossing a blocker
# never makes the score jump.
_OVERCONFIDENCE_RAMP_START = 0.05
_FAIRNESS_RAMP_START = 0.10
_WEAK_DIMENSION_RAMP_START = 30.0  # sub-score where the weak-dimension ceiling reaches 59

# Score caps keep the number consistent with the grade band.
_BLOCKED_SCORE_CAP = 39  # grade D
_CAPPED_SCORE_CAP = 59  # grade C
_WEAK_DIMENSION = 40.0  # below this an assessed sub-score starts lowering the score ceiling


# ---------------------------------------------------------------------------
# Sub-score computers
# ---------------------------------------------------------------------------


def _equalized_odds_failed(bias_data: object) -> bool:
    """True when the equalized-odds computation raised inside the pipeline."""
    eo = bias_data.get("equalized_odds") if isinstance(bias_data, dict) else None
    return isinstance(eo, dict) and eo.get("reason") == "computation_error"


def _is_assessed(dimension: str, data: object) -> bool:
    """Return True when a module result carries the evidence its sub-score needs.

    A skipped module, or a failure analysis that ran without probabilities
    (``status == "degraded"``), has not assessed its dimension. Scoring it from
    defaults would report "unknown" as "worst case" (TL-02).
    """
    if not isinstance(data, dict) or data.get("status") in ("skipped", "degraded"):
        return False
    if dimension == "calibration":
        return data.get("ece") is not None
    if dimension == "failure":
        # Needs an error rate; an empty or placeholder failure block is not
        # evidence (GB-05).
        overall = data.get("misclassification_summary", {}).get("__overall__", {})
        return (
            overall.get("overall_error_rate") is not None
            and data.get("confidence_gap", {}).get("status") != "skipped"
        )
    if dimension == "bias":
        # A crashed equalized-odds computation leaves fairness unassessed: the
        # subgroup gap alone would make the model look fairer than it is (GB-06).
        return not _equalized_odds_failed(data) and bool(_fairness_gaps(data))
    if dimension == "representation":
        silhouette = data.get("separability", {}).get("silhouette_score")
        return silhouette is not None and bool(np.isfinite(float(silhouette)))
    return True


def _calibration_score(cal_data: dict) -> float:
    """
    Compute calibration sub-score (0–100).

    CalibrationScore = 100 × clip(1 − ECE / 0.25, 0, 1)

    ECE is scale-free across the number of classes (top-label ECE for
    multiclass), unlike the multiclass Brier score, which ranges over [0, 2] and
    mixes accuracy into calibration (TL-05). Brier is still reported.
    """
    ece = float(cal_data["ece"])
    return 100.0 * float(np.clip(1.0 - ece / _CAL_ECE_AT_ZERO, 0.0, 1.0))


def _failure_score(fail_data: dict) -> float:
    """
    Compute failure sub-score (0–100).

    FailureScore = 100 × (1 − clip(error_rate / 0.20, 0, 1) × (1 − DetectionScore))
    DetectionScore = clip(2 × AUROC − 1, 0, 1)

    The risk this dimension measures is errors that confidence does not flag.
    It scales with how often errors occur: one undetectable error in a thousand
    costs half a point, not 80 (methodology 2.1, GA-01). AUROC is the
    error-detection AUROC of top-label confidence; without errors the score is
    100. Reports saved before 2.0 lack the AUROC and use the confidence gap
    normalised by its attainable maximum ``1 − 1/K`` as DetectionScore.
    """
    misc = fail_data.get("misclassification_summary", {})
    error_rate = float(misc["__overall__"]["overall_error_rate"])

    auroc = fail_data.get("confidence_auroc")
    if error_rate <= 0.0:
        detection = 1.0
    elif auroc is not None:
        detection = float(np.clip(2.0 * float(auroc) - 1.0, 0.0, 1.0))
    else:
        gap = float(fail_data.get("confidence_gap", {}).get("gap", 0.0))
        n_classes = fail_data.get("n_classes")
        if n_classes is None:
            n_classes = sum(1 for k in misc if not str(k).startswith("__"))
        max_gap = 1.0 - 1.0 / max(int(n_classes), 2)
        detection = float(np.clip(gap / max_gap, 0.0, 1.0))

    error_weight = float(np.clip(error_rate / _FAILURE_ERROR_RATE_AT_FULL_WEIGHT, 0.0, 1.0))
    return 100.0 * (1.0 - error_weight * (1.0 - detection))


def _fairness_gaps(bias_data: dict) -> list[float]:
    """All defined fairness gaps: subgroup accuracy gaps and equalized-odds TPR/FPR gaps."""
    gaps: list[float] = []
    for feat_data in bias_data.get("subgroup_performance", {}).values():
        if isinstance(feat_data, dict):
            gap = feat_data.get("__summary__", {}).get("performance_gap")
            if gap is not None:
                gaps.append(float(gap))
    for feat_data in bias_data.get("equalized_odds", {}).values():
        if isinstance(feat_data, dict):
            summary = feat_data.get("__summary__", {})
            gaps.extend(
                float(summary[k]) for k in ("tpr_gap", "fpr_gap") if summary.get(k) is not None
            )
    return gaps


def _bias_score(bias_data: dict) -> float:
    """
    Compute bias (fairness) sub-score (0–100).

    BiasScore = 100 × clip(1 − max_gap / 0.30, 0, 1)

    ``max_gap`` is the largest defined gap across sensitive features: subgroup
    accuracy gap, equalized-odds TPR gap and FPR gap. The dimension is only
    scored when sensitive features were supplied (TL-15); class imbalance is a
    property of the data and is reported, not scored.
    """
    max_gap = max(_fairness_gaps(bias_data), default=0.0)
    return 100.0 * float(np.clip(1.0 - max_gap / _BIAS_GAP_AT_ZERO, 0.0, 1.0))


def _representation_score(rep_data: dict) -> float:
    """
    Compute representation sub-score (0–100).

    RepScore = 100 × clip(0.5 + 0.5 × silhouette, 0, 1)
    """
    sil = float(rep_data["separability"]["silhouette_score"])
    return 100.0 * float(np.clip(0.5 + 0.5 * sil, 0.0, 1.0))


# ---------------------------------------------------------------------------
# TrustScoreResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrustScoreResult:
    """
    Structured result from the Trust Score computation.

    Attributes
    ----------
    score : int
      Overall Trust Score in [0, 100].
    grade : str
      Letter grade: A / B / C / D, or ``"N/A"`` when no dimension could be
      scored (insufficient evidence; ``score`` is then 0 and meaningless).
    verdict : str
      Plain-English deployment recommendation.
    sub_scores : dict
      Per-dimension scores in [0, 100].
    weights_used : dict
      Actual weights used (after redistribution for missing dimensions).
    breakdown : dict
      Weighted contribution of each dimension to the final score.
    task_type : str
      The task the score was computed for — ``"classification"`` (default) or
      ``"regression"``. Both share this interface (0–100, A–D, verdicts), but a
      regression ``75`` and a classification ``75`` are **not** directly
      comparable: they aggregate different underlying dimensions.
    informativeness_status : str or None
      Regression only (``None`` for classification). Explains how the Uncertainty
      Informativeness dimension was resolved: ``"present"`` (a sharpness proxy or
      error-variance correlation was scored), ``"unusable_uncertainty"``
      (multi-level intervals were supplied but every level missed nominal
      coverage by at least twice the calibration tolerance,
      so the dimension is scored a truthful ``0.0`` rather than dropped), or
      ``"absent"`` (no uncertainty evidence was supplied and the dimension's
      weight was redistributed). Lets a downstream consumer tell "0.0 because the
      supplied uncertainty was unusable" apart from "dropped because none was
      supplied."
    is_partial : bool
      True when a core dimension (calibration or failure) was not assessed, for
      example without probabilities or with a ``modules=`` subset. A partial
      assessment is capped at grade C (ADR-001).
    missing_dimensions : list[str]
      The core dimensions that were not assessed.
    blockers : list[str]
      Critical conditions that forced grade D (score capped at 39).
    caps_applied : list[str]
      Conditions that capped the result at grade C (score capped at 59).
    base_score : int
      Weighted score of the assessed dimensions before blockers and caps.
    penalties_applied : dict
      Deprecated since methodology 2.0 and always empty: signals are counted
      once, in their sub-score. Kept for backward compatibility.
    score_version : str
      Trust Score methodology version (see ADR-001).
    """

    score: int
    grade: str
    verdict: str
    sub_scores: dict[str, float] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)
    breakdown: dict[str, float] = field(default_factory=dict)
    penalties_applied: dict[str, float] = field(default_factory=dict)
    base_score: int = 0
    is_blocked: bool = False
    task_type: str = "classification"
    informativeness_status: str | None = None
    is_partial: bool = False
    missing_dimensions: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    caps_applied: list[str] = field(default_factory=list)
    score_version: str = SCORE_VERSION

    def __str__(self) -> str:
        lines = [
            f"Trust Score: {self.score}/100 [{self.grade}]",
            f"Assessment : {self.verdict}",
            "\nDimension Breakdown:",
        ]
        for dim, score in self.sub_scores.items():
            lines.append(f"  - {dim:<18} {score:5.1f}/100")
        if self.informativeness_status == "unusable_uncertainty":
            lines.append(
                "  (informativeness = 0.0: intervals supplied but no level is calibrated enough)"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"TrustScoreResult(score={self.score}, grade={self.grade!r})"

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks (text is HTML-escaped)."""
        from trustlens._report.html import render_trust_score_html

        return render_trust_score_html(self)


def _score_bar(score: float, width: int = 12) -> str:
    """Return empty string (ASCII bars removed for professional output)."""
    return ""


# ---------------------------------------------------------------------------
# Main computation function
# ---------------------------------------------------------------------------


def _redistribute(weights: dict[str, float], present: list[str]) -> dict[str, float]:
    """Renormalise the weights of the dimensions that were actually scored."""
    total = sum(weights[d] for d in present)
    return {d: weights[d] / total for d in present} if total > 0 else {}


def _ramp_ceiling(value: float, start: float, end: float, floor: float) -> float | None:
    """Maximum allowed score for a risk signal between ``start`` (100) and ``end`` (``floor``)."""
    if value <= start:
        return None
    fraction = min((value - start) / (end - start), 1.0)
    return 100.0 - (100.0 - floor) * fraction


def _finalize(
    raw_score: float,
    blockers: list[str],
    caps: list[str],
    ceilings: list[tuple[float, str]] | None = None,
) -> tuple[int, int, str, str, list[str]]:
    """Turn the weighted score, blockers, caps and ceilings into the reported result.

    Returns ``(score, base_score, grade, verdict, binding)`` where ``binding``
    lists the caps and ceilings that actually lowered the score. A blocked
    result is capped at 39 (grade D), a capped result at 59 (grade C), and each
    ceiling at its own limit; the reported score always lies inside its grade
    band. ``base_score`` keeps the weighted evidence score before any limit.
    """
    base_score = int(round(float(np.clip(raw_score, 0.0, 100.0))))
    limits: list[tuple[float, str]] = [(_CAPPED_SCORE_CAP, c) for c in caps]
    limits += list(ceilings or [])
    score = base_score
    binding: list[str] = []
    if blockers:
        score = min(score, _BLOCKED_SCORE_CAP)
    else:
        for limit, reason in sorted(limits):
            if limit < score:
                score = int(np.floor(limit))
            if limit < base_score:
                binding.append(reason)

    grade, verdict = "D", _GRADE_THRESHOLDS[-1][2]
    for threshold, g, v in _GRADE_THRESHOLDS:
        if score >= threshold:
            grade, verdict = g, v
            break
    if blockers:
        verdict = f"Low Trust - {blockers[0]}"
    elif binding:
        verdict = f"{verdict.split(' - ')[0]} - {binding[0]}"
    return score, base_score, grade, verdict, binding


def _validated_weights(weights: dict[str, float] | None, defaults: dict[str, float]) -> dict:
    """Merge custom weights over the defaults; reject unknown keys and negative values."""
    w = dict(defaults)
    if weights:
        unknown = sorted(set(weights) - set(defaults))
        if unknown:
            raise ValueError(f"Unknown weight key(s) {unknown}. Valid keys: {sorted(defaults)}.")
        invalid = sorted(k for k, v in weights.items() if not np.isfinite(v) or v < 0)
        if invalid:
            raise ValueError(
                f"Weights must be finite and non-negative; got invalid values for {invalid}."
            )
        w.update(weights)
    if sum(w.values()) <= 0:
        raise ValueError("At least one weight must be positive.")
    return w


def _skill_inputs(
    results: dict, failure: dict, calibration: dict, sub_scores: dict
) -> tuple[float | None, float | None, bool]:
    """Return (accuracy, majority baseline, legacy_input) for the blockers.

    Results saved before methodology 2.0 lack the error-detection AUROC, the
    overconfidence error and the accuracy/baseline pair. Accuracy and baseline
    are then rebuilt from the error rate and the class frequencies, and the
    caller marks the score as computed from legacy input (review F4).
    """
    legacy = ("failure" in sub_scores and "confidence_auroc" not in failure) or (
        "calibration" in sub_scores and "overconfidence_error" not in calibration
    )
    accuracy = failure.get("accuracy")
    baseline = failure.get("baseline_accuracy")
    if accuracy is None:
        error_rate = failure.get("misclassification_summary", {}).get("__overall__", {})
        if error_rate.get("overall_error_rate") is not None:
            accuracy = 1.0 - float(error_rate["overall_error_rate"])
    if baseline is None:
        freqs = (results.get("bias") or {}).get("class_imbalance", {}).get("class_frequencies")
        if freqs:
            baseline = float(max(freqs.values()))
    if legacy:
        warnings.warn(
            "Scoring results saved before Trust Score methodology 2.0: the error-detection "
            "AUROC and/or overconfidence error are missing, so the failure sub-score uses the "
            "confidence-gap fallback and the overconfidence blocker cannot fire. Re-run "
            "analyze() for a full 2.0 score.",
            UserWarning,
            stacklevel=3,
        )
    return accuracy, baseline, legacy


def compute_trust_score(
    results: dict,
    weights: dict[str, float] | None = None,
) -> TrustScoreResult:
    """
    Compute the classification Trust Score (methodology ``SCORE_VERSION``) from a results dict.

    Parameters
    ----------
    results : dict
      The ``TrustReport.results`` dictionary.
    weights : dict, optional
      Custom dimension weights for ``"calibration"``, ``"failure"``, ``"bias"``
      and ``"representation"``; merged over the defaults and renormalised over
      the dimensions that were scored. Unknown keys and negative values raise
      ``ValueError``.

    Returns
    -------
    TrustScoreResult
      Structured score result with per-dimension breakdown.

    Notes
    -----
    See the module docstring and ADR-001 for the formulas. In short: the
    weighted mean of the assessed sub-scores, then blockers (grade D, score
    ≤ 39) and caps (grade C, score ≤ 59). Every blocker is approached by a
    ceiling that falls linearly towards 39, so the score has no cliffs. Signals
    are counted once; there are no additive penalties.

    Examples
    --------
    >>> from trustlens.trust_score import compute_trust_score  # doctest: +SKIP
    >>> result = compute_trust_score(report.results)  # doctest: +SKIP
    >>> print(result.score, result.grade)  # e.g. 74 'B'  # doctest: +SKIP
    """
    w = _validated_weights(weights, _DEFAULT_WEIGHTS)

    # 1. Sub-scores for the dimensions that were actually assessed
    scorers = {
        "calibration": _calibration_score,
        "failure": _failure_score,
        "bias": _bias_score,
        "representation": _representation_score,
    }
    sub_scores = {
        dim: fn(results[dim]) for dim, fn in scorers.items() if _is_assessed(dim, results.get(dim))
    }
    missing_dimensions = [d for d in _CORE_DIMENSIONS if d not in sub_scores]
    bias_raw = results.get("bias")
    if _equalized_odds_failed(bias_raw):
        missing_dimensions.append("bias (equalized odds failed)")
    elif (
        isinstance(bias_raw, dict)
        and bias_raw.get("subgroup_performance")
        and "bias" not in sub_scores
    ):
        # Sensitive features were supplied but no two groups were large enough
        # to compare: fairness was requested and not assessed (NF3-06).
        missing_dimensions.append("bias (no two groups large enough to compare)")
    is_partial = bool(missing_dimensions)

    # 2. Weighted mean over the assessed dimensions
    weights_used = _redistribute(w, [d for d in w if d in sub_scores])
    if sub_scores and not weights_used:
        raise ValueError(
            f"All assessed dimensions {sorted(sub_scores)} have weight 0; give at least one "
            "of them a positive weight."
        )
    raw_score = sum(sub_scores[d] * weights_used[d] for d in weights_used)

    failure_raw = results.get("failure")
    calibration_raw = results.get("calibration")
    failure: dict = failure_raw if isinstance(failure_raw, dict) else {}
    calibration: dict = calibration_raw if isinstance(calibration_raw, dict) else {}
    accuracy, baseline, legacy_input = _skill_inputs(results, failure, calibration, sub_scores)

    # 3. Blockers: critical signals that must not be averaged away
    blockers: list[str] = []
    caps: list[str] = []
    ceilings: list[tuple[float, str]] = []
    auroc = failure.get("confidence_auroc")
    if accuracy is not None and baseline is not None and baseline < 1.0:
        # With a single class in y_true every correct model "ties" the baseline,
        # so the check needs at least two classes (baseline < 1).
        skill = (accuracy - baseline) / (1.0 - baseline)
        informative = auroc is not None and float(auroc) >= _NO_SKILL_AUROC
        if skill <= 0 and not informative:
            blockers.append(
                f"Blocked by no predictive skill (accuracy {accuracy:.3f} does not beat "
                f"the majority-class baseline {baseline:.3f})"
            )
        else:
            # Confidence that ranks errors (e.g. a calibrated rare-event model
            # whose scores never cross 0.5) lifts the floor from 39 towards 59:
            # the decisions lack skill, the model may not.
            detection = 0.0
            if auroc is not None:
                detection = float(
                    np.clip(
                        (float(auroc) - _NO_SKILL_AUROC) / (_NO_SKILL_AUROC_FULL - _NO_SKILL_AUROC),
                        0.0,
                        1.0,
                    )
                )
            floor = _BLOCKED_SCORE_CAP + (_CAPPED_SCORE_CAP - _BLOCKED_SCORE_CAP) * detection
            ramp_end = _NO_SKILL_RAMP_END
            n_eval = failure.get("n_samples", calibration.get("n_samples"))
            if n_eval is None:
                # Results saved before n_samples was recorded (NF4-03).
                counts = (results.get("bias") or {}).get("class_imbalance", {}).get("class_counts")
                if counts:
                    n_eval = sum(int(c) for c in counts.values())
            if n_eval:
                n_minority = int(n_eval) * (1.0 - baseline)
                if n_minority > 0:
                    # Not capped at 1: with fewer than 10 non-majority samples
                    # even a perfect model cannot show enough skill to lift the
                    # ceiling fully, and each correct prediction stays a small
                    # step (NF4-02).
                    ramp_end = max(ramp_end, _NO_SKILL_RAMP_MIN_SAMPLES / n_minority)
            limit = _ramp_ceiling(ramp_end - skill, 0.0, ramp_end, floor)
            if limit is not None:
                if skill <= 0:
                    reason = (
                        f"Decisions do not beat the majority-class baseline (accuracy "
                        f"{accuracy:.3f} vs {baseline:.3f}); review the decision threshold"
                    )
                else:
                    reason = (
                        f"Accuracy {accuracy:.3f} barely beats the majority-class baseline "
                        f"{baseline:.3f}"
                    )
                ceilings.append((limit, f"{reason}; limits the score to {int(np.floor(limit))}"))
    oce = calibration.get("overconfidence_error")
    if oce is not None:
        oce_value = float(oce)
        n_cal = calibration.get("n_samples")
        low_support = n_cal is not None and int(n_cal) < _MIN_SAMPLES_OVERCONFIDENCE
        # On small samples the estimate is too noisy to block on: the ceiling
        # stops at 59 (grade C) instead of 39.
        floor = float(_CAPPED_SCORE_CAP if low_support else _BLOCKED_SCORE_CAP)
        limit = _ramp_ceiling(oce_value, _OVERCONFIDENCE_RAMP_START, _BLOCK_OVERCONFIDENCE, floor)
        if oce_value > _BLOCK_OVERCONFIDENCE and not low_support:
            blockers.append(
                f"Blocked by overconfidence (overconfidence error {oce_value:.3f} "
                f"> {_BLOCK_OVERCONFIDENCE})"
            )
        elif limit is not None:
            support = f" on only {n_cal} samples (low support)" if low_support else ""
            ceilings.append(
                (
                    limit,
                    f"Overconfidence error {oce_value:.3f}{support} limits the score "
                    f"to {int(np.floor(limit))}",
                )
            )
    fairness_gaps = _fairness_gaps(results.get("bias", {}) or {})
    if fairness_gaps:
        max_gap = max(fairness_gaps)
        if max_gap > _BLOCK_FAIRNESS_GAP:
            blockers.append(
                f"Blocked by severe fairness violation (largest gap {max_gap:.3f} "
                f"> {_BLOCK_FAIRNESS_GAP})"
            )
        else:
            limit = _ramp_ceiling(
                max_gap, _FAIRNESS_RAMP_START, _BLOCK_FAIRNESS_GAP, float(_BLOCKED_SCORE_CAP)
            )
            if limit is not None:
                ceilings.append(
                    (
                        limit,
                        f"Fairness gap {max_gap:.3f} limits the score to {int(np.floor(limit))}",
                    )
                )

    # 4. Caps: an incomplete assessment or a very weak dimension cannot pass
    n_samples = failure.get("n_samples", calibration.get("n_samples"))
    if n_samples is not None and int(n_samples) < _MIN_SAMPLES_FOR_GRADE:
        caps.append(
            f"Only {int(n_samples)} samples; at least {_MIN_SAMPLES_FOR_GRADE} are needed for a "
            "passing grade (capped at grade C)"
        )
    if is_partial:
        caps.insert(
            0,
            "Incomplete assessment, not assessed: "
            + ", ".join(missing_dimensions)
            + " (capped at grade C)",
        )
    for dim, sub in sorted(sub_scores.items()):
        if sub < _WEAK_DIMENSION:
            # 100 at a sub-score of 40, falling to 59 at 30 and below (no cliff).
            limit = _ramp_ceiling(
                _WEAK_DIMENSION - sub,
                0.0,
                _WEAK_DIMENSION - _WEAK_DIMENSION_RAMP_START,
                float(_CAPPED_SCORE_CAP),
            )
            if limit is not None:
                ceilings.append(
                    (
                        limit,
                        f"Weak dimension: {dim} ({sub:.1f}/100) limits the score to "
                        f"{int(np.floor(limit))}",
                    )
                )

    score, base_score, grade, verdict, binding = _finalize(raw_score, blockers, caps, ceilings)
    if not sub_scores and not blockers:
        # Nothing could be scored: report insufficient evidence instead of a
        # misleading 0/D (ADR-001; SPOS NEEDS_EVIDENCE).
        grade = NOT_ASSESSED_GRADE
        verdict = (
            "Not assessed - no dimension could be scored; provide y_prob "
            "(calibration, failure) and/or sensitive_features (fairness)"
        )

    return TrustScoreResult(
        score=score,
        grade=grade,
        verdict=verdict,
        sub_scores={d: round(sub_scores[d], 1) for d in weights_used},
        weights_used={d: round(weights_used[d], 3) for d in weights_used},
        breakdown={d: round(sub_scores[d] * weights_used[d], 2) for d in weights_used},
        penalties_applied={},
        base_score=base_score,
        is_blocked=bool(blockers),
        task_type="classification",
        is_partial=is_partial,
        missing_dimensions=missing_dimensions,
        blockers=blockers,
        caps_applied=binding,
        score_version=f"{SCORE_VERSION}-legacy-input" if legacy_input else SCORE_VERSION,
    )


# ---------------------------------------------------------------------------
# Regression Trust Score
# ---------------------------------------------------------------------------
#
# A regression-specific scorer that reuses the classification interface
# (``TrustScoreResult``, the 0–100 scale, the A/B/C/D bands, the deployment
# verdicts and the weight-redistribution mechanism) but scores three
# regression-native dimensions instead of the classification four.
#
# Design (RFC #145, converged with the maintainer):
#   Accuracy / Skill            0.30   skill S = 1 − MSE/Var(y)  (= R² vs the
#                                      mean-predictor baseline), docked by a
#                                      p90/median heavy-tail penalty.
#   Interval Calibration        0.40   ICE (mean |emp(tau)-tau| across levels)
#                                      when multi-level intervals are supplied,
#                                      else single-level PICP |calibration_error|,
#                                      through a tolerance (regression analog of ECE).
#   Uncertainty Informativeness 0.30   calibration-conditioned sharpness proxy vs a
#                                      climatology reference (RFC #155) when multi-
#                                      level intervals are supplied, else max(pearson,
#                                      spearman) of predicted uncertainty vs error.
#
# Point-prediction-only reports score on Accuracy alone (the other two are
# redistributed away), exactly as a no-embeddings classification report drops
# Representation today.
#
# Blockers (→ grade D, score ≤ 39): negative skill (S < 0); severe interval
#                        miscoverage (calibration_error < −0.10 — materially
#                        over-confident). Ceilings lead up to both (methodology
#                        2.2): S 0.10 → 0 and shortfall 0.05 → 0.10 lower the
#                        maximum score from 100 to 39.
# Inside a sub-score only: a heavy tail docks the Accuracy/Skill dimension. Weak
#                        uncertainty correlation lowers Informativeness and is not
#                        penalised a second time (methodology 2.0, TL-09).

_REGRESSION_DEFAULT_WEIGHTS: dict[str, float] = {
    "accuracy": 0.30,
    "interval_calibration": 0.40,
    "uncertainty_informativeness": 0.30,
}

# Interval-calibration sub-score: the |calibration_error| (or ICE) at which the
# sub-score reaches 0. A ±0.10 miss → 50/100; ±0.20 → 0/100.
#
# NB: this SCORING tolerance (0.20) is deliberately distinct from — and looser
# than — the 0.05 calibration GATE in metrics/regression.py
# (``multilevel_interval_coverage``'s ``tolerance``). They answer different
# questions and should NOT be unified:
#   * 0.05 (metric layer) decides which levels are calibrated *enough* to count
#     fully in the sharpness proxy, so over-confident intervals cannot be
#     rewarded for looking "sharp". Since methodology 2.2 a level's weight falls
#     linearly from 1 at 0.05 to 0 at 0.10 instead of a hard pass/fail (NF3-02).
#   * 0.20 (here) is a smooth ramp mapping the continuous ICE / |calibration_error|
#     onto the 0–100 sub-score, so calibration quality degrades gracefully rather
#     than cliff-edging. A gradient, not a gate.
# Using 0.05 for the ramp would make the sub-score far too steep; using 0.20 for
# admission would let materially miscalibrated levels inflate the sharpness proxy.
_REG_CALIBRATION_TOLERANCE = 0.20

# Heavy-tail penalty (docks the Accuracy/Skill dimension). The p90/median
# absolute-error ratio at/below the threshold incurs no dock; the dock then ramps
# linearly with the excess up to a capped fraction of the sub-score.
_REG_TAIL_RATIO_THRESHOLD = 3.0
_REG_TAIL_RATIO_SCALE = 7.0
_REG_MAX_TAIL_DOCK_FRACTION = 0.50


# Severe-miscoverage blocker: realised coverage this far below nominal means the
# intervals are materially over-confident (the regression "confidently wrong").
_REG_SEVERE_MISCOVERAGE = -0.10
# Ceiling ramps before the regression blockers (methodology 2.2, NF-03): the
# ceiling falls from 100 to 39 as the skill drops from 0.10 to 0, and as the
# coverage shortfall grows from 0.05 to 0.10, so neither blocker is a cliff.
_REG_SKILL_RAMP_END = 0.10
_REG_MISCOVERAGE_RAMP_START = -0.05


def _regression_accuracy_score(error_dist: dict, target_variance: float) -> dict[str, float]:
    """
    Accuracy/Skill sub-score (0–100) plus the diagnostics needed downstream.

    The skill score ``S = 1 − MSE / Var(y)`` is the coefficient of determination
    against a predict-the-mean baseline (scale-free and comparable across
    datasets). The base sub-score is ``100 × clip(S, 0, 1)``, then docked by a
    heavy-tail penalty derived from the ``p90 / median`` absolute-error ratio so
    a handful of catastrophic errors hidden by the aggregate still lower the
    dimension.

    Returns a dict with ``score`` (the docked sub-score), ``skill`` (the raw
    ``S``, which may be negative → blocker), ``tail_ratio`` and ``tail_dock``
    (points removed by the heavy-tail penalty).
    """
    rmse = float(error_dist.get("rmse", 0.0))
    mse = rmse * rmse
    if target_variance > 0.0:
        skill = 1.0 - mse / target_variance
    else:
        # Constant target: a mean-predictor is already exact, so skill "over the
        # mean" is undefined. Treat a perfect fit as full skill, else none.
        # NB: a 0.0 here unambiguously means a genuine constant target — callers
        # resolve a *missing* variance to a ValueError upstream, never to 0.0
        # (issue #150), so this branch is never reached for absent ground truth.
        skill = 1.0 if mse <= 1e-12 else 0.0

    base = 100.0 * float(np.clip(skill, 0.0, 1.0))

    median = float(error_dist.get("median_absolute_error", 0.0))
    p90 = float(error_dist.get("p90_absolute_error", 0.0))
    if median > 1e-12:
        tail_ratio = p90 / median
    elif p90 > 1e-12:
        # Degenerate (median ~0 but a real tail) → treat as maximally heavy.
        tail_ratio = _REG_TAIL_RATIO_THRESHOLD + _REG_TAIL_RATIO_SCALE
    else:
        tail_ratio = 1.0  # all-zero errors → no tail

    excess = max(0.0, tail_ratio - _REG_TAIL_RATIO_THRESHOLD)
    dock_fraction = float(np.clip(excess / _REG_TAIL_RATIO_SCALE, 0.0, _REG_MAX_TAIL_DOCK_FRACTION))
    tail_dock = base * dock_fraction
    return {
        "score": base - tail_dock,
        "skill": skill,
        "tail_ratio": tail_ratio,
        "tail_dock": tail_dock,
    }


def _interval_calibration_score(coverage: dict) -> float:
    """
    Interval-calibration sub-score (0–100).

    Uses the multi-level ICE (``mean |emp(tau) - tau|``) when present —
    ``100 × (1 − clip(ICE / T, 0, 1))`` — otherwise the single-level PICP
    ``|calibration_error|`` through the same tolerance. Both are the regression
    analog of how :func:`_calibration_score` maps ECE for classification (RFC #155).
    """
    if coverage.get("ice") is not None:
        ice = float(coverage["ice"])
        return 100.0 * (1.0 - float(np.clip(ice / _REG_CALIBRATION_TOLERANCE, 0.0, 1.0)))
    cal_err = float(coverage.get("calibration_error", 0.0))
    return 100.0 * (1.0 - float(np.clip(abs(cal_err) / _REG_CALIBRATION_TOLERANCE, 0.0, 1.0)))


def _uncertainty_informativeness_score(corr: dict) -> float:
    """
    Uncertainty-informativeness sub-score (0–100).

    ``100 × clip(max(pearson, spearman), 0, 1)`` — rewards uncertainty that is
    larger exactly where the realised error is larger.
    """
    strongest = max(float(corr.get("pearson", 0.0)), float(corr.get("spearman", 0.0)))
    return 100.0 * float(np.clip(strongest, 0.0, 1.0))


def _informativeness_from_sharpness(coverage: dict, fallback: float | None = None) -> float:
    """
    Uncertainty-informativeness sub-score (0–100) from the calibration-conditioned
    sharpness proxy (RFC #155).

    ``w × 100 × clip(sharpness_skill, 0, 1) + (1 − w) × fallback`` with ``w`` the
    ``sharpness_weight`` (1 when a level is within the calibration tolerance,
    0 at twice it) and ``fallback`` the correlation score or 0 — rewards
    intervals sharper than the climatology baseline *among well-calibrated levels*, the
    CRPS-Resolution analog of the correlation-based score. Preferred over the
    error-variance correlation when multi-level intervals are available.
    """
    skill = float(coverage.get("sharpness_skill") or 0.0)
    # Weighted by the best level's calibration weight (NF3-02). As the last
    # usable level leaves the calibration band the score moves continuously to
    # what applies once no level is usable: the error-variance correlation
    # score when predicted variance was supplied (``fallback``), else the
    # "unusable uncertainty" 0 (NF4-01). Older results lack the weight.
    weight = float(np.clip(float(coverage.get("sharpness_weight", 1.0)), 0.0, 1.0))
    sharpness = 100.0 * float(np.clip(skill, 0.0, 1.0))
    return weight * sharpness + (1.0 - weight) * (fallback or 0.0)


def _reg_metric_present(metric: dict | None) -> bool:
    """True when an optional regression metric was actually computed (not skipped)."""
    return isinstance(metric, dict) and metric.get("status") != "skipped"


def regression_trust_score(
    results: dict,
    y_true: np.ndarray | None = None,
    weights: dict[str, float] | None = None,
) -> TrustScoreResult:
    """
    Compute the regression Trust Score from a regression report's results.

    Mirrors :func:`compute_trust_score` but scores three regression-native
    dimensions — Accuracy/Skill, Interval Calibration and Uncertainty
    Informativeness — while reusing the same :class:`TrustScoreResult`
    interface, the 0–100 scale, the A/B/C/D bands, the deployment verdicts and
    the weight-redistribution mechanism (see the module-level notes / RFC #145).

    Parameters
    ----------
    results : dict
      Either the full report results dict (with a ``"regression"`` key, as built
      by the regression pipeline) or the inner regression metrics dict directly.
      Expected keys: ``error_distribution`` (always present) plus the optional
      ``interval_coverage`` / ``error_variance_correlation`` (which may be
      ``status="skipped"`` dicts when their inputs were absent).
    y_true : np.ndarray, optional
      Ground-truth targets, used to compute ``Var(y)`` for the skill score. May
      be omitted **only** when ``results`` carries a persisted
      ``regression["target_variance"]`` (as emitted by the regression pipeline,
      issue #150), which is then used as the fallback so a regression Trust Score
      can be recomputed from a stored report alone. If both are supplied, the
      explicitly-passed ``y_true`` wins and a mismatch beyond tolerance warns. If
      neither is available a ``ValueError`` is raised.
    weights : dict, optional
      Custom dimension weights (keys: ``"accuracy"``, ``"interval_calibration"``,
      ``"uncertainty_informativeness"``). Defaults to ``0.30 / 0.40 / 0.30``.

    Returns
    -------
    TrustScoreResult
      With ``task_type="regression"``. A regression ``75`` and a classification
      ``75`` share this interface but are **not** directly comparable.

    Examples
    --------
    >>> from trustlens.trust_score import regression_trust_score  # doctest: +SKIP
    >>> result = regression_trust_score(report.results, report.y_true)  # doctest: +SKIP
    >>> print(result.score, result.grade)  # doctest: +SKIP
    """
    reg = results.get("regression", results)
    error_dist = reg.get("error_distribution", {}) or {}
    coverage = reg.get("interval_coverage", {})
    corr = reg.get("error_variance_correlation", {})

    # Resolve Var(y) for the skill denominator. Priority (issue #150):
    #   1. explicit y_true → np.var (population, ddof=0), no behaviour change;
    #   2. else a persisted results["regression"]["target_variance"];
    #   3. else raise — never silently fall through to 0.0, which would route a
    #      missing variance through the genuine "constant target" branch and
    #      yield a misleading skill score.
    stored_variance = reg.get("target_variance")
    if y_true is not None:
        y_true_arr = np.asarray(y_true, dtype=float)
        target_variance = float(np.var(y_true_arr)) if y_true_arr.size else 0.0
        if stored_variance is not None and not np.isclose(
            target_variance, float(stored_variance), rtol=1e-6, atol=1e-9
        ):
            warnings.warn(
                "regression_trust_score: Var(y) from the supplied y_true "
                f"({target_variance:.6g}) disagrees with the persisted "
                f"target_variance ({float(stored_variance):.6g}); using y_true.",
                stacklevel=2,
            )
    elif stored_variance is not None:
        target_variance = float(stored_variance)
    else:
        raise ValueError(
            "regression_trust_score needs the target variance Var(y) to compute "
            "the skill score, but neither was available: pass y_true, or use a "
            "report whose regression results carry a persisted 'target_variance' "
            "(emitted by the regression pipeline; see issue #150)."
        )

    w = _validated_weights(weights, _REGRESSION_DEFAULT_WEIGHTS)

    sub_scores: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 1. Sub-scores (Accuracy always available; the other two are optional)
    # ------------------------------------------------------------------
    accuracy = _regression_accuracy_score(error_dist, target_variance)
    sub_scores["accuracy"] = accuracy["score"]
    skill = accuracy["skill"]

    coverage_present = _reg_metric_present(coverage)
    interval_present = coverage_present and ("ice" in coverage or "calibration_error" in coverage)
    calibration_error: float | None = None
    if interval_present:
        # Multi-level reports carry the worst per-level gap (most over-confident);
        # single-level PICP reports carry calibration_error. Either drives the
        # severe-miscoverage blocker below.
        calibration_error = float(
            coverage.get("worst_calibration_error", coverage.get("calibration_error", 0.0))
        )
        sub_scores["interval_calibration"] = _interval_calibration_score(coverage)

    # Uncertainty Informativeness: prefer the calibration-conditioned sharpness
    # proxy (multi-level intervals, RFC #155); fall back to the error-variance
    # correlation when only predicted variance is available.
    sharpness_skill = coverage.get("sharpness_skill") if coverage_present else None
    n_interval_levels = int(coverage.get("n_levels", 0)) if coverage_present else 0
    n_calibrated_levels = int(coverage.get("n_calibrated_levels", 0)) if coverage_present else 0
    corr_present = _reg_metric_present(corr) and ("pearson" in corr or "spearman" in corr)
    informativeness_status = "absent"
    if sharpness_skill is not None:
        fallback = _uncertainty_informativeness_score(corr) if corr_present else None
        sub_scores["uncertainty_informativeness"] = _informativeness_from_sharpness(
            coverage, fallback
        )
        informativeness_status = "present"
    elif corr_present:
        sub_scores["uncertainty_informativeness"] = _uncertainty_informativeness_score(corr)
        informativeness_status = "present"
    elif n_interval_levels >= 2 and n_calibrated_levels == 0:
        # RFC #155 follow-up (PR #161 review): multi-level intervals WERE
        # supplied, but every level fell outside the calibration band, so
        # ``sharpness_skill`` is None because the uncertainty was *unusable*,
        # not because it was absent — and there is no error-variance-correlation
        # fallback either. Score a truthful ``Informativeness = 0.0`` instead of
        # dropping the dimension and redistributing its 0.30 weight: "the
        # supplied uncertainty delivered zero usable resolution" is a real,
        # scorable failure, distinct from "no uncertainty was provided at all"
        # (which stays on the redistribute path). Scoped to the multi-level path
        # (``n_levels >= 2``): the single-level PICP path has no
        # "we tried and it was unusable" signal and keeps redistributing.
        sub_scores["uncertainty_informativeness"] = 0.0
        informativeness_status = "unusable_uncertainty"

    # ------------------------------------------------------------------
    # 2. Redistribute weights across the dimensions actually present
    #    (a point-only report collapses to Accuracy = 1.00)
    # ------------------------------------------------------------------
    active_dims = [d for d in w if d in sub_scores]
    total_active_weight = sum(w[d] for d in active_dims)
    weights_used: dict[str, float] = {}
    if total_active_weight > 0:
        for dim in active_dims:
            weights_used[dim] = w[dim] / total_active_weight
    else:
        for dim in active_dims:
            weights_used[dim] = 1.0 / len(active_dims) if active_dims else 0.0

    raw_score = sum(sub_scores[d] * weights_used[d] for d in active_dims)
    breakdown = {d: round(sub_scores[d] * weights_used[d], 2) for d in active_dims}

    # ------------------------------------------------------------------
    # 3. Blockers → grade D (negative skill; severe interval miscoverage).
    #    Weak uncertainty correlation is scored only in its sub-score; the
    #    former extra composite penalty counted the same signal twice (TL-09).
    # ------------------------------------------------------------------
    blockers: list[str] = []
    ceilings: list[tuple[float, str]] = []
    if skill < 0.0:
        blockers.append("Blocked by negative skill (worse than predicting the mean; R^2 < 0)")
    else:
        limit = _ramp_ceiling(
            _REG_SKILL_RAMP_END - skill, 0.0, _REG_SKILL_RAMP_END, float(_BLOCKED_SCORE_CAP)
        )
        if limit is not None:
            ceilings.append(
                (
                    limit,
                    f"Skill R^2 {skill:.3f} is close to the mean predictor; limits the score "
                    f"to {int(np.floor(limit))}",
                )
            )
    if interval_present and calibration_error is not None:
        if calibration_error < _REG_SEVERE_MISCOVERAGE:
            blockers.append(
                f"Blocked by severe interval miscoverage (coverage {calibration_error:+.2f} "
                "below nominal - over-confident intervals)"
            )
        else:
            limit = _ramp_ceiling(
                -calibration_error,
                -_REG_MISCOVERAGE_RAMP_START,
                -_REG_SEVERE_MISCOVERAGE,
                float(_BLOCKED_SCORE_CAP),
            )
            if limit is not None:
                ceilings.append(
                    (
                        limit,
                        f"Interval coverage {calibration_error:+.3f} below nominal limits the "
                        f"score to {int(np.floor(limit))}",
                    )
                )
    final_score, base_score, grade, verdict, binding = _finalize(raw_score, blockers, [], ceilings)

    return TrustScoreResult(
        score=final_score,
        grade=grade,
        verdict=verdict,
        sub_scores={d: round(sub_scores[d], 1) for d in active_dims},
        weights_used={d: round(weights_used[d], 3) for d in active_dims},
        breakdown=breakdown,
        penalties_applied={},
        base_score=base_score,
        is_blocked=bool(blockers),
        blockers=blockers,
        caps_applied=binding,
        task_type="regression",
        informativeness_status=informativeness_status,
    )
