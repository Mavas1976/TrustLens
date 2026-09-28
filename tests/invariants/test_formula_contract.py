"""
Formula contract for the Trust Score methodology (TL-10).

The formulas below are transcribed by hand from ADR-001 and
docs/trust_score_explained.md and evaluated independently of the
implementation, so a change to the code alone fails these tests. They do not
read the documentation; the published before/after table is checked against
the code by tests/reference/test_published_table.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from trustlens import compute_trust_score
from trustlens.trust_score import SCORE_VERSION


def documented_calibration(ece):
    return 100 * min(max(1 - ece / 0.25, 0), 1)


def documented_failure(auroc, error_rate):
    detection = 1.0 if error_rate == 0 else min(max(2 * auroc - 1, 0), 1)
    return 100 * (1 - min(max(error_rate / 0.20, 0), 1) * (1 - detection))


def documented_ceiling(value, start, end, floor):
    if value <= start:
        return 100.0
    return 100 - (100 - floor) * min((value - start) / (end - start), 1)


def documented_bias(max_gap):
    return 100 * min(max(1 - max_gap / 0.30, 0), 1)


def _results(ece, auroc, error_rate, gaps=None, oce=0.0, accuracy=None, baseline=0.5):
    accuracy = 1 - error_rate if accuracy is None else accuracy
    results = {
        "calibration": {"brier_score": 0.1, "ece": ece, "overconfidence_error": oce},
        "failure": {
            "misclassification_summary": {"__overall__": {"overall_error_rate": error_rate}},
            "confidence_gap": {"gap": 0.2},
            "confidence_auroc": auroc,
            "n_classes": 2,
            "accuracy": accuracy,
            "baseline_accuracy": baseline,
        },
    }
    if gaps is not None:
        results["bias"] = {
            "class_imbalance": {"imbalance_ratio": 1.0},
            "subgroup_performance": {"g": {"__summary__": {"performance_gap": gaps}}},
        }
    return results


def test_score_version():
    assert SCORE_VERSION == "2.2"
    assert compute_trust_score(_results(0.02, 0.9, 0.1)).score_version == "2.2"


@pytest.mark.parametrize("ece", [0.0, 0.03, 0.1, 0.2, 0.3])
def test_calibration_formula(ece):
    ts = compute_trust_score(_results(ece, 0.9, 0.1))
    assert ts.sub_scores["calibration"] == pytest.approx(documented_calibration(ece), abs=0.05)


@pytest.mark.parametrize(
    ("auroc", "error_rate"), [(0.5, 0.2), (0.75, 0.1), (0.95, 0.05), (0.3, 0.3)]
)
def test_failure_formula(auroc, error_rate):
    ts = compute_trust_score(_results(0.02, auroc, error_rate))
    assert ts.sub_scores["failure"] == pytest.approx(
        documented_failure(auroc, error_rate), abs=0.05
    )


def test_failure_formula_without_errors():
    ts = compute_trust_score(_results(0.02, None, 0.0))
    assert ts.sub_scores["failure"] == pytest.approx(documented_failure(None, 0.0), abs=0.05)


@pytest.mark.parametrize("gap", [0.0, 0.05, 0.12, 0.3])
def test_bias_formula(gap):
    ts = compute_trust_score(_results(0.02, 0.9, 0.1, gaps=gap))
    assert ts.sub_scores["bias"] == pytest.approx(documented_bias(gap), abs=0.05)


def test_weighted_mean_uses_documented_weights_and_redistribution():
    ts = compute_trust_score(_results(0.05, 0.85, 0.1, gaps=0.03))
    cal, fail, bias = (
        documented_calibration(0.05),
        documented_failure(0.85, 0.1),
        documented_bias(0.03),
    )
    expected = (0.35 * cal + 0.30 * fail + 0.25 * bias) / 0.90
    assert ts.base_score == round(expected)
    assert ts.weights_used == pytest.approx(
        {"calibration": 0.389, "failure": 0.333, "bias": 0.278}, abs=1e-3
    )


def test_bias_not_scored_without_sensitive_features():
    ts = compute_trust_score(_results(0.05, 0.85, 0.1))
    assert "bias" not in ts.sub_scores


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"oce": 0.11}, "overconfidence"),
        ({"gaps": 0.16}, "fairness"),
    ],
)
def test_blockers_cap_score_at_39(kwargs, fragment):
    ts = compute_trust_score(_results(0.02, 0.95, 0.05, **kwargs))
    assert ts.is_blocked and ts.grade == "D"
    assert ts.score <= 39 < ts.base_score
    assert any(fragment in b for b in ts.blockers), ts.blockers


def test_blocker_thresholds_are_strict_inequalities():
    assert not compute_trust_score(_results(0.02, 0.95, 0.05, oce=0.10)).is_blocked
    assert not compute_trust_score(_results(0.02, 0.95, 0.05, gaps=0.15)).is_blocked


def test_weak_dimension_limits_the_score():
    # Calibration ≈ 4 (at or below 30) while the weighted score is ≈ 95.
    weights = {"calibration": 0.05, "failure": 0.95}
    ts = compute_trust_score(_results(0.24, 0.99, 0.01), weights=weights)
    assert ts.base_score > 59
    assert ts.grade == "C" and ts.score == 59
    assert any("Weak dimension" in c for c in ts.caps_applied)


@pytest.mark.parametrize(
    ("oce", "gap"),
    [(0.04, None), (0.075, None), (0.0999, None), (0.02, 0.12), (0.02, 0.149)],
)
def test_ceiling_ramps_follow_the_documented_formula(oce, gap):
    ts = compute_trust_score(_results(0.02, 0.99, 0.01, oce=oce, gaps=gap))
    limit = min(
        documented_ceiling(oce, 0.05, 0.10, 39),
        documented_ceiling(gap, 0.10, 0.15, 39) if gap is not None else 100.0,
    )
    assert ts.score == min(ts.base_score, int(np.floor(limit)))


def test_score_is_continuous_across_blocker_thresholds():
    """GA-02: no jump when a signal crosses its blocker threshold."""
    for key, values in (
        ("oce", np.linspace(0.04, 0.12, 161)),
        ("gaps", np.linspace(0.08, 0.17, 181)),
    ):
        scores = [
            compute_trust_score(_results(0.02, 0.99, 0.01, **{key: float(v)})).score for v in values
        ]
        steps = np.abs(np.diff(scores))
        assert steps.max() <= 2, (key, int(steps.max()))
        assert all(a >= b for a, b in zip(scores, scores[1:])), key


@pytest.mark.parametrize("seed", range(20))
def test_grade_always_matches_score_band(seed):
    rng = np.random.default_rng(seed)
    ts = compute_trust_score(
        _results(
            ece=float(rng.uniform(0, 0.3)),
            auroc=float(rng.uniform(0.3, 1)),
            error_rate=float(rng.uniform(0.01, 0.5)),
            gaps=float(rng.uniform(0, 0.3)),
            oce=float(rng.uniform(0, 0.15)),
        )
    )
    band = "A" if ts.score >= 80 else "B" if ts.score >= 60 else "C" if ts.score >= 40 else "D"
    assert ts.grade == band, (ts.score, ts.grade)
    assert ts.penalties_applied == {}


def test_custom_weights_are_validated():
    results = _results(0.05, 0.85, 0.1)
    with pytest.raises(ValueError, match="Unknown weight"):
        compute_trust_score(results, weights={"fairness": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        compute_trust_score(results, weights={"calibration": -0.5})
    ts = compute_trust_score(results, weights={"calibration": 1.0, "failure": 1.0})
    assert ts.weights_used == pytest.approx({"calibration": 0.5, "failure": 0.5})


def test_no_skill_blocks_when_confidence_is_uninformative():
    ts = compute_trust_score(_results(0.02, 0.55, 0.4, accuracy=0.6, baseline=0.6))
    assert ts.is_blocked and any("no predictive skill" in b for b in ts.blockers)


def test_no_skill_with_informative_confidence_is_capped_not_blocked():
    """Review F2: a calibrated rare-event model whose scores never cross 0.5."""
    ts = compute_trust_score(_results(0.02, 0.77, 0.185, accuracy=0.815, baseline=0.815))
    assert not ts.is_blocked
    assert ts.grade == "C" and any("decision threshold" in c for c in ts.caps_applied)


def test_single_class_evaluation_set_is_not_no_skill():
    """Review F1: with one class every correct model ties the baseline of 1.0."""
    ts = compute_trust_score(_results(0.02, None, 0.0, accuracy=1.0, baseline=1.0))
    assert not ts.is_blocked


def test_overconfidence_on_small_sample_is_capped_not_blocked():
    """Review F3: the overconfidence error is too noisy to block on below n = 100."""
    results = _results(0.02, 0.95, 0.05, oce=0.15)
    results["calibration"]["n_samples"] = 40
    ts = compute_trust_score(results)
    assert not ts.is_blocked and ts.grade == "C"
    results["calibration"]["n_samples"] = 400
    assert compute_trust_score(results).is_blocked


def test_legacy_results_are_flagged():
    """Review F4: v0.5.0 results lack the 2.0 inputs; the score says so."""
    results = _results(0.02, 0.9, 0.1)
    del results["failure"]["confidence_auroc"]
    del results["calibration"]["overconfidence_error"]
    with pytest.warns(UserWarning, match="before Trust Score methodology 2.0"):
        ts = compute_trust_score(results)
    assert ts.score_version == "2.2-legacy-input"


def test_zero_weight_on_every_assessed_dimension_is_rejected():
    """Review F7."""
    with pytest.raises(ValueError, match="weight 0"):
        compute_trust_score(_results(0.02, 0.9, 0.1), weights={"calibration": 0.0, "failure": 0.0})


def documented_no_skill_ceiling(accuracy, baseline, auroc, n=None):
    skill = (accuracy - baseline) / (1 - baseline)
    if accuracy >= 1:
        floor = 59.0  # no errors: full detection
    else:
        floor = 39.0 if auroc is None else 39 + 20 * min(max((auroc - 0.6) / 0.1, 0), 1)
    ramp_end = 0.10 if n is None else max(0.10, 10 / (n * (1 - baseline)))
    return documented_ceiling(ramp_end - skill, 0.0, ramp_end, floor)


@pytest.mark.parametrize(
    ("accuracy", "auroc"),
    [(0.61, 0.55), (0.63, 0.55), (0.6, 0.65), (0.62, 0.65), (0.64, 0.9), (0.6, 0.9)],
)
def test_no_skill_ceiling_follows_the_documented_formula(accuracy, auroc):
    ts = compute_trust_score(_results(0.02, auroc, 1 - accuracy, accuracy=accuracy, baseline=0.6))
    limit = documented_no_skill_ceiling(accuracy, 0.6, auroc)
    assert not ts.is_blocked
    assert ts.score <= int(np.floor(limit))
    assert ts.score == min(ts.base_score, int(np.floor(limit))) or any(
        "Weak dimension" in c for c in ts.caps_applied
    )


def test_no_skill_is_continuous_in_accuracy_and_detection():
    """NF-01: one extra correct prediction cannot move a model from D to A."""
    for auroc in (0.5, 0.65, 0.9):
        scores = [
            compute_trust_score(
                _results(0.02, auroc, 1 - float(a), accuracy=float(a), baseline=0.95)
            ).score
            for a in np.linspace(0.94, 0.97, 301)
        ]
        steps = np.abs(np.diff(scores))
        assert steps.max() <= 2, (auroc, int(steps.max()))
    scores = [
        compute_trust_score(_results(0.02, float(u), 0.05, accuracy=0.95, baseline=0.95)).score
        for u in np.linspace(0.5, 0.8, 301)
    ]
    assert np.abs(np.diff(scores)).max() <= 2
    assert scores == sorted(scores)


def test_weak_dimension_ramp_is_linear_between_30_and_40():
    """GA-02: a calibration sub-score of 35 limits the score to 79, not 59."""
    weights = {"calibration": 0.05, "failure": 0.95}
    ts = compute_trust_score(_results(0.25 * (1 - 0.35), 0.99, 0.01), weights=weights)
    assert ts.score == min(ts.base_score, int(np.floor(documented_ceiling(5, 0, 10, 59))))
    scores = [
        compute_trust_score(_results(float(e), 0.99, 0.01), weights=weights).score
        for e in np.linspace(0.14, 0.18, 201)
    ]
    assert np.abs(np.diff(scores)).max() <= 2


@pytest.mark.parametrize(("auroc", "expected"), [(0.55, 39), (0.62, 43), (0.65, 49), (0.75, 59)])
def test_no_skill_end_point_follows_the_published_auroc_constants(auroc, expected):
    """NF3-04: end point 39 at AUROC <= 0.6, rising linearly to 59 at 0.7."""
    ts = compute_trust_score(_results(0.02, auroc, 0.05, accuracy=0.95, baseline=0.95))
    assert ts.base_score > 59
    assert ts.score == expected
    assert ts.is_blocked == (auroc < 0.6)


@pytest.mark.parametrize(
    ("n", "baseline", "correct"),
    [
        (2000, 0.99, 1),
        (2000, 0.99, 3),
        (200, 0.9, 1),
        (60, 0.9, 2),
        (200, 0.995, 1),
        (500, 0.996, 2),
    ],
)
def test_no_skill_ramp_spans_at_least_ten_minority_samples(n, baseline, correct):
    """NF3-01: with few minority samples one correct prediction is a small step."""
    accuracy = baseline + correct / n
    results = _results(0.02, 0.5, 1 - accuracy, accuracy=accuracy, baseline=baseline)
    results["failure"]["n_samples"] = n
    ts = compute_trust_score(results)
    limit = documented_no_skill_ceiling(accuracy, baseline, 0.5, n=n)
    assert ts.score == min(ts.base_score, int(np.floor(limit)))
    # One more correct prediction moves the ceiling by at most 61/10 (NF4-02).
    previous = documented_no_skill_ceiling(accuracy - 1 / n, baseline, 0.5, n=n)
    assert limit - previous <= 61 / 10 + 1e-9 or accuracy >= 1


def test_legacy_results_take_the_sample_count_from_class_counts():
    """NF4-03: without n_samples the ramp still spans ten minority samples."""
    results = _results(0.02, 0.9, 1 - 0.62, accuracy=0.62, baseline=0.6)
    results["bias"] = {"class_imbalance": {"class_counts": {0: 60, 1: 40}}}
    ts = compute_trust_score(results)
    limit = documented_no_skill_ceiling(0.62, 0.6, 0.9, n=100)  # n = sum of the counts
    assert ts.base_score > limit
    assert ts.score == int(np.floor(limit))


def test_perfect_model_is_never_below_the_same_model_with_a_miss():
    """NF5-02: with few minority samples, one more error never raises the score."""
    for n, n_minority in ((500, 1), (500, 2), (1000, 3), (2000, 5), (2000, 9)):
        baseline = 1 - n_minority / n
        scores = []
        for missed in range(n_minority + 1):
            accuracy = 1 - missed / n
            results = _results(
                0.02, None if missed == 0 else 1.0, missed / n, accuracy=accuracy, baseline=baseline
            )
            results["failure"]["n_samples"] = n
            ts = compute_trust_score(results)
            scores.append(ts.score)
            if missed == 0 and n_minority < 10:
                # NF6-04: the limit names the thin evidence, not "barely beats".
                assert any(f"Only {n_minority} non-majority" in c for c in ts.caps_applied)
        assert scores == sorted(scores, reverse=True), (n, n_minority, scores)
