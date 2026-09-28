"""
Formula contract for Trust Score methodology 2.0 (TL-10).

Each formula below is transcribed from ADR-001 / docs/trust_score_explained.md
and evaluated independently of the implementation. If the code or the
documentation changes alone, these tests fail, so both must change together.
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
    return 100 * (0.8 * detection + 0.2 * (1 - error_rate))


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
    assert SCORE_VERSION == "2.0"
    assert compute_trust_score(_results(0.02, 0.9, 0.1)).score_version == "2.0"


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
        ({"accuracy": 0.6, "baseline": 0.6}, "no predictive skill"),
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


def test_weak_dimension_caps_at_grade_c():
    ts = compute_trust_score(_results(0.24, 0.99, 0.01))  # calibration ≈ 4
    assert ts.grade == "C" and ts.score <= 59
    assert any("Weak dimension" in c for c in ts.caps_applied)


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
