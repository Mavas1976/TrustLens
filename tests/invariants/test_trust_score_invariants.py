"""
Trust Score invariants.

Properties any trustworthy composite score must satisfy regardless of how its
weights and thresholds are tuned. Each test names the audit issue it guards
(``audit/2026-09-trustlens-audit.md``). A test marked ``xfail(strict=True)``
documents a known defect: the fix for that issue must turn it green, and the
strict marker then forces the marker's removal in the same change.
"""

from __future__ import annotations

import numpy as np
import pytest

from trustlens import analyze, compute_trust_score

RNG_SEED = 20260928


def _calibrated_binary(n: int, sharpness: float, seed: int = RNG_SEED):
    """Perfectly calibrated binary predictions: y ~ Bernoulli(p)."""
    rng = np.random.default_rng(seed)
    p = 1.0 / (1.0 + np.exp(-sharpness * rng.normal(size=n)))
    y = (rng.random(n) < p).astype(int)
    y_prob = np.column_stack([1.0 - p, p])
    y_pred = (p >= 0.5).astype(int)
    return y, y_pred, y_prob


def _perfect_binary(n: int = 400, confidence: float = 0.99):
    y = np.tile([0, 1], n // 2)
    p = np.where(y == 1, confidence, 1.0 - confidence)
    return y, y.copy(), np.column_stack([1.0 - p, p])


# ---------------------------------------------------------------------------
# A good model must never be blocked (TL-01)
# ---------------------------------------------------------------------------


def test_all_correct_confident_model_is_not_blocked():
    """TL-01: a model that is always right with 0.99 confidence is the ideal case."""
    y, y_pred, y_prob = _perfect_binary()
    ts = analyze(None, None, y, y_pred=y_pred, y_prob=y_prob, verbose=False).trust_score
    assert not ts.is_blocked, ts.verdict
    assert ts.grade == "A", (ts.score, ts.sub_scores)


@pytest.mark.parametrize("sharpness", [4.0, 8.0])
def test_calibrated_accurate_binary_model_is_not_blocked(sharpness):
    """TL-01: perfectly calibrated models with high accuracy must not be blocked."""
    y, y_pred, y_prob = _calibrated_binary(5000, sharpness)
    assert (y == y_pred).mean() > 0.85
    ts = analyze(None, None, y, y_pred=y_pred, y_prob=y_prob, verbose=False).trust_score
    assert not ts.is_blocked, (ts.verdict, ts.sub_scores)


# ---------------------------------------------------------------------------
# A dimension that was not assessed is never scored as 0 (TL-02, TL-03)
# ---------------------------------------------------------------------------


def test_skipped_calibration_is_not_scored_as_zero():
    """TL-02: without y_prob, calibration is unknown, not worst-case."""
    y, y_pred, _ = _perfect_binary()
    ts = analyze(None, None, y, y_pred=y_pred, verbose=False).trust_score
    assert "calibration" not in ts.sub_scores, ts.sub_scores
    assert ts.is_partial
    assert "calibration" in ts.missing_dimensions


def test_skipped_module_dict_is_dropped_in_compute_trust_score():
    """TL-02: the public scorer redistributes a skipped module's weight."""
    results = {
        "calibration": {"status": "skipped", "reason": "missing_probabilities"},
        "bias": {"class_imbalance": {"imbalance_ratio": 1.0}},
    }
    ts = compute_trust_score(results)
    assert "calibration" not in ts.sub_scores
    assert ts.is_partial


def test_partial_run_cannot_earn_a_passing_grade():
    """TL-03: a subset of modules must never yield 'production-ready'."""
    y, y_pred, y_prob = _calibrated_binary(2000, 1.0)
    groups = np.where(np.arange(len(y)) % 2 == 0, "a", "b")
    report = analyze(
        None,
        None,
        y,
        y_pred=y_pred,
        y_prob=y_prob,
        sensitive_features={"g": groups},
        modules=["bias"],
        verbose=False,
    )
    ts = report.trust_score
    assert ts.is_partial
    assert ts.grade not in ("A", "B"), (ts.score, ts.grade, ts.verdict)
    assert report.metadata["partial"] is True


def test_unknown_module_name_is_rejected():
    """TL-03: a typo in ``modules=`` must fail loudly instead of scoring 0/D."""
    y, y_pred, y_prob = _perfect_binary()
    with pytest.raises(ValueError, match="calibartion"):
        analyze(None, None, y, y_pred=y_pred, y_prob=y_prob, modules=["calibartion"])


# ---------------------------------------------------------------------------
# Scale invariance of the regression score (TL-04)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scale",
    [
        1e-5,
        1e-3,
        1e3,
        1e5,
    ],
)
def test_regression_score_is_scale_invariant(scale):
    """TL-04: rescaling target and prediction together must not change the score."""
    rng = np.random.default_rng(RNG_SEED)
    y = rng.normal(size=500)
    noisy = y + rng.normal(scale=0.5, size=500)
    bad = rng.normal(size=500)
    for y_pred in (noisy, bad):
        base = analyze(None, None, y, y_pred=y_pred, task="regression", verbose=False)
        scaled = analyze(
            None, None, y * scale, y_pred=y_pred * scale, task="regression", verbose=False
        )
        assert scaled.trust_score.score == base.trust_score.score
        assert scaled.trust_score.grade == base.trust_score.grade


# ---------------------------------------------------------------------------
# Fairness gaps only from defined rates (TL-06)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="TL-06")
def test_perfect_classifier_has_no_fairness_violation():
    """TL-06: a group without positives has an undefined TPR, not a TPR of 0."""
    y, y_pred, y_prob = _perfect_binary(400)
    y[:40] = 0
    y_pred[:40] = 0
    p = np.where(y == 1, 0.99, 0.01)
    y_prob = np.column_stack([1.0 - p, p])
    groups = np.where(np.arange(len(y)) < 40, "A", "B")
    ts = analyze(
        None,
        None,
        y,
        y_pred=y_pred,
        y_prob=y_prob,
        sensitive_features={"g": groups},
        verbose=False,
    ).trust_score
    assert "Fairness" not in ts.penalties_applied, ts.penalties_applied
    assert not ts.is_blocked, ts.verdict


@pytest.mark.xfail(strict=True, reason="TL-06")
def test_tiny_group_does_not_trigger_fairness_block():
    """TL-06: a single-sample group carries no statistical evidence of bias."""
    y, y_pred, y_prob = _calibrated_binary(3000, 8.0)
    groups = np.array(["big"] * len(y), dtype=object)
    wrong = np.flatnonzero(y != y_pred)[0]
    groups[wrong] = "tiny"
    ts = analyze(
        None,
        None,
        y,
        y_pred=y_pred,
        y_prob=y_prob,
        sensitive_features={"g": groups},
        verbose=False,
    ).trust_score
    assert "Fairness" not in ts.penalties_applied, ts.penalties_applied


# ---------------------------------------------------------------------------
# Task routing (TL-07)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="TL-07")
def test_integer_valued_regression_target_is_not_classification():
    """TL-07: a count-like target with many distinct values is regression."""
    rng = np.random.default_rng(RNG_SEED)
    y = np.round(rng.normal(100, 30, size=600)).astype(int)
    y_pred = y + rng.integers(-5, 6, size=600)
    report = analyze(None, None, y, y_pred=y_pred, verbose=False)
    assert report.task_type == "regression"


def test_small_label_set_stays_classification():
    """TL-07 guard: ordinary class labels must keep routing to classification."""
    y, y_pred, y_prob = _perfect_binary()
    report = analyze(None, None, y, y_pred=y_pred, y_prob=y_prob, verbose=False)
    assert report.task_type == "classification"


# ---------------------------------------------------------------------------
# HTML output never executes user-supplied text (TL-08)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="TL-08")
def test_html_repr_escapes_feature_names():
    """TL-08: feature names come from data and must be escaped in HTML."""
    payload = "<script>alert(1)</script>"
    y, y_pred, y_prob = _calibrated_binary(2000, 8.0)
    groups = np.where(np.arange(len(y)) % 3 == 0, "a", "b")
    # Degrade group "a" so the subgroup-gap insight (which names the feature) fires.
    flip = (groups == "a") & (np.arange(len(y)) % 2 == 0)
    y_pred = np.where(flip, 1 - y_pred, y_pred)
    report = analyze(
        None,
        None,
        y,
        y_pred=y_pred,
        y_prob=y_prob,
        sensitive_features={payload: groups},
        verbose=False,
    )
    html_out = report._repr_html_()
    assert payload not in html_out
    assert payload not in report.trust_score._repr_html_()


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


def test_worse_calibration_never_raises_the_score():
    """Distorting calibrated probabilities (same predictions) must not help."""
    y, y_pred, y_prob = _calibrated_binary(4000, 3.0)
    p = y_prob[:, 1]
    scores = []
    for temperature in (1.0, 0.7, 0.5, 0.3):
        logit = np.log(p / (1.0 - p)) / temperature
        q = np.clip(1.0 / (1.0 + np.exp(-logit)), 1e-6, 1.0 - 1e-6)
        ts = analyze(
            None,
            None,
            y,
            y_pred=y_pred,
            y_prob=np.column_stack([1.0 - q, q]),
            verbose=False,
        ).trust_score
        scores.append(ts.score)
    assert scores == sorted(scores, reverse=True), scores


# ---------------------------------------------------------------------------
# Trust Score v2 (phase 2) — tracked, not yet fixed
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="TL-05: multiclass Brier unnormalised (phase 2)")
def test_perfectly_calibrated_multiclass_has_high_calibration_score():
    rng = np.random.default_rng(RNG_SEED)
    k, n = 10, 5000
    y = rng.integers(0, k, n)
    y_prob = np.full((n, k), 1.0 / k)
    ts = analyze(
        None, None, y, y_pred=y_prob.argmax(axis=1), y_prob=y_prob, verbose=False
    ).trust_score
    assert ts.sub_scores["calibration"] >= 80.0, ts.sub_scores


def test_compare_never_recommends_a_partial_assessment(capsys):
    """TL-11: a modules= subset or a report without probabilities is not deployable."""
    from trustlens import compare

    y, y_pred, y_prob = _calibrated_binary(2000, 8.0)
    bias_only = analyze(
        None, None, y, y_pred=y_pred, y_prob=y_prob, modules=["bias"], verbose=False
    )
    no_probs = analyze(None, None, y, y_pred=y_pred, verbose=False)
    compare([bias_only, no_probs])
    out = capsys.readouterr().out
    assert "DO NOT DEPLOY" in out
    assert "Recommendation: Deploy" not in out
