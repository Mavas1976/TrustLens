"""Regression blockers are approached by ceiling ramps, not cliffs (NF-03)."""

from __future__ import annotations

import warnings

import numpy as np

from trustlens import analyze

RNG = np.random.default_rng(0)
Y_TRUE = RNG.normal(size=4000)
NOISE = RNG.normal(size=4000)


def _score(y_pred, half_width):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = analyze(
            None,
            None,
            Y_TRUE,
            y_pred=y_pred,
            task="regression",
            prediction_intervals=(y_pred - half_width, y_pred + half_width),
            confidence_level=0.90,
            verbose=False,
        )
    return report.trust_score, report.results["regression"]


def _assert_lipschitz(signal, scores, slope):
    """No jump: each score step is bounded by the ramp slope times the signal step.

    The measured signal (skill, coverage) itself moves in small discrete steps
    on a finite sample, so the bound is on score change per signal change,
    plus one point for integer rounding.
    """
    d_signal = np.abs(np.diff(signal))
    d_score = np.abs(np.diff(scores))
    assert np.all(d_score <= slope * d_signal + 1), list(zip(signal, scores))


def test_score_is_continuous_as_skill_crosses_zero():
    skills, scores = [], []
    for s in np.linspace(0.97, 1.03, 121):
        noise = s * NOISE * np.std(Y_TRUE) / np.std(NOISE)
        ts, reg = _score(Y_TRUE + noise, 1.645 * np.std(noise))
        skills.append(1 - np.mean(noise**2) / np.var(Y_TRUE))
        scores.append(ts.score)
    assert min(skills) < 0 < max(skills)
    assert min(scores) <= 39 < max(scores)
    # Ceiling slope 61 / 0.10 plus the accuracy sub-score's own slope.
    _assert_lipschitz(skills, scores, slope=61 / 0.10 + 100)


def test_score_is_continuous_as_coverage_crosses_the_miscoverage_blocker():
    y_pred = Y_TRUE + 0.3 * NOISE
    errors, scores = [], []
    for z in np.linspace(1.15, 1.45, 121):
        ts, reg = _score(y_pred, z * 0.3)
        errors.append(reg["interval_coverage"]["calibration_error"])
        scores.append(ts.score)
    assert min(errors) < -0.10 < -0.05 < max(errors)
    assert min(scores) <= 39 < 60 < max(scores)
    # Ceiling slope 61 / 0.05 plus the interval-calibration sub-score's slope.
    _assert_lipschitz(errors, scores, slope=61 / 0.05 + 500)


def test_informativeness_has_no_cliff_at_the_calibration_gate():
    """NF3-02: a level drifting across the calibration tolerance moves the score gradually."""
    from scipy.stats import norm

    y_pred = Y_TRUE + 0.5 * NOISE
    levels = (0.5, 0.8, 0.9)
    errors, scores = [], []
    for s in np.linspace(0.5, 1.2, 281):
        intervals = {
            lvl: (
                y_pred - s * 0.5 * norm.ppf(0.5 + lvl / 2),
                y_pred + s * 0.5 * norm.ppf(0.5 + lvl / 2),
            )
            for lvl in levels
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyze(
                None,
                None,
                Y_TRUE,
                y_pred=y_pred,
                task="regression",
                prediction_intervals=intervals,
                verbose=False,
            )
        cov = report.results["regression"]["interval_coverage"]
        errors.append(max(abs(p["calibration_error"]) for p in cov["per_level"]))
        scores.append(report.trust_score.score)
    # Largest slope any ramp can have: blocker ceiling (61 / 0.05) plus the
    # informativeness weight (100 / 0.05) plus interval calibration (500).
    _assert_lipschitz(errors, scores, slope=61 / 0.05 + 100 / 0.05 + 500)
    assert max(np.abs(np.diff(scores))) <= 6


def test_informativeness_blends_into_the_correlation_score():
    """NF4-01: with predicted variance, worse intervals never raise the score."""
    from scipy.stats import norm

    sigma = 0.3 + 0.4 * np.abs(Y_TRUE)
    y_pred = Y_TRUE + sigma * NOISE
    errors, scores = [], []
    for s in np.linspace(1.0, 1.5, 201):
        intervals = {
            lvl: (
                y_pred - s * sigma * norm.ppf(0.5 + lvl / 2),
                y_pred + s * sigma * norm.ppf(0.5 + lvl / 2),
            )
            for lvl in (0.3, 0.5, 0.7)
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyze(
                None,
                None,
                Y_TRUE,
                y_pred=y_pred,
                task="regression",
                prediction_intervals=intervals,
                predicted_variance=sigma**2,
                verbose=False,
            )
        cov = report.results["regression"]["interval_coverage"]
        # The best-calibrated level sets the sharpness weight.
        errors.append(min(abs(p["calibration_error"]) for p in cov["per_level"]))
        scores.append(report.trust_score.score)
    assert min(errors) < 0.05 and max(errors) > 0.10
    # Old behaviour jumped 12 points at the band edge; now every step is small.
    assert max(np.abs(np.diff(scores))) <= 4, scores
