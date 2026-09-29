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


def test_widening_an_over_covering_level_never_raises_the_score():
    """NF6-01: a wide level dropping out of the sharpness proxy must not help."""
    from scipy.stats import norm

    y = Y_TRUE + 0.5 * NOISE
    y_pred = Y_TRUE
    q_lo, q_hi = np.quantile(y, [0.1, 0.9])
    centre, half = (q_lo + q_hi) / 2, (q_hi - q_lo) / 2
    z50 = norm.ppf(0.75)
    scores = []
    for k in np.linspace(1.0, 1.6, 61):
        intervals = {
            0.5: (y_pred - 0.5 * z50, y_pred + 0.5 * z50),
            0.8: (np.full(y.size, centre - half * k), np.full(y.size, centre + half * k)),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyze(
                None,
                None,
                y,
                y_pred=y_pred,
                task="regression",
                prediction_intervals=intervals,
                verbose=False,
            )
        scores.append(report.trust_score.score)
    rises = [b - a for a, b in zip(scores, scores[1:]) if b > a]
    assert not rises, scores


def test_narrowing_into_over_confidence_gains_at_most_rounding():
    """NF7-01 (methodology 2.3): with no free calibration zone, narrowing a level
    until it is over-confident no longer buys points (64 -> 72 in 2.2)."""
    from scipy.stats import norm

    rng = np.random.default_rng(7)
    f = rng.normal(size=4000)
    y = f + rng.normal(size=4000)
    levels = (0.1, 0.5, 0.9)
    scores, errors = [], []
    for k in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.3):
        intervals = {
            lvl: (
                f - norm.ppf(0.5 + lvl / 2) * (k if lvl == 0.1 else 1.0),
                f + norm.ppf(0.5 + lvl / 2) * (k if lvl == 0.1 else 1.0),
            )
            for lvl in levels
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyze(
                None,
                None,
                y,
                y_pred=f,
                task="regression",
                prediction_intervals=intervals,
                verbose=False,
            )
        cov = report.results["regression"]["interval_coverage"]
        errors.append(next(p["calibration_error"] for p in cov["per_level"] if p["level"] == 0.1))
        scores.append(report.trust_score.score)
    assert errors[-1] < -0.05 < errors[0]
    assert max(scores) <= scores[0] + 1, scores
    assert scores[-1] < scores[0], scores


def test_calibration_asymmetry_is_deliberate():
    """Under-coverage (over-confident) is penalised harder than equal over-coverage
    (conservative), and the score is monotone on each side of nominal coverage."""
    from scipy.stats import norm

    rng = np.random.default_rng(0)
    f = rng.normal(size=4000)
    y = f + rng.normal(size=4000)

    def run(scale, levels=(0.5, 0.8, 0.9)):
        intervals = {
            lvl: (
                f - norm.ppf(0.5 + lvl / 2) * (scale if lvl == 0.8 else 1.0),
                f + norm.ppf(0.5 + lvl / 2) * (scale if lvl == 0.8 else 1.0),
            )
            for lvl in levels
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyze(
                None,
                None,
                y,
                y_pred=f,
                task="regression",
                prediction_intervals=intervals,
                verbose=False,
            )
        cov = report.results["regression"]["interval_coverage"]
        err = next(p["calibration_error"] for p in cov["per_level"] if p["level"] == 0.8)
        return err, report.trust_score

    under = [run(s) for s in (0.8, 0.85, 0.9, 0.95, 1.0)]
    over = [run(s) for s in (1.0, 1.1, 1.2, 1.3, 1.5)]
    # Monotone on each side: further from nominal never scores higher.
    assert [t.score for _, t in under] == sorted(t.score for _, t in under)
    assert [t.score for _, t in over] == sorted((t.score for _, t in over), reverse=True)
    # Asymmetric across sides at a similar miss (about 0.10): under-coverage blocks.
    err_under, ts_under = under[0]
    err_over, ts_over = next((e, t) for e, t in over if e > 0.09)
    assert err_under < -0.10 and 0.09 < err_over < abs(err_under)
    assert ts_under.is_blocked and not ts_over.is_blocked
    assert ts_under.score < ts_over.score
    # Over-coverage is not rewarded: grossly wide intervals end low.
    assert run(2.0, levels=(0.8,))[1].score < under[-1][1].score
