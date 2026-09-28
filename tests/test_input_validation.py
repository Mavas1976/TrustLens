"""Input contract of analyze() (TL-12, TL-13): one validation layer, clear errors."""

import logging

import numpy as np
import pandas as pd
import pytest

from trustlens import analyze

rng = np.random.default_rng(7)
N = 300


def _binary():
    p = rng.uniform(0.05, 0.95, N)
    y = (rng.random(N) < p).astype(int)
    return y, (p >= 0.5).astype(int), np.column_stack([1 - p, p])


def test_length_mismatch_names_the_argument():
    y, y_pred, y_prob = _binary()
    with pytest.raises(ValueError, match=r"X has 10"):
        analyze(None, np.zeros((10, 3)), y, y_pred=y_pred, y_prob=y_prob, verbose=False)
    with pytest.raises(ValueError, match=r"sensitive_features\['g'\] has 5"):
        analyze(
            None,
            None,
            y,
            y_pred=y_pred,
            y_prob=y_prob,
            sensitive_features={"g": np.array(["a"] * 5)},
            verbose=False,
        )


def test_unnormalised_probabilities_are_rejected():
    y, y_pred, y_prob = _binary()
    with pytest.raises(ValueError, match="sum to 1"):
        analyze(None, None, y, y_pred=y_pred, y_prob=y_prob * 0.5, verbose=False)


def test_array_likes_are_accepted():
    y, y_pred, y_prob = _binary()
    reference = analyze(None, None, y, y_pred=y_pred, y_prob=y_prob, verbose=False)
    as_lists = analyze(
        None, None, list(y), y_pred=list(y_pred), y_prob=y_prob.tolist(), verbose=False
    )
    as_pandas = analyze(
        None,
        None,
        pd.Series(y),
        y_pred=pd.Series(y_pred),
        y_prob=pd.DataFrame(y_prob),
        verbose=False,
    )
    one_d = analyze(None, None, y, y_pred=y_pred, y_prob=y_prob[:, 1], verbose=False)
    for report in (as_lists, as_pandas, one_d):
        assert report.trust_score.score == reference.trust_score.score
        assert isinstance(report.y_true, np.ndarray)


def test_pandas_index_mismatch_warns(caplog):
    y, y_pred, y_prob = _binary()
    shuffled = pd.Series(y_pred, index=rng.permutation(N))
    with caplog.at_level(logging.WARNING, logger="trustlens.core.inputs"):
        analyze(None, None, pd.Series(y), y_pred=shuffled, y_prob=y_prob, verbose=False)
    assert "different index" in caplog.text


def test_sensitive_features_dataframe_and_missing_values():
    y, y_pred, y_prob = _binary()
    groups = rng.choice(["a", "b"], N).astype(object)
    groups[:40] = None
    frame = pd.DataFrame({"region": groups, "tier": rng.choice(["x", "y"], N)})
    report = analyze(
        None, None, y, y_pred=y_pred, y_prob=y_prob, sensitive_features=frame, verbose=False
    )
    subgroups = report.results["bias"]["subgroup_performance"]
    assert set(subgroups) == {"region", "tier"}
    assert "<missing>" in subgroups["region"]


def test_manual_labels_that_are_not_column_indices_are_inferred():
    """TL-13: labels {1, 2} or strings with y_pred + y_prob no longer crash."""
    y, y_pred, y_prob = _binary()
    base = analyze(None, None, y, y_pred=y_pred, y_prob=y_prob, verbose=False).trust_score.score
    shifted = analyze(None, None, y + 1, y_pred=y_pred + 1, y_prob=y_prob, verbose=False)
    names = np.array(["no", "yes"])
    named = analyze(None, None, names[y], y_pred=names[y_pred], y_prob=y_prob, verbose=False)
    assert shifted.trust_score.score == base
    assert named.trust_score.score == base


def test_unencodable_labels_raise_a_clear_error():
    y, _, y_prob = _binary()
    labels = np.array([10, 20, 30])[rng.integers(0, 3, N)]
    with pytest.raises(ValueError, match="class_labels"):
        analyze(None, None, labels, y_pred=labels, y_prob=y_prob, verbose=False)


def test_conformal_coverage_uses_encoded_labels():
    """TL-13: labels 1..3 must be matched against the right prediction-set column."""
    k = 3
    y_idx = rng.integers(0, k, N)
    y_prob = rng.dirichlet(np.ones(k), N)
    sets = np.zeros((N, k), dtype=int)
    sets[np.arange(N), y_idx] = 1  # every set contains the true class
    report = analyze(
        None,
        None,
        y_idx + 1,
        y_pred=y_prob.argmax(axis=1) + 1,
        y_prob=y_prob,
        class_labels=np.array([1, 2, 3]),
        y_pred_sets=sets,
        verbose=False,
    )
    assert report.results["calibration"]["conformal"]["marginal_coverage"] == pytest.approx(1.0)


def test_verbose_false_prints_nothing(capsys):
    """TL-26: progress output is gated by verbose."""
    y, y_pred, y_prob = _binary()
    analyze(None, None, y, y_pred=y_pred, y_prob=y_prob, verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ""
    analyze(None, None, y, y_pred=y_pred, y_prob=y_prob, verbose=True)
    assert "Running calibration analysis" in capsys.readouterr().out
