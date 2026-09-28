"""Task auto-detection (TL-07): class labels vs. integer-valued regression targets."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from trustlens.api import _detect_task

rng = np.random.default_rng(0)


@pytest.mark.parametrize(
    ("y", "expected"),
    [
        (np.array([0, 1] * 50), "classification"),
        (np.array(["cat", "dog"] * 50), "classification"),
        (np.arange(25, dtype=float).repeat(4), "classification"),  # float-encoded labels
        (np.arange(1, 51).repeat(2), "classification"),  # contiguous labels 1..50
        (rng.integers(0, 60, size=10_000) * 7, "classification"),  # few distinct per sample
        (np.round(rng.normal(100, 30, size=600)).astype(int), "regression"),  # counts
        (rng.normal(size=300), "regression"),  # continuous floats
    ],
)
def test_auto_detection_from_target(y, expected):
    assert _detect_task(y, "auto") == expected


def test_fitted_regressor_routes_to_regression():
    X = rng.normal(size=(200, 3))
    y = np.round(X[:, 0] * 3).astype(int)  # few integer values, still a regressor
    assert _detect_task(y, "auto", model=LinearRegression().fit(X, y)) == "regression"


def test_fitted_classifier_routes_to_classification():
    X = rng.normal(size=(600, 3))
    y = np.round(rng.normal(100, 30, size=600)).astype(int)
    model = LogisticRegression(max_iter=50).fit(X[:, :1], (y > 100).astype(int))
    assert _detect_task(y, "auto", model=model) == "classification"


def test_probabilities_imply_classification():
    y = np.round(rng.normal(100, 30, size=600)).astype(int)
    assert _detect_task(y, "auto", y_prob=np.full((600, 2), 0.5)) == "classification"


def test_explicit_task_is_honoured_and_validated():
    y = rng.normal(size=100)
    assert _detect_task(y, "classification") == "classification"
    with pytest.raises(ValueError, match="Invalid task"):
        _detect_task(y, "Regression")
