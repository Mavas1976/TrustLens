"""
Reference-model benchmark.

Real, deterministic scikit-learn models on bundled datasets with the grade band a
reasonable practitioner would expect. The suite turns any change to the Trust
Score methodology into a visible, reviewed diff instead of a silent shift for
users. When a methodology change legitimately moves a band, update the
expectation here together with the ADR (docs/adr/) and the CHANGELOG.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trustlens import analyze


def _split(loader, stratify=True):
    X, y = loader(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if stratify else None)


def _classifier_report(loader, model):
    X_train, X_test, y_train, y_test = _split(loader)
    model.fit(X_train, y_train)
    return analyze(model, X_test, y_test, verbose=False), model.score(X_test, y_test)


def _scaled_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


# (id, loader, model factory, allowed grades)
GOOD_CLASSIFIERS = [
    pytest.param(
        "breast_cancer-logreg",
        load_breast_cancer,
        _scaled_logreg,
        {"A", "B"},
        marks=pytest.mark.xfail(strict=True, reason="TL-01"),
    ),
    (
        "breast_cancer-rf",
        load_breast_cancer,
        lambda: RandomForestClassifier(200, random_state=0),
        {"A", "B"},
    ),
    ("iris-logreg", load_iris, _scaled_logreg, {"A", "B", "C"}),
    pytest.param(
        "wine-rf",
        load_wine,
        lambda: RandomForestClassifier(200, random_state=0),
        {"A"},
        marks=pytest.mark.xfail(strict=True, reason="TL-01"),
    ),
]


def _as_param(case):
    values, marks = (case.values, case.marks) if hasattr(case, "values") else (case, ())
    return pytest.param(*values[1:], id=values[0], marks=marks)


@pytest.mark.parametrize(
    ("loader", "factory", "allowed"),
    [_as_param(case) for case in GOOD_CLASSIFIERS],
)
def test_good_classifier_is_deployable(loader, factory, allowed):
    report, accuracy = _classifier_report(loader, factory())
    ts = report.trust_score
    assert accuracy > 0.9
    assert not ts.is_blocked, (ts.verdict, ts.sub_scores)
    assert ts.grade in allowed, (ts.score, ts.grade, ts.sub_scores, ts.penalties_applied)


def test_confidently_wrong_classifier_is_blocked():
    X_train, X_test, y_train, y_test = _split(load_breast_cancer)
    model = _scaled_logreg().fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, ::-1]  # swap columns: confident and wrong
    y_pred = y_prob.argmax(axis=1)
    ts = analyze(None, None, y_test, y_pred=y_pred, y_prob=y_prob, verbose=False).trust_score
    assert ts.is_blocked
    assert ts.grade == "D"


def test_uninformative_classifier_scores_low():
    rng = np.random.default_rng(0)
    _, _, _, y_test = _split(load_breast_cancer)
    p = np.clip(0.5 + rng.normal(scale=0.02, size=len(y_test)), 0.0, 1.0)
    y_prob = np.column_stack([1.0 - p, p])
    ts = analyze(
        None, None, y_test, y_pred=y_prob.argmax(axis=1), y_prob=y_prob, verbose=False
    ).trust_score
    assert ts.grade == "D", (ts.score, ts.sub_scores)


def test_regression_ridge_on_diabetes_is_moderate():
    X_train, X_test, y_train, y_test = _split(load_diabetes, stratify=False)
    y_pred = Ridge(alpha=1.0).fit(X_train, y_train).predict(X_test)
    ts = analyze(None, None, y_test, y_pred=y_pred, task="regression", verbose=False).trust_score
    assert not ts.is_blocked
    assert ts.grade in {"B", "C"}, (ts.score, ts.sub_scores)


def test_regression_worse_than_mean_is_blocked():
    X_train, X_test, y_train, y_test = _split(load_diabetes, stratify=False)
    y_pred = np.random.default_rng(0).permutation(y_test)  # right scale, no skill
    ts = analyze(None, None, y_test, y_pred=y_pred, task="regression", verbose=False).trust_score
    assert ts.is_blocked
    assert ts.grade == "D"
