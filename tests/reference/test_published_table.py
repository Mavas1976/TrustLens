"""
The before/after table in docs/trust_score_explained.md is recomputed here.

Each row of the "current" column is parsed from the published page and
compared with a fresh computation, so the documentation cannot drift from the
code (GA-10). The recipes match the ones used to produce the table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trustlens import analyze

DOC = Path(__file__).resolve().parents[2] / "docs" / "trust_score_explained.md"
LOADERS = {"breast_cancer": load_breast_cancer, "iris": load_iris, "wine": load_wine}
MODELS = {
    "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "RandomForest": lambda: RandomForestClassifier(200, random_state=0),
    "GaussianNB": GaussianNB,
}


def _published_rows() -> dict[str, str]:
    text = DOC.read_text(encoding="utf-8")
    section = text.split("## Changes from methodology 1.x", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and " · " in cells[0]:
            rows[cells[0]] = cells[2]
    return rows


def _split(loader, stratify=True):
    X, y = loader(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if stratify else None)


def _format(ts) -> str:
    if ts.grade == "N/A":
        return "N/A"
    return f"{ts.score}/{ts.grade}" + (" blocked" if ts.is_blocked else "")


def _compute(name: str) -> str:
    dataset, variant = name.split(" · ", 1)
    if dataset == "diabetes":
        X_train, X_test, y_train, y_test = _split(load_diabetes, stratify=False)
        y_pred = Ridge().fit(X_train, y_train).predict(X_test)
        return _format(
            analyze(None, None, y_test, y_pred=y_pred, task="regression", verbose=False).trust_score
        )
    X_train, X_test, y_train, y_test = _split(LOADERS[dataset])
    if variant in MODELS:
        model = MODELS[variant]().fit(X_train, y_train)
        return _format(analyze(model, X_test, y_test, verbose=False).trust_score)
    model = MODELS["LogReg"]().fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    if variant == "LogReg, y_pred only":
        ts = analyze(None, None, y_test, y_pred=model.predict(X_test), verbose=False).trust_score
    elif variant.startswith("swapped probabilities"):
        swapped = probs[:, ::-1]
        ts = analyze(
            None, None, y_test, y_pred=swapped.argmax(axis=1), y_prob=swapped, verbose=False
        ).trust_score
    elif variant == "coin-flip model":
        rng = np.random.default_rng(0)
        p = np.clip(0.5 + rng.normal(scale=0.02, size=len(y_test)), 0, 1)
        ts = analyze(
            None,
            None,
            y_test,
            y_pred=(p >= 0.5).astype(int),
            y_prob=np.column_stack([1 - p, p]),
            verbose=False,
        ).trust_score
    else:
        raise KeyError(name)
    return _format(ts)


PUBLISHED = _published_rows()


def test_table_is_present_and_complete():
    assert len(PUBLISHED) == 13, sorted(PUBLISHED)


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_published_value_matches_code(name):
    assert _compute(name) == PUBLISHED[name]
