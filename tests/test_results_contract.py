"""The results dict produced by analyze() follows trustlens.results_schema (TL-17)."""

import numpy as np
import pytest

from trustlens import analyze
from trustlens.results_schema import check_results_contract

rng = np.random.default_rng(3)
N = 400


def _binary():
    p = rng.uniform(0.02, 0.98, N)
    y = (rng.random(N) < p).astype(int)
    return y, (p >= 0.5).astype(int), np.column_stack([1 - p, p])


@pytest.mark.parametrize(
    "case",
    ["full", "no_probabilities", "subset", "multiclass", "regression", "embeddings"],
)
def test_pipeline_output_follows_contract(case):
    y, y_pred, y_prob = _binary()
    groups = rng.choice(["a", "b"], N)
    if case == "full":
        report = analyze(
            None,
            None,
            y,
            y_pred=y_pred,
            y_prob=y_prob,
            sensitive_features={"g": groups},
            verbose=False,
        )
    elif case == "no_probabilities":
        report = analyze(None, None, y, y_pred=y_pred, verbose=False)
    elif case == "subset":
        report = analyze(
            None, None, y, y_pred=y_pred, y_prob=y_prob, modules=["failure"], verbose=False
        )
    elif case == "multiclass":
        prob = rng.dirichlet(np.ones(4), N)
        labels = prob.argmax(axis=1)
        report = analyze(None, None, labels, y_pred=labels, y_prob=prob, verbose=False)
    elif case == "regression":
        target = rng.normal(size=N)
        report = analyze(
            None,
            None,
            target,
            y_pred=target + rng.normal(scale=0.3, size=N),
            task="regression",
            verbose=False,
        )
    else:
        report = analyze(
            None,
            None,
            y,
            y_pred=y_pred,
            y_prob=y_prob,
            embeddings=rng.normal(size=(N, 5)),
            verbose=False,
        )
    assert check_results_contract(report.results) == []


def test_contract_checker_flags_violations():
    bad = {
        "calibration": {"ece": 1.7},
        "failure": {"accuracy": "high"},
        "bias": {"status": "broken"},
        "surprise": {},
    }
    problems = check_results_contract(bad)
    assert any("calibration.ece" in p for p in problems)
    assert any("failure.accuracy" in p for p in problems)
    assert any("unknown status" in p for p in problems)
    assert any("unknown module" in p for p in problems)
