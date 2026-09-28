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


def test_top_label_metrics_use_argmax_not_thresholded_predictions(caplog):
    """GA-03 / TL-14: ECE judges argmax(y_prob), and a differing y_pred is reported."""
    from trustlens.metrics.calibration import expected_calibration_error

    y, _, y_prob = _binary()
    argmax = y_prob.argmax(axis=1)
    y_pred = np.where(np.arange(N) % 3 == 0, 1 - argmax, argmax)  # a third disagree
    with caplog.at_level(logging.WARNING, logger="trustlens.core.pipeline"):
        report = analyze(None, None, y, y_pred=y_pred, y_prob=y_prob, verbose=False)
    assert "differs from argmax" in caplog.text
    assert report.results["failure"]["confidence_auroc"] is not None
    k = rng.integers(0, 3, N)
    multi = rng.dirichlet(np.ones(3), N)
    top = multi.argmax(axis=1)
    multi_report = analyze(
        None,
        None,
        k,
        y_pred=np.where(np.arange(N) % 2 == 0, (top + 1) % 3, top),
        y_prob=multi,
        verbose=False,
    )
    multi_expected = expected_calibration_error((top == k).astype(float), multi.max(axis=1))
    assert multi_report.results["calibration"]["ece"] == pytest.approx(multi_expected)


def test_swapped_probability_columns_are_reported(caplog):
    """GA-03: columns in the wrong order make y_pred disagree with argmax."""
    y, y_pred, y_prob = _binary()
    with caplog.at_level(logging.WARNING, logger="trustlens.core.pipeline"):
        analyze(None, None, y, y_pred=y_pred, y_prob=y_prob[:, ::-1], verbose=False)
    assert "differs from argmax" in caplog.text


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_non_finite_targets_are_rejected(task):
    """GA-07: NaN targets raise instead of producing NaN scores and invalid JSON."""
    y = rng.normal(size=N) if task == "regression" else rng.integers(0, 2, N).astype(float)
    y[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        analyze(None, None, y, y_pred=np.zeros(N), task=task, verbose=False)


def test_numeric_sensitive_feature_with_nan_forms_missing_group():
    """GB-04: NaN in a float feature must not crash on mixed str/float sorting."""
    y, y_pred, y_prob = _binary()
    feature = rng.normal(size=N).round(0)
    feature[:10] = np.nan
    report = analyze(
        None,
        None,
        y,
        y_pred=y_pred,
        y_prob=y_prob,
        sensitive_features={"f": feature},
        verbose=False,
    )
    assert "<missing>" in report.results["bias"]["subgroup_performance"]["f"]


def test_equalized_odds_runs_for_any_two_labels():
    """GA-05: binary labels other than 0/1 still get TPR/FPR gaps."""
    y, y_pred, y_prob = _binary()
    names = np.array(["neg", "pos"])
    groups = rng.choice(["a", "b"], N)
    report = analyze(
        None,
        None,
        names[y],
        y_pred=names[y_pred],
        y_prob=y_prob,
        sensitive_features={"g": groups},
        verbose=False,
    )
    eo = report.results["bias"]["equalized_odds"]
    assert "g" in eo and eo["g"]["__summary__"]["tpr_gap"] is not None


def test_ragged_semantic_prediction_sets_are_encoded():
    """GA-08: sets written in class labels map onto probability columns."""
    k = rng.integers(0, 3, N)
    labels = np.array(["x", "y", "z"])
    y_prob = rng.dirichlet(np.ones(3), N)
    sets = [[labels[i]] for i in k]  # each set holds exactly the true label
    report = analyze(
        None,
        None,
        labels[k],
        y_pred=labels[y_prob.argmax(axis=1)],
        y_prob=y_prob,
        class_labels=labels,
        y_pred_sets=sets,
        verbose=False,
    )
    assert report.results["calibration"]["conformal"]["marginal_coverage"] == pytest.approx(1.0)


def test_modules_accepts_a_single_name():
    """GA-11: a string is one module, not an iterable of characters."""
    y, y_pred, y_prob = _binary()
    report = analyze(
        None, None, y, y_pred=y_pred, y_prob=y_prob, modules="calibration", verbose=False
    )
    assert "calibration" in report.results and "failure" not in report.results


def test_tiny_sample_cannot_pass():
    """GA-11: a handful of samples is not enough evidence for a passing grade."""
    y = np.array([0, 1] * 5)
    p = np.where(y == 1, 0.95, 0.05)
    ts = analyze(
        None, None, y, y_pred=y, y_prob=np.column_stack([1 - p, p]), verbose=False
    ).trust_score
    assert ts.grade == "C"
    assert any("samples" in c for c in ts.caps_applied)


def test_equalized_odds_crash_caps_the_grade(monkeypatch):
    """GB-06: a failed fairness computation must not make fairness look better."""
    import trustlens.core.pipeline as pipeline

    def boom(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(pipeline, "equalized_odds", boom)
    # Own generator and a sharp, calibrated model: the uncapped score must
    # clearly exceed 59 whatever ran before this test.
    local = np.random.default_rng(11)
    p = local.choice([0.03, 0.97], N)
    y = (local.random(N) < p).astype(int)
    ts = analyze(
        None,
        None,
        y,
        y_pred=(p >= 0.5).astype(int),
        y_prob=np.column_stack([1 - p, p]),
        sensitive_features={"g": local.choice(["a", "b"], N)},
        verbose=False,
    ).trust_score
    assert ts.base_score > 59  # the cap is what brings the grade down
    assert ts.grade == "C" and ts.score == 59
    assert any("equalized odds failed" in c for c in ts.caps_applied)
    # NF-06: fairness is not assessed (no score from the subgroup gap alone),
    # so the report is partial and compare() will not recommend it.
    assert "bias" not in ts.sub_scores
    assert ts.is_partial and "bias (equalized odds failed)" in ts.missing_dimensions


def test_quick_analyze_refuses_model_without_data():
    """GB-17: the caller's model is never swapped for a demo model."""
    from sklearn.linear_model import LogisticRegression

    from trustlens import quick_analyze

    with pytest.raises(ValueError, match="pass X and y"):
        quick_analyze(LogisticRegression())


@pytest.mark.parametrize("dtype", ["string", "object"])
def test_pandas_na_in_sensitive_feature_forms_missing_group(dtype):
    """NF-02: pd.NA in a string/object column is a missing value, not a crash."""
    y, y_pred, y_prob = _binary()
    feature = pd.Series(rng.choice(["a", "b"], N), dtype=dtype)
    feature.iloc[:10] = pd.NA
    report = analyze(
        None,
        None,
        y,
        y_pred=y_pred,
        y_prob=y_prob,
        sensitive_features={"f": feature},
        verbose=False,
    )
    assert "<missing>" in report.results["bias"]["subgroup_performance"]["f"]


def test_low_support_groups_are_named_in_a_warning(caplog):
    """GA-04: excluded groups are logged by name, not silently dropped."""
    y, y_pred, y_prob = _binary()
    groups = np.array(["big"] * (N - 10) + ["tinygroupX"] * 10, dtype=object)
    with caplog.at_level(logging.WARNING, logger="trustlens.core.pipeline"):
        analyze(
            None,
            None,
            y,
            y_pred=y_pred,
            y_prob=y_prob,
            sensitive_features={"g": groups},
            verbose=False,
        )
    assert "tinygroupX" in caplog.text and "excluded from fairness gaps" in caplog.text


def test_quick_analyze_refuses_data_without_model():
    """NF-08: the caller's data is never swapped for the demo dataset."""
    from trustlens import quick_analyze

    X = rng.normal(size=(20, 3))
    with pytest.raises(ValueError, match="fitted model"):
        quick_analyze(None, X, np.zeros(20, dtype=int))


def test_missing_labels_in_object_targets_raise_clearly():
    """NF3-07: None in an object y_true is a validation error, not a sort crash."""
    y = np.array(["a", "b"] * (N // 2), dtype=object)
    y[3] = None
    with pytest.raises(ValueError, match="missing value"):
        analyze(None, None, y, y_pred=np.array(["a"] * N, dtype=object), verbose=False)


def test_single_class_target_warns(caplog):
    """NF3-05: the no-skill check cannot run on one class; say so."""
    y = np.ones(N, dtype=int)
    p = rng.uniform(0.6, 0.99, N)
    with caplog.at_level(logging.WARNING, logger="trustlens.core.pipeline"):
        analyze(
            None,
            None,
            y,
            y_pred=np.ones(N, dtype=int),
            y_prob=np.column_stack([1 - p, p]),
            class_labels=np.array([0, 1]),
            verbose=False,
        )
    assert "single class" in caplog.text


def test_fairness_without_comparable_groups_is_partial():
    """NF3-06: requested but unassessable fairness makes the report partial."""
    y, y_pred, y_prob = _binary()
    groups = np.array(["big"] * (N - 10) + ["small"] * 10, dtype=object)
    ts = analyze(
        None,
        None,
        y,
        y_pred=y_pred,
        y_prob=y_prob,
        sensitive_features={"g": groups},
        verbose=False,
    ).trust_score
    assert "bias" not in ts.sub_scores
    assert ts.is_partial and ts.score <= 59
    assert any("no two groups" in d for d in ts.missing_dimensions)
