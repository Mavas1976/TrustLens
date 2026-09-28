"""
trustlens.core.inputs
=====================
Single validation layer for everything a user passes to ``analyze()`` (TL-12).

Responsibilities
----------------
* Convert array-likes (lists, pandas Series/DataFrames) to numpy.
* Reject inputs whose lengths disagree, naming the offending argument.
* Reject probability rows that do not sum to 1 (the prediction contract).
* Warn when pandas inputs carry different indexes, because TrustLens aligns
  by position, not by index.
* Normalise sensitive features: accept a DataFrame, and map missing values
  to an explicit ``"<missing>"`` group instead of crashing deep in sklearn.

``X`` is only length-checked and passed through unchanged, so models that rely
on column names keep working.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from trustlens.utils import validate_array

logger = logging.getLogger(__name__)

MISSING_GROUP = "<missing>"
_ROW_SUM_TOLERANCE = 1e-3


@dataclass
class PreparedInputs:
    """Validated user inputs, ready for the backends and the pipeline."""

    y_true: np.ndarray
    y_pred: Optional[np.ndarray]
    y_prob: Optional[np.ndarray]
    sensitive_features: Optional[dict[str, np.ndarray]]
    embeddings: Optional[np.ndarray]


def _as_1d(values: Any, name: str) -> np.ndarray:
    arr = validate_array(values, name)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}.")
    return arr


def _require_finite(arr: np.ndarray, name: str) -> None:
    """Targets and predictions must not contain NaN/Inf or missing labels (GA-07, NF3-07)."""
    if arr.dtype.kind == "O":
        missing = int(sum(_is_missing(v) for v in arr))
        if missing:
            raise ValueError(
                f"{name} contains {missing} missing value(s) (None, NaN or NA); remove or "
                "impute them first."
            )
    if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
        bad = int(np.sum(~np.isfinite(arr)))
        raise ValueError(
            f"{name} contains {bad} non-finite value(s) (NaN or Inf); remove or impute them first."
        )


def _pandas_index(values: Any) -> Optional[Any]:
    index = getattr(values, "index", None)
    return index if index is not None and hasattr(values, "iloc") else None


def _warn_on_index_mismatch(named_values: dict[str, Any]) -> None:
    indexes: dict[str, Any] = {}
    for name, values in named_values.items():
        idx = _pandas_index(values)
        if idx is not None:
            indexes[name] = idx
    if len(indexes) < 2:
        return
    names = list(indexes)
    reference = indexes[names[0]]
    differing = [
        name
        for name in names[1:]
        if len(indexes[name]) != len(reference) or not indexes[name].equals(reference)
    ]
    if differing:
        logger.warning(
            "pandas inputs %s have a different index than %s. TrustLens aligns rows by "
            "position, not by index; reset or align the indexes if rows do not correspond.",
            differing,
            names[0],
        )


def _prepare_probabilities(y_prob: Any) -> np.ndarray:
    prob = np.asarray(y_prob, dtype=float)
    if prob.ndim == 2 and prob.shape[1] == 1:
        prob = prob.ravel()
    if prob.ndim == 1:
        # Binary positive-class probabilities: expand to the (n, 2) contract.
        prob = np.column_stack([1.0 - prob, prob])
    if prob.ndim != 2 or prob.shape[1] < 2:
        raise ValueError(f"'y_prob' must have shape (n_samples, n_classes); got {prob.shape}.")
    if not np.all(np.isfinite(prob)):
        raise ValueError("'y_prob' contains non-finite values (NaN or Inf).")
    row_sums = prob.sum(axis=1)
    bad = np.abs(row_sums - 1.0) > _ROW_SUM_TOLERANCE
    if bad.any():
        raise ValueError(
            f"'y_prob' rows must sum to 1 (tolerance {_ROW_SUM_TOLERANCE}); {int(bad.sum())} of "
            f"{len(prob)} rows do not (e.g. row {int(np.argmax(bad))} sums to "
            f"{row_sums[np.argmax(bad)]:.4f}). Pass normalised class probabilities, not scores."
        )
    return prob


def _is_missing(value: Any) -> bool:
    """None, NaN, pandas ``NA`` and ``NaT`` count as missing (GB-04, NF-02)."""
    if value is None or type(value).__name__ in ("NAType", "NaTType"):
        return True
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def _prepare_sensitive_features(features: Any) -> Optional[dict[str, np.ndarray]]:
    if features is None:
        return None
    if hasattr(features, "columns") and hasattr(features, "iloc"):  # pandas DataFrame
        features = {str(col): features[col] for col in features.columns}
    if not isinstance(features, dict):
        raise TypeError(
            "'sensitive_features' must be a dict of {name: 1-D array} or a pandas DataFrame."
        )
    prepared: dict[str, np.ndarray] = {}
    for name, values in features.items():
        arr = _as_1d(values, f"sensitive_features['{name}']").astype(object)
        missing = np.array([_is_missing(v) for v in arr], dtype=bool)
        if missing.any():
            logger.warning(
                "sensitive_features['%s'] has %d missing value(s); they form the group '%s'.",
                name,
                int(missing.sum()),
                MISSING_GROUP,
            )
            # Mixed str/float values cannot be sorted by np.unique, so all
            # group values become strings once a missing group exists (GB-04).
            arr = np.array([str(v) for v in arr], dtype=object)
            arr[missing] = MISSING_GROUP
        prepared[str(name)] = arr
    return prepared


def check_lengths(n_samples: int, **named: Any) -> None:
    """Raise a ValueError naming every argument whose length differs from ``n_samples``."""
    wrong = {
        name: len(value)
        for name, value in named.items()
        if value is not None and hasattr(value, "__len__") and len(value) != n_samples
    }
    if wrong:
        details = ", ".join(f"{name} has {length}" for name, length in wrong.items())
        raise ValueError(
            f"All inputs must have one row per sample: y_true has {n_samples}, but {details}."
        )


def prepare_inputs(
    y_true: Any,
    X: Any = None,
    y_pred: Any = None,
    y_prob: Any = None,
    sensitive_features: Any = None,
    embeddings: Any = None,
    task: str = "classification",
) -> PreparedInputs:
    """Validate and convert the user-facing inputs of ``analyze()``.

    Raises
    ------
    ValueError
        On empty or wrongly shaped arrays, disagreeing lengths, non-finite
        probabilities, or probability rows that do not sum to 1.
    TypeError
        When ``sensitive_features`` is neither a dict nor a DataFrame.
    """
    named: dict[str, Any] = {"y_true": y_true, "X": X, "y_pred": y_pred, "y_prob": y_prob}
    if isinstance(sensitive_features, dict):
        named.update({f"sensitive_features['{k}']": v for k, v in sensitive_features.items()})
    elif sensitive_features is not None:
        named["sensitive_features"] = sensitive_features
    _warn_on_index_mismatch({k: v for k, v in named.items() if v is not None})

    y_true_arr = _as_1d(y_true, "y_true")
    _require_finite(y_true_arr, "y_true")
    y_pred_arr = _as_1d(y_pred, "y_pred") if y_pred is not None else None
    if y_pred_arr is not None:
        _require_finite(y_pred_arr, "y_pred")
    y_prob_arr = None
    if y_prob is not None:
        if task == "regression":
            raise ValueError("'y_prob' is a classification input; regression uses y_pred.")
        y_prob_arr = _prepare_probabilities(y_prob)
    features = _prepare_sensitive_features(sensitive_features)
    embeddings_arr = validate_array(embeddings, "embeddings") if embeddings is not None else None

    check_lengths(
        len(y_true_arr),
        X=X,
        y_pred=y_pred_arr,
        y_prob=y_prob_arr,
        embeddings=embeddings_arr,
        **{f"sensitive_features['{k}']": v for k, v in (features or {}).items()},
    )
    return PreparedInputs(
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        y_prob=y_prob_arr,
        sensitive_features=features,
        embeddings=embeddings_arr,
    )
