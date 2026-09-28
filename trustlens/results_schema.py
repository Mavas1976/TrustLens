"""
trustlens.results_schema
========================
Typed contract for ``TrustReport.results`` (TL-17).

``analyze()`` returns results as nested dicts, which keeps them JSON-friendly.
This module documents their shape as ``TypedDict`` classes (usable by type
checkers) and provides :func:`check_results_contract`, which the test suite
runs against real pipeline output so the contract cannot drift silently.

Conventions
-----------
* A module that ran carries its metrics. A module that could not run carries
  ``status`` (``"skipped"`` or ``"degraded"``), ``reason`` and ``details``.
* A metric that could not be computed is ``None`` or absent, never a
  placeholder ``0.0``. Consumers must treat ``None`` as "unknown".
* Keys starting with ``__`` (``__overall__``, ``__summary__``) hold
  aggregates next to per-class or per-group entries.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class SkippedModule(TypedDict, total=False):
    status: str  # "skipped" | "degraded"
    reason: str
    details: str


class CalibrationResult(TypedDict, total=False):
    brier_score: float
    ece: float
    mce: float
    overconfidence_error: float
    n_samples: int
    reliability_curve: Any
    conformal: dict[str, Any]


class ConfidenceGap(TypedDict, total=False):
    gap: float
    correct_confidence_mean: float
    incorrect_confidence_mean: float
    status: str


class FailureResult(TypedDict, total=False):
    misclassification_summary: dict[str, dict[str, Any]]
    confidence_gap: ConfidenceGap
    confidence_auroc: Optional[float]
    n_classes: int
    accuracy: float
    baseline_accuracy: float
    status: str
    reason: str


class BiasResult(TypedDict, total=False):
    class_imbalance: dict[str, Any]
    subgroup_performance: dict[str, dict[str, Any]]
    equalized_odds: dict[str, Any]


class RepresentationResult(TypedDict, total=False):
    separability: dict[str, Any]


class ClassificationResults(TypedDict, total=False):
    calibration: CalibrationResult
    failure: FailureResult
    bias: BiasResult
    representation: RepresentationResult


_NUMERIC_FIELDS: dict[str, tuple[str, ...]] = {
    "calibration": ("brier_score", "ece", "mce", "overconfidence_error"),
    "failure": ("accuracy", "baseline_accuracy"),
}
_UNIT_INTERVAL_FIELDS = {
    ("calibration", "ece"),
    ("calibration", "mce"),
    ("calibration", "overconfidence_error"),
    ("failure", "accuracy"),
    ("failure", "baseline_accuracy"),
    ("failure", "confidence_auroc"),
}
_KNOWN_MODULES = {"calibration", "failure", "bias", "representation", "regression"}


def check_results_contract(results: dict[str, Any]) -> list[str]:
    """Return a list of contract violations in a classification or regression results dict.

    An empty list means the dict follows the conventions above. Plugin results
    (``plugin_*`` keys) are not checked.
    """
    problems: list[str] = []
    for module, data in results.items():
        if module.startswith("plugin_"):
            continue
        if module not in _KNOWN_MODULES:
            problems.append(f"unknown module '{module}'")
            continue
        if not isinstance(data, dict):
            problems.append(f"{module}: expected a dict, got {type(data).__name__}")
            continue
        status = data.get("status")
        if status is not None and status not in ("skipped", "degraded"):
            problems.append(f"{module}: unknown status {status!r}")
        if status == "skipped":
            continue
        for field in _NUMERIC_FIELDS.get(module, ()):
            if field in data and data[field] is not None:
                if not isinstance(data[field], (int, float)):
                    problems.append(f"{module}.{field}: expected a number or None")
        for mod, field in _UNIT_INTERVAL_FIELDS:
            if mod == module and isinstance(data.get(field), (int, float)):
                value = float(data[field])
                if not 0.0 <= value <= 1.0:
                    problems.append(f"{module}.{field}={value} is outside [0, 1]")
    return problems
