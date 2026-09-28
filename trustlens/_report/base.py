"""Shared state and constants for the TrustReport mixins (TL-18)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

# A sub-score below this is named as a reason in explanations (methodology 2.0).
_WEAK_EXPLAIN = 60.0


class ReportBase:
    """Attributes and cross-mixin methods every TrustReport mixin may rely on."""

    results: dict[str, Any]
    model: Any
    X: Any
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray]
    embeddings: Optional[np.ndarray]
    framework: Optional[str]
    backend_metadata: dict[str, Any]
    task_type: str
    prediction_intervals: Any
    predicted_variance: Optional[np.ndarray]
    metadata: dict[str, Any]
    trust_score: Any
    _patterns: list[str]

    if TYPE_CHECKING:

        @property
        def patterns(self) -> list[str]: ...

        @property
        def deployment_explanation(self) -> dict[str, Any]: ...

        @property
        def deployment_summary(self) -> str: ...

        def _require_classification(self, feature: str) -> None: ...

        def _require_regression(self, feature: str) -> None: ...

        def _generate_insights(self) -> list[str]: ...

        def _generate_conclusion(self) -> str: ...

        def _format_score_explanation(self) -> list[str]: ...

        def _generate_text_report(self, verbose: bool = False) -> str: ...

        def _generate_regression_text(self, verbose: bool = False) -> str: ...

        def _to_serializable(self, obj: Any) -> Any: ...

        def _max_confidence(self) -> np.ndarray: ...

        def summary_plot(self, save_path: Optional[str] = None, show: bool = True) -> Any: ...

        def plot(self, module: Optional[str] = None, save_dir: Optional[str] = None) -> Any: ...
