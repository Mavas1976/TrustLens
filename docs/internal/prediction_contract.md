# Internal RFC: Prediction Semantics Contract

This document defines the strict contract for how predictions must be normalized and structured within TrustLens backends. All `resolve()` functions must adhere to these rules.

## 1. Classification Semantics

### Binary Classification
TrustLens metrics expect binary probabilities as a 2D array. Rows must sum to
1.0 (enforced at the API boundary).
- **Allowed Input Shapes**: `(n,)`, `(n, 1)`, `(n, 2)`.
- **Normalization Rule**: All binary outputs must be normalized to `(n, 2)` where index 1 represents the positive class.
- **Example**: If a model returns a 1D array of positive probabilities, the backend must expand it to `[1-p, p]`.

### Multiclass Classification
- **Allowed Input Shape**: `(n, C)`.
- **Normalization Rule**: Probabilities must sum to 1.0 per sample. No logits allowed in the `PredictionBundle`.

---

## 2. Regression Semantics
Regression is supported through `analyze(..., task="regression")` (or
auto-detection) and bypasses the classification backends: point predictions
come from `y_pred` or `model.predict(X)`.
- **Contract**: classification backends still raise `NotImplementedError` when
  asked to resolve a regressor with `task="classification"`.

## 2a. Input Validation
`trustlens.core.inputs.prepare_inputs` validates every user input once before
any backend runs: one row per sample for every argument, finite probability
rows that sum to 1 (tolerance 1e-3), 1-D binary probabilities expanded to
`(n, 2)`, and a warning when pandas inputs carry different indexes.

## 2b. Results Contract
The shape of `TrustReport.results` is documented as `TypedDict` classes in
`trustlens/results_schema.py`. A metric that could not be computed is `None`
or absent, never a placeholder `0.0`. A module that could not run carries
`status` (`skipped` / `degraded`), `reason` and `details`.
`check_results_contract()` is run against real pipeline output in the test
suite.

---

### Logits Policy
Raw logits are strictly forbidden inside the `PredictionBundle`.

Backends are responsible for converting logits to calibrated probabilities (e.g., via Softmax for multiclass or Sigmoid for binary tasks) before constructing the bundle. This ensures that downstream trust metrics, calibration plots, and uncertainty estimations receive standardized probability scores.

---

## 3. Label Consistency
- **Class Labels**: The `class_labels` field in `PredictionBundle` should be populated whenever possible to ensure that metric reports use the correct semantic labels (e.g., "Malignant" instead of `1`).

---

## 4. Metadata Hooks
Every `PredictionBundle` should include detection metadata for debugging:
```python
metadata = {
    "resolver": "sklearn",
    "detection_method": "module_name", # or "explicit", "capability"
    "framework_version": "1.3.0"
}
```

---

## 5. Manual Overrides (`model=None`)
TrustLens supports auditing prediction data without a direct model reference.
- **Contract**: If `model=None` is passed to `analyze()`, the `manual` resolver is used.
- **Requirement**: Either `y_pred` or `y_prob` must be provided.
- **Derivation**: If `y_prob` is provided but `y_pred` is missing, `y_pred` will be derived via `argmax`.

---

## 6. Degraded Mode Transparency
When critical components (like `y_prob`) are missing, TrustLens enters a **Degraded Mode**.
- **Audit Trail**: Metadata will include `"degraded_mode": True` and a list of `"missing_components"`.
- **Module Behavior**: Analysis modules must skip confidence-based metrics (e.g., Calibration, ECE) and report a `"skipped"` status instead of crashing.
