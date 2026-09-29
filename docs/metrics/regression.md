# Regression Metrics

Regression metrics evaluate the accuracy and reliability of continuous target predictions, particularly focusing on the quality of the model's uncertainty bounds.

## Why This Matters

For regression tasks, a point prediction is rarely enough for high-stakes decision-making. Teams need to know both *how close* the prediction is on average (Skill) and *how reliable* the uncertainty bounds are (Interval Calibration). Over-confident intervals that routinely miss the target can lead to disastrous downstream decisions.

## When to Use

- when evaluating any regression or continuous-target model
- when validating the reliability of prediction intervals (e.g., confidence intervals or quantiles)
- when you need to understand if a model's uncertainty bounds actually correlate with its errors

## Core Concepts

TrustLens evaluates regression models across three primary dimensions:

### 1. Accuracy and Skill
Measures the overall fit of the model's point predictions against the ground truth.
*   **RMSE & MSE:** Standard distance metrics measuring average error.
*   **Skill Score (R²):** The coefficient of determination against a predict-the-mean baseline (`1 - MSE/Var(y)`). It provides a scale-free measure of how much better the model is than simply guessing the average value.
*   **Heavy-Tail Penalty:** TrustLens automatically analyzes the error distribution (e.g., the ratio of the p90 error to the median error) to identify and penalize models that hide catastrophic misses behind an acceptable average error.

### 2. Interval Calibration
Checks whether the predicted uncertainty intervals are honest.
*   **Single-Level PICP:** For a single prediction interval (e.g., a 90% confidence interval), TrustLens calculates the Prediction Interval Coverage Probability (PICP) and its associated `calibration_error`.
*   **Multi-Level Interval Calibration Error (ICE):** When multi-level intervals are provided, TrustLens calculates ICE, which is the mean absolute coverage gap across all supplied levels (`mean |emp(tau) - tau|`). This is the regression equivalent of Expected Calibration Error (ECE) for classification.

### 3. Uncertainty Informativeness
Evaluates whether the uncertainty bounds are actually useful. A model can be perfectly calibrated by predicting the marginal distribution every time, but this provides no discriminative value.
*   **Calibration-Conditioned Sharpness Proxy:** Evaluates how much tighter the model's intervals are than a naive climatology baseline at each level (width ratio `model_width / climatology_width`). Each level gets a calibration weight: 1 at nominal coverage, falling linearly to 0 at `|emp - tau| = 0.10` (twice the 0.05 tolerance). The reported `sharpness_skill` is `1 - weighted_mean(width ratio)`; the scored `sharpness_evidence` is the best level's `weight × clip(1 - width ratio, 0, 1)`, which never rises when a level is widened or shifted away from nominal coverage. Narrowing a level trades sharpness against calibration (the resolution/reliability trade-off of the CRPS decomposition); because the weight has no free zone, the gain from narrowing into over-confidence is small and peaks near the 0.05 tolerance (at most +4 in a 51-configuration grid; up to about +8 in a random search, mostly on levels that start at or above climatology width or at low nominal levels).
*   **Error-Variance Correlation:** Used when `predicted_variance` is supplied (the Uncertainty Informativeness sub-score takes the larger of this and the sharpness evidence); calculates the Spearman/Pearson correlation between the predicted variance and the realized absolute error.

## API Reference

```{eval-rst}
.. automodule:: trustlens.metrics.regression
   :members:
   :show-inheritance:
```

## Related Pages

- [Features and Modules](../features.md)
- [Trust Score Explained](../trust_score_explained.md)
- [Known Limitations](../known_limitations.md)
