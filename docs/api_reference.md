# API Reference

This page provides the technical documentation for all public components of TrustLens.

```{eval-rst}
.. toctree::
   :maxdepth: 2

   metrics/bias
   metrics/calibration
   metrics/failure
   metrics/representation
   metrics/regression
   metrics/conformal
```

---

## Core API

### `trustlens.analyze`
```{eval-rst}
.. autofunction:: trustlens.api.analyze
```

### `trustlens.quick_analyze`
```{eval-rst}
.. autofunction:: trustlens.api.quick_analyze
```

### `trustlens.compare`
```{eval-rst}
.. autofunction:: trustlens.comparison.compare
```

### `trustlens.compute_trust_score`
```{eval-rst}
.. autofunction:: trustlens.trust_score.compute_trust_score
```

### `trustlens.regression_trust_score`
```{eval-rst}
.. autofunction:: trustlens.trust_score.regression_trust_score
```

### Results contract
```{eval-rst}
.. automodule:: trustlens.results_schema
   :members: check_results_contract
```

### `trustlens.TrustReport`
```{eval-rst}
.. autoclass:: trustlens.report.TrustReport
   :members:
   :inherited-members:
```

### `trustlens.TrustScoreResult`
```{eval-rst}
.. autoclass:: trustlens.trust_score.TrustScoreResult
   :members:
   :show-inheritance:
```

---

## Visualization

### `trustlens.visualization.plot_module`
```{eval-rst}
.. autofunction:: trustlens.visualization.plot_module
```

### `trustlens.visualization.fairness`
```{eval-rst}
.. automodule:: trustlens.visualization.fairness
   :members:
   :show-inheritance:
```
