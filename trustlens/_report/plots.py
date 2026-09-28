"""Plotting entry points of a TrustReport.

Part of the TrustReport split (TL-18); behaviour is unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

from trustlens._report.base import ReportBase

logger = logging.getLogger("trustlens.report")


class PlotMixin(ReportBase):
    """Plotting entry points of a TrustReport."""

    def plot_residuals(
        self,
        *,
        title: str = "Residuals vs. Predicted",
        save_path: str | None = None,
        show: bool = True,
    ) -> Any:
        """Residuals (``y_true - y_pred``) vs. predicted value, for spotting
        heteroscedasticity and bias. Regression reports only.

        If prediction intervals were supplied to :func:`analyze`, they are overlaid
        as a band in residual space. See
        :func:`trustlens.visualization.regression_plots.plot_residuals`.
        """
        self._require_regression("plot_residuals")
        from trustlens.visualization.regression_plots import plot_residuals

        return plot_residuals(
            self.y_true,
            self.y_pred,
            prediction_intervals=self.prediction_intervals,
            title=title,
            save_path=save_path,
            show=show,
        )

    def plot_error_distribution(
        self,
        *,
        bins: int = 30,
        title: str = "Error Distribution",
        save_path: str | None = None,
        show: bool = True,
    ) -> Any:
        """Histogram of signed errors (``y_true - y_pred``) against a fitted normal,
        for spotting skew and heavy tails. Regression reports only. See
        :func:`trustlens.visualization.regression_plots.plot_error_distribution`.
        """
        self._require_regression("plot_error_distribution")
        from trustlens.visualization.regression_plots import plot_error_distribution

        return plot_error_distribution(
            self.y_true,
            self.y_pred,
            bins=bins,
            title=title,
            save_path=save_path,
            show=show,
        )

    def _max_confidence(self) -> np.ndarray:
        """Return per-sample max predicted confidence."""
        if self.y_prob is None:
            return np.zeros(len(self.y_true))
        yp = np.asarray(self.y_prob)
        return cast(np.ndarray, yp.max(axis=1) if yp.ndim == 2 else yp)

    def summary_plot(
        self,
        save_path: str | None = None,
        show: bool = True,
    ) -> Any:
        """
        Render the TrustLens Summary Dashboard — a single-figure overview
        of the model's trustworthiness.

        Layout (2×3 grid):

          Trust Score Gauge Reliability Diag  Confidence Gap

          Error Rate Dist.  Class Dist.    Sub-score Bars


        Parameters
        ----------
        save_path : str, optional
          If provided, saves the figure to this path (PNG or PDF).
        show : bool
          If True, calls ``plt.show()`` for interactive display.
          Default True. Set to False in non-interactive environments.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._require_classification("summary_plot()")
        from trustlens.visualization.summary_plot import plot_summary_dashboard

        fig = plot_summary_dashboard(
            trust_score=self.trust_score,
            results=self.results,
            y_true=self.y_true,
            y_pred=self.y_pred,
            y_prob=self.y_prob,
            model_name=self.metadata["model_class"],
            save_path=save_path,
        )
        if show:
            try:
                import matplotlib.pyplot as plt

                if "agg" not in plt.get_backend().lower():
                    plt.show()
            except Exception:
                pass

        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass

        return fig

    def show_failures(
        self,
        top_k: int = 10,
        images: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        save_path: str | None = None,
    ) -> None:
        """
        Display the most alarming model failures — high-confidence wrong
        predictions that deserve immediate attention.

        For each failure, reports:
        * Predicted class and confidence level
        * True class
        * A "danger rating" based on confidence level
        * Feature values (if ``feature_names`` provided)

        Parameters
        ----------
        top_k : int
          Number of top failures to show. Default 10.
        images : np.ndarray, optional
          Image array shape (n_samples, H, W, C) or (n_samples, H, W).
          If provided, renders a grid of the most-confident wrong predictions.
        feature_names : list[str], optional
          Column names for tabular features in ``self.X``.
        save_path : str, optional
          If provided and ``images`` is given, saves the failure grid as PNG.

        Examples
        --------
        >>> report.show_failures(top_k=10)  # doctest: +SKIP
        >>> report.show_failures(top_k=5, images=X_images)  # doctest: +SKIP
        """
        self._require_classification("show_failures()")
        max_conf = self._max_confidence()
        y_true = np.asarray(self.y_true)
        y_pred = np.asarray(self.y_pred)

        # Identify wrong predictions
        wrong_mask = y_true != y_pred
        if not wrong_mask.any():
            print(" No misclassifications found — perfect predictions!")
            return

        wrong_indices = np.where(wrong_mask)[0]
        wrong_confidence = max_conf[wrong_mask]

        # Sort by confidence descending (worst offenders first)
        sorted_order = np.argsort(wrong_confidence)[::-1]
        top_indices = wrong_indices[sorted_order[:top_k]]

        print("\nCRITICAL FAILURES")
        print(
            f"{self.metadata['model_class']} | "
            f"{wrong_mask.sum()} total errors / "
            f"{len(y_true)} samples ({100 * wrong_mask.mean():.1f}%)"
        )
        print(f"\n{'#':<4} {'Sample':<8} {'True':>6} {'Pred':>6} {'Confidence':>12} {'Danger':>8}")

        for rank, idx in enumerate(top_indices, start=1):
            conf = float(max_conf[idx])
            true_cls = int(y_true[idx])
            pred_cls = int(y_pred[idx])
            danger = _danger_rating(conf)
            print(f"{rank:<4} {idx:<8} {true_cls:>6} {pred_cls:>6} {conf:>11.1%} {danger:>8}")

            # Show top features if names provided
            if feature_names is not None:
                feats = np.asarray(self.X)[idx]
                top_feat_idx = np.argsort(np.abs(feats))[::-1][:3]
                feat_strs = [
                    f"{feature_names[i]}={feats[i]:.3g}"
                    for i in top_feat_idx
                    if i < len(feature_names)
                ]
                if feat_strs:
                    print(f"    Top features: {', '.join(feat_strs)}")

        # Summary insight
        top_conf = max_conf[top_indices]
        print("\n Insights:")
        print(f"   Mean confidence on top failures: {top_conf.mean():.1%}")
        print("   These are high-confidence mistakes - the model is")
        print("   certain it is right, but it is wrong.")
        if top_conf.mean() > 0.85:
            print("   Overconfidence detected - consider calibration.")
        print()

        # Optional: image grid
        if images is not None:
            fig = _plot_failure_grid(
                images=images,
                indices=top_indices,
                y_true=y_true,
                y_pred=y_pred,
                confidences=max_conf,
                save_path=save_path,
            )
            _ = fig

    def plot(
        self,
        module: str | None = None,
        save_dir: str | None = None,
    ) -> None:
        """
        Render per-module visualisations.

        Parameters
        ----------
        module : str, optional
          Which module to plot (e.g., ``"calibration"``).
          If None, all available modules are plotted.
        save_dir : str, optional
          Directory path where figures are saved as PNG files.
        """
        self._require_classification("plot()")
        from trustlens.visualization import plot_module

        modules_to_plot = [module] if module else list(self.results.keys())
        for m in modules_to_plot:
            if m in self.results:
                plot_module(
                    module_name=m,
                    data=self.results[m],
                    save_dir=save_dir,
                    embeddings=self.embeddings if m == "representation" else None,
                    labels=self.y_true if m == "representation" else None,
                )
            else:
                logger.warning("Module '%s' not found in results.", m)

    def plot_embedding_2d(
        self,
        method: str = "umap",
        n_max: int = 5000,
        save_path: str | None = None,
        show: bool = True,
    ) -> Any:
        """
        Project stored embeddings to 2D and render a class-colored scatter plot.

        Delegates to the underlying ``plot_embedding_2d`` in the visualization
        sub-package, forwarding silhouette score (when available) so the plot
        is annotated automatically.

        Guarantees:
        - Returns matplotlib.figure.Figure
        - Raises ValueError when embeddings were not supplied to ``analyze()``
        - Projection falls back gracefully (UMAP -> t-SNE -> PCA) when optional libraries are missing

        Parameters
        ----------
        method : str
          Projection algorithm: ``"umap"`` (default), ``"tsne"``, or ``"pca"``.
        n_max : int
          Max samples plotted.  Subsampled randomly when the embedding matrix
          exceeds this limit.
        save_path : str, optional
          File path to save the figure (e.g. ``"clusters.png"``).
        show : bool
          Whether to display the figure interactively.  Default True.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If embeddings are not available (i.e. not passed to ``analyze()``).
        """
        from trustlens.visualization.representation_plots import (
            plot_embedding_2d as _plot,
        )

        if self.embeddings is None:
            raise ValueError(
                "No embeddings available. "
                "Pass 'embeddings' to analyze() to enable 2D embedding visualization."
            )

        sil = self.results.get("representation", {}).get("separability", {}).get("silhouette_score")

        return _plot(
            embeddings=self.embeddings,
            labels=self.y_true,
            silhouette_score=sil,
            method=method,
            n_max=n_max,
            save_path=save_path,
            show=show,
        )


def _danger_rating(confidence: float) -> str:
    """Map confidence level to a danger string."""
    if confidence >= 0.95:
        return "CRITICAL"
    if confidence >= 0.85:
        return "HIGH"
    if confidence >= 0.70:
        return "MEDIUM"
    return "LOW"


def _plot_failure_grid(
    images: np.ndarray,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    save_path: str | None = None,
) -> Any:
    """Render a grid of failure images with prediction annotations."""
    import matplotlib.pyplot as plt

    n = len(indices)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax, idx in zip(axes, indices):
        img = images[idx]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)

        conf = confidences[idx]
        color = "#FF3B30" if conf >= 0.85 else "#FF9F0A"
        ax.set_title(
            f"True: {y_true[idx]} Pred: {y_pred[idx]}\nConf: {conf:.1%}",
            fontsize=9,
            color=color,
            fontweight="bold",
        )
        ax.axis("off")

    for ax in axes[len(indices) :]:
        ax.set_visible(False)

    fig.suptitle(
        "High-Confidence Failures",
        fontsize=13,
        fontweight="bold",
        color="#FF3B30",
    )
    plt.tight_layout()

    if save_path:
        from trustlens.visualization.style import save_figure

        save_figure(fig, save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig
