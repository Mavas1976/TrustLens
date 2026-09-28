"""Fairness plotting entry point of a TrustReport (plot_bias).

Part of the TrustReport split (TL-18, GB-08); behaviour is unchanged.
"""

from __future__ import annotations

import logging
import os

from trustlens._report.base import ReportBase

logger = logging.getLogger("trustlens.report")


class BiasPlotMixin(ReportBase):
    """Fairness plotting entry point of a TrustReport (plot_bias)."""

    def plot_bias(
        self,
        mode: str = "summary",
        show: bool = True,
        save_path: str | None = None,
        multi_feature: bool = False,
    ):
        """
        Generate fairness/bias visualizations from report results.

        ``mode`` and ``multi_feature`` are independent dimensions; the return
        shape is fully determined by their combination, with no flattening.

        **Guarantees**

        - Return shape is fixed by the ``(mode, multi_feature)`` combination
          (see the table below). The structure never collapses across calls.
        - Multi-feature outputs never contain ``None`` values; missing
          components are represented by empty dicts ``{}``.
        - Raises ``ValueError`` for invalid ``mode`` or unusable data, regardless
          of ``multi_feature``.

        Return shape by ``(mode, multi_feature)``:

        +-----------------+----------------+--------------------------------------+
        | mode            | multi_feature  | Return type                          |
        +=================+================+======================================+
        | single mode     | False          | ``Figure``                           |
        | (summary,       |                |                                      |
        | subgroup,       |                |                                      |
        | equalized_odds, |                |                                      |
        | gap)            |                |                                      |
        +-----------------+----------------+--------------------------------------+
        | summary         | True           | ``Figure`` (same as ``False``)       |
        +-----------------+----------------+--------------------------------------+
        | subgroup,       | True           | ``dict[str, Figure]`` keyed by       |
        | equalized_odds, |                | feature name (sorted)                |
        | gap             |                |                                      |
        +-----------------+----------------+--------------------------------------+
        | all             | False          | ``dict[str, Figure | None]`` keyed   |
        |                 |                | by mode (existing behavior)          |
        +-----------------+----------------+--------------------------------------+
        | all             | True           | ``dict[str, dict[str, Figure]]``     |
        |                 |                | keyed by mode, then feature.         |
        |                 |                | Always contains exactly the keys     |
        |                 |                | 'subgroup', 'equalized_odds',        |
        |                 |                | 'gap' (in that order). Components    |
        |                 |                | with no data return ``{}``.          |
        +-----------------+----------------+--------------------------------------+

        Returned dict for ``mode="all"`` (regardless of ``multi_feature``)
        ALWAYS contains exactly three keys in order: ``'subgroup'`` ->
        ``'equalized_odds'`` -> ``'gap'``. Within each multi-feature dict,
        feature order follows ``sorted(feature_names)`` for determinism.

        Note: ``mode="all"`` returns a structured dict and does NOT display
        figures unless ``show=True``.

        Parameters
        ----------
        mode : str, optional
            Visualization mode. One of {"summary", "all", "subgroup",
            "equalized_odds", "gap"}. Default "summary".
        show : bool
            Whether to display the figure interactively. Default True.
        save_path : str, optional
            If provided, saves the figure(s) to this path. Only honored when
            ``multi_feature=False`` (single-feature behavior).
            - Single modes: Treated as full file path. Defaults to ``.png`` if extension missing.
            - ``mode="all"``: Treated as base name. Appends suffixes and ``.png``.
            When ``multi_feature=True``, ``save_path`` is ignored (per-feature
            saving is intentionally not exposed here; use the lower-level
            ``plot_*_multi`` helpers if you need on-disk output).
        multi_feature : bool, optional
            If True, return per-feature figures rather than only the first
            sensitive feature. Default False (backward compatible).

        Returns
        -------
        matplotlib.figure.Figure | dict
            The return type depends on the ``(mode, multi_feature)`` combination.
            Possible shapes:
            - ``Figure``
            - ``dict[str, Figure | None]``
            - ``dict[str, Figure]``
            - ``dict[str, dict[str, Figure]]``
            See the return-shape table above.

        Notes
        -----
        New modes can be added by extending ``ALLOWED_MODES`` and dispatch
        logic without breaking existing behavior.
        Function behavior MUST be deterministic given identical inputs (no
        randomness, no state mutation). This includes consistent plot ordering,
        consistent key ordering in dicts, and no randomness in visualization.
        Each returned Figure must be independent (no shared axes, state, or
        references between plots). Each plot must be independently renderable
        and savable.

        Caution: Figures are returned open. If calling this method repeatedly
        in a loop, ensure you call ``plt.close(fig)`` on the returned figures
        to avoid memory accumulation.

        Raises
        ------
        ValueError
            If ``mode`` is invalid, ``"bias"`` is not present in
            ``self.results``, or the data is unusable.
        """
        self._require_classification("plot_bias()")
        import matplotlib.pyplot as plt

        from trustlens.visualization import _plot_bias
        from trustlens.visualization.fairness import (
            plot_equalized_odds,
            plot_equalized_odds_multi,
            plot_fairness_gap,
            plot_fairness_gap_multi,
            plot_subgroup_performance,
            plot_subgroup_performance_multi,
        )

        ALLOWED_MODES = {"summary", "all", "subgroup", "equalized_odds", "gap"}
        if mode not in ALLOWED_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Allowed: {ALLOWED_MODES}")

        if "bias" not in self.results:
            raise ValueError(
                "Bias results not available in report. "
                "Ensure 'bias' module was included in analyze()."
            )

        bias_data = self.results["bias"]

        # Validation: check for usable structure (valid, non-empty)
        has_subgroup = bool(bias_data.get("subgroup_performance"))
        has_eo = bool(bias_data.get("equalized_odds"))
        has_imbalance = bool(bias_data.get("class_imbalance"))

        if not (has_subgroup or has_eo or has_imbalance):
            raise ValueError("Bias data is present but not usable for visualization.")

        # Reserved meta keys that should never be treated as feature names.
        _META_KEYS = ("status", "reason", "details")

        def _get_first(key):
            d = bias_data.get(key, {})
            for k, v in d.items():
                if k not in _META_KEYS:
                    return k, v
            return None, None

        def _sorted_feature_dict(key):
            """Return ``{feature: data}`` ordered by ``sorted(feature_names)``.

            Drops reserved meta keys so wrapper functions iterate only over
            real features.
            """
            d = bias_data.get(key, {})
            return {fname: d[fname] for fname in sorted(k for k in d if k not in _META_KEYS)}

        def _get_save_path(base_path, suffix=None):
            if base_path is None:
                return None

            name, ext = os.path.splitext(base_path)
            if suffix:
                # mode="all": Strip extension if present, then append suffix and .png
                return f"{name}_{suffix}.png"

            # Single modes: If no extension -> append .png
            if not ext:
                ext = ".png"
            return f"{name}{ext}"

        # ------------------------------------------------------------------
        # multi_feature=True dispatch
        #
        # Implemented as a thin transformation layer over the same plotting
        # primitives the single-feature path uses. The "summary" mode simply
        # falls through to the existing single-figure path because a summary
        # plot is feature-agnostic by design.
        # ------------------------------------------------------------------
        if multi_feature and mode != "summary":
            if save_path is not None:
                logger.warning(
                    "save_path is ignored when multi_feature=True; "
                    "use plot_*_multi helpers directly for per-feature files."
                )

            if mode == "subgroup":
                feat_dict = _sorted_feature_dict("subgroup_performance")
                if not feat_dict:
                    raise ValueError("Missing 'subgroup_performance' data for 'subgroup' mode.")
                return plot_subgroup_performance_multi(feat_dict, show=show)

            if mode == "equalized_odds":
                feat_dict = _sorted_feature_dict("equalized_odds")
                if not feat_dict:
                    raise ValueError("Missing 'equalized_odds' data for 'equalized_odds' mode.")
                return plot_equalized_odds_multi(feat_dict, show=show)

            if mode == "gap":
                # Priority: subgroup_performance -> equalized_odds, mirroring
                # the single-feature path. The gap plot internally uses the
                # equalized-odds-shaped data, so we use whichever is present.
                feat_dict = _sorted_feature_dict("subgroup_performance")
                if not feat_dict:
                    feat_dict = _sorted_feature_dict("equalized_odds")
                if not feat_dict:
                    raise ValueError(
                        "Missing sufficient data for 'gap' mode "
                        "(either subgroup_performance or equalized_odds)."
                    )
                return plot_fairness_gap_multi(feat_dict, show=show)

            if mode == "all":
                # Return shape is fixed: three keys, in this order, every
                # call. Components with no data map to {} (never None).
                multi_results: dict[str, dict[str, plt.Figure]] = {
                    "subgroup": {},
                    "equalized_odds": {},
                    "gap": {},
                }

                sub_dict = _sorted_feature_dict("subgroup_performance")
                eo_dict = _sorted_feature_dict("equalized_odds")

                if sub_dict:
                    multi_results["subgroup"] = plot_subgroup_performance_multi(
                        sub_dict, show=False
                    )

                if eo_dict:
                    multi_results["equalized_odds"] = plot_equalized_odds_multi(eo_dict, show=False)

                # 'gap' uses the same priority rule as the single-feature
                # path: subgroup_performance first, falling back to
                # equalized_odds.
                gap_dict = sub_dict if sub_dict else eo_dict
                if gap_dict:
                    multi_results["gap"] = plot_fairness_gap_multi(gap_dict, show=False)

                if show:
                    backend = plt.get_backend().lower()
                    if "agg" not in backend:
                        try:
                            plt.show()
                        except Exception:
                            pass

                return multi_results

        if mode == "summary":
            # "summary" mode requires data compatible with _plot_bias()
            fig = _plot_bias(bias_data)
            if fig is None:
                raise ValueError("Failed to generate summary bias plot.")
            if save_path:
                from trustlens.visualization.style import save_figure

                save_figure(fig, _get_save_path(save_path), dpi=150, bbox_inches="tight")
            if show:
                backend = plt.get_backend().lower()
                if "agg" not in backend:
                    try:
                        plt.show()
                    except Exception:
                        pass
            return fig

        if mode == "subgroup":
            feat_name, feat_data = _get_first("subgroup_performance")
            if feat_data is None:
                raise ValueError("Missing 'subgroup_performance' data for 'subgroup' mode.")
            return plot_subgroup_performance(
                feat_data, feat_name, show=show, save_path=_get_save_path(save_path)
            )

        if mode == "equalized_odds":
            feat_name, feat_data = _get_first("equalized_odds")
            if feat_data is None:
                raise ValueError("Missing 'equalized_odds' data for 'equalized_odds' mode.")
            return plot_equalized_odds(
                feat_data, feat_name, show=show, save_path=_get_save_path(save_path)
            )

        if mode == "gap":
            # Priority: subgroup_performance > equalized_odds
            feat_name, feat_data = _get_first("subgroup_performance")
            if feat_data is None:
                feat_name, feat_data = _get_first("equalized_odds")

            if feat_data is None:
                raise ValueError(
                    "Missing sufficient data for 'gap' mode (either subgroup_performance or equalized_odds)."
                )
            return plot_fairness_gap(
                feat_data, feat_name, show=show, save_path=_get_save_path(save_path)
            )

        if mode == "all":
            results = {}
            plot_errors: list[str] = []

            # 1. Subgroup
            f_name_s, f_data_s = _get_first("subgroup_performance")
            fig_s = None
            if f_data_s:
                try:
                    fig_s = plot_subgroup_performance(
                        f_data_s,
                        f_name_s,
                        show=False,
                        save_path=_get_save_path(save_path, "subgroup"),
                    )
                except Exception as exc:  # noqa: BLE001 - reported below
                    plot_errors.append(f"s: {type(exc).__name__}: {exc}")
                    logger.warning("Bias plot failed: %s", exc)
                    fig_s = None
            results["subgroup"] = fig_s

            # 2. Equalized Odds
            f_name_eo, f_data_eo = _get_first("equalized_odds")
            fig_eo = None
            if f_data_eo:
                try:
                    fig_eo = plot_equalized_odds(
                        f_data_eo,
                        f_name_eo,
                        show=False,
                        save_path=_get_save_path(save_path, "equalized_odds"),
                    )
                except Exception as exc:  # noqa: BLE001 - reported below
                    plot_errors.append(f"eo: {type(exc).__name__}: {exc}")
                    logger.warning("Bias plot failed: %s", exc)
                    fig_eo = None
            results["equalized_odds"] = fig_eo

            # 3. Gap
            fig_g = None
            # Re-use best available data for gap priority
            g_name, g_data = (f_name_s, f_data_s) if f_data_s else (f_name_eo, f_data_eo)
            if g_data:
                try:
                    fig_g = plot_fairness_gap(
                        g_data, g_name, show=False, save_path=_get_save_path(save_path, "gap")
                    )
                except Exception as exc:  # noqa: BLE001 - reported below
                    plot_errors.append(f"g: {type(exc).__name__}: {exc}")
                    logger.warning("Bias plot failed: %s", exc)
                    fig_g = None
            results["gap"] = fig_g

            if all(v is None for v in results.values()):
                raise ValueError(
                    "Failed to generate any bias plots in 'all' mode"
                    + (": " + "; ".join(plot_errors) if plot_errors else ".")
                )

            if show:
                backend = plt.get_backend().lower()
                if "agg" not in backend:
                    try:
                        plt.show()
                    except Exception:
                        pass

            return results
