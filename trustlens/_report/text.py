"""Console and plain-text rendering of a TrustReport.

Part of the TrustReport split (TL-18); behaviour is unchanged.
"""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stdout
from typing import Any

import numpy as np

from trustlens._report.base import _WEAK_EXPLAIN, ReportBase

logger = logging.getLogger("trustlens.report")


class TextReportMixin(ReportBase):
    """Console and plain-text rendering of a TrustReport."""

    # A |coverage gap| below this magnitude reads as "on target" in the panel
    # (rounds to 0.0000 at the 4-decimal display precision).
    _GAP_ON_TARGET_EPS: float = 5e-5

    def _format_score_explanation(self) -> list[str]:
        """Rank and format top penalties for explanation."""
        penalties = self.trust_score.penalties_applied
        if not penalties:
            # Methodology 2.0: name the weakest assessed dimensions instead.
            weak = sorted(
                (s, d) for d, s in self.trust_score.sub_scores.items() if s < _WEAK_EXPLAIN
            )[:3]
            if not weak:
                return []
            labels = ["Dominant Issue", "Secondary Issue", "Minor Impact"]
            lines = ["Score Explanation:"]
            for label, (s, d) in zip(labels, weak):
                lines.append(f"  - {label:<16}: {d.replace('_', ' ').title()} ({s:.1f}/100)")
            return lines

        # Sort by magnitude descending
        sorted_p = sorted(penalties.items(), key=lambda x: x[1], reverse=True)
        top_p = sorted_p[:3]

        lines = ["Score Explanation:"]
        labels = ["Dominant Issue", "Secondary Issue", "Minor Impact"]

        for i, (name, val) in enumerate(top_p):
            if val > 0:
                label = labels[i] if i < len(labels) else "Other Impact"
                lines.append(f"  - {label:<16}: {name} (-{val:.1f})")
        return lines

    @property
    def deployment_summary(self) -> str:
        """Format the deployment explanation into a human-readable string."""
        exp = self.deployment_explanation
        lines = [f"Deployment Verdict: {exp['verdict']}", "", "Reasons:"]
        for reason in exp["reasons"]:
            icon = "✗" if reason["status"] == "fail" else "✓"
            lines.append(f"{icon} {reason['message']}")

        if exp["primary_risk"]:
            lines.extend(["", "Primary Risk:", str(exp["primary_risk"]["metric"])])

        lines.extend(["", "Recommendations:"])
        for rec in exp["recommendations"]:
            lines.append(f"• {rec}")

        return "\n".join(lines)

    def _print_score_methodology(self) -> None:
        """Display the mathematical composition and notes section."""
        ts = self.trust_score
        print("\n[ SCORE METHODOLOGY ]")
        weights_str = " + ".join(
            [f"{k.capitalize()} ({round(v * 100)}%)" for k, v in ts.weights_used.items()]
        )
        print(f"  Formula     : {weights_str}")
        for line in _methodology_lines(ts):
            print(line)

    def show(self, verbose: bool = False) -> None:
        """
        Print a rich, structured summary of all analysis results to stdout.

        Displays the Trust Score prominently at the top, followed by
        key insights and then delimited per-module metric summaries.

        For regression reports the classification Trust Score is not computed;
        the regression reliability metrics are shown instead.
        """
        if self.task_type == "regression":
            self._show_regression(verbose=verbose)
            return

        print("\nTrustLens Analysis Report")
        print(f"Timestamp : {self.metadata['timestamp']}")
        print(f"Model     : {self.metadata['model_class']}")
        print(f"Samples   : {self.metadata['n_samples']:,}")
        print(f"Classes   : {self.metadata['n_classes']}")

        # Trust Score section
        ts = self.trust_score
        print(f"\nTRUST SCORE: {ts.score}/100 [{ts.grade}]")
        print(f"Assessment : {ts.verdict}")

        for line in _score_summary_lines(ts):
            print(line)

        explanation = self._format_score_explanation()
        if explanation:
            print()
            for line in explanation:
                print(line)

        print(f"\n{self.deployment_summary}")

        # Print Key Observations/Insights
        print("\nKey Observations:")
        insights = self._generate_insights()
        if not insights:
            print("- No critical issues found.")
        for insight in insights:
            print(f"- {insight}")

        print("\nDimension Breakdown:")
        for dim, score in ts.sub_scores.items():
            print(f"- {dim.capitalize() + ' Score':<18}: {score:5.1f}/100")

        for module_name, module_data in self.results.items():
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                self._print_module(module_data, indent=0, verbose=verbose)
            out = f.getvalue().strip()
            if out:
                print(f"\n{module_name.title()} Analysis")
                print(out)
            # Conformal prediction sits as a sub-block of calibration; render it
            # as its own panel (the generic printer hides nested dicts unless
            # verbose, and the panel adds explicit over/under gap labelling).
            if (
                module_name == "calibration"
                and isinstance(module_data, dict)
                and "conformal" in module_data
            ):
                self._print_conformal_panel(module_data["conformal"])

        conclusion = self._generate_conclusion()
        print(f"\nConclusion:\n{conclusion}")

        # Methodology section at the end
        self._print_score_methodology()
        print()

    def _show_regression(self, verbose: bool = False) -> None:
        """Render the regression reliability report (no classification trust score)."""
        reg = self.results.get("regression", {})
        print("\nTrustLens Regression Reliability Report")
        print(f"Timestamp : {self.metadata['timestamp']}")
        print(f"Model     : {self.metadata['model_class']}")
        print(f"Samples   : {self.metadata['n_samples']:,}")
        print("Task      : regression")

        ts = self.trust_score
        print(f"\nTRUST SCORE: {ts.score}/100 [{ts.grade}]")
        print(f"Assessment : {ts.verdict}")
        if ts.sub_scores:
            print("Dimensions :")
            for dim, dim_score in ts.sub_scores.items():
                label = dim.replace("_", " ").title()
                print(f"  - {label:<28}: {dim_score:5.1f}/100")
        for line in _score_summary_lines(ts):
            print(line)

        ed = reg.get("error_distribution", {})
        if ed and ed.get("status") != "skipped":
            print("\nError Distribution (|y_true - y_pred|):")
            print(f"  MAE                : {ed.get('mean_absolute_error'):.4f}")
            print(f"  RMSE               : {ed.get('rmse'):.4f}")
            print(f"  Median abs error   : {ed.get('median_absolute_error'):.4f}")
            print(f"  90th-pct abs error : {ed.get('p90_absolute_error'):.4f}")
            print(f"  Max abs error      : {ed.get('max_error'):.4f}")
            medae = ed.get("median_absolute_error") or 0.0
            p90 = ed.get("p90_absolute_error") or 0.0
            if medae > 0 and p90 / medae > 3:
                print(
                    f"  ! Heavy error tail : p90 is {p90 / medae:.1f}x the median "
                    "— a tail of large mistakes worth investigating."
                )

        pic = reg.get("interval_coverage", {})
        if pic:
            if pic.get("status") == "skipped":
                print("\nPrediction Interval Coverage:")
                print(f"  skipped — {pic.get('details', pic.get('reason', 'no intervals'))}")
            elif pic.get("ice") is not None:
                # Multi-level interval calibration (RFC #155): ICE + sharpness proxy.
                print("\nInterval Calibration (multi-level):")
                print(
                    f"  ICE                : {pic.get('ice'):.4f}  "
                    f"({pic.get('n_calibrated_levels')}/{pic.get('n_levels')} levels calibrated)"
                )
                ss = pic.get("sharpness_skill")
                print(
                    "  Sharpness skill    : "
                    + (
                        f"{ss:+.4f}  (vs climatology, calibrated levels)"
                        if ss is not None
                        else "n/a (no calibrated level)"
                    )
                )
                print(f"  Worst level gap    : {pic.get('worst_calibration_error'):+.4f}")
                print(f"  Mean interval width: {pic.get('mean_interval_width'):.4f}")
                print(f"  Verdict            : {pic.get('verdict')}")
            else:
                print("\nPrediction Interval Coverage (PICP):")
                print(
                    f"  Coverage (PICP)    : {pic.get('picp'):.4f}  "
                    f"(target {pic.get('target_coverage')})"
                )
                print(f"  Calibration error  : {pic.get('calibration_error'):+.4f}")
                print(f"  Mean interval width: {pic.get('mean_interval_width'):.4f}")
                print(f"  Verdict            : {pic.get('verdict')}")

        evc = reg.get("error_variance_correlation", {})
        if evc:
            print("\nError-Uncertainty Correlation:")
            if evc.get("status") == "skipped":
                print(f"  skipped — {evc.get('details', evc.get('reason', 'no variance'))}")
            else:
                print(f"  Pearson            : {evc.get('pearson'):+.4f}")
                print(f"  Spearman           : {evc.get('spearman'):+.4f}")
                print(f"  Verdict            : {evc.get('verdict')}")

        # Render any additional / plugin modules generically.
        for module_name, module_data in self.results.items():
            if module_name == "regression":
                continue
            f = io.StringIO()
            with redirect_stdout(f):
                self._print_module(module_data, indent=0, verbose=verbose)
            out = f.getvalue().strip()
            if out:
                print(f"\n{module_name.title()} Analysis")
                print(out)

        if pic.get("status") == "skipped" and evc.get("status") == "skipped":
            print(
                "\nNote: uncertainty metrics (PICP, error-variance correlation) need "
                "prediction intervals / predicted variance. Pass prediction_intervals "
                "and/or predicted_variance to analyze() to enable them."
            )
        print()

    def _generate_regression_text(self, verbose: bool = False) -> str:
        """Plain-text regression report (captures _show_regression's output)."""
        f = io.StringIO()
        with redirect_stdout(f):
            self._show_regression(verbose=verbose)
        return f.getvalue().strip()

    def _generate_text_report(self, verbose: bool = False) -> str:
        """
        Generate a clean, human-readable text report without ANSI colors.
        Mirroring the structure of show().
        """
        if self.task_type == "regression":
            return self._generate_regression_text(verbose=verbose)

        lines = []
        lines.append("TrustLens Analysis Report")
        lines.append(f"Timestamp : {self.metadata['timestamp']}")
        lines.append(f"Model     : {self.metadata['model_class']}")
        lines.append(f"Samples   : {self.metadata['n_samples']:,}")
        lines.append(f"Classes   : {self.metadata['n_classes']}")

        ts = self.trust_score
        lines.append(f"\nTRUST SCORE: {ts.score}/100 [{ts.grade}]")
        lines.append(f"Assessment : {ts.verdict}")
        lines.extend(_score_summary_lines(ts))
        explanation = self._format_score_explanation()
        if explanation:
            lines.append("")
            lines.extend(explanation)

        lines.append("")
        lines.extend(self.deployment_summary.split("\n"))

        lines.append("\nKey Observations:")
        insights = self._generate_insights()
        if not insights:
            lines.append("- No critical issues found.")
        else:
            for insight in insights:
                lines.append(f"- {insight}")

        lines.append("\nDimension Breakdown:")
        for dim, score in ts.sub_scores.items():
            lines.append(f"- {dim.capitalize() + ' Score':<18}: {score:5.1f}/100")

        for module_name, module_data in self.results.items():
            line_buf: list[str] = []
            self._get_module_text_lines(module_data, line_buf, indent=0, verbose=verbose)
            if line_buf:
                lines.append(f"\n{module_name.title()} Analysis")
                lines.extend(line_buf)
            # Conformal panel — keep the saved text report consistent with show().
            if (
                module_name == "calibration"
                and isinstance(module_data, dict)
                and "conformal" in module_data
            ):
                lines.extend(self._conformal_panel_lines(module_data["conformal"]))

        lines.append(f"\nConclusion:\n{self._generate_conclusion()}")

        # Text methodology lines
        lines.append("\n[ SCORE METHODOLOGY ]")
        weights_str = " + ".join(
            [f"{k.capitalize()} ({round(v * 100)}%)" for k, v in ts.weights_used.items()]
        )
        lines.append(f"  Formula     : {weights_str}")
        lines.extend(_methodology_lines(ts))
        return "\n".join(lines)

    def _get_module_text_lines(
        self, data: Any, buf: list[str], indent: int = 0, verbose: bool = False
    ) -> None:
        """Helper for _generate_text_report to recursively collect lines."""
        prefix = " " * indent
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str) and key.startswith("__") and key.endswith("__"):
                    continue
                display_key = str(key).replace("_", " ").title()

                if isinstance(value, dict):
                    if verbose:
                        buf.append(f"{prefix}- {display_key}:")
                        self._get_module_text_lines(value, buf, indent + 2, verbose)
                elif isinstance(value, (list, np.ndarray, tuple)):
                    if verbose:
                        buf.append(
                            f"{prefix}- {display_key}: [data structure of size {len(value)}]"
                        )
                elif isinstance(value, float):
                    buf.append(f"{prefix}- {display_key}: {value:.4f}")
                else:
                    buf.append(f"{prefix}- {display_key}: {value}")
        else:
            if verbose:
                buf.append(f"{prefix}- {data}")

    def _generate_conclusion(self) -> str:
        """Generate a short 1-2 line conclusion that follows the Trust Score verdict."""
        ts = self.trust_score
        if ts.grade == "N/A":
            return (
                "Not assessed: no trust dimension could be scored. Supply predicted "
                "probabilities (y_prob) and, for fairness, sensitive_features."
            )
        blockers = getattr(ts, "blockers", []) or []
        if blockers:
            return f"Do not deploy. {blockers[0]}."
        if getattr(ts, "is_partial", False):
            missing = ", ".join(getattr(ts, "missing_dimensions", []) or [])
            return f"Incomplete assessment ({missing} not assessed). Complete it before deciding."
        if ts.grade == "A":
            return "No critical issues detected across the measured dimensions."
        if ts.grade == "B":
            return "Model is generally reliable, but minor issues should be addressed before broad deployment."
        if ts.grade == "C":
            return (
                "Model shows moderate risk. Investigate the flagged dimensions before proceeding."
            )
        return "Model exhibits critical issues and should not be deployed until fundamental problems are resolved."

    def _generate_insights(self) -> list[str]:
        """Generate plain-text insights based on results."""
        insight_list = []

        def add_insight(msg: str, priority: int):
            insight_list.append((priority, msg))

        # Surfaced Patterns
        if self.patterns:
            pattern_lines = [f"  - {p}" for p in self.patterns]
            pattern_msg = "Patterns Detected:\n" + "\n".join(pattern_lines)
            add_insight(pattern_msg, 2)

        # Core signals

        failure_score = self.trust_score.sub_scores.get("failure", 100.0)
        conf_gap = self.results.get("failure", {}).get("confidence_gap", {}).get("gap", 0.0)
        silhouette = (
            self.results.get("representation", {})
            .get("separability", {})
            .get("silhouette_score", 0.0)
        )

        # Legacy pattern checks removed. Patterns are now sourced from self.patterns.
        is_confidently_wrong = "Confidently Wrong" in self.patterns

        # Pattern: Generalization Risk
        if silhouette < 0 and failure_score < 50:
            add_insight(
                "⚠ Warning: Poor latent representation correlates with high failure risk.\n    → The network struggles to differentiate classes; investigate feature quality.",
                2,
            )

        cal_score = self.trust_score.sub_scores.get("calibration", 100.0)

        # Check Calibration
        # Only comment on dimensions that were actually scored (skipped modules
        # have no calibration or confidence evidence).
        if "calibration" in self.trust_score.sub_scores:
            if not is_confidently_wrong:
                if cal_score < 75:
                    add_insight(
                        "Critical: Calibration is poor (score < 75).\n    → Consider temperature scaling or isotonic regression.",
                        1,
                    )
                elif cal_score < 90:
                    add_insight(
                        "Warning: Calibration is acceptable (score 75-89), but could be improved.",
                        2,
                    )
                else:
                    add_insight("ℹ Info: Calibration quality is excellent (score 90+).", 3)

        # Check Failure
        failure_module = self.results.get("failure", {})
        error_rate = (
            failure_module.get("misclassification_summary", {})
            .get("__overall__", {})
            .get("overall_error_rate", 0.0)
        )
        error_pct = round(error_rate * 100) if error_rate is not None else 0

        avg_err_conf = failure_module.get("confidence_gap", {}).get(
            "incorrect_confidence_mean", 0.0
        )
        conf_str = (
            f"~{avg_err_conf:.2f} confidence" if avg_err_conf > 0 else "confidence-weighted error"
        )

        if not is_confidently_wrong:
            if failure_score < 40:
                add_insight(
                    f"Critical: High failure risk detected ({error_pct}% error rate).\n    → Heavily penalized because errors are dangerously concentrated in the {conf_str} range.",
                    1,
                )
            elif failure_score < 60:
                add_insight(
                    f"Warning: Moderate failure risk ({error_pct}% error rate).\n    → Model exhibits concerning confidence (~{avg_err_conf:.2f}) on incorrect predictions.",
                    2,
                )

        if "failure" in self.trust_score.sub_scores and not is_confidently_wrong:
            # With no errors there is no "incorrect" group to be overconfident on (GA-11).
            if error_rate > 0 and conf_gap < 0.05:
                add_insight(
                    "Warning: Model is overconfident on incorrect predictions (low confidence gap).",
                    2,
                )

        # Check Bias
        if "bias" in self.results:
            bias_module = self.results["bias"]
            for feat_name, feat_data in bias_module.get("subgroup_performance", {}).items():
                excluded = (feat_data.get("__summary__") or {}).get("excluded_groups")
                if excluded:
                    # GA-04: say which groups were too small to judge.
                    add_insight(
                        f"Info: {feat_name} groups {', '.join(map(str, excluded))} have too few "
                        "samples for a fairness gap and were excluded.",
                        2,
                    )
            ratio = bias_module.get("class_imbalance", {}).get("imbalance_ratio", 1.0)
            if ratio > 5.0:
                add_insight(
                    "Warning: Severe class imbalance may affect performance.\n    → Consider rebalancing or fairness constraints.",
                    2,
                )

            subgroups = bias_module.get("subgroup_performance", {})
            for feat_name, feat_data in subgroups.items():
                gap = feat_data.get("__summary__", {}).get("performance_gap") or 0.0
                if gap > 0.1:
                    add_insight(
                        f"Warning: Significant performance gap detected across {feat_name}.\n    → Investigate subgroup disparities.",
                        2,
                    )

            eq_odds = bias_module.get("equalized_odds", {})
            has_eq_odds_skipped = eq_odds.get("status") == "skipped"

            has_severe = False
            if has_eq_odds_skipped:
                add_insight(
                    "ℹ Info: Fairness analysis skipped due to insufficient subgroup diversity (or non-binary target).",
                    3,
                )
            elif eq_odds:
                for k, val in eq_odds.items():
                    if isinstance(val, dict) and k not in ("status", "reason", "details"):
                        summary = val.get("__summary__", {})
                        if (
                            summary.get("tpr_violation") == "severe"
                            or summary.get("fpr_violation") == "severe"
                        ):
                            has_severe = True
                if has_severe:
                    add_insight(
                        "Critical: Severe fairness disparity detected between subgroups.\n    → Investigate subgroup disparities and consider rebalancing.",
                        1,
                    )

            if not has_severe and not has_eq_odds_skipped:
                # Check if gap from subgroup is low and no class imbalance severity.
                if ratio <= 5.0:
                    max_gap = 0.0
                    for feat_data in subgroups.values():
                        gap = feat_data.get("__summary__", {}).get("performance_gap") or 0.0
                        if gap > max_gap:
                            max_gap = gap
                    if max_gap <= 0.1:
                        margin = 0.1 - max_gap
                        has_penalty = "Fairness" in getattr(
                            self.trust_score, "penalties_applied", {}
                        )
                        if has_penalty:
                            if margin < 0.01:
                                msg = "ℹ Info: At threshold boundary (0.00 margin from 0.10 limit). Minimal penalty applied."
                            else:
                                msg = f"ℹ Info: Minor fairness variations detected (margin: {margin:.2f} from 0.10 limit). Small penalty applied."
                            add_insight(msg, 3)
                        else:
                            add_insight(
                                f"ℹ Info: No bias detected (margin: {margin:.2f} from 0.10 limit).",
                                3,
                            )

        # Sort by priority, then deduplicate while preserving order
        insight_list.sort(key=lambda x: x[0])
        seen = set()
        final_insights = []
        for _, msg in insight_list:
            if msg not in seen:
                seen.add(msg)
                final_insights.append(msg)

        return final_insights

    def _print_module(self, data: Any, indent: int = 0, verbose: bool = False) -> None:
        """Recursively pretty-print a module's result dictionary."""
        prefix = " " * indent
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str) and key.startswith("__") and key.endswith("__"):
                    continue
                display_key = str(key).replace("_", " ").title()

                if isinstance(value, dict):
                    if verbose:
                        print(f"{prefix}- {display_key}:")
                        self._print_module(value, indent + 2, verbose)
                elif isinstance(value, (list, np.ndarray, tuple)):
                    if verbose:
                        print(f"{prefix}- {display_key}: [data structure of size {len(value)}]")
                elif isinstance(value, float):
                    print(f"{prefix}- {display_key}: {value:.4f}")
                else:
                    print(f"{prefix}- {display_key}: {value}")
        else:
            if verbose:
                print(f"{prefix}- {data}")

    @classmethod
    def _cov_gap_label(cls, gap: float | None, over_is_positive: bool) -> str:
        """Format a coverage gap as an explicit over-/under-coverage annotation.

        The conformal block deliberately carries two ``*_gap`` fields with
        opposite sign conventions: ``coverage_gap = marginal − nominal``
        (positive ⇒ *over*-coverage, ``over_is_positive=True``) and
        ``worst_class_gap = nominal − worst`` (positive ⇒ *under*-coverage,
        ``over_is_positive=False``). Spelling the direction out here means a
        reader never has to remember which field means which.
        """
        if gap is None:
            return ""
        if abs(gap) < cls._GAP_ON_TARGET_EPS:
            return "  (on target)"
        over = (gap > 0) == over_is_positive
        return f"  ({'over' if over else 'under'} by {abs(gap):.4f})"

    def _conformal_panel_lines(self, conf: Any) -> list[str]:
        """Build the conformal-prediction panel as a list of text lines.

        Shared by ``show()`` (printed) and ``_generate_text_report`` (buffered)
        so the interactive and saved outputs stay identical. Diagnostic-only:
        none of these values influence the Trust Score. Layout follows the
        signal order — headline marginal coverage vs target, then the two
        conditional diagnostics (worst-class gap, size-stratified violation),
        then informativeness (set size, singleton rate, efficiency).
        """
        if not isinstance(conf, dict):
            return []
        lines = ["\nConformal Prediction (prediction sets)"]

        if conf.get("status") == "skipped":
            reason = conf.get("reason", "unknown")
            details = conf.get("details", "")
            line = f"  Skipped: {reason}"
            if details:
                line += f" ({details})"
            lines.append(line)
            return lines

        nominal = conf.get("nominal")
        if nominal is not None:
            lines.append(f"  Nominal coverage     : {nominal:.4f}")
        else:
            lines.append("  Nominal coverage     : (not supplied — coverage gaps unavailable)")

        marg = conf.get("marginal_coverage")
        if marg is not None:
            label = self._cov_gap_label(conf.get("coverage_gap"), over_is_positive=True)
            lines.append(f"  Marginal coverage    : {marg:.4f}{label}")

        wcc = conf.get("worst_class_coverage")
        if wcc is not None:
            label = self._cov_gap_label(conf.get("worst_class_gap"), over_is_positive=False)
            lines.append(f"  Worst-class coverage : {wcc:.4f}{label}")

        ssc = conf.get("ssc_violation")
        if ssc is not None:
            flag = "  (conditionally honest)" if ssc <= 0.0 else ""
            lines.append(f"  SSC violation        : {ssc:.4f}{flag}")

        for label, key in (
            ("Avg set size", "avg_set_size"),
            ("Singleton rate", "singleton_rate"),
            ("Size efficiency", "size_efficiency"),
            ("Empty rate", "empty_rate"),
        ):
            val = conf.get(key)
            if val is not None:
                lines.append(f"  {label:<20} : {val:.4f}")

        n_s, n_c = conf.get("n_samples"), conf.get("n_classes")
        if n_s is not None and n_c is not None:
            lines.append(f"  (n={n_s}, classes={n_c})")
        return lines

    def _print_conformal_panel(self, conf: Any) -> None:
        """Print the conformal-prediction panel (interactive ``show()`` path)."""
        for line in self._conformal_panel_lines(conf):
            print(line)


def _methodology_lines(ts: Any) -> list[str]:
    """Definitions printed under the score formula (methodology 2.0)."""
    return [
        f"  Method      : Trust Score methodology {getattr(ts, 'score_version', '1.x')}",
        "  Definitions :",
        "    - Calibration : 100 x (1 - ECE / 0.25)",
        "    - Failure     : 100 x (1 - min(error rate / 0.20, 1) x (1 - detection)), detection from AUROC",
        "    - Bias        : 100 x (1 - largest fairness gap / 0.30), with sensitive features only",
        "    - Blockers    : no skill, overconfidence > 0.10, fairness gap > 0.15 (grade D);",
        "                    ceilings fall continuously towards each threshold",
        "    - Caps        : incomplete assessment (grade C); sub-scores below 40 lower the ceiling",
    ]


def _score_summary_lines(ts: Any) -> list[str]:
    """Explain how the reported score was reached.

    Methodology 2.0 reports blockers and caps; results from methodology 1.x
    (for example rebuilt from older saved reports) still list their penalties.
    """
    penalties = getattr(ts, "penalties_applied", None) or {}
    if penalties:
        penalties_str = ", ".join(f"{k} (-{v})" for k, v in penalties.items())
        return [
            "\nScore Summary:",
            f"  Base Score        : {ts.base_score}",
            f"  Penalties Applied : -{sum(penalties.values()):.1f} [{penalties_str}]",
            f"  Final Score       : {ts.score}",
        ]
    blockers = list(getattr(ts, "blockers", []) or [])
    caps = list(getattr(ts, "caps_applied", []) or [])
    if not (blockers or caps):
        return []
    lines = ["\nScore Summary:", f"  Weighted Score    : {ts.base_score}"]
    lines.extend(f"  Blocker           : {b}" for b in blockers)
    if ts.grade == "N/A":
        caps = [c.replace(" (capped at grade C)", "") for c in caps]
    lines.extend(f"  Cap               : {c}" for c in caps)
    lines.append(f"  Final Score       : {ts.score}")
    return lines
