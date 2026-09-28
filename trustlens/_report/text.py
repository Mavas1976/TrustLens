"""Console and plain-text rendering of a TrustReport.

Part of the TrustReport split (TL-18); behaviour is unchanged.
"""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stdout
from typing import Any

import numpy as np

from trustlens._report.base import ReportBase

logger = logging.getLogger("trustlens.report")


class TextReportMixin(ReportBase):
    """Console and plain-text rendering of a TrustReport."""

    # A |coverage gap| below this magnitude reads as "on target" in the panel
    # (rounds to 0.0000 at the 4-decimal display precision).
    _GAP_ON_TARGET_EPS: float = 5e-5

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
