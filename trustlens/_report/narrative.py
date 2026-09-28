"""Narrative text of a TrustReport: insights, conclusion and score explanation.

Part of the TrustReport split (TL-18, GB-08); behaviour is unchanged.
"""

from __future__ import annotations

import logging

from trustlens._report.base import _WEAK_EXPLAIN, ReportBase

logger = logging.getLogger("trustlens.report")


class NarrativeMixin(ReportBase):
    """Narrative text of a TrustReport: insights, conclusion and score explanation."""

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
