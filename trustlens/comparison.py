"""
trustlens.comparison
====================
Utility for comparative analysis across multiple TrustReports.
"""

from __future__ import annotations

from typing import Any, Optional

from trustlens.report import TrustReport


def _is_deployable(trust_score: Any) -> bool:
    """A report can be recommended only when it is complete, unblocked and not grade D.

    Partial assessments (``modules=`` subsets, missing probabilities) and failing
    grades are never recommended for deployment (ADR-001, TL-11).
    """
    return (
        not trust_score.is_blocked
        and not getattr(trust_score, "is_partial", False)
        and trust_score.grade != "D"
    )


def _exclusion_reason(trust_score: Any) -> str:
    blockers = getattr(trust_score, "blockers", None)
    if blockers:
        return str(blockers[0])
    if getattr(trust_score, "is_partial", False):
        return "incomplete assessment (" + ", ".join(trust_score.missing_dimensions) + ")"
    return str(trust_score.verdict)


def _weakest_dimension(trust_score: Any) -> Optional[tuple[str, float]]:
    if not trust_score.sub_scores:
        return None
    dim = min(trust_score.sub_scores, key=trust_score.sub_scores.get)
    return dim, float(trust_score.sub_scores[dim])


def compare(
    reports: list[TrustReport],
    names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Compare multiple models and recommend the safest candidate.

    Only complete, unblocked reports with a grade above D are eligible. Scores
    are ranked only when the eligible reports were scored on the same
    dimensions; otherwise the result carries a warning, because a report
    without (for example) a fairness dimension is not directly comparable to
    one with it.

    Parameters
    ----------
    reports : list[TrustReport]
        A list of generated TrustReport objects of the same task type.
    names : list[str], optional
        Display names, one per report. Defaults to each report's model class,
        suffixed with its position when several reports share a class.

    Returns
    -------
    dict
        ``{"recommended": name or None, "ranking": [...], "excluded": [...],
        "warnings": [...]}``. ``ranking`` lists eligible reports by score
        (``name``, ``score``, ``grade``, ``weakest_dimension``); ``excluded``
        lists the others with the reason. ``recommended`` is ``None`` when no
        report is eligible, or when the eligible reports were scored on
        different dimensions (their scores are then not comparable). The same
        summary is printed.

    Raises
    ------
    ValueError
        If the reports mix task types or ``names`` has the wrong length.
    """
    result: dict[str, Any] = {"recommended": None, "ranking": [], "excluded": [], "warnings": []}
    if not reports:
        print("========== Model Comparison & Recommendation ==========")
        print("No reports provided for comparison.\n")
        return result

    # Trust Scores are only comparable within a single task type: a regression
    # score and a classification score share this interface (0–100, A–D) but
    # aggregate different dimensions, so cross-task ranking is meaningless (see
    # RFC #145). Refuse to rank a mixed batch rather than recommend silently.
    task_types = {getattr(rep, "task_type", "classification") for rep in reports}
    if len(task_types) > 1:
        raise ValueError(
            "compare() cannot rank reports across different task types "
            f"({sorted(task_types)}): regression and classification Trust Scores are "
            "not directly comparable. Compare reports within a single task type."
        )

    if names is None:
        classes = [rep.metadata.get("model_class", "UnknownModel") for rep in reports]
        names = [f"{c} #{i + 1}" if classes.count(c) > 1 else c for i, c in enumerate(classes)]
    elif len(names) != len(reports):
        raise ValueError(f"names has {len(names)} entries for {len(reports)} reports.")
    if len(set(names)) != len(names):
        raise ValueError(f"names must be unique; got {names}.")

    print("========== Model Comparison & Recommendation ==========")
    for name, rep in zip(names, reports):
        ts = rep.trust_score
        print(f"{name:<20} Score: {ts.score:3d}/100 [{ts.grade}] | Verdict: {ts.verdict}")
    print()

    eligible = [(n, r) for n, r in zip(names, reports) if _is_deployable(r.trust_score)]
    for name, rep in zip(names, reports):
        if not _is_deployable(rep.trust_score):
            result["excluded"].append(
                {
                    "name": name,
                    "grade": rep.trust_score.grade,
                    "reason": _exclusion_reason(rep.trust_score),
                }
            )

    eligible.sort(key=lambda item: item[1].trust_score.score, reverse=True)
    for name, rep in eligible:
        weakest = _weakest_dimension(rep.trust_score)
        result["ranking"].append(
            {
                "name": name,
                "score": rep.trust_score.score,
                "grade": rep.trust_score.grade,
                "weakest_dimension": weakest,
            }
        )

    if len(eligible) > 1 and eligible[0][1].trust_score.score == eligible[1][1].trust_score.score:
        tied = [n for n, r in eligible if r.trust_score.score == eligible[0][1].trust_score.score]
        result["warnings"].append(
            f"Tie at {eligible[0][1].trust_score.score}/100 between {tied}; the recommendation "
            "follows input order. Compare their sub-scores before choosing."
        )

    dimension_sets = {tuple(sorted(r.trust_score.sub_scores)) for _, r in eligible}
    comparable = len(dimension_sets) <= 1
    if not comparable:
        # A model scored without fairness is not better than one whose fairness
        # was measured and found wanting (R-011): refuse to pick either.
        result["warnings"].append(
            "Eligible reports were scored on different dimensions "
            f"({[list(s) for s in sorted(dimension_sets)]}); scores are not directly comparable, "
            "so no model is recommended. Re-run analyze() with the same inputs "
            "(y_prob, sensitive_features, embeddings) for every model."
        )

    if not comparable:
        print("Recommendation: NONE - the reports were scored on different dimensions.")
    elif not eligible:
        print("Recommendation: DO NOT DEPLOY any model.")
        print(
            "  * No model has a complete assessment without critical diagnostic blocks"
            " or a failing grade."
        )
    else:
        best_name, best = eligible[0]
        result["recommended"] = best_name
        print(f"Recommendation: Deploy {best_name}.")
        print("  * Highest Trust Score among complete, unblocked assessments.")
        if len(eligible) > 1:
            runner_name, runner = eligible[1]
            diff = best.trust_score.score - runner.trust_score.score
            print(f"  * Ahead of {runner_name} by {diff} point(s).")
        weakest = _weakest_dimension(best.trust_score)
        if weakest:
            print(f"  * Weakest dimension of {best_name}: {weakest[0]} ({weakest[1]:.1f}/100).")

    if result["excluded"]:
        print("\nExcluded:")
        for item in result["excluded"]:
            print(f"  * {item['name']:<20}: {item['reason']}")
    for warning in result["warnings"]:
        print(f"\nWarning: {warning}")
    print()
    return result
