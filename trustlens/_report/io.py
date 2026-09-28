"""Saving and serialising a TrustReport.

Part of the TrustReport split (TL-18); behaviour is unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, cast

import numpy as np

from trustlens._report.base import ReportBase

logger = logging.getLogger("trustlens.report")


class PersistenceMixin(ReportBase):
    """Saving and serialising a TrustReport."""

    def save(
        self,
        path: str | os.PathLike[str] = "trust_report",
        overwrite: bool = True,
        **kwargs: Any,
    ) -> Path:
        """
        Save the analysis report.

        If ``path`` ends with '.json' or '.txt', saves a single file. A path
        without a suffix is treated as a directory and receives a full report
        bundle (JSON, metadata, plots). Any other suffix (e.g. ``.png``) raises
        ``ValueError`` instead of silently creating a directory with that name.

        Parameters
        ----------
        path : str or os.PathLike
          Target file path (e.g., "report.json") or directory path.
        overwrite : bool, default=True
          When False, refuse to replace an existing file or a non-empty bundle
          directory (``FileExistsError``).
        **kwargs : Any
          Backward compatibility for ``directory`` argument.

        Returns
        -------
        Path
          Resolved path to the saved file or directory.
        """
        if "directory" in kwargs:
            path = kwargs.pop("directory")
        if kwargs:
            raise TypeError(f"save() got unexpected keyword argument(s) {sorted(kwargs)}.")

        path = os.fspath(path)
        p = Path(path).resolve()
        suffix = p.suffix.lower()
        # Any suffix other than .json/.txt is refused unless the path is an
        # existing directory, so 'report.png' or 'report.v2' never silently
        # becomes a bundle directory (GB-18).
        if suffix and suffix not in (".json", ".txt") and not p.is_dir():
            raise ValueError(
                f"Unsupported report file type '{p.suffix}'. Use '.json', '.txt', or a "
                "directory path without a suffix (or an existing directory) for the full bundle."
            )
        if not overwrite and p.exists() and (p.is_file() or any(p.iterdir())):
            raise FileExistsError(f"{p} already exists; pass overwrite=True to replace it.")

        is_classification = self.task_type != "regression"

        # 1. Single-file JSON export
        if suffix == ".json":
            p.parent.mkdir(parents=True, exist_ok=True)
            data: dict[str, Any] = {
                "results": self.results,
                "metadata": self.metadata,
                "task_type": self.task_type,
                "trust_score": self.trust_score.score,
                "grade": self.trust_score.grade,
                "sub_scores": self.trust_score.sub_scores,
                "score_version": self.trust_score.score_version,
            }
            if is_classification:
                data["deployment_explanation"] = self.deployment_explanation
            p.write_text(self._dumps(data), encoding="utf-8")
            logger.info("Report JSON saved to: %s", p)
            return p

        # 2. Single-file TXT export
        if suffix == ".txt":
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(self._generate_text_report(), encoding="utf-8")
            logger.info("Report TXT saved to: %s", p)
            return p

        # 3. Directory bundle: metrics, metadata, Trust Score (+ plots for classification)
        out_dir = p
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(self._dumps(self.results), encoding="utf-8")
        (out_dir / "metadata.json").write_text(self._dumps(self.metadata), encoding="utf-8")
        (out_dir / "trust_score.json").write_text(
            self._dumps(self._trust_score_payload()), encoding="utf-8"
        )
        if is_classification:
            try:
                self.summary_plot(save_path=str(out_dir / "summary_plot.png"), show=False)
            except Exception as exc:  # noqa: BLE001 - plots are optional artefacts
                logger.warning("Summary plot skipped: %s", exc)
            try:
                self.plot(save_dir=str(out_dir))
            except Exception as exc:  # noqa: BLE001 - plots are optional artefacts
                logger.warning("Plot generation skipped: %s", exc)
        logger.info("Report bundle saved to: %s", out_dir)
        return out_dir

    def _trust_score_payload(self) -> dict[str, Any]:
        """Every Trust Score field, for both task types (one serializer; GB-08)."""
        ts = self.trust_score
        payload: dict[str, Any] = {
            "score": ts.score,
            "grade": ts.grade,
            "verdict": ts.verdict,
            "task_type": ts.task_type,
            "score_version": ts.score_version,
            "sub_scores": ts.sub_scores,
            "weights_used": ts.weights_used,
            "breakdown": ts.breakdown,
            "base_score": ts.base_score,
            "is_blocked": ts.is_blocked,
            "blockers": ts.blockers,
            "caps_applied": ts.caps_applied,
            "is_partial": ts.is_partial,
            "missing_dimensions": ts.missing_dimensions,
            "penalties_applied": ts.penalties_applied,
        }
        if self.task_type != "regression":
            payload["deployment_explanation"] = self.deployment_explanation
        return payload

    def to_dict(self) -> dict[str, Any]:
        """
        Return all results as a flat, JSON-serializable dictionary.

        Useful for logging to MLflow, W&B, or any experiment tracker.

        Returns
        -------
        dict
          Flat dict with keys like ``"calibration.brier_score"``.
        """
        from trustlens.utils import flatten_dict

        flat = flatten_dict(self._to_serializable(self.results))

        # Regression reports carry a regression-specific Trust Score (different
        # dimensions from classification) alongside the reliability metrics. The
        # classification-only deployment verdict block is omitted.
        if self.task_type == "regression":
            flat["task_type"] = "regression"
            flat["n_samples"] = self.metadata["n_samples"]
            flat["model"] = self.metadata["model_class"]
            flat["timestamp"] = self.metadata["timestamp"]
            flat["framework"] = self.metadata.get("framework", "unknown")
            flat["trustlens_version"] = self.metadata["trustlens_version"]
            flat["trust_score"] = self.trust_score.score
            flat["trust_grade"] = self.trust_score.grade
            flat["trust_score_version"] = self.trust_score.score_version
            for dim, score in self.trust_score.sub_scores.items():
                flat[f"trust_{dim}_score"] = score
            return cast(dict[str, Any], self._to_serializable(flat))

        flat["trust_score"] = self.trust_score.score
        flat["trust_grade"] = self.trust_score.grade
        flat["framework"] = self.metadata.get("framework", "unknown")
        flat["trustlens_version"] = self.metadata["trustlens_version"]

        # Flatten deployment explanation for MLflow/W&B tracking
        exp = self.deployment_explanation
        flat["deployment_verdict"] = exp["verdict"]
        flat["deployment_primary_risk_metric"] = (
            exp["primary_risk"].get("metric") if exp["primary_risk"] else None
        )
        flat["deployment_primary_risk_value"] = (
            exp["primary_risk"].get("value") if exp["primary_risk"] else None
        )

        flat["trust_score_version"] = self.trust_score.score_version
        flat["trust_partial"] = self.trust_score.is_partial
        for dim, score in self.trust_score.sub_scores.items():
            flat[f"trust_{dim}_score"] = score
        return cast(dict[str, Any], self._to_serializable(flat))

    def _to_serializable(self, obj: Any) -> Any:
        """Recursively convert numpy / non-JSON-native types.

        Non-finite floats (NaN, +/-inf) become ``None``: ``json.dumps`` would
        otherwise emit ``NaN``/``Infinity``, which is not valid JSON (GB-16).
        """
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return self._to_serializable(obj.tolist())
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (float, np.floating)):
            value = float(obj)
            return value if np.isfinite(value) else None
        return obj

    def _dumps(self, obj: Any) -> str:
        """Strict JSON: serialisable types only, never NaN/Infinity tokens."""
        return json.dumps(self._to_serializable(obj), indent=2, allow_nan=False)
