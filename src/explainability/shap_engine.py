"""
SHAP Explainability Engine
===========================
Generates global and per-customer explanations using SHAP.
Produces human-readable explanation narratives for each high-risk customer.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # headless backend — must be set before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports/shap")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── SHAP Engine ────────────────────────────────────────────────────────────────


class SHAPExplainer:
    """Wraps SHAP TreeExplainer (or KernelExplainer fallback) for churn models."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.base_value = None

    def fit(self, X_background: np.ndarray, model_type: str = "tree"):
        """Initialise the SHAP explainer."""
        if model_type == "tree":
            try:
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Using TreeExplainer")
            except Exception:
                logger.warning("TreeExplainer failed — falling back to KernelExplainer")
                background = shap.sample(X_background, 100)
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
        else:
            background = shap.sample(X_background, 100)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
        return self

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for X. Returns array shape (n, features)."""
        sv = self.explainer.shap_values(X)
        # Binary classification — take class-1 SHAP values.
        # SHAP returns different shapes across versions/models:
        #   • list [class0, class1]                      (older API)
        #   • ndarray (n_samples, n_features, n_classes) (SHAP >=0.45 for sklearn trees)
        #   • ndarray (n_samples, n_features)            (single-output, e.g. XGBoost binary)
        if isinstance(sv, list):
            self.shap_values = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            self.shap_values = sv[:, :, -1]
        else:
            self.shap_values = sv
        try:
            self.base_value = self.explainer.expected_value[1]
        except (TypeError, IndexError):
            self.base_value = float(self.explainer.expected_value)
        return self.shap_values

    # ── Plot helpers ───────────────────────────────────────────────────

    def plot_summary(self, X: np.ndarray, save_path: str = None):
        """Global SHAP summary (beeswarm) plot."""
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Summary plot saved to {save_path}")
        plt.close()

    def plot_bar_importance(self, save_path: str = None):
        """Global feature importance bar chart."""
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = [
            "#e63946" if v > importance_df["mean_abs_shap"].median() else "#457b9d"
            for v in importance_df["mean_abs_shap"]
        ]
        ax.barh(
            importance_df["feature"][:20], importance_df["mean_abs_shap"][:20], color=colors[:20]
        )
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Global Feature Importance (SHAP)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return importance_df

    def plot_waterfall(self, idx: int, X: np.ndarray, save_path: str = None):
        """Waterfall plot for a single customer."""
        sv = self.shap_values[idx]
        exp = shap.Explanation(
            values=sv,
            base_values=self.base_value,
            data=X[idx],
            feature_names=self.feature_names,
        )
        shap.plots.waterfall(exp, show=False)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    # ── Narrative generation ───────────────────────────────────────────

    def explain_customer(
        self,
        customer_id: str,
        idx: int,
        X_row: np.ndarray,
        churn_prob: float,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """Return structured + narrative explanation for one customer."""
        sv = self.shap_values[idx]
        sorted_idx = np.argsort(np.abs(sv))[::-1]

        top_drivers = []
        for rank, fi in enumerate(sorted_idx[:top_n]):
            direction = "increases" if sv[fi] > 0 else "decreases"
            top_drivers.append(
                {
                    "rank": rank + 1,
                    "feature": self.feature_names[fi],
                    "shap_value": round(float(sv[fi]), 4),
                    "feature_value": round(float(X_row[fi]), 4),
                    "direction": direction,
                }
            )

        primary_driver = top_drivers[0]["feature"]
        bullet_lines = "\n".join(
            f"  • {d['feature']} (SHAP={d['shap_value']:+.3f}) — {d['direction']} churn risk"
            for d in top_drivers
        )
        narrative = (
            f"Customer {customer_id} has a {churn_prob:.0%} churn probability.\n"
            f"Top drivers:\n{bullet_lines}"
        )

        return {
            "customer_id": customer_id,
            "churn_probability": round(churn_prob, 4),
            "primary_driver": primary_driver,
            "top_drivers": top_drivers,
            "narrative": narrative,
        }

    def explain_batch(
        self,
        customer_ids: List[str],
        X: np.ndarray,
        churn_probs: np.ndarray,
        churn_threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Explain all high-risk customers; return DataFrame."""
        records = []
        high_risk_idx = np.where(churn_probs >= churn_threshold)[0]
        logger.info(f"Explaining {len(high_risk_idx)} high-risk customers …")

        for idx in high_risk_idx:
            exp = self.explain_customer(
                customer_id=customer_ids[idx],
                idx=idx,
                X_row=X[idx],
                churn_prob=churn_probs[idx],
            )
            flat = {
                "customer_id": exp["customer_id"],
                "churn_probability": exp["churn_probability"],
                "primary_driver": exp["primary_driver"],
            }
            for d in exp["top_drivers"]:
                flat[f"driver_{d['rank']}_feature"] = d["feature"]
                flat[f"driver_{d['rank']}_shap"] = d["shap_value"]
            records.append(flat)

        return pd.DataFrame(records)

    def get_global_importance(self) -> pd.DataFrame:
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "mean_abs_shap": mean_abs,
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def save_global_importance(self, path: str = "reports/shap/global_importance.csv"):
        df = self.get_global_importance()
        df.to_csv(path, index=False)
        logger.info(f"Global importance saved to {path}")
        return df
