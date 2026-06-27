"""
Model Drift & Data Quality Monitor
====================================
Detects feature drift, prediction drift, and data quality issues
between a reference (training) dataset and a new production batch.

Usage:
    from src.monitoring.drift_monitor import DriftMonitor
    monitor = DriftMonitor(reference_df, feature_names)
    report = monitor.run(new_df)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ── Thresholds ─────────────────────────────────────────────────────────────────

PSI_THRESHOLDS = {
    "no_drift": 0.10,
    "minor_drift": 0.20,
    "major_drift": float("inf"),
}

KS_ALPHA = 0.05  # significance level for KS test


# ── Drift metrics ──────────────────────────────────────────────────────────────


def compute_psi(reference: np.ndarray, production: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.10  → no drift
    PSI 0.10–0.20 → minor drift (monitor)
    PSI > 0.20  → major drift (retrain)
    """
    ref = reference[~np.isnan(reference)]
    prod = production[~np.isnan(production)]
    if len(ref) == 0 or len(prod) == 0:
        return 0.0

    breakpoints = np.percentile(ref, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    ref_pct = np.histogram(ref, bins=breakpoints)[0] / len(ref)
    prod_pct = np.histogram(prod, bins=breakpoints)[0] / len(prod)

    # Clip to avoid log(0)
    ref_pct = np.clip(ref_pct, 1e-6, None)
    prod_pct = np.clip(prod_pct, 1e-6, None)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return round(float(psi), 4)


def compute_ks(reference: np.ndarray, production: np.ndarray) -> Tuple[float, float]:
    """Kolmogorov-Smirnov test. Returns (statistic, p-value)."""
    ref = reference[~np.isnan(reference)]
    prod = production[~np.isnan(production)]
    if len(ref) < 5 or len(prod) < 5:
        return 0.0, 1.0
    stat, pval = stats.ks_2samp(ref, prod)
    return round(float(stat), 4), round(float(pval), 4)


def psi_label(psi: float) -> str:
    if psi < PSI_THRESHOLDS["no_drift"]:
        return "✅ No Drift"
    if psi < PSI_THRESHOLDS["minor_drift"]:
        return "⚠️ Minor Drift"
    return "🚨 Major Drift"


# ── Report dataclasses ─────────────────────────────────────────────────────────


@dataclass
class FeatureDriftResult:
    feature: str
    psi: float
    psi_label: str
    ks_statistic: float
    ks_pvalue: float
    ref_mean: float
    prod_mean: float
    ref_std: float
    prod_std: float
    drift_detected: bool


@dataclass
class PredictionDriftResult:
    ref_mean_score: float
    prod_mean_score: float
    psi: float
    psi_label: str
    ks_statistic: float
    ks_pvalue: float
    drift_detected: bool


@dataclass
class DataQualityResult:
    feature: str
    ref_null_pct: float
    prod_null_pct: float
    null_increase: float
    ref_min: float
    ref_max: float
    prod_min: float
    prod_max: float
    out_of_range_pct: float
    quality_ok: bool


@dataclass
class MonitoringReport:
    run_timestamp: str = ""
    n_reference: int = 0
    n_production: int = 0
    feature_drift: List[FeatureDriftResult] = field(default_factory=list)
    prediction_drift: Optional[PredictionDriftResult] = None
    data_quality: List[DataQualityResult] = field(default_factory=list)
    drifted_features: List[str] = field(default_factory=list)
    overall_status: str = "OK"
    recommendation: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for fd in self.feature_drift:
            rows.append(
                {
                    "feature": fd.feature,
                    "psi": fd.psi,
                    "status": fd.psi_label,
                    "ks_stat": fd.ks_statistic,
                    "ks_pval": fd.ks_pvalue,
                    "ref_mean": fd.ref_mean,
                    "prod_mean": fd.prod_mean,
                    "drift": fd.drift_detected,
                }
            )
        return pd.DataFrame(rows).sort_values("psi", ascending=False)

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "  MODEL MONITORING REPORT",
            f"  {self.run_timestamp}",
            f"{'='*60}",
            f"  Reference samples : {self.n_reference:,}",
            f"  Production samples: {self.n_production:,}",
            f"  Overall status    : {self.overall_status}",
            f"  Drifted features  : {len(self.drifted_features)}",
        ]
        if self.drifted_features:
            lines.append(f"  Features drifting : {', '.join(self.drifted_features[:5])}")
        if self.prediction_drift:
            pd_result = self.prediction_drift
            lines += [
                "\n  Prediction Drift:",
                f"    Ref avg score  : {pd_result.ref_mean_score:.4f}",
                f"    Prod avg score : {pd_result.prod_mean_score:.4f}",
                f"    PSI            : {pd_result.psi:.4f} ({pd_result.psi_label})",
            ]
        lines += [
            f"\n  Recommendation: {self.recommendation}",
            f"{'='*60}\n",
        ]
        return "\n".join(lines)


# ── Monitor class ──────────────────────────────────────────────────────────────


class DriftMonitor:
    """
    Compares a reference (training) dataset to a production batch
    and generates a comprehensive drift & quality report.
    """

    def __init__(self, reference_df: pd.DataFrame, feature_names: List[str]):
        self.reference_df = reference_df
        self.feature_names = [f for f in feature_names if f in reference_df.columns]
        self._ref_stats = self._compute_stats(reference_df)
        logger.info(
            f"DriftMonitor initialised with {len(reference_df):,} reference samples, "
            f"{len(self.feature_names)} features"
        )

    def _compute_stats(self, df: pd.DataFrame) -> Dict:
        stats_dict = {}
        for feat in self.feature_names:
            if feat in df.columns:
                col = pd.to_numeric(df[feat], errors="coerce")
                stats_dict[feat] = {
                    "mean": float(col.mean()),
                    "std": float(col.std()),
                    "min": float(col.min()),
                    "max": float(col.max()),
                    "null_pct": float(col.isna().mean()),
                    "values": col.dropna().values,
                }
        return stats_dict

    def check_feature_drift(self, production_df: pd.DataFrame) -> List[FeatureDriftResult]:
        results = []
        for feat in self.feature_names:
            if feat not in production_df.columns:
                continue
            ref_vals = self._ref_stats[feat]["values"]
            prod_col = pd.to_numeric(production_df[feat], errors="coerce")
            prod_vals = prod_col.dropna().values

            psi = compute_psi(ref_vals, prod_vals)
            ks_stat, ks_pval = compute_ks(ref_vals, prod_vals)
            drift_detected = psi >= PSI_THRESHOLDS["minor_drift"] or ks_pval < KS_ALPHA

            results.append(
                FeatureDriftResult(
                    feature=feat,
                    psi=psi,
                    psi_label=psi_label(psi),
                    ks_statistic=ks_stat,
                    ks_pvalue=ks_pval,
                    ref_mean=round(self._ref_stats[feat]["mean"], 4),
                    prod_mean=round(float(prod_col.mean()), 4),
                    ref_std=round(self._ref_stats[feat]["std"], 4),
                    prod_std=round(float(prod_col.std()), 4),
                    drift_detected=drift_detected,
                )
            )
        return sorted(results, key=lambda x: x.psi, reverse=True)

    def check_prediction_drift(
        self,
        ref_scores: np.ndarray,
        prod_scores: np.ndarray,
    ) -> PredictionDriftResult:
        psi = compute_psi(ref_scores, prod_scores)
        ks_stat, ks_pval = compute_ks(ref_scores, prod_scores)
        return PredictionDriftResult(
            ref_mean_score=round(float(ref_scores.mean()), 4),
            prod_mean_score=round(float(prod_scores.mean()), 4),
            psi=psi,
            psi_label=psi_label(psi),
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            drift_detected=psi >= PSI_THRESHOLDS["minor_drift"],
        )

    def check_data_quality(self, production_df: pd.DataFrame) -> List[DataQualityResult]:
        results = []
        for feat in self.feature_names:
            if feat not in production_df.columns:
                continue
            prod_col = pd.to_numeric(production_df[feat], errors="coerce")
            ref_stat = self._ref_stats[feat]

            prod_null_pct = float(prod_col.isna().mean())
            null_increase = prod_null_pct - ref_stat["null_pct"]

            prod_vals = prod_col.dropna().values
            if len(prod_vals) > 0:
                out_of_range = (
                    (prod_vals < ref_stat["min"] - 3 * ref_stat["std"])
                    | (prod_vals > ref_stat["max"] + 3 * ref_stat["std"])
                ).mean()
                prod_min = float(prod_vals.min())
                prod_max = float(prod_vals.max())
            else:
                out_of_range = 1.0
                prod_min = prod_max = float("nan")

            quality_ok = (null_increase < 0.10) and (out_of_range < 0.05)

            results.append(
                DataQualityResult(
                    feature=feat,
                    ref_null_pct=round(ref_stat["null_pct"], 4),
                    prod_null_pct=round(prod_null_pct, 4),
                    null_increase=round(null_increase, 4),
                    ref_min=round(ref_stat["min"], 4),
                    ref_max=round(ref_stat["max"], 4),
                    prod_min=round(prod_min, 4) if not np.isnan(prod_min) else float("nan"),
                    prod_max=round(prod_max, 4) if not np.isnan(prod_max) else float("nan"),
                    out_of_range_pct=round(float(out_of_range), 4),
                    quality_ok=quality_ok,
                )
            )
        return results

    def run(
        self,
        production_df: pd.DataFrame,
        ref_scores: np.ndarray = None,
        prod_scores: np.ndarray = None,
    ) -> MonitoringReport:
        from datetime import datetime

        report = MonitoringReport(
            run_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            n_reference=len(self.reference_df),
            n_production=len(production_df),
        )

        # Feature drift
        report.feature_drift = self.check_feature_drift(production_df)
        report.drifted_features = [r.feature for r in report.feature_drift if r.drift_detected]

        # Prediction drift
        if ref_scores is not None and prod_scores is not None:
            report.prediction_drift = self.check_prediction_drift(ref_scores, prod_scores)

        # Data quality
        report.data_quality = self.check_data_quality(production_df)

        # Overall status
        n_major = sum(1 for r in report.feature_drift if r.psi >= PSI_THRESHOLDS["minor_drift"])
        pred_drift = report.prediction_drift and report.prediction_drift.drift_detected
        quality_fails = sum(1 for q in report.data_quality if not q.quality_ok)

        if n_major >= 5 or pred_drift:
            report.overall_status = "🚨 RETRAIN REQUIRED"
            report.recommendation = (
                f"{n_major} features show major drift. "
                "Trigger model retraining pipeline immediately."
            )
        elif n_major >= 2 or quality_fails >= 3:
            report.overall_status = "⚠️ MONITOR CLOSELY"
            report.recommendation = (
                f"{n_major} features drifting, {quality_fails} quality issues. "
                "Schedule retraining within 2 weeks."
            )
        else:
            report.overall_status = "✅ HEALTHY"
            report.recommendation = (
                "Model performing within expected parameters. Next review in 30 days."
            )

        logger.info(f"Monitoring complete: {report.overall_status}")
        return report

    def save_report(self, report: MonitoringReport, output_dir: str = "reports/monitoring"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ts = report.run_timestamp.replace(" ", "_").replace(":", "-")

        # Feature drift CSV
        drift_df = report.to_dataframe()
        drift_df.to_csv(f"{output_dir}/feature_drift_{ts}.csv", index=False)

        # Text summary
        with open(f"{output_dir}/monitoring_summary_{ts}.txt", "w") as f:
            f.write(report.summary())

        logger.info(f"Monitoring report saved to {output_dir}/")
        return drift_df


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    import warnings

    warnings.filterwarnings("ignore")

    from src.data.generate_data import generate_dataset
    from src.features.feature_engineering import MODEL_FEATURES, engineer_features

    logging.basicConfig(level=logging.INFO)

    print("Generating reference dataset (training)...")
    df_ref_raw = generate_dataset(n=2000, output_path=None)
    df_ref = engineer_features(df_ref_raw)

    print("Generating simulated production batch (with slight drift)...")
    df_prod_raw = generate_dataset(n=500, output_path=None)
    # Inject drift: inflate billing visits and reduce sessions
    df_prod_raw["billing_page_visits"] = df_prod_raw["billing_page_visits"] * 2
    df_prod_raw["sessions_30d"] = (df_prod_raw["sessions_30d"] * 0.5).astype(int)
    df_prod = engineer_features(df_prod_raw)

    monitor = DriftMonitor(df_ref, MODEL_FEATURES)
    report = monitor.run(df_prod)

    print(report.summary())
    drift_df = monitor.save_report(report)
    print("\nTop drifted features:")
    print(drift_df[drift_df["drift"]].head(10).to_string(index=False))
