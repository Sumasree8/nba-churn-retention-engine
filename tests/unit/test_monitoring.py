"""
Monitoring Tests — Drift Detection & Data Quality
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.generate_data import generate_dataset
from src.features.feature_engineering import MODEL_FEATURES, engineer_features
from src.monitoring.drift_monitor import (
    PSI_THRESHOLDS,
    DriftMonitor,
    compute_ks,
    compute_psi,
    psi_label,
)


@pytest.fixture(scope="module")
def reference_df():
    np.random.seed(0)
    df = generate_dataset(n=500)
    return engineer_features(df)


@pytest.fixture(scope="module")
def production_df_clean():
    np.random.seed(99)
    df = generate_dataset(n=200)
    return engineer_features(df)


@pytest.fixture(scope="module")
def production_df_drifted():
    np.random.seed(7)
    df = generate_dataset(n=200)
    eng = engineer_features(df)
    # Artificially drift key features
    eng["billing_page_visits"] = eng["billing_page_visits"] * 4
    eng["sessions_30d"] = (eng["sessions_30d"] * 0.2).astype(int)
    eng["engagement_decay_score"] = (eng["engagement_decay_score"] * 1.5).clip(0, 1)
    return eng


class TestPSIMetric:
    def test_identical_distributions_psi_near_zero(self):
        arr = np.random.normal(0, 1, 1000)
        psi = compute_psi(arr, arr)
        assert psi < 0.05, f"PSI of identical distributions should be near 0, got {psi}"

    def test_very_different_distributions_high_psi(self):
        ref = np.random.normal(0, 1, 1000)
        prod = np.random.normal(5, 1, 1000)
        psi = compute_psi(ref, prod)
        assert (
            psi > PSI_THRESHOLDS["minor_drift"]
        ), "Very different distributions should have high PSI"

    def test_psi_non_negative(self):
        ref = np.random.uniform(0, 1, 500)
        prod = np.random.uniform(0, 2, 500)
        assert compute_psi(ref, prod) >= 0

    def test_psi_label_thresholds(self):
        assert "No Drift" in psi_label(0.05)
        assert "Minor" in psi_label(0.15)
        assert "Major" in psi_label(0.25)

    def test_psi_handles_empty(self):
        psi = compute_psi(np.array([]), np.array([1, 2, 3]))
        assert psi == 0.0

    def test_psi_handles_nan(self):
        ref = np.array([1.0, 2.0, float("nan"), 3.0])
        prod = np.array([1.5, 2.5, float("nan"), 3.5])
        psi = compute_psi(ref, prod)
        assert psi >= 0


class TestKSMetric:
    def test_same_distribution_high_pvalue(self):
        arr = np.random.normal(0, 1, 500)
        stat, pval = compute_ks(arr, arr)
        assert pval > 0.05, "Identical distributions should not reject H0"

    def test_different_distributions_low_pvalue(self):
        ref = np.random.normal(0, 1, 500)
        prod = np.random.normal(10, 1, 500)
        stat, pval = compute_ks(ref, prod)
        assert pval < 0.01, "Very different distributions should reject H0"

    def test_returns_tuple(self):
        result = compute_ks(np.random.normal(0, 1, 100), np.random.normal(0, 1, 100))
        assert len(result) == 2


class TestDriftMonitor:
    def test_init_with_valid_data(self, reference_df):
        monitor = DriftMonitor(reference_df, MODEL_FEATURES)
        assert len(monitor.feature_names) > 0

    def test_check_feature_drift_returns_results(self, reference_df, production_df_clean):
        monitor = DriftMonitor(reference_df, MODEL_FEATURES)
        results = monitor.check_feature_drift(production_df_clean)
        assert len(results) > 0
        assert all(hasattr(r, "psi") for r in results)
        assert all(hasattr(r, "feature") for r in results)

    def test_clean_data_mostly_no_drift(self, reference_df, production_df_clean):
        monitor = DriftMonitor(reference_df, MODEL_FEATURES)
        results = monitor.check_feature_drift(production_df_clean)
        drifted = [r for r in results if r.drift_detected]
        # Clean production data should have minimal drift
        drift_rate = len(drifted) / max(len(results), 1)
        assert drift_rate < 0.50, f"Too many features drifting in clean data: {drift_rate:.0%}"

    def test_drifted_data_detected(self, reference_df, production_df_drifted):
        monitor = DriftMonitor(reference_df, MODEL_FEATURES)
        results = monitor.check_feature_drift(production_df_drifted)
        drifted = [r for r in results if r.drift_detected]
        assert len(drifted) >= 1, "Should detect at least 1 drifted feature in drifted data"

    def test_prediction_drift_check(self, reference_df):
        monitor = DriftMonitor(reference_df, MODEL_FEATURES)
        ref_scores = np.random.beta(3, 7, 500)
        prod_scores_same = np.random.beta(3, 7, 200)
        prod_scores_diff = np.random.beta(8, 2, 200)

        result_clean = monitor.check_prediction_drift(ref_scores, prod_scores_same)
        result_drift = monitor.check_prediction_drift(ref_scores, prod_scores_diff)

        assert not result_clean.drift_detected or result_clean.psi < result_drift.psi
        assert result_drift.psi > result_clean.psi

    def test_run_returns_monitoring_report(self, reference_df, production_df_clean):
        monitor = DriftMonitor(reference_df, MODEL_FEATURES)
        report = monitor.run(production_df_clean)
        assert report.n_reference == len(reference_df)
        assert report.n_production == len(production_df_clean)
        assert report.overall_status in {"✅ HEALTHY", "⚠️ MONITOR CLOSELY", "🚨 RETRAIN REQUIRED"}
        assert len(report.feature_drift) > 0

    def test_report_to_dataframe(self, reference_df, production_df_clean):
        monitor = DriftMonitor(reference_df, MODEL_FEATURES)
        report = monitor.run(production_df_clean)
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "psi" in df.columns

    def test_data_quality_check(self, reference_df, production_df_clean):
        monitor = DriftMonitor(reference_df, MODEL_FEATURES)
        quality = monitor.check_data_quality(production_df_clean)
        assert len(quality) > 0
        assert all(hasattr(q, "feature") for q in quality)
        assert all(hasattr(q, "quality_ok") for q in quality)
