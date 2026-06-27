"""
Unit Tests — Feature Engineering & Archetype Classification
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.clv.clv_calculator import add_clv_columns, assign_clv_tier, calculate_clv
from src.features.feature_engineering import MODEL_FEATURES, FeatureEngineer, engineer_features
from src.nba_engine.archetype_classifier import ARCHETYPES, classify_archetypes
from src.nba_engine.nba_engine import (
    NBAEngine,
    add_revenue_impact,
    calculate_expected_revenue_saved,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_raw_df():
    np.random.seed(0)
    n = 100
    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:04d}" for i in range(n)],
            "subscription_type": np.random.choice(
                ["Starter", "Professional", "Business", "Enterprise"], n
            ),
            "monthly_revenue": np.random.uniform(29, 999, n),
            "tenure_months": np.random.randint(1, 60, n),
            "country": "US",
            "industry": "SaaS",
            "sessions_30d": np.random.randint(0, 60, n),
            "sessions_90d": np.random.randint(0, 150, n),
            "days_since_last_login": np.random.randint(0, 90, n),
            "days_since_last_core_action": np.random.randint(0, 90, n),
            "login_frequency_change": np.random.uniform(-1, 1, n),
            "session_duration_change": np.random.uniform(-1, 1, n),
            "ticket_count": np.random.randint(0, 15, n),
            "support_sentiment": np.random.uniform(-1, 1, n),
            "error_rate": np.random.uniform(0, 0.3, n),
            "failed_api_calls": np.random.randint(0, 20, n),
            "rage_click_count": np.random.randint(0, 10, n),
            "billing_page_visits": np.random.randint(0, 10, n),
            "downgrade_page_visits": np.random.randint(0, 5, n),
            "invoice_download_frequency": np.random.randint(0, 5, n),
            "trial_expiry_days": np.random.choice([-1, 7, 14], n),
            "tier_capacity_utilization": np.random.uniform(0.1, 0.99, n),
            "export_frequency": np.random.randint(0, 20, n),
            "api_usage": np.random.randint(0, 5000, n),
            "api_limit_usage": np.random.uniform(0, 1, n),
            "seat_limit_usage": np.random.uniform(0.1, 1, n),
            "churned": np.random.randint(0, 2, n),
        }
    )


@pytest.fixture
def sample_engineered_df(sample_raw_df):
    return engineer_features(sample_raw_df)


# ── Feature Engineering Tests ──────────────────────────────────────────────────


class TestFeatureEngineering:
    def test_output_shape(self, sample_raw_df, sample_engineered_df):
        assert len(sample_engineered_df) == len(sample_raw_df)
        assert sample_engineered_df.shape[1] > sample_raw_df.shape[1]

    def test_velocity_range(self, sample_engineered_df):
        v = sample_engineered_df["velocity_30d_vs_90d"]
        assert (v >= 0).all(), "velocity must be non-negative"
        assert (v <= 5).all(), "velocity clipped to 5"

    def test_scores_in_unit_interval(self, sample_engineered_df):
        score_cols = [
            "engagement_decay_score",
            "friction_score",
            "pricing_sensitivity_score",
            "growth_pressure_score",
            "customer_health_score",
            "retention_risk_score",
        ]
        for col in score_cols:
            assert col in sample_engineered_df.columns, f"Missing column: {col}"
            vals = sample_engineered_df[col]
            assert (vals >= 0).all() and (vals <= 1).all(), f"{col} out of [0,1]"

    def test_model_features_present(self, sample_engineered_df):
        for feat in MODEL_FEATURES:
            assert feat in sample_engineered_df.columns, f"Missing model feature: {feat}"

    def test_no_infinite_values(self, sample_engineered_df):
        numeric = sample_engineered_df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Infinite values found"

    def test_fit_transform_consistent(self, sample_raw_df):
        fe = FeatureEngineer()
        fe.fit(sample_raw_df)
        df1 = fe.transform(sample_raw_df)
        df2 = fe.transform(sample_raw_df)
        pd.testing.assert_frame_equal(df1, df2)


# ── Archetype Tests ────────────────────────────────────────────────────────────


class TestArchetypeClassifier:
    def test_all_archetypes_present(self):
        assert set(ARCHETYPES.keys()) == {"A", "B", "C", "D"}

    def test_archetype_column_created(self, sample_engineered_df):
        df = classify_archetypes(sample_engineered_df)
        assert "churn_archetype" in df.columns

    def test_valid_archetype_values(self, sample_engineered_df):
        df = classify_archetypes(sample_engineered_df)
        assert df["churn_archetype"].isin(["A", "B", "C", "D"]).all()

    def test_archetype_confidence_range(self, sample_engineered_df):
        df = classify_archetypes(sample_engineered_df)
        assert (df["archetype_confidence"] >= 0).all()
        assert (df["archetype_confidence"] <= 1).all()

    def test_ghost_high_decay(self, sample_raw_df):
        df = sample_raw_df.copy()
        df.loc[0, "days_since_last_login"] = 80
        df.loc[0, "sessions_30d"] = 0
        df.loc[0, "login_frequency_change"] = -0.9
        df_eng = engineer_features(df)
        df_class = classify_archetypes(df_eng)
        assert df_class.iloc[0]["churn_archetype"] == "A", "High-decay user should be Ghost"


# ── CLV Tests ──────────────────────────────────────────────────────────────────


class TestCLVCalculator:
    def test_clv_positive(self):
        clv = calculate_clv(200, 0.2)
        assert clv > 0

    def test_high_churn_lower_clv(self):
        clv_low = calculate_clv(200, 0.1)
        clv_high = calculate_clv(200, 0.8)
        assert clv_low > clv_high

    def test_tier_assignment(self):
        assert assign_clv_tier(10000) == "Enterprise"
        assert assign_clv_tier(2000) == "High"
        assert assign_clv_tier(800) == "Medium"
        assert assign_clv_tier(100) == "Low"

    def test_clv_columns_added(self, sample_raw_df):
        df = sample_raw_df.copy()
        df["churn_probability"] = 0.3
        df = add_clv_columns(df)
        assert "clv" in df.columns
        assert "clv_tier" in df.columns


# ── NBA Engine Tests ───────────────────────────────────────────────────────────


class TestNBAEngine:
    def test_revenue_saved_formula(self):
        saved = calculate_expected_revenue_saved(0.8, 100, 0.25, 12)
        assert abs(saved - 240.0) < 0.01

    def test_zero_churn_zero_saved(self):
        assert calculate_expected_revenue_saved(0.0, 100, 0.25) == 0.0

    def test_fallback_action_returned(self, sample_engineered_df):
        engine = NBAEngine()
        # Provide a minimal row with archetype Z (unknown → fallback)
        row = sample_engineered_df.iloc[0].copy()
        result = engine.recommend(row, archetype="A")
        assert "action" in result
        assert "action_code" in result

    def test_batch_recommend_shape(self, sample_engineered_df):
        df = sample_engineered_df.copy()
        df["churn_probability"] = 0.6
        df = classify_archetypes(df)
        engine = NBAEngine()
        result = engine.recommend_batch(df)
        assert len(result) == len(df)
        assert "action" in result.columns

    def test_revenue_impact_columns(self, sample_engineered_df):
        df = sample_engineered_df.copy()
        df["churn_probability"] = 0.6
        df["expected_conversion_rate"] = 0.2
        df["action_cost_usd"] = 5.0
        df = add_revenue_impact(df)
        assert "expected_revenue_saved" in df.columns
        assert "campaign_roi" in df.columns
        assert (df["expected_revenue_saved"] >= 0).all()
