"""
Integration Tests — End-to-End Pipeline
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.business_impact.impact_engine import simulate_business_impact
from src.clv.clv_calculator import add_clv_columns
from src.data.generate_data import generate_dataset
from src.features.feature_engineering import MODEL_FEATURES, engineer_features
from src.nba_engine.archetype_classifier import classify_archetypes
from src.nba_engine.nba_engine import NBAEngine, run_nba_pipeline


@pytest.fixture(scope="module")
def pipeline_output():
    """Run the full pipeline (light version) and return results."""
    df_raw = generate_dataset(n=500)
    df_eng = engineer_features(df_raw)

    # Fake churn probabilities (skip model training in integration test)
    np.random.seed(42)
    df_eng["churn_probability"] = np.random.beta(3, 7, len(df_eng))
    df_eng["primary_driver"] = "engagement_decay_score"

    df_high_risk = df_eng[df_eng["churn_probability"] >= 0.4].copy().reset_index(drop=True)
    df_classified = classify_archetypes(df_high_risk)
    df_clv = add_clv_columns(df_classified)
    df_nba = run_nba_pipeline(df_clv)
    report, df_impact = simulate_business_impact(df_nba)

    return {
        "df_raw": df_raw,
        "df_eng": df_eng,
        "df_classified": df_classified,
        "df_nba": df_nba,
        "df_impact": df_impact,
        "report": report,
    }


class TestEndToEndPipeline:
    def test_data_generated(self, pipeline_output):
        df = pipeline_output["df_raw"]
        assert len(df) == 500
        assert "churned" in df.columns

    def test_feature_engineering_produces_expected_columns(self, pipeline_output):
        df = pipeline_output["df_eng"]
        for feat in MODEL_FEATURES:
            assert feat in df.columns, f"Missing: {feat}"

    def test_high_risk_filtered(self, pipeline_output):
        df = pipeline_output["df_classified"]
        assert (df["churn_probability"] >= 0.4).all()

    def test_archetypes_assigned_to_all(self, pipeline_output):
        df = pipeline_output["df_classified"]
        assert df["churn_archetype"].notna().all()
        assert df["churn_archetype"].isin(["A", "B", "C", "D"]).all()

    def test_nba_action_assigned_to_all(self, pipeline_output):
        df = pipeline_output["df_nba"]
        assert "action" in df.columns
        assert df["action"].notna().all()

    def test_revenue_saved_non_negative(self, pipeline_output):
        df = pipeline_output["df_impact"]
        assert (df["revenue_saved"] >= 0).all()

    def test_impact_report_totals_positive(self, pipeline_output):
        report = pipeline_output["report"]
        assert report.total_at_risk_users > 0
        assert report.expected_revenue_saved >= 0
        assert report.total_campaign_cost >= 0

    def test_clv_tiers_present(self, pipeline_output):
        df = pipeline_output["df_nba"]
        assert "clv_tier" in df.columns
        assert df["clv_tier"].isin(["Low", "Medium", "High", "Enterprise"]).all()

    def test_all_archetypes_produce_valid_actions(self):
        """Each archetype should get a recommendation, even without matching conditions."""
        engine = NBAEngine()
        for arch in ["A", "B", "C", "D"]:
            row = pd.Series(
                {
                    "engagement_decay_score": 0.1,
                    "friction_score": 0.1,
                    "pricing_sensitivity_score": 0.1,
                    "growth_pressure_score": 0.1,
                    "days_since_last_login": 5,
                    "ticket_count": 1,
                    "billing_page_visits": 1,
                    "downgrade_page_visits": 0,
                    "tier_capacity_utilization": 0.3,
                    "api_limit_usage": 0.2,
                    "support_sentiment": 0.1,
                    "error_rate": 0.02,
                    "rage_click_count": 0,
                    "seat_limit_usage": 0.3,
                }
            )
            rec = engine.recommend(row, archetype=arch)
            assert "action" in rec
            assert "action_code" in rec
            assert len(rec["action"]) > 0

    def test_pipeline_output_columns(self, pipeline_output):
        df = pipeline_output["df_nba"]
        required_cols = [
            "customer_id",
            "churn_probability",
            "churn_archetype",
            "archetype_name",
            "action",
            "urgency",
            "expected_conversion_rate",
            "action_cost_usd",
            "clv",
            "clv_tier",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing output column: {col}"

    def test_revenue_saved_correlates_with_mrr(self, pipeline_output):
        """Higher MRR customers should generally have higher revenue saved."""
        df = pipeline_output["df_impact"]
        corr = df["monthly_revenue"].corr(df["revenue_saved"])
        assert corr > 0, "Revenue saved should positively correlate with MRR"
