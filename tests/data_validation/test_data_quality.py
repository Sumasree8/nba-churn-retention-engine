"""
Data Validation Tests
======================
Great Expectations–style validation for the raw and engineered datasets.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.generate_data import generate_dataset


@pytest.fixture(scope="module")
def raw_df():
    return generate_dataset(n=500)


class TestRawDataSchema:
    REQUIRED_COLUMNS = [
        "customer_id",
        "subscription_type",
        "monthly_revenue",
        "tenure_months",
        "country",
        "industry",
        "sessions_30d",
        "sessions_90d",
        "days_since_last_login",
        "ticket_count",
        "support_sentiment",
        "billing_page_visits",
        "downgrade_page_visits",
        "tier_capacity_utilization",
        "churned",
    ]

    def test_required_columns_present(self, raw_df):
        for col in self.REQUIRED_COLUMNS:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_no_duplicate_customer_ids(self, raw_df):
        assert raw_df["customer_id"].nunique() == len(raw_df)

    def test_churn_binary(self, raw_df):
        assert set(raw_df["churned"].unique()).issubset({0, 1})

    def test_churn_rate_reasonable(self, raw_df):
        rate = raw_df["churned"].mean()
        assert 0.05 <= rate <= 0.45, f"Churn rate {rate:.2%} outside expected range"

    def test_mrr_positive(self, raw_df):
        assert (raw_df["monthly_revenue"] > 0).all()

    def test_tenure_positive(self, raw_df):
        assert (raw_df["tenure_months"] >= 1).all()

    def test_subscription_types_valid(self, raw_df):
        valid = {"Starter", "Professional", "Business", "Enterprise"}
        assert set(raw_df["subscription_type"].unique()).issubset(valid)

    def test_sentiment_range(self, raw_df):
        assert (raw_df["support_sentiment"] >= -1).all()
        assert (raw_df["support_sentiment"] <= 1).all()

    def test_no_null_critical_columns(self, raw_df):
        for col in ["customer_id", "monthly_revenue", "churned"]:
            assert raw_df[col].isna().sum() == 0, f"Nulls in {col}"

    def test_sessions_non_negative(self, raw_df):
        assert (raw_df["sessions_30d"] >= 0).all()
        assert (raw_df["sessions_90d"] >= 0).all()

    def test_capacity_util_range(self, raw_df):
        assert (raw_df["tier_capacity_utilization"] >= 0).all()
        assert (raw_df["tier_capacity_utilization"] <= 1).all()

    def test_record_count(self, raw_df):
        assert len(raw_df) >= 100, "Should have at least 100 records"

    def test_class_imbalance_exists(self, raw_df):
        """Churned class should be minority (<50%)."""
        churn_pct = raw_df["churned"].mean()
        assert churn_pct < 0.5, "Churned should be minority class"


class TestDataDistributions:
    def test_mrr_distribution_by_tier(self, raw_df):
        starter_mrr = raw_df[raw_df["subscription_type"] == "Starter"]["monthly_revenue"]
        enterprise_mrr = raw_df[raw_df["subscription_type"] == "Enterprise"]["monthly_revenue"]
        assert starter_mrr.mean() < enterprise_mrr.mean(), "Enterprise MRR should be higher"

    def test_countries_present(self, raw_df):
        assert raw_df["country"].nunique() >= 5, "Should have at least 5 countries"

    def test_industries_present(self, raw_df):
        assert raw_df["industry"].nunique() >= 5, "Should have at least 5 industries"
