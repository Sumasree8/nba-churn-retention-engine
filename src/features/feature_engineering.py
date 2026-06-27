"""
Feature Engineering Pipeline
==============================
Transforms raw SaaS behavioral data into ML-ready features.
Creates advanced derived features for churn modelling and archetype classification.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Core feature engineering transformer — sklearn compatible."""

    def fit(self, X: pd.DataFrame, y=None):
        self._subscription_encoder = LabelEncoder()
        self._subscription_encoder.fit(X["subscription_type"])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # ── 1. Velocity: 30d vs 90d engagement ratio ──────────────────
        df["velocity_30d_vs_90d"] = np.where(
            df["sessions_90d"] > 0,
            df["sessions_30d"] / (df["sessions_90d"] / 3 + 1e-6),
            0.0,
        )
        df["velocity_30d_vs_90d"] = df["velocity_30d_vs_90d"].clip(0, 5)

        # ── 2. Engagement decay score (Ghost archetype signal) ─────────
        df["engagement_decay_score"] = (
            0.35 * (df["days_since_last_login"] / 90)
            + 0.35 * (df["days_since_last_core_action"] / 90)
            + 0.15 * (1 - df["velocity_30d_vs_90d"].clip(0, 1))
            + 0.15
            * np.where(df["login_frequency_change"] < 0, abs(df["login_frequency_change"]), 0)
        ).clip(0, 1)

        # ── 3. Friction score (Frustrated Professional signal) ─────────
        ticket_density_norm = (df["ticket_count"] / 20).clip(0, 1)
        df["ticket_density_30d"] = df["ticket_count"]
        df["friction_score"] = (
            0.30 * ticket_density_norm
            + 0.25 * df["error_rate"]
            + 0.20 * (df["failed_api_calls"] / 50).clip(0, 1)
            + 0.15 * (df["rage_click_count"] / 30).clip(0, 1)
            + 0.10 * np.where(df["support_sentiment"] < 0, abs(df["support_sentiment"]), 0)
        ).clip(0, 1)

        # ── 4. Pricing sensitivity score (Price-Sensitive Optimizer) ───
        df["pricing_sensitivity_score"] = (
            0.35 * (df["billing_page_visits"] / 20).clip(0, 1)
            + 0.35 * (df["downgrade_page_visits"] / 10).clip(0, 1)
            + 0.20 * (df["invoice_download_frequency"] / 12).clip(0, 1)
            + 0.10
            * np.where(
                df["trial_expiry_days"] >= 0, (14 - df["trial_expiry_days"].clip(0, 14)) / 14, 0
            )
        ).clip(0, 1)

        # ── 5. Growth pressure score (Outgrown User signal) ───────────
        df["growth_pressure_score"] = (
            0.35 * df["tier_capacity_utilization"]
            + 0.25 * df["api_limit_usage"]
            + 0.25 * df["seat_limit_usage"]
            + 0.15 * (df["export_frequency"] / 40).clip(0, 1)
        ).clip(0, 1)

        # ── 6. Customer health score (composite) ──────────────────────
        df["customer_health_score"] = (
            0.30 * (1 - df["engagement_decay_score"])
            + 0.25 * (1 - df["friction_score"])
            + 0.25 * (1 - df["pricing_sensitivity_score"])
            + 0.20 * (df["sessions_30d"] / 120).clip(0, 1)
        ).clip(0, 1)

        # ── 7. Retention risk score ────────────────────────────────────
        df["retention_risk_score"] = (
            0.30 * df["engagement_decay_score"]
            + 0.25 * df["friction_score"]
            + 0.25 * df["pricing_sensitivity_score"]
            + 0.20 * (1 - df["customer_health_score"])
        ).clip(0, 1)

        # ── 8. Support ticket sentiment (normalised) ───────────────────
        df["support_ticket_sentiment"] = df["support_sentiment"]

        # ── 9. Subscription type encoding ─────────────────────────────
        sub_order = {"Starter": 0, "Professional": 1, "Business": 2, "Enterprise": 3}
        df["subscription_tier"] = df["subscription_type"].map(sub_order).fillna(0).astype(int)

        # ── 10. Tenure buckets ─────────────────────────────────────────
        df["tenure_bucket"] = pd.cut(
            df["tenure_months"],
            bins=[0, 3, 12, 24, 84],
            labels=[0, 1, 2, 3],
        ).astype(int)

        # ── 11. Log-transform heavy-tailed features ────────────────────
        df["log_api_usage"] = np.log1p(df["api_usage"])
        df["log_monthly_revenue"] = np.log1p(df["monthly_revenue"])

        logger.debug(f"Feature engineering produced {df.shape[1]} columns")
        return df

    def get_feature_names(self) -> list:
        return ENGINEERED_FEATURES + BASE_FEATURES


# Feature groups used downstream
BASE_FEATURES = [
    "sessions_30d",
    "sessions_90d",
    "days_since_last_login",
    "days_since_last_core_action",
    "login_frequency_change",
    "session_duration_change",
    "ticket_count",
    "support_sentiment",
    "error_rate",
    "failed_api_calls",
    "rage_click_count",
    "billing_page_visits",
    "downgrade_page_visits",
    "invoice_download_frequency",
    "tier_capacity_utilization",
    "export_frequency",
    "api_usage",
    "api_limit_usage",
    "seat_limit_usage",
    "tenure_months",
    "monthly_revenue",
]

ENGINEERED_FEATURES = [
    "velocity_30d_vs_90d",
    "engagement_decay_score",
    "friction_score",
    "pricing_sensitivity_score",
    "growth_pressure_score",
    "customer_health_score",
    "retention_risk_score",
    "support_ticket_sentiment",
    "ticket_density_30d",
    "subscription_tier",
    "tenure_bucket",
    "log_api_usage",
    "log_monthly_revenue",
]

MODEL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES


def build_feature_pipeline() -> Pipeline:
    """Returns sklearn Pipeline with feature engineering + scaling."""
    return Pipeline(
        [
            ("engineer", FeatureEngineer()),
            ("scaler", StandardScaler()),
        ]
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function — returns engineered DataFrame (unscaled)."""
    eng = FeatureEngineer()
    eng.fit(df)
    return eng.transform(df)


def save_pipeline(pipeline, path: str = "models/artifacts/feature_pipeline.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Feature pipeline saved to {path}")


def load_pipeline(path: str = "models/artifacts/feature_pipeline.pkl"):
    return joblib.load(path)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from src.data.generate_data import generate_dataset

    df_raw = generate_dataset(n=1000)
    df_eng = engineer_features(df_raw)
    print(df_eng[ENGINEERED_FEATURES].describe().T.round(3))
    print(
        f"\nEngineered {len(ENGINEERED_FEATURES)} new features on top of {len(BASE_FEATURES)} base features"
    )
