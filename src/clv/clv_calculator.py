"""
Customer Lifetime Value (CLV) Calculator
==========================================
Estimates CLV and assigns customer tiers for use in NBA prioritization.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


TIER_THRESHOLDS = {
    "Enterprise": 5000,
    "High": 1500,
    "Medium": 500,
    "Low": 0,
}


def calculate_clv(
    monthly_revenue: float,
    churn_probability: float,
    discount_rate: float = 0.10,
    months: int = 36,
) -> float:
    """
    Discounted CLV over a finite horizon.

    CLV = Σ [ MRR × (1 - churn_prob)^t / (1 + monthly_discount)^t ]
    """
    monthly_discount = discount_rate / 12
    retention_prob = 1 - churn_probability
    clv = sum(
        monthly_revenue * (retention_prob**t) / ((1 + monthly_discount) ** t)
        for t in range(1, months + 1)
    )
    return round(clv, 2)


def assign_clv_tier(clv: float) -> str:
    for tier, threshold in TIER_THRESHOLDS.items():
        if clv >= threshold:
            return tier
    return "Low"


def add_clv_columns(df: pd.DataFrame, churn_prob_col: str = "churn_probability") -> pd.DataFrame:
    """Append clv and clv_tier columns to a DataFrame."""
    df = df.copy()

    churn_probs = (
        df[churn_prob_col] if churn_prob_col in df.columns else pd.Series(0.2, index=df.index)
    )
    revenues = (
        df["monthly_revenue"]
        if "monthly_revenue" in df.columns
        else pd.Series(100.0, index=df.index)
    )

    df["clv"] = [calculate_clv(mrr, cp) for mrr, cp in zip(revenues, churn_probs)]
    df["clv_tier"] = df["clv"].apply(assign_clv_tier)
    df["annualised_revenue"] = (revenues * 12).round(2)

    logger.info(f"CLV calculated. Tiers: {df['clv_tier'].value_counts().to_dict()}")
    return df


def get_clv_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary stats grouped by CLV tier."""
    if "clv_tier" not in df.columns:
        df = add_clv_columns(df)
    return (
        df.groupby("clv_tier")
        .agg(
            count=("clv", "count"),
            avg_clv=("clv", "mean"),
            total_clv=("clv", "sum"),
            avg_mrr=("monthly_revenue", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("avg_clv", ascending=False)
    )
