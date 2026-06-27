"""
Synthetic SaaS User Behavior Data Generator
============================================
Generates 50,000+ realistic records with churn labels.
Business-realistic distributions, noise, outliers, and class imbalance.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)

COUNTRIES = [
    "US",
    "UK",
    "Germany",
    "France",
    "India",
    "Canada",
    "Australia",
    "Brazil",
    "Netherlands",
    "Singapore",
]
COUNTRY_WEIGHTS = [0.35, 0.12, 0.10, 0.07, 0.08, 0.06, 0.05, 0.05, 0.06, 0.06]

INDUSTRIES = [
    "SaaS",
    "FinTech",
    "HealthTech",
    "E-Commerce",
    "EdTech",
    "MarTech",
    "HR Tech",
    "Legal Tech",
    "PropTech",
    "Other",
]

SUBSCRIPTION_TYPES = ["Starter", "Professional", "Business", "Enterprise"]
SUB_WEIGHTS = [0.40, 0.30, 0.20, 0.10]

MRR_MAP = {
    "Starter": (29, 79),
    "Professional": (99, 299),
    "Business": (399, 999),
    "Enterprise": (1500, 5000),
}


def _clamp(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate_dataset(n: int = 50000, output_path: str = None) -> pd.DataFrame:
    logger.info(f"Generating {n:,} synthetic SaaS user records …")

    # ── Identifiers & demographics ─────────────────────────────────────
    customer_ids = [f"CUST-{i:06d}" for i in range(1, n + 1)]
    subscription_type = np.random.choice(SUBSCRIPTION_TYPES, size=n, p=SUB_WEIGHTS)
    country = np.random.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS)
    industry = np.random.choice(INDUSTRIES, size=n)
    tenure_months = np.random.exponential(scale=18, size=n).astype(int)
    tenure_months = _clamp(tenure_months, 1, 84)

    # ── Revenue ────────────────────────────────────────────────────────
    mrr = np.array([np.random.uniform(*MRR_MAP[s]) for s in subscription_type])
    mrr = np.round(mrr, 2)

    # ── Engagement signals ─────────────────────────────────────────────
    base_sessions_90d = np.random.poisson(lam=30, size=n)
    sessions_90d = _clamp(base_sessions_90d + np.random.randint(-5, 10, n), 0, 300)
    # 30d sessions correlated with 90d but with decay signal for churners
    sessions_30d = _clamp((sessions_90d / 3 * np.random.uniform(0.2, 1.3, n)).astype(int), 0, 120)
    days_since_last_login = np.random.exponential(scale=7, size=n).astype(int)
    days_since_last_login = _clamp(days_since_last_login, 0, 90)
    days_since_last_core_action = days_since_last_login + np.random.randint(0, 10, n)
    days_since_last_core_action = _clamp(days_since_last_core_action, 0, 90)
    login_frequency_change = np.random.normal(loc=0, scale=0.3, size=n)  # fraction change
    session_duration_change = np.random.normal(loc=0, scale=0.25, size=n)

    # ── Support signals ────────────────────────────────────────────────
    ticket_count = np.random.poisson(lam=1.5, size=n)
    ticket_count = _clamp(ticket_count, 0, 20)
    support_sentiment = np.random.normal(loc=0.1, scale=0.4, size=n)  # -1 bad, +1 good
    support_sentiment = _clamp(support_sentiment, -1, 1)
    error_rate = np.random.beta(a=1.2, b=8, size=n)
    failed_api_calls = np.random.poisson(lam=2, size=n)
    failed_api_calls = _clamp(failed_api_calls, 0, 50)
    rage_click_count = np.random.poisson(lam=0.8, size=n)
    rage_click_count = _clamp(rage_click_count, 0, 30)

    # ── Billing signals ────────────────────────────────────────────────
    billing_page_visits = np.random.poisson(lam=1.2, size=n)
    billing_page_visits = _clamp(billing_page_visits, 0, 20)
    downgrade_page_visits = np.random.poisson(lam=0.4, size=n)
    downgrade_page_visits = _clamp(downgrade_page_visits, 0, 10)
    invoice_download_frequency = np.random.poisson(lam=0.8, size=n)
    invoice_download_frequency = _clamp(invoice_download_frequency, 0, 12)
    trial_expiry_days = np.where(tenure_months < 1, np.random.randint(0, 14, n), -1)

    # ── Capacity / growth signals ──────────────────────────────────────
    tier_capacity_utilization = np.random.beta(a=2, b=3, size=n)
    tier_capacity_utilization = _clamp(tier_capacity_utilization, 0.05, 1.0)
    export_frequency = np.random.poisson(lam=3, size=n)
    export_frequency = _clamp(export_frequency, 0, 40)
    api_usage = np.random.exponential(scale=500, size=n).astype(int)
    api_usage = _clamp(api_usage, 0, 10000)
    api_limit_usage = _clamp(api_usage / 10000, 0, 1.0)
    seat_limit_usage = np.random.beta(a=1.5, b=2.5, size=n)
    seat_limit_usage = _clamp(seat_limit_usage, 0.05, 1.0)

    # ── Churn label (realistic ~18% churn rate) ────────────────────────
    churn_logit = (
        -2.5
        + 0.04 * days_since_last_login
        + 0.03 * days_since_last_core_action
        - 0.015 * sessions_30d
        + 0.08 * ticket_count
        - 1.2 * support_sentiment
        + 1.5 * billing_page_visits / 5
        + 1.2 * downgrade_page_visits / 3
        + 0.6 * error_rate
        - 0.01 * tenure_months
        + 0.5 * login_frequency_change * -1
        + np.random.normal(0, 0.5, n)  # noise
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churned = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    # ── Assemble DataFrame ─────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "subscription_type": subscription_type,
            "monthly_revenue": mrr,
            "tenure_months": tenure_months,
            "country": country,
            "industry": industry,
            "sessions_30d": sessions_30d,
            "sessions_90d": sessions_90d,
            "days_since_last_login": days_since_last_login,
            "days_since_last_core_action": days_since_last_core_action,
            "login_frequency_change": np.round(login_frequency_change, 4),
            "session_duration_change": np.round(session_duration_change, 4),
            "ticket_count": ticket_count,
            "support_sentiment": np.round(support_sentiment, 4),
            "error_rate": np.round(error_rate, 4),
            "failed_api_calls": failed_api_calls,
            "rage_click_count": rage_click_count,
            "billing_page_visits": billing_page_visits,
            "downgrade_page_visits": downgrade_page_visits,
            "invoice_download_frequency": invoice_download_frequency,
            "trial_expiry_days": trial_expiry_days,
            "tier_capacity_utilization": np.round(tier_capacity_utilization, 4),
            "export_frequency": export_frequency,
            "api_usage": api_usage,
            "api_limit_usage": np.round(api_limit_usage, 4),
            "seat_limit_usage": np.round(seat_limit_usage, 4),
            "churned": churned,
        }
    )

    # ── Inject outliers (1%) ───────────────────────────────────────────
    outlier_idx = np.random.choice(n, size=int(n * 0.01), replace=False)
    df.loc[outlier_idx, "ticket_count"] = np.random.randint(15, 30, len(outlier_idx))
    df.loc[outlier_idx, "billing_page_visits"] = np.random.randint(10, 25, len(outlier_idx))

    churn_rate = df["churned"].mean()
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Churn rate: {churn_rate:.2%}")
    logger.info(
        f"Revenue range: ${df['monthly_revenue'].min():.0f} – ${df['monthly_revenue'].max():.0f}"
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic SaaS churn dataset")
    parser.add_argument("--n", type=int, default=50000, help="Number of records")
    parser.add_argument("--output", type=str, default="data/raw/saas_churn_dataset.csv")
    args = parser.parse_args()
    generate_dataset(n=args.n, output_path=args.output)
