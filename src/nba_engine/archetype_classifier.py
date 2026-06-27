"""
Churn Archetype Classifier
===========================
Classifies high-risk users into one of four archetypes:
  A – The Ghost
  B – Frustrated Professional
  C – Price-Sensitive Optimizer
  D – Outgrown User
Uses a rule-based scoring approach driven by SHAP-derived feature signals.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Archetype definitions ──────────────────────────────────────────────────────

ARCHETYPES = {
    "A": {
        "name": "The Ghost",
        "description": "Lost momentum and engagement — quietly drifting away.",
        "psychology": "User lost momentum and engagement.",
        "color": "#6c757d",
        "emoji": "👻",
    },
    "B": {
        "name": "Frustrated Professional",
        "description": "Experiencing product friction and support issues.",
        "psychology": "Product friction and support issues driving frustration.",
        "color": "#dc3545",
        "emoji": "😤",
    },
    "C": {
        "name": "Price-Sensitive Optimizer",
        "description": "Concerned about cost — actively evaluating alternatives.",
        "psychology": "Concerned about pricing and ROI justification.",
        "color": "#fd7e14",
        "emoji": "💸",
    },
    "D": {
        "name": "Outgrown User",
        "description": "Hitting plan limits — ready for the next tier.",
        "psychology": "Needs a larger plan but hasn't upgraded yet.",
        "color": "#198754",
        "emoji": "🚀",
    },
}


def compute_archetype_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute archetype affinity scores [0-1] for each customer row.
    Expects the feature-engineered DataFrame.
    """
    out = df.copy()

    # Ghost score — engagement decay signals
    ghost_score = (
        0.40 * out.get("engagement_decay_score", pd.Series(0, index=out.index)).clip(0, 1)
        + 0.25 * (out.get("days_since_last_login", pd.Series(0, index=out.index)) / 90).clip(0, 1)
        + 0.20 * (1 - out.get("velocity_30d_vs_90d", pd.Series(1, index=out.index)).clip(0, 1))
        + 0.15
        * np.where(
            out.get("login_frequency_change", pd.Series(0, index=out.index)) < 0,
            abs(out.get("login_frequency_change", pd.Series(0, index=out.index))),
            0,
        )
    )

    # Frustrated Professional score — friction signals
    frustrated_score = (
        0.35 * out.get("friction_score", pd.Series(0, index=out.index)).clip(0, 1)
        + 0.25 * (out.get("ticket_count", pd.Series(0, index=out.index)) / 20).clip(0, 1)
        + 0.20 * out.get("error_rate", pd.Series(0, index=out.index)).clip(0, 1)
        + 0.20
        * np.where(
            out.get("support_sentiment", pd.Series(0, index=out.index)) < 0,
            abs(out.get("support_sentiment", pd.Series(0, index=out.index))),
            0,
        )
    )

    # Price-Sensitive Optimizer score
    price_score = (
        0.40 * out.get("pricing_sensitivity_score", pd.Series(0, index=out.index)).clip(0, 1)
        + 0.30 * (out.get("billing_page_visits", pd.Series(0, index=out.index)) / 20).clip(0, 1)
        + 0.30 * (out.get("downgrade_page_visits", pd.Series(0, index=out.index)) / 10).clip(0, 1)
    )

    # Outgrown User score
    outgrown_score = (
        0.40 * out.get("growth_pressure_score", pd.Series(0, index=out.index)).clip(0, 1)
        + 0.30 * out.get("tier_capacity_utilization", pd.Series(0, index=out.index)).clip(0, 1)
        + 0.30 * out.get("api_limit_usage", pd.Series(0, index=out.index)).clip(0, 1)
    )

    out["score_ghost"] = ghost_score.clip(0, 1).round(4)
    out["score_frustrated"] = frustrated_score.clip(0, 1).round(4)
    out["score_price_sensitive"] = price_score.clip(0, 1).round(4)
    out["score_outgrown"] = outgrown_score.clip(0, 1).round(4)

    score_matrix = out[
        ["score_ghost", "score_frustrated", "score_price_sensitive", "score_outgrown"]
    ]
    out["churn_archetype"] = score_matrix.idxmax(axis=1).map(
        {
            "score_ghost": "A",
            "score_frustrated": "B",
            "score_price_sensitive": "C",
            "score_outgrown": "D",
        }
    )
    out["archetype_confidence"] = score_matrix.max(axis=1).round(4)
    out["archetype_name"] = out["churn_archetype"].map(
        {k: v["name"] for k, v in ARCHETYPES.items()}
    )

    return out


def classify_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    """Public entry point — returns DataFrame with archetype columns appended."""
    return compute_archetype_scores(df)


def get_archetype_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Summary table of archetype distribution."""
    if "churn_archetype" not in df.columns:
        df = classify_archetypes(df)
    dist = (
        df.groupby("churn_archetype")
        .agg(
            count=("churn_archetype", "count"),
            avg_churn_prob=("churn_probability", "mean"),
            avg_monthly_revenue=("monthly_revenue", "mean"),
        )
        .reset_index()
    )
    dist["archetype_name"] = dist["churn_archetype"].map(
        {k: v["name"] for k, v in ARCHETYPES.items()}
    )
    return dist


def get_archetype_info(code: str) -> Dict:
    return ARCHETYPES.get(code.upper(), {})
