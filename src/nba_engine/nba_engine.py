"""
Next-Best-Action (NBA) Recommendation Engine
=============================================
Rule-based engine that maps churn archetype + feature signals
to the optimal retention action.

Rules loaded from config/nba_rules.yaml for easy business configuration.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

DEFAULT_RULES_PATH = Path(__file__).parent.parent.parent / "config" / "nba_rules.yaml"


def load_rules(path: str = None) -> Dict:
    rules_path = Path(path) if path else DEFAULT_RULES_PATH
    with open(rules_path) as f:
        return yaml.safe_load(f)


# ── Condition evaluator ────────────────────────────────────────────────────────

CONDITION_OPS = {
    "_gte": lambda val, threshold: val >= threshold,
    "_lte": lambda val, threshold: val <= threshold,
    "_eq": lambda val, threshold: val == threshold,
    "_gt": lambda val, threshold: val > threshold,
    "_lt": lambda val, threshold: val < threshold,
}


def evaluate_conditions(row: pd.Series, conditions: Dict) -> bool:
    """Check whether all conditions in a rule are met for a customer row."""
    for cond_key, threshold in conditions.items():
        matched = False
        for suffix, op in CONDITION_OPS.items():
            if cond_key.endswith(suffix):
                feature_name = cond_key[: -len(suffix)]
                if feature_name not in row.index:
                    matched = True  # Skip missing features
                    break
                matched = op(row[feature_name], threshold)
                break
        if not matched:
            return False
    return True


# ── NBA Engine ─────────────────────────────────────────────────────────────────


class NBAEngine:
    """Recommendation engine that loads rules from YAML and scores customers."""

    def __init__(self, rules_path: str = None):
        self.config = load_rules(rules_path)
        self.rules = self.config.get("rules", [])
        self.defaults = self.config.get("defaults", {})
        self.thresholds = self.config.get("thresholds", {})
        logger.info(f"NBA engine loaded {len(self.rules)} rules")

    def recommend(self, row: pd.Series, archetype: str) -> Dict[str, Any]:
        """Return the best matching action for a single customer."""
        archetype_rules = [r for r in self.rules if r["archetype"] == archetype]
        archetype_rules.sort(key=lambda r: r["priority"])

        for rule in archetype_rules:
            if evaluate_conditions(row, rule.get("conditions", {})):
                return {
                    "rule_id": rule["id"],
                    "action": rule["action"],
                    "action_code": rule["action_code"],
                    "channel": rule["channel"],
                    "urgency": rule["urgency"],
                    "expected_conversion_rate": rule["expected_conversion_rate"],
                    "action_cost_usd": rule["cost_usd"],
                    "description": rule["description"],
                }

        # Fallback
        return {
            "rule_id": "FALLBACK",
            "action": self.defaults["fallback_action"],
            "action_code": self.defaults["fallback_action_code"],
            "channel": self.defaults["fallback_channel"],
            "urgency": "low",
            "expected_conversion_rate": self.defaults["fallback_expected_conversion_rate"],
            "action_cost_usd": self.defaults["fallback_cost_usd"],
            "description": "Generic retention email for unmatched conditions.",
        }

    def recommend_batch(
        self,
        df: pd.DataFrame,
        churn_prob_col: str = "churn_probability",
        archetype_col: str = "churn_archetype",
    ) -> pd.DataFrame:
        """Add NBA columns to a DataFrame of customers."""
        results = []
        for _, row in df.iterrows():
            archetype = row.get(archetype_col, "A")
            rec = self.recommend(row, archetype)
            results.append(rec)

        rec_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), rec_df.reset_index(drop=True)], axis=1)


# ── Business Impact Calculator ─────────────────────────────────────────────────


def calculate_expected_revenue_saved(
    churn_prob: float,
    monthly_revenue: float,
    conversion_rate: float,
    months_retained: int = 12,
) -> float:
    """
    Expected Revenue Saved = P(saved) × MRR × months
    P(saved) = churn_prob × conversion_rate
    """
    p_saved = churn_prob * conversion_rate
    return round(p_saved * monthly_revenue * months_retained, 2)


def add_revenue_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Append expected_revenue_saved and campaign_roi columns."""
    df = df.copy()
    df["expected_revenue_saved"] = df.apply(
        lambda r: calculate_expected_revenue_saved(
            r.get("churn_probability", 0.5),
            r.get("monthly_revenue", 100),
            r.get("expected_conversion_rate", 0.1),
        ),
        axis=1,
    )
    df["campaign_roi"] = (
        (df["expected_revenue_saved"] - df["action_cost_usd"])
        / df["action_cost_usd"].clip(lower=0.01)
    ).round(2)
    return df


# ── End-to-End Pipeline ────────────────────────────────────────────────────────


def run_nba_pipeline(df_with_archetypes: pd.DataFrame, rules_path: str = None) -> pd.DataFrame:
    """Full pipeline: rules matching → revenue impact → final output table."""
    engine = NBAEngine(rules_path)
    df_nba = engine.recommend_batch(df_with_archetypes)
    df_nba = add_revenue_impact(df_nba)
    logger.info(
        f"NBA pipeline complete. Total expected revenue saved: "
        f"${df_nba['expected_revenue_saved'].sum():,.0f}"
    )
    return df_nba


# ── Column definitions for output table ───────────────────────────────────────

NBA_OUTPUT_COLUMNS = [
    "customer_id",
    "churn_probability",
    "primary_driver",
    "churn_archetype",
    "archetype_name",
    "action",
    "channel",
    "urgency",
    "expected_conversion_rate",
    "action_cost_usd",
    "expected_revenue_saved",
    "campaign_roi",
    "monthly_revenue",
]
