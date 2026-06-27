"""
Business Impact Engine
=======================
Calculates expected revenue saved, ROI, and incremental profit
from retention campaigns.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ImpactAssumptions:
    """Configurable assumptions for the business impact simulation."""

    campaign_reach_rate: float = 0.85  # % of identified users actually contacted
    base_churn_without_intervention: float = 0.18  # baseline churn rate
    months_retained: int = 12  # retention horizon
    discount_rate_annual: float = 0.10  # cost of capital
    overhead_multiplier: float = 1.20  # overhead on top of direct cost
    incremental_margin: float = 0.70  # SaaS gross margin


@dataclass
class ImpactReport:
    total_at_risk_users: int = 0
    total_at_risk_mrr: float = 0.0
    expected_users_retained: float = 0.0
    expected_revenue_saved: float = 0.0
    total_campaign_cost: float = 0.0
    net_incremental_profit: float = 0.0
    roi_pct: float = 0.0
    payback_months: float = 0.0
    by_archetype: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_urgency: pd.DataFrame = field(default_factory=pd.DataFrame)


def simulate_business_impact(
    df_nba: pd.DataFrame,
    assumptions: ImpactAssumptions = None,
) -> ImpactReport:
    """Run the business impact simulation on the NBA output DataFrame."""
    if assumptions is None:
        assumptions = ImpactAssumptions()

    report = ImpactReport()
    df = df_nba.copy()

    # Ensure required columns exist
    for col, default in [
        ("churn_probability", 0.5),
        ("monthly_revenue", 100.0),
        ("expected_conversion_rate", 0.1),
        ("action_cost_usd", 5.0),
    ]:
        if col not in df.columns:
            df[col] = default

    # ── Compute per-customer impact ────────────────────────────────────
    df["p_saved"] = (
        df["churn_probability"] * df["expected_conversion_rate"] * assumptions.campaign_reach_rate
    )
    df["revenue_saved"] = df["p_saved"] * df["monthly_revenue"] * assumptions.months_retained
    df["campaign_cost_total"] = df["action_cost_usd"] * assumptions.overhead_multiplier
    df["incremental_profit"] = (
        df["revenue_saved"] * assumptions.incremental_margin - df["campaign_cost_total"]
    )

    # ── Aggregate totals ───────────────────────────────────────────────
    report.total_at_risk_users = len(df)
    report.total_at_risk_mrr = round(df["monthly_revenue"].sum(), 2)
    report.expected_users_retained = round(df["p_saved"].sum(), 1)
    report.expected_revenue_saved = round(df["revenue_saved"].sum(), 2)
    report.total_campaign_cost = round(df["campaign_cost_total"].sum(), 2)
    report.net_incremental_profit = round(df["incremental_profit"].sum(), 2)
    report.roi_pct = round(
        (report.net_incremental_profit / max(report.total_campaign_cost, 1)) * 100, 1
    )
    report.payback_months = round(
        report.total_campaign_cost
        / max(df["revenue_saved"].mean() * assumptions.incremental_margin, 1),
        1,
    )

    # ── By archetype ───────────────────────────────────────────────────
    if "churn_archetype" in df.columns:
        report.by_archetype = (
            df.groupby("churn_archetype")
            .agg(
                users=("churn_archetype", "count"),
                revenue_saved=("revenue_saved", "sum"),
                campaign_cost=("campaign_cost_total", "sum"),
                net_profit=("incremental_profit", "sum"),
            )
            .round(2)
            .reset_index()
        )

    # ── By urgency ─────────────────────────────────────────────────────
    if "urgency" in df.columns:
        report.by_urgency = (
            df.groupby("urgency")
            .agg(
                users=("urgency", "count"),
                revenue_saved=("revenue_saved", "sum"),
                avg_conversion=("expected_conversion_rate", "mean"),
            )
            .round(2)
            .reset_index()
        )

    logger.info(
        f"Impact Report: {report.total_at_risk_users:,} users | "
        f"${report.expected_revenue_saved:,.0f} revenue saved | "
        f"ROI {report.roi_pct:.0f}%"
    )
    return report, df


def print_impact_report(report: ImpactReport):
    print("\n" + "=" * 60)
    print("  BUSINESS IMPACT REPORT — NBA Retention Campaign")
    print("=" * 60)
    print(f"  Users at Risk            : {report.total_at_risk_users:>10,}")
    print(f"  At-Risk MRR              : ${report.total_at_risk_mrr:>12,.2f}")
    print(f"  Expected Retained Users  : {report.expected_users_retained:>10,.0f}")
    print(f"  Expected Revenue Saved   : ${report.expected_revenue_saved:>12,.2f}")
    print(f"  Total Campaign Cost      : ${report.total_campaign_cost:>12,.2f}")
    print(f"  Net Incremental Profit   : ${report.net_incremental_profit:>12,.2f}")
    print(f"  Campaign ROI             : {report.roi_pct:>9.1f}%")
    print(f"  Payback Period           : {report.payback_months:>9.1f} months")
    print("=" * 60)
    if not report.by_archetype.empty:
        print("\n  By Archetype:")
        print(report.by_archetype.to_string(index=False))
    print()
