"""
Report Generator
=================
Generates formatted executive-ready reports from pipeline outputs.
Run after pipeline.py to produce HTML and Markdown report artifacts.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
NBA_OUTPUT = Path("data/sample_outputs/nba_recommendations.csv")
SHAP_IMPORTANCE = Path("reports/shap/global_importance.csv")


ARCHETYPE_NAMES = {
    "A": "👻 The Ghost",
    "B": "😤 Frustrated Professional",
    "C": "💸 Price-Sensitive Optimizer",
    "D": "🚀 Outgrown User",
}

URGENCY_EMOJI = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🟢",
}


def load_nba_output() -> pd.DataFrame:
    if not NBA_OUTPUT.exists():
        raise FileNotFoundError(f"NBA output not found at {NBA_OUTPUT}. Run pipeline.py first.")
    return pd.read_csv(NBA_OUTPUT)


def generate_markdown_report(df: pd.DataFrame) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n = len(df)
    total_rev = df["expected_revenue_saved"].sum() if "expected_revenue_saved" in df.columns else 0
    avg_churn = df["churn_probability"].mean()
    total_cost = df["action_cost_usd"].sum() if "action_cost_usd" in df.columns else 0
    net_profit = total_rev * 0.70 - total_cost * 1.20
    roi = (net_profit / max(total_cost, 1)) * 100

    # Archetype table
    arch_rows = []
    if "churn_archetype" in df.columns:
        for arch, grp in df.groupby("churn_archetype"):
            arch_rows.append(
                {
                    "Archetype": ARCHETYPE_NAMES.get(arch, arch),
                    "Customers": len(grp),
                    "Avg Churn %": f"{grp['churn_probability'].mean():.1%}",
                    "Revenue Saved": f"${grp.get('expected_revenue_saved', pd.Series(0)).sum():,.0f}",
                    "Top Action": grp["action"].mode()[0] if "action" in grp.columns else "N/A",
                }
            )
    arch_df = pd.DataFrame(arch_rows)

    # Urgency breakdown
    urgency_rows = []
    if "urgency" in df.columns:
        for urg, grp in df.groupby("urgency"):
            urgency_rows.append(
                {
                    "Urgency": f"{URGENCY_EMOJI.get(urg, '')} {urg.title()}",
                    "Count": len(grp),
                    "% of Total": f"{len(grp)/n:.1%}",
                }
            )
    urg_df = pd.DataFrame(urgency_rows)

    # Top 10 highest-value customers
    top10_cols = [
        c
        for c in [
            "customer_id",
            "subscription_type",
            "monthly_revenue",
            "churn_probability",
            "archetype_name",
            "action",
            "expected_revenue_saved",
        ]
        if c in df.columns
    ]
    top10 = (
        df.nlargest(10, "expected_revenue_saved")[top10_cols]
        if "expected_revenue_saved" in df.columns
        else df.head(10)[top10_cols]
    )

    lines = [
        "# NBA Churn Retention Engine — Executive Report",
        f"*Generated: {now}*\n",
        "---\n",
        "## 📊 Campaign Overview\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| High-Risk Customers | **{n:,}** |",
        f"| Average Churn Probability | **{avg_churn:.1%}** |",
        f"| Expected Revenue Saved (12 mo) | **${total_rev:,.0f}** |",
        f"| Total Campaign Cost | **${total_cost:,.0f}** |",
        f"| Net Incremental Profit | **${net_profit:,.0f}** |",
        f"| Campaign ROI | **{roi:.0f}%** |",
        "",
        "---\n",
        "## 👥 Churn Archetype Distribution\n",
        arch_df.to_markdown(index=False) if not arch_df.empty else "_No archetype data available._",
        "",
        "---\n",
        "## 🚨 Action Urgency Breakdown\n",
        urg_df.to_markdown(index=False) if not urg_df.empty else "_No urgency data available._",
        "",
        "---\n",
        "## 💰 Top 10 Highest-Value Customers to Retain\n",
        top10.to_markdown(index=False),
        "",
        "---\n",
        "## 🧠 Top 10 Churn Drivers (SHAP)\n",
    ]

    if SHAP_IMPORTANCE.exists():
        shap_df = pd.read_csv(SHAP_IMPORTANCE).head(10)
        lines.append(shap_df.to_markdown(index=False))
    else:
        lines.append("_Run pipeline with SHAP enabled to see feature importance._")

    lines += [
        "",
        "---\n",
        "## 📋 Recommended Actions Summary\n",
    ]

    if "action" in df.columns:
        action_summary = (
            df.groupby("action")
            .agg(
                count=("action", "count"),
                total_rev_saved=("expected_revenue_saved", "sum"),
                avg_conversion=("expected_conversion_rate", "mean"),
            )
            .round(3)
            .sort_values("total_rev_saved", ascending=False)
            .reset_index()
        )
        action_summary.columns = ["Action", "Customers", "Total Rev Saved ($)", "Avg Conversion"]
        action_summary["Total Rev Saved ($)"] = action_summary["Total Rev Saved ($)"].map(
            "${:,.0f}".format
        )
        lines.append(action_summary.to_markdown(index=False))

    lines += [
        "",
        "---",
        "\n*Report generated by NBA Churn Retention Engine v1.0.0*",
    ]

    return "\n".join(lines)


def generate_html_report(df: pd.DataFrame) -> str:
    """Wrap the markdown report in a simple HTML template."""
    try:
        import markdown

        md_content = generate_markdown_report(df)
        body = markdown.markdown(md_content, extensions=["tables", "fenced_code"])
    except ImportError:
        body = f"<pre>{generate_markdown_report(df)}</pre>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NBA Churn Retention Engine — Executive Report</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 40px auto;
            padding: 0 24px; background: #f8f9fa; color: #212529; }}
    h1 {{ color: #e94560; border-bottom: 3px solid #e94560; padding-bottom: 8px; }}
    h2 {{ color: #0f3460; margin-top: 36px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; background: white;
             border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    th {{ background: #0f3460; color: white; padding: 10px 14px; text-align: left; font-weight: 600; }}
    td {{ padding: 9px 14px; border-bottom: 1px solid #e9ecef; }}
    tr:hover td {{ background: #f1f3f5; }}
    hr {{ border: none; border-top: 1px solid #dee2e6; margin: 28px 0; }}
    em {{ color: #6c757d; }}
    strong {{ color: #e94560; }}
    code {{ background: #f1f3f5; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""
    return html


def run_report_generation():
    logger.info("Generating executive reports …")
    df = load_nba_output()

    # Markdown
    md = generate_markdown_report(df)
    md_path = REPORTS_DIR / "nba" / "executive_report.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md)
    logger.info(f"Markdown report saved: {md_path}")

    # HTML
    html = generate_html_report(df)
    html_path = REPORTS_DIR / "nba" / "executive_report.html"
    html_path.write_text(html)
    logger.info(f"HTML report saved: {html_path}")

    print("\n✓ Reports generated:")
    print(f"  {md_path}")
    print(f"  {html_path}")
    return md_path, html_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_report_generation()
