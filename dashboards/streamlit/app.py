"""
NBA Churn Retention Engine — Streamlit Executive Dashboard
============================================================
Pages:
  1. Executive Overview
  2. Churn Prediction
  3. Customer Explorer
  4. SHAP Explainability
  5. NBA Recommendations
  6. Revenue Impact Simulator
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="NBA Churn Retention Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e94560; }
    .metric-label { font-size: 0.85rem; color: #a8b2d8; margin-top: 4px; }
    .archetype-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stDataFrame { border-radius: 8px; }
    div[data-testid="stSidebarNav"] { background-color: #0f3460; }
</style>
""", unsafe_allow_html=True)

ARCHETYPE_COLORS = {
    "A": "#6c757d", "B": "#dc3545", "C": "#fd7e14", "D": "#198754"
}
ARCHETYPE_NAMES = {
    "A": "👻 The Ghost",
    "B": "😤 Frustrated Pro",
    "C": "💸 Price Sensitive",
    "D": "🚀 Outgrown User",
}
URGENCY_COLORS = {
    "critical": "#dc3545", "high": "#fd7e14", "medium": "#ffc107", "low": "#198754"
}

# ── Currency (INR) ──────────────────────────────────────────────────────────────
# Underlying data is in USD; we display everything in Indian Rupees.
# Override the rate without code changes via env var, e.g. USD_TO_INR=86.5
USD_TO_INR = float(os.getenv("USD_TO_INR", "83.0"))
CURRENCY_COLS = ("monthly_revenue", "expected_revenue_saved", "clv", "action_cost_usd")


def _indian_group(n) -> str:
    """Format an integer with Indian digit grouping: 1234567 -> 12,34,567."""
    s = f"{abs(int(round(n))):d}"
    neg = float(n) < 0
    if len(s) > 3:
        last3, rest = s[-3:], s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        s = ",".join(parts) + "," + last3
    return ("-" if neg else "") + s


def inr(amount, compact: bool = False) -> str:
    """Format a rupee amount with the ₹ symbol and Indian conventions."""
    a = float(amount)
    if compact:
        if abs(a) >= 1e7:
            return f"₹{a / 1e7:,.2f} Cr"
        if abs(a) >= 1e5:
            return f"₹{a / 1e5:,.2f} L"
    return "₹" + _indian_group(a)


def usd_to_inr(amount, compact: bool = False) -> str:
    """Convert a USD amount to INR and format it."""
    return inr(float(amount) * USD_TO_INR, compact=compact)


def to_inr_display(frame: pd.DataFrame, cols=CURRENCY_COLS) -> pd.DataFrame:
    """Return a copy with USD currency columns converted to INR strings + ₹ labels."""
    out = frame.copy()
    rename = {}
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: usd_to_inr(v) if pd.notna(v) else v)
            rename[c] = c.replace("_usd", "").replace("_", " ").title() + " (₹)"
    return out.rename(columns=rename)


# ── Data Loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_nba_output() -> pd.DataFrame:
    path = Path("data/sample_outputs/nba_recommendations.csv")
    if path.exists():
        return pd.read_csv(path)
    return _generate_demo_data()


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    path = Path("data/raw/saas_churn_dataset.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_shap_importance() -> pd.DataFrame:
    path = Path("reports/shap/global_importance.csv")
    if path.exists():
        return pd.read_csv(path)
    return _demo_shap_importance()


def _generate_demo_data() -> pd.DataFrame:
    """Generate in-memory demo data if pipeline hasn't been run."""
    np.random.seed(42)
    n = 500
    archetypes = np.random.choice(list("ABCD"), n, p=[0.30, 0.25, 0.25, 0.20])
    archetype_names = {
        "A": "The Ghost", "B": "Frustrated Professional",
        "C": "Price-Sensitive Optimizer", "D": "Outgrown User"
    }
    actions = {
        "A": "Personalized Feature Discovery Email",
        "B": "Priority Support & CS Outreach",
        "C": "Personalised Discount Offer (20-30%)",
        "D": "Free 30-Day Premium Tier Trial",
    }
    urgency_map = {"A": "medium", "B": "critical", "C": "high", "D": "high"}
    mrr = np.random.choice([49, 99, 199, 499, 999, 2499], n, p=[0.25, 0.30, 0.25, 0.12, 0.05, 0.03])
    churn_prob = np.random.beta(5, 3, n)
    conv_rate = np.random.uniform(0.08, 0.38, n)
    rev_saved = churn_prob * conv_rate * mrr * 12

    return pd.DataFrame({
        "customer_id": [f"CUST-{i:06d}" for i in range(1, n+1)],
        "subscription_type": np.random.choice(["Starter","Professional","Business","Enterprise"], n),
        "monthly_revenue": mrr,
        "tenure_months": np.random.randint(1, 60, n),
        "churn_probability": churn_prob.round(4),
        "primary_driver": np.random.choice(
            ["engagement_decay_score","friction_score","pricing_sensitivity_score","growth_pressure_score"], n
        ),
        "churn_archetype": archetypes,
        "archetype_name": [archetype_names[a] for a in archetypes],
        "action": [actions[a] for a in archetypes],
        "channel": np.random.choice(["email","in-app","phone","push"], n),
        "urgency": [urgency_map[a] for a in archetypes],
        "expected_conversion_rate": conv_rate.round(4),
        "action_cost_usd": np.random.choice([0.5, 5, 15, 25, 50, 75], n),
        "expected_revenue_saved": rev_saved.round(2),
        "campaign_roi": np.random.uniform(2, 50, n).round(2),
        "clv": (mrr * 18 * np.random.uniform(0.5, 1.5, n)).round(2),
        "clv_tier": np.random.choice(["Low","Medium","High","Enterprise"], n, p=[0.30,0.40,0.20,0.10]),
        "customer_health_score": np.random.beta(2, 5, n).round(4),
        "retention_risk_score": np.random.beta(5, 2, n).round(4),
    })


def _demo_shap_importance() -> pd.DataFrame:
    features = [
        "engagement_decay_score", "days_since_last_login", "friction_score",
        "pricing_sensitivity_score", "retention_risk_score", "support_sentiment",
        "billing_page_visits", "ticket_count", "error_rate",
        "velocity_30d_vs_90d", "growth_pressure_score", "tier_capacity_utilization",
        "login_frequency_change", "sessions_30d", "downgrade_page_visits",
    ]
    values = np.sort(np.random.exponential(0.05, len(features)))[::-1]
    return pd.DataFrame({"feature": features, "mean_abs_shap": values.round(4)})


def calculate_clv_simple(mrr, churn_prob, months=36):
    r = 1 - churn_prob
    return sum(mrr * r**t / (1.008**t) for t in range(1, months + 1))


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.shields.io/badge/NBA-Churn%20Engine-e94560?style=for-the-badge", width="stretch")
    st.markdown("### Navigation")
    page = st.selectbox("Select Page", [
        "🏠 Executive Overview",
        "🔮 Churn Prediction",
        "🔍 Customer Explorer",
        "🧠 SHAP Explainability",
        "🎯 NBA Recommendations",
        "💰 Revenue Impact Simulator",
    ])
    st.markdown("---")
    st.markdown("**Data Source**")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df):,} rows")
    else:
        df = load_nba_output()
    st.markdown(f"*{len(df):,} customers loaded*")
    st.markdown("---")
    churn_threshold = st.slider("Churn Risk Threshold", 0.30, 0.90, 0.50, 0.05)
    df_filtered = df[df["churn_probability"] >= churn_threshold]
    st.markdown(f"**{len(df_filtered):,}** customers above threshold")


# ── PAGE 1: Executive Overview ─────────────────────────────────────────────────

if page == "🏠 Executive Overview":
    st.title("🎯 NBA Churn Retention Engine")
    st.caption("Next-Best-Action Churn Retention Platform — Executive Dashboard")

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    total_rev = df_filtered["expected_revenue_saved"].sum()
    avg_churn = df["churn_probability"].mean()
    high_risk = len(df_filtered)
    avg_roi = df_filtered["campaign_roi"].mean() if "campaign_roi" in df_filtered.columns else 0
    arch_dist = df_filtered["churn_archetype"].value_counts()

    with col1:
        st.metric("High-Risk Customers", f"{high_risk:,}", f"{high_risk/len(df):.0%} of base")
    with col2:
        st.metric("Avg Churn Probability", f"{avg_churn:.1%}", delta=None)
    with col3:
        st.metric("Expected Revenue Saved", usd_to_inr(total_rev, compact=True), "annual")
    with col4:
        st.metric("Avg Campaign ROI", f"{avg_roi:.0f}x", delta=None)
    with col5:
        total_cost = df_filtered["action_cost_usd"].sum() if "action_cost_usd" in df_filtered.columns else 0
        st.metric("Campaign Cost", usd_to_inr(total_cost, compact=True), delta=None)

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        # Archetype distribution
        arch_data = df_filtered["churn_archetype"].value_counts().reset_index()
        arch_data.columns = ["archetype", "count"]
        arch_data["archetype_name"] = arch_data["archetype"].map(ARCHETYPE_NAMES)
        arch_data["color"] = arch_data["archetype"].map(ARCHETYPE_COLORS)
        fig = px.bar(
            arch_data, x="archetype_name", y="count",
            color="archetype",
            color_discrete_map=ARCHETYPE_COLORS,
            title="Customer Distribution by Churn Archetype",
            labels={"count": "Customers", "archetype_name": ""},
        )
        fig.update_layout(showlegend=False, height=300, margin=dict(t=40))
        st.plotly_chart(fig, width="stretch")

    with col_right:
        # Revenue at risk by tier
        if "clv_tier" in df_filtered.columns:
            tier_data = df_filtered.groupby("clv_tier")["monthly_revenue"].sum().reset_index()
            fig2 = px.pie(
                tier_data, values="monthly_revenue", names="clv_tier",
                title="MRR at Risk by CLV Tier",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.45,
            )
            fig2.update_layout(height=300, margin=dict(t=40))
            st.plotly_chart(fig2, width="stretch")

    # Churn probability distribution
    fig3 = px.histogram(
        df, x="churn_probability", nbins=50,
        title="Churn Probability Distribution",
        color_discrete_sequence=["#e94560"],
        labels={"churn_probability": "Churn Probability"},
    )
    fig3.update_traces(
        marker=dict(
            line=dict(width=0),
            color="#e94560", opacity=0.85,
        ),
    )
    fig3.add_vline(
        x=churn_threshold, line_dash="dash", line_color="#fbbf24", line_width=2,
        annotation_text=f"Threshold {churn_threshold:.0%}",
        annotation_position="top", annotation_font_color="#fbbf24",
    )
    fig3.update_layout(
        height=300, margin=dict(t=50, b=10),
        bargap=0.04,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "#e6e9f5"},
        xaxis=dict(showgrid=False, tickformat=".0%"),
        yaxis=dict(showgrid=True, gridcolor="rgba(168,178,216,0.12)", title="Customers"),
    )
    st.plotly_chart(fig3, width="stretch")


# ── PAGE 2: Churn Prediction ───────────────────────────────────────────────────

elif page == "🔮 Churn Prediction":
    st.title("🔮 Churn Prediction")
    st.caption("Score individual customers in real time")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Customer Input")
        sub_type = st.selectbox("Subscription Type", ["Starter", "Professional", "Business", "Enterprise"])
        mrr = st.number_input("Monthly Revenue (₹)", 2000.0, 8000000.0, 16000.0, step=1000.0)
        tenure = st.slider("Tenure (months)", 1, 84, 14)
        sessions_30 = st.slider("Sessions (30d)", 0, 120, 8)
        days_login = st.slider("Days Since Last Login", 0, 90, 12)
        tickets = st.slider("Support Tickets", 0, 20, 3)
        sentiment = st.slider("Support Sentiment", -1.0, 1.0, -0.3)
        billing_visits = st.slider("Billing Page Visits", 0, 20, 3)
        downgrade_visits = st.slider("Downgrade Page Visits", 0, 10, 2)
        capacity_util = st.slider("Tier Capacity Utilization", 0.0, 1.0, 0.55)

    with col2:
        st.subheader("Prediction Results")

        # Simulate prediction
        risk_score = (
            0.25 * (days_login / 90)
            + 0.20 * (tickets / 20)
            + 0.15 * max(0, -sentiment)
            + 0.20 * (billing_visits / 20 + downgrade_visits / 10) / 2
            + 0.10 * max(0, 1 - sessions_30 / 30)
            - 0.10 * (tenure / 84)
        )
        churn_prob = min(max(risk_score + 0.1, 0.05), 0.97)

        if churn_prob > 0.7:
            color, risk_label, risk_emoji = "#e94560", "CRITICAL", "🔴"
        elif churn_prob > 0.5:
            color, risk_label, risk_emoji = "#fd7e14", "HIGH", "🟠"
        elif churn_prob > 0.3:
            color, risk_label, risk_emoji = "#ffc107", "MEDIUM", "🟡"
        else:
            color, risk_label, risk_emoji = "#2dd4a7", "LOW", "🟢"

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob * 100,
            number={"suffix": "%", "font": {"size": 60, "color": color, "family": "Inter"}},
            delta={
                "reference": 50, "suffix": "%",
                "increasing": {"color": "#e94560"}, "decreasing": {"color": "#2dd4a7"},
                "font": {"size": 16},
            },
            title={"text": "<span style='font-size:0.95rem;letter-spacing:0.12em;"
                           "color:#a8b2d8'>CHURN PROBABILITY</span>"},
            gauge={
                "axis": {
                    "range": [0, 100], "tickwidth": 1, "tickcolor": "#3a4368",
                    "tickfont": {"color": "#a8b2d8", "size": 11},
                },
                "bar": {"color": color, "thickness": 0.30},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(45,212,167,0.18)"},
                    {"range": [30, 50], "color": "rgba(255,193,7,0.18)"},
                    {"range": [50, 75], "color": "rgba(253,126,20,0.22)"},
                    {"range": [75, 100], "color": "rgba(233,69,96,0.28)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.85, "value": churn_prob * 100,
                },
            },
        ))
        fig.update_layout(
            height=320, margin=dict(t=70, b=10, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter", "color": "white"},
        )
        st.plotly_chart(fig, width="stretch")

        clv_value = calculate_clv_simple(mrr, churn_prob)
        st.markdown(f"""
<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
            border:1px solid {color};border-left:5px solid {color};
            border-radius:14px;padding:18px 22px;margin-top:6px;
            box-shadow:0 4px 18px rgba(0,0,0,0.35);">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <div style="font-size:0.72rem;letter-spacing:0.14em;color:#a8b2d8;
                  text-transform:uppercase;">Risk Level</div>
      <div style="font-size:1.7rem;font-weight:700;color:{color};line-height:1.2;">
        {risk_emoji} {risk_label}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:0.72rem;letter-spacing:0.14em;color:#a8b2d8;
                  text-transform:uppercase;">Estimated CLV</div>
      <div style="font-size:1.5rem;font-weight:700;color:#ffffff;line-height:1.2;">
        {inr(clv_value, compact=True)}</div>
    </div>
  </div>
</div>
        """, unsafe_allow_html=True)

        # Top drivers
        drivers = {
            "Days since login": 0.25 * (days_login / 90),
            "Support tickets": 0.20 * (tickets / 20),
            "Negative sentiment": 0.15 * max(0, -sentiment),
            "Billing page visits": 0.20 * (billing_visits / 20 + downgrade_visits / 10) / 2,
            "Low engagement": 0.10 * max(0, 1 - sessions_30 / 30),
        }
        drivers_df = pd.DataFrame({"driver": list(drivers.keys()), "impact": list(drivers.values())})
        drivers_df = drivers_df.sort_values("impact", ascending=True)
        fig2 = px.bar(drivers_df, x="impact", y="driver", orientation="h",
                      title="Top Churn Drivers", color_discrete_sequence=["#e94560"])
        fig2.update_layout(height=280, margin=dict(t=40))
        st.plotly_chart(fig2, width="stretch")


# ── PAGE 3: Customer Explorer ──────────────────────────────────────────────────

elif page == "🔍 Customer Explorer":
    st.title("🔍 Customer Explorer")

    col1, col2, col3 = st.columns(3)
    with col1:
        arch_filter = st.multiselect("Archetype", options=list("ABCD"),
                                      default=list("ABCD"), format_func=lambda x: ARCHETYPE_NAMES[x])
    with col2:
        urgency_filter = st.multiselect("Urgency", ["critical", "high", "medium", "low"],
                                         default=["critical", "high"])
    with col3:
        min_prob = st.slider("Min Churn Probability", 0.3, 0.9, churn_threshold, 0.05)

    df_view = df[
        (df["churn_probability"] >= min_prob) &
        (df["churn_archetype"].isin(arch_filter)) &
        (df["urgency"].isin(urgency_filter) if "urgency" in df.columns else True)
    ].sort_values("churn_probability", ascending=False)

    st.markdown(f"**{len(df_view):,}** customers match filters")

    display_cols = [c for c in [
        "customer_id", "subscription_type", "monthly_revenue", "tenure_months",
        "churn_probability", "churn_archetype", "archetype_name",
        "action", "urgency", "expected_revenue_saved", "clv_tier"
    ] if c in df_view.columns]

    st.dataframe(
        to_inr_display(df_view[display_cols].head(200)),
        width="stretch",
        hide_index=True,
    )

    if st.button("📥 Download Filtered Results"):
        csv = df_view[display_cols].to_csv(index=False)
        st.download_button("Download CSV", csv, "filtered_customers.csv", "text/csv")


# ── PAGE 4: SHAP Explainability ────────────────────────────────────────────────

elif page == "🧠 SHAP Explainability":
    st.title("🧠 SHAP Explainability")
    st.caption("Global and individual-level model explanations")

    shap_df = load_shap_importance()

    col1, col2 = st.columns([1.5, 1])

    with col1:
        fig = px.bar(
            shap_df.head(15).sort_values("mean_abs_shap"),
            x="mean_abs_shap", y="feature",
            orientation="h",
            title="Global Feature Importance (Mean |SHAP|)",
            color="mean_abs_shap",
            color_continuous_scale="Reds",
            labels={"mean_abs_shap": "Mean |SHAP Value|", "feature": "Feature"},
        )
        fig.update_layout(height=450, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Feature Groups")
        feature_groups = {
            "Engagement": ["engagement_decay_score", "velocity_30d_vs_90d", "sessions_30d", "login_frequency_change"],
            "Friction": ["friction_score", "ticket_count", "error_rate", "rage_click_count"],
            "Pricing": ["pricing_sensitivity_score", "billing_page_visits", "downgrade_page_visits"],
            "Growth": ["growth_pressure_score", "tier_capacity_utilization", "api_limit_usage"],
        }
        group_importance = {}
        for group, features in feature_groups.items():
            group_imp = shap_df[shap_df["feature"].isin(features)]["mean_abs_shap"].sum()
            group_importance[group] = group_imp

        fig2 = px.pie(
            values=list(group_importance.values()),
            names=list(group_importance.keys()),
            title="SHAP Importance by Feature Group",
            hole=0.4,
        )
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, width="stretch")

    # Individual explanation
    st.subheader("Individual Customer Explanation")
    if len(df_filtered) > 0:
        selected_id = st.selectbox("Select Customer", df_filtered["customer_id"].head(50).tolist())
        cust = df_filtered[df_filtered["customer_id"] == selected_id].iloc[0]
        st.markdown(f"""
**Customer:** `{selected_id}`  
**Churn Probability:** `{cust.get('churn_probability', 0):.1%}`  
**Archetype:** `{ARCHETYPE_NAMES.get(cust.get('churn_archetype', 'A'), 'Unknown')}`  
**Primary Driver:** `{cust.get('primary_driver', 'N/A')}`

> This customer is at **{cust.get('churn_probability', 0):.0%} churn risk** primarily because of elevated `{cust.get('primary_driver', 'engagement signals')}`.
        """)


# ── PAGE 5: NBA Recommendations ────────────────────────────────────────────────

elif page == "🎯 NBA Recommendations":
    st.title("🎯 Next-Best-Action Recommendations")

    # Summary by archetype
    if "churn_archetype" in df_filtered.columns and "action" in df_filtered.columns:
        arch_summary = (
            df_filtered.groupby(["churn_archetype", "action"])
            .agg(count=("customer_id", "count"), rev_saved=("expected_revenue_saved", "sum"))
            .reset_index()
            .sort_values("count", ascending=False)
        )
        arch_summary["archetype_name"] = arch_summary["churn_archetype"].map(ARCHETYPE_NAMES)

        col1, col2 = st.columns([1.5, 1])
        with col1:
            fig = px.bar(
                arch_summary, x="archetype_name", y="count",
                color="churn_archetype", color_discrete_map=ARCHETYPE_COLORS,
                title="Recommended Actions by Archetype",
                labels={"count": "Customers", "archetype_name": ""},
                text="count",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, width="stretch")

        with col2:
            urgency_counts = df_filtered["urgency"].value_counts() if "urgency" in df_filtered.columns else pd.Series()
            if not urgency_counts.empty:
                fig2 = px.pie(
                    values=urgency_counts.values,
                    names=urgency_counts.index,
                    title="Action Urgency Distribution",
                    color=urgency_counts.index,
                    color_discrete_map=URGENCY_COLORS,
                    hole=0.4,
                )
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, width="stretch")

    # Action recommendations table
    st.subheader("Action Priority Queue")
    rec_cols = [c for c in [
        "customer_id", "churn_probability", "archetype_name",
        "action", "urgency", "channel",
        "expected_conversion_rate", "expected_revenue_saved",
    ] if c in df_filtered.columns]
    st.dataframe(
        to_inr_display(
            df_filtered[rec_cols].sort_values("expected_revenue_saved", ascending=False).head(100)
        ),
        width="stretch",
        hide_index=True,
    )


# ── PAGE 6: Revenue Impact Simulator ──────────────────────────────────────────

elif page == "💰 Revenue Impact Simulator":
    st.title("💰 Revenue Impact Simulator")
    st.caption("Adjust assumptions to model different retention campaign scenarios")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("Assumptions")
        reach_rate = st.slider("Campaign Reach Rate", 0.5, 1.0, 0.85, 0.05)
        conv_lift = st.slider("Conversion Rate Multiplier", 0.5, 2.0, 1.0, 0.1)
        months = st.slider("Retention Horizon (months)", 6, 24, 12)
        margin = st.slider("Gross Margin", 0.5, 0.9, 0.70, 0.05)
        overhead = st.slider("Overhead Multiplier", 1.0, 1.5, 1.2, 0.05)

    with col2:
        # Recalculate based on sliders
        df_sim = df_filtered.copy()
        if "expected_conversion_rate" in df_sim.columns:
            adj_conv = df_sim["expected_conversion_rate"] * conv_lift
            df_sim["p_saved"] = df_sim["churn_probability"] * adj_conv * reach_rate
            df_sim["rev_saved_sim"] = df_sim["p_saved"] * df_sim["monthly_revenue"] * months
            df_sim["cost_sim"] = df_sim.get("action_cost_usd", 5) * overhead
            df_sim["profit_sim"] = df_sim["rev_saved_sim"] * margin - df_sim["cost_sim"]

            total_rev = df_sim["rev_saved_sim"].sum()
            total_cost = df_sim["cost_sim"].sum()
            total_profit = df_sim["profit_sim"].sum()
            roi = total_profit / max(total_cost, 1) * 100

            st.subheader("Simulation Results")
            r1, r2 = st.columns(2)
            with r1:
                st.metric("Revenue Saved", usd_to_inr(total_rev, compact=True))
                st.metric("Campaign Cost", usd_to_inr(total_cost, compact=True))
            with r2:
                st.metric("Net Profit", usd_to_inr(total_profit, compact=True))
                st.metric("ROI", f"{roi:.0f}%")

            # Sensitivity: ROI vs conversion rate
            conv_range = np.arange(0.3, 2.1, 0.1)
            rois = []
            for c in conv_range:
                adj = df_sim["expected_conversion_rate"] * c
                p = df_sim["churn_probability"] * adj * reach_rate
                rev = (p * df_sim["monthly_revenue"] * months).sum()
                cost = total_cost
                rois.append((rev * margin - cost) / max(cost, 1) * 100)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=conv_range, y=rois, mode="lines+markers",
                                     line=dict(color="#e94560", width=3)))
            fig.add_vline(x=conv_lift, line_dash="dash", line_color="orange",
                          annotation_text="Current")
            fig.update_layout(
                title="ROI Sensitivity to Conversion Rate Multiplier",
                xaxis_title="Conversion Rate Multiplier",
                yaxis_title="ROI (%)",
                height=300, margin=dict(t=40),
            )
            st.plotly_chart(fig, width="stretch")
