# Data Dictionary — NBA Churn Retention Engine

## Raw Dataset (`data/raw/saas_churn_dataset.csv`)

### Customer Demographics

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `customer_id` | string | Unique customer identifier (format: CUST-XXXXXX) | `CUST-007823` |
| `subscription_type` | categorical | Plan tier | `Professional` |
| `monthly_revenue` | float | Monthly recurring revenue in USD | `199.00` |
| `tenure_months` | integer | Months since first subscription (1–84) | `14` |
| `country` | categorical | ISO country code | `US` |
| `industry` | categorical | Customer's industry vertical | `SaaS` |

### Engagement Signals

| Field | Type | Description | Business Meaning |
|-------|------|-------------|-----------------|
| `sessions_30d` | integer | Product sessions in last 30 days (0–120) | Core activity |
| `sessions_90d` | integer | Product sessions in last 90 days (0–300) | Trend baseline |
| `days_since_last_login` | integer | Days since most recent login (0–90) | Recency signal |
| `days_since_last_core_action` | integer | Days since last value-creating action (0–90) | Depth of usage |
| `login_frequency_change` | float | MoM change in login frequency (−1 to +1) | Trend direction |
| `session_duration_change` | float | MoM change in avg session duration (−1 to +1) | Engagement quality |

### Support & Friction Signals

| Field | Type | Description | Business Meaning |
|-------|------|-------------|-----------------|
| `ticket_count` | integer | Support tickets filed in 30 days (0–20) | Issue volume |
| `support_sentiment` | float | NLP sentiment of support interactions (−1 bad, +1 good) | Customer mood |
| `error_rate` | float | Application/API error rate (0–0.3) | Product quality experience |
| `failed_api_calls` | integer | Failed API calls in 30 days (0–50) | Technical friction |
| `rage_click_count` | integer | Detected UI rage clicks (0–30) | UX frustration |

### Billing & Pricing Signals

| Field | Type | Description | Business Meaning |
|-------|------|-------------|-----------------|
| `billing_page_visits` | integer | Visits to pricing/billing pages (0–20) | Price concern |
| `downgrade_page_visits` | integer | Visits to downgrade/cancel pages (0–10) | Downgrade intent |
| `invoice_download_frequency` | integer | Invoices downloaded in 30 days (0–12) | Finance scrutiny |
| `trial_expiry_days` | integer | Days until trial expires (−1 = not on trial) | Urgency |

### Capacity & Growth Signals

| Field | Type | Description | Business Meaning |
|-------|------|-------------|-----------------|
| `tier_capacity_utilization` | float | % of plan capacity used (0.05–1.0) | Plan fit |
| `export_frequency` | integer | Data exports in 30 days (0–40) | Power usage |
| `api_usage` | integer | API calls in 30 days (0–10,000) | Integration depth |
| `api_limit_usage` | float | % of API quota consumed (0–1.0) | API headroom |
| `seat_limit_usage` | float | % of seat allowance used (0.05–1.0) | Team scaling |

### Target Variable

| Field | Type | Description | Distribution |
|-------|------|-------------|-------------|
| `churned` | integer | Binary churn label: 1=churned, 0=retained | ~18–22% positive |

---

## Engineered Features (`data/processed/engineered_features.csv`)

All raw fields are retained, plus the following derived features:

| Feature | Formula | Range | Detects |
|---------|---------|-------|---------|
| `velocity_30d_vs_90d` | sessions_30d / (sessions_90d/3 + ε) | 0–5 | Engagement trend direction |
| `engagement_decay_score` | Weighted: login_recency + velocity decline + frequency change | 0–1 | Ghost archetype |
| `friction_score` | Weighted: tickets + errors + rage_clicks + negative_sentiment | 0–1 | Frustrated Professional |
| `pricing_sensitivity_score` | Weighted: billing_visits + downgrade_visits + invoice_freq | 0–1 | Price-Sensitive Optimizer |
| `growth_pressure_score` | Weighted: capacity_util + api_limit + seat_limit + exports | 0–1 | Outgrown User |
| `customer_health_score` | Inverse composite of engagement + friction + pricing | 0–1 | Overall health |
| `retention_risk_score` | Composite risk aggregation | 0–1 | Holistic churn risk |
| `ticket_density_30d` | Raw ticket_count (pass-through, named for clarity) | 0–20 | Support load |
| `support_ticket_sentiment` | Pass-through of support_sentiment | −1 to +1 | Friction quality |
| `subscription_tier` | Ordinal: Starter=0, Pro=1, Business=2, Enterprise=3 | 0–3 | Plan context |
| `tenure_bucket` | Bucketed: 0–3m=0, 3–12m=1, 12–24m=2, 24m+=3 | 0–3 | Lifecycle stage |
| `log_api_usage` | log1p(api_usage) | 0–9.2 | Stabilise heavy tail |
| `log_monthly_revenue` | log1p(monthly_revenue) | 3.4–8.5 | Stabilise heavy tail |

---

## NBA Recommendations Output (`data/sample_outputs/nba_recommendations.csv`)

| Field | Type | Description |
|-------|------|-------------|
| `customer_id` | string | Unique customer identifier |
| `subscription_type` | string | Plan tier |
| `monthly_revenue` | float | MRR in USD |
| `tenure_months` | integer | Months as customer |
| `churn_probability` | float | Model-predicted churn probability (0–1) |
| `primary_driver` | string | Top SHAP feature driving churn risk |
| `churn_archetype` | string | A / B / C / D |
| `archetype_name` | string | Human-readable archetype name |
| `action` | string | Recommended next-best action |
| `action_code` | string | Machine-readable action identifier |
| `channel` | string | Delivery channel (email / in-app / phone / push) |
| `urgency` | string | Priority level (critical / high / medium / low) |
| `expected_conversion_rate` | float | Historical conversion rate for this action type |
| `action_cost_usd` | float | Direct cost of executing this action |
| `expected_revenue_saved` | float | Expected revenue retained (12-month horizon) |
| `campaign_roi` | float | (Revenue Saved − Cost) / Cost |
| `clv` | float | Customer lifetime value (discounted, 36-month) |
| `clv_tier` | string | Low / Medium / High / Enterprise |
| `customer_health_score` | float | Composite health signal (0–1, higher = healthier) |
| `retention_risk_score` | float | Composite risk signal (0–1, higher = riskier) |

---

## Subscription Tiers & Revenue Ranges

| Tier | MRR Range | Typical Archetype Mix | CLV Tier |
|------|-----------|----------------------|----------|
| Starter | $29–$79 | Ghost (high), Price-Sensitive (medium) | Low |
| Professional | $99–$299 | All archetypes | Medium |
| Business | $399–$999 | Frustrated, Outgrown (high) | High |
| Enterprise | $1,500–$5,000 | Outgrown, Frustrated | Enterprise |

---

## Churn Rate Benchmarks (SaaS Industry)

| Segment | Monthly Churn | Annual Churn | Source |
|---------|--------------|--------------|--------|
| SMB SaaS | 3–5% | 31–46% | Industry average |
| Mid-Market | 1–2% | 12–22% | Industry average |
| Enterprise | 0.5–1% | 6–12% | Industry average |
| **This dataset** | **~1.7%** | **~18–22%** | Synthetic, realistic |
