# NBA Churn Retention Engine — Executive Report
*Generated: 2026-06-15 15:58*

---

## 📊 Campaign Overview

| Metric | Value |
|--------|-------|
| High-Risk Customers | **3,050** |
| Average Churn Probability | **66.0%** |
| Expected Revenue Saved (12 mo) | **$1,111,006** |
| Total Campaign Cost | **$3,528** |
| Net Incremental Profit | **$773,470** |
| Campaign ROI | **21921%** |

---

## 👥 Churn Archetype Distribution

| Archetype                   |   Customers | Avg Churn %   | Revenue Saved   | Top Action              |
|:----------------------------|------------:|:--------------|:----------------|:------------------------|
| 👻 The Ghost                 |         939 | 67.7%         | $352,500        | General Retention Email |
| 😤 Frustrated Professional   |         246 | 68.9%         | $148,857        | General Retention Email |
| 💸 Price-Sensitive Optimizer |          29 | 67.3%         | $7,458          | General Retention Email |
| 🚀 Outgrown User             |        1836 | 64.7%         | $602,191        | General Retention Email |

---

## 🚨 Action Urgency Breakdown

| Urgency    |   Count | % of Total   |
|:-----------|--------:|:-------------|
| 🔴 Critical |      66 | 2.2%         |
| 🟠 High     |     276 | 9.0%         |
| 🟢 Low      |    2697 | 88.4%        |
| 🟡 Medium   |      11 | 0.4%         |

---

## 💰 Top 10 Highest-Value Customers to Retain

| customer_id   | subscription_type   |   monthly_revenue |   churn_probability | archetype_name          | action                          |   expected_revenue_saved |
|:--------------|:--------------------|------------------:|--------------------:|:------------------------|:--------------------------------|-------------------------:|
| CUST-008902   | Enterprise          |           4866.51 |            0.904095 | Frustrated Professional | Priority Support & CS Outreach  |                 14783.3  |
| CUST-002034   | Enterprise          |           4300.18 |            0.921933 | Frustrated Professional | Priority Support & CS Outreach  |                 13320.6  |
| CUST-004322   | Enterprise          |           4153.14 |            0.766194 | Frustrated Professional | Priority Support & CS Outreach  |                 10691.9  |
| CUST-001335   | Enterprise          |           3503.93 |            0.743771 | Frustrated Professional | Priority Support & CS Outreach  |                  8756.56 |
| CUST-003077   | Enterprise          |           3010.35 |            0.740163 | Outgrown User           | Free 30-Day Premium Tier Trial  |                  8556.09 |
| CUST-005710   | Enterprise          |           3441.34 |            0.608824 | Outgrown User           | Free 30-Day Premium Tier Trial  |                  8045.46 |
| CUST-004471   | Enterprise          |           4401.65 |            0.861163 | The Ghost               | Re-Engagement Push Notification |                  6822.97 |
| CUST-001024   | Enterprise          |           4757.54 |            0.759109 | The Ghost               | Re-Engagement Push Notification |                  6500.68 |
| CUST-008087   | Enterprise          |           4275.56 |            0.786599 | The Ghost               | Re-Engagement Push Notification |                  6053.67 |
| CUST-009001   | Enterprise          |           4739.44 |            0.709068 | The Ghost               | Re-Engagement Push Notification |                  6049.05 |

---

## 🧠 Top 10 Churn Drivers (SHAP)

| feature                   |   mean_abs_shap |
|:--------------------------|----------------:|
| engagement_decay_score    |           0.142 |
| days_since_last_login     |           0.121 |
| friction_score            |           0.098 |
| pricing_sensitivity_score |           0.084 |
| retention_risk_score      |           0.076 |
| support_sentiment         |           0.068 |
| billing_page_visits       |           0.059 |
| ticket_count              |           0.051 |
| error_rate                |           0.044 |
| velocity_30d_vs_90d       |           0.041 |

---

## 📋 Recommended Actions Summary

| Action                                           |   Customers | Total Rev Saved ($)   |   Avg Conversion |
|:-------------------------------------------------|------------:|:----------------------|-----------------:|
| General Retention Email                          |        2663 | $803,640              |             0.08 |
| Re-Engagement Push Notification                  |         232 | $153,401              |             0.15 |
| Priority Support & CS Outreach                   |          66 | $95,323               |             0.28 |
| Free 30-Day Premium Tier Trial                   |          40 | $45,331               |             0.32 |
| Recommended Content Campaign                     |          27 | $5,323                |             0.09 |
| Advanced Feature Unlock & Upgrade Guide          |           6 | $4,240                |             0.2  |
| Annual Plan Upgrade Incentive                    |           7 | $1,445                |             0.16 |
| Personalised Discount Offer (20-30%)             |           4 | $886                  |             0.25 |
| Product Feedback Survey + Engineering Escalation |           3 | $853                  |             0.14 |
| Personalized Feature Discovery Email             |           2 | $564                  |             0.12 |

---

*Report generated by NBA Churn Retention Engine v1.0.0*