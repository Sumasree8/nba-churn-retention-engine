# Power BI Assets — NBA Churn Retention Engine
# ===============================================
# This file documents the star schema, DAX measures, and dashboard layout
# for the Power BI version of the retention analytics platform.

## Star Schema

### Fact Table: fact_churn_predictions
| Column                     | Type     | Description                          |
|---------------------------|----------|--------------------------------------|
| prediction_id (PK)         | INT      | Surrogate key                        |
| customer_id (FK)           | VARCHAR  | Links to dim_customers               |
| archetype_id (FK)          | INT      | Links to dim_archetypes              |
| action_id (FK)             | INT      | Links to dim_actions                 |
| date_id (FK)               | INT      | Links to dim_date                    |
| churn_probability          | DECIMAL  | 0.0–1.0                              |
| churn_risk_level           | VARCHAR  | LOW/MEDIUM/HIGH/CRITICAL             |
| primary_driver             | VARCHAR  | Top SHAP feature                     |
| expected_revenue_saved     | DECIMAL  | USD                                  |
| campaign_roi               | DECIMAL  | Multiple                             |
| clv                        | DECIMAL  | Customer lifetime value USD          |
| expected_conversion_rate   | DECIMAL  | 0.0–1.0                              |
| action_cost_usd            | DECIMAL  | USD                                  |

### Dimension: dim_customers
| Column              | Type     | Description                          |
|--------------------|----------|--------------------------------------|
| customer_id (PK)    | VARCHAR  | Unique customer identifier           |
| subscription_type   | VARCHAR  | Starter/Professional/Business/Enterprise |
| monthly_revenue     | DECIMAL  | MRR in USD                          |
| tenure_months       | INT      | Months since first subscription      |
| country             | VARCHAR  | ISO country code                     |
| industry            | VARCHAR  | Customer industry vertical           |
| clv_tier            | VARCHAR  | Low/Medium/High/Enterprise           |

### Dimension: dim_archetypes
| Column        | Type    | Description                              |
|--------------|---------|------------------------------------------|
| archetype_id  | INT     | 1=Ghost, 2=Frustrated, 3=Price, 4=Outgrown |
| archetype_code| CHAR(1) | A/B/C/D                                  |
| archetype_name| VARCHAR | Human-readable archetype name            |
| psychology    | VARCHAR | Archetype psychology description         |
| color_hex     | VARCHAR | Dashboard color code                     |

### Dimension: dim_actions
| Column                   | Type    | Description                      |
|-------------------------|---------|----------------------------------|
| action_id (PK)           | INT     | Surrogate key                    |
| action_code              | VARCHAR | Unique action identifier         |
| action_name              | VARCHAR | Human-readable action            |
| channel                  | VARCHAR | email/in-app/phone/push          |
| urgency                  | VARCHAR | critical/high/medium/low         |
| expected_conversion_rate | DECIMAL | Historical conversion rate       |
| base_cost_usd            | DECIMAL | Cost per action                  |

### Dimension: dim_date
| Column      | Type | Description        |
|------------|------|--------------------|
| date_id     | INT  | YYYYMMDD format    |
| date        | DATE | Calendar date      |
| year        | INT  | Year               |
| quarter     | INT  | Quarter (1-4)      |
| month       | INT  | Month (1-12)       |
| month_name  | VARCHAR | January–December |
| week        | INT  | ISO week number    |
| day_of_week | VARCHAR | Monday–Sunday    |

---

## DAX Measures

```dax
-- Total Expected Revenue Saved
Total Revenue Saved = SUM(fact_churn_predictions[expected_revenue_saved])

-- Average Churn Probability
Avg Churn Probability = AVERAGE(fact_churn_predictions[churn_probability])

-- High Risk Customer Count
High Risk Customers = 
CALCULATE(
    COUNTROWS(fact_churn_predictions),
    fact_churn_predictions[churn_risk_level] IN {"HIGH", "CRITICAL"}
)

-- Campaign ROI
Campaign ROI = 
DIVIDE(
    SUM(fact_churn_predictions[expected_revenue_saved]) - SUM(fact_churn_predictions[action_cost_usd]),
    SUM(fact_churn_predictions[action_cost_usd]),
    0
)

-- Revenue at Risk (MRR of high-risk customers × 12)
Revenue at Risk = 
CALCULATE(
    SUMX(
        fact_churn_predictions,
        RELATED(dim_customers[monthly_revenue]) * 12
    ),
    fact_churn_predictions[churn_probability] >= 0.5
)

-- Customer Health Index (inverse of avg risk)
Customer Health Index = 
1 - AVERAGE(fact_churn_predictions[churn_probability])

-- Top Archetype by Revenue at Risk
Top Archetype = 
CALCULATE(
    FIRSTNONBLANK(dim_archetypes[archetype_name], 1),
    TOPN(1, SUMMARIZE(fact_churn_predictions, dim_archetypes[archetype_name], "rev", [Revenue at Risk]), [rev], DESC)
)

-- Retention Rate (saved / at-risk)
Predicted Retention Rate = 
DIVIDE(
    SUMX(fact_churn_predictions, fact_churn_predictions[churn_probability] * fact_churn_predictions[expected_conversion_rate]),
    COUNTROWS(fact_churn_predictions),
    0
)

-- CLV Weighted Average
Weighted Avg CLV = 
AVERAGEX(fact_churn_predictions, RELATED(dim_customers[monthly_revenue]) * 18)

-- Critical Actions Count
Critical Actions = 
CALCULATE(
    COUNTROWS(fact_churn_predictions),
    RELATED(dim_actions[urgency]) = "critical"
)
```

---

## Dashboard Layout

### Page 1: Executive Summary
- KPI cards: Total Revenue Saved, High Risk Customers, Avg Churn Prob, ROI
- Donut chart: Customer distribution by Archetype
- Bar chart: Revenue at Risk by Subscription Tier
- Line chart: Churn probability trend over time
- Gauge: Customer Health Index

### Page 2: Customer Health
- Scatter plot: Churn Probability vs CLV (bubble = MRR)
- Matrix: Archetype × CLV Tier heatmap (count)
- Table: Top 20 highest-risk customers with action
- Bar: Top 10 churn drivers (SHAP importance)
- Slicer: Country, Industry, Subscription Type

### Page 3: Retention Impact
- Waterfall: Revenue impact breakdown by campaign type
- Funnel: At-Risk → Contacted → Converted
- Bar: ROI by Archetype
- Table: Action recommendations with cost and expected ROI
- Sensitivity: Revenue saved vs conversion rate assumption
