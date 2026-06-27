<div align="center">

# 🎯 NBA Churn Retention Engine

### Predict churn → explain *why* → prescribe the next-best action → quantify the ₹ impact.

**An end-to-end, production-grade ML platform that turns raw SaaS usage data into revenue-saving retention decisions — not just risk scores.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.44-purple?style=flat-square)](https://shap.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)](https://docker.com)
[![Tests](https://img.shields.io/badge/tests-78%20passing-brightgreen?style=flat-square)](#testing)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**[🚀 Live Demo](https://your-app.streamlit.app) · [📊 Dashboard Tour](#-dashboard-guide) · [🔌 API](#-api-reference) · [🏗 Architecture](#-system-architecture)**

</div>

---

## ⚡ At a Glance

| 🎯 Model Quality | 💰 Revenue Saved / yr | 📈 Retention ROI | 🧠 Explainability | ⚙️ Stack |
|:---:|:---:|:---:|:---:|:---:|
| **0.93 ROC-AUC** | **₹28+ Cr** | **~19×** | **Per-customer SHAP** | XGBoost · FastAPI · Streamlit · Docker |

> Feed in one customer record → get the **churn probability**, the **behavioral archetype**, the **exact retention play to run**, and the **rupee value** of running it. This is the complete **detect → explain → segment → act → measure** loop that most churn projects stop one step into.

---

## 📸 Dashboard Preview

A 6-page executive dashboard built in Streamlit — dark, board-room ready, all figures in **₹**.

| 🏠 Executive Overview | 🔮 Churn Prediction | 🎯 NBA Recommendations |
|:---:|:---:|:---:|
| KPI band · archetype mix · risk distribution | Live gauge · risk & CLV card · top drivers | Action priority queue · urgency split |

<!--
📷 To add live screenshots (app is at http://localhost:8501):
   1. Open each page, press Cmd+Shift+4 (macOS) and capture.
   2. Save into docs/screenshots/ with the names below.
   3. Uncomment the image block; GitHub will render them inline.

<p align="center">
  <img src="docs/screenshots/01_executive_overview.png" width="32%" alt="Executive Overview"/>
  <img src="docs/screenshots/02_churn_prediction.png"  width="32%" alt="Churn Prediction"/>
  <img src="docs/screenshots/03_nba_recommendations.png" width="32%" alt="NBA Recommendations"/>
</p>
-->

> ▶️ **Best experienced live:** [open the deployed app](https://your-app.streamlit.app) or run `make dashboard`.

---

## 📋 Table of Contents

1. [Business Problem](#business-problem)
2. [System Architecture](#system-architecture)
3. [Churn Archetypes](#churn-archetypes)
4. [Project Structure](#project-structure)
5. [Quick Start](#quick-start)
6. [Data Dictionary](#data-dictionary)
7. [Feature Engineering](#feature-engineering)
8. [Model Selection & Performance](#model-selection--performance)
9. [Explainable AI (SHAP)](#explainable-ai-shap)
10. [NBA Logic](#nba-logic)
11. [CLV & Business Impact](#clv--business-impact)
12. [Dashboard Guide](#dashboard-guide)
13. [API Reference](#api-reference)
14. [Deployment Guide](#deployment-guide)
15. [Testing](#testing)
16. [ROI Analysis](#roi-analysis)
17. [Future Improvements](#future-improvements)
18. [Portfolio Highlights](#portfolio-highlights)

---

## 🏢 Business Problem

Most churn analytics projects stop at prediction — generating a list of "at-risk" customers but providing no decision support for what to do next.

This project solves the **complete churn retention loop**:

| Stage | Traditional Approach | This System |
|-------|---------------------|-------------|
| Detection | ✅ Churn probability score | ✅ Churn probability score |
| Explanation | ❌ Black box | ✅ SHAP-driven natural language explanation |
| Segmentation | ❌ Generic "at-risk" bucket | ✅ 4 behavioral archetypes |
| Action | ❌ Mass email blast | ✅ Personalised Next-Best-Action |
| Impact | ❌ No measurement | ✅ Revenue saved + ROI simulation |

**Business value:** A SaaS company with **₹40 Cr ARR** and 18% annual churn — bleeding **~₹7.5 Cr/year** — can retain **₹1.6–3.3 Cr+ annually** with targeted interventions, achieving **10–40× ROI** on retention spend.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: CHURN PREDICTION                    │
│                                                                  │
│  Raw SaaS Data → Feature Engineering → Model Ensemble           │
│                      ↓                      ↓                   │
│              50+ Features           XGBoost / LightGBM          │
│              (13 derived)           (champion auto-selected)    │
│                                          ↓                       │
│                              Churn Probability Score [0–1]       │
└──────────────────────────────────────────────────────────────────┘
                                   ↓
                     HIGH RISK CUSTOMERS (P > 0.50)
                                   ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: NBA ENGINE                           │
│                                                                  │
│  SHAP Explanation → Primary Driver → Archetype Classification    │
│        ↓                                      ↓                  │
│  "WHY churn?"                      Ghost / Frustrated /          │
│                                    Price-Sensitive / Outgrown    │
│                                             ↓                    │
│                              YAML Rule Engine → NBA Action       │
│                                             ↓                    │
│                              CLV Scoring → Revenue Impact        │
└─────────────────────────────────────────────────────────────────┘
                                   ↓
              OUTPUT: Customer ID | Churn% | Driver | Archetype
                      Action | Channel | Urgency | Revenue Saved
```

---

## 👥 Churn Archetypes

### 👻 Archetype A — The Ghost
**Psychology:** Lost momentum and engagement; quietly drifting away.

| Signal | Feature | Weight |
|--------|---------|--------|
| Inactivity | `engagement_decay_score` | 40% |
| Login drop | `days_since_last_login` | 25% |
| Velocity decline | `velocity_30d_vs_90d` | 20% |
| Frequency change | `login_frequency_change` | 15% |

**Recommended Actions:** Personalized feature discovery email · Recommended content campaign · Re-engagement push notification

---

### 😤 Archetype B — Frustrated Professional
**Psychology:** Product friction and unresolved support issues breeding frustration.

| Signal | Feature | Weight |
|--------|---------|--------|
| Support issues | `ticket_density_30d` | 35% |
| Technical errors | `error_rate` + `failed_api_calls` | 25% |
| UI friction | `rage_click_count` | 20% |
| Sentiment | `support_ticket_sentiment` | 20% |

**Recommended Actions:** Priority CS outreach · Live onboarding session · Engineering escalation

---

### 💸 Archetype C — Price-Sensitive Optimizer
**Psychology:** Actively questioning ROI; exploring downgrade or cancel options.

| Signal | Feature | Weight |
|--------|---------|--------|
| Pricing concern | `billing_page_visits` | 35% |
| Downgrade intent | `downgrade_page_visits` | 35% |
| Invoice review | `invoice_download_frequency` | 20% |
| Trial proximity | `trial_expiry_days` | 10% |

**Recommended Actions:** Personalised discount · Subscription pause · Annual plan incentive

---

### 🚀 Archetype D — Outgrown User
**Psychology:** Hitting plan limits and ready for the next tier but hasn't been prompted.

| Signal | Feature | Weight |
|--------|---------|--------|
| Capacity usage | `tier_capacity_utilization` | 40% |
| API limits | `api_limit_usage` | 25% |
| Seat limits | `seat_limit_usage` | 25% |
| Export frequency | `export_frequency` | 10% |

**Recommended Actions:** Free premium trial · Enterprise consultation · Advanced feature unlock

---

## 📁 Project Structure

```
nba-churn-retention-engine/
│
├── 📂 data/
│   ├── raw/                    # Raw synthetic SaaS dataset (50K records)
│   ├── processed/              # Feature-engineered dataset
│   └── sample_outputs/         # NBA recommendation output table
│
├── 📂 src/
│   ├── data/
│   │   └── generate_data.py    # Synthetic data generator
│   ├── features/
│   │   └── feature_engineering.py  # Feature pipeline (13 derived features)
│   ├── models/
│   │   └── train.py            # 5-model training + auto-selection
│   ├── explainability/
│   │   └── shap_engine.py      # SHAP global + individual explanations
│   ├── nba_engine/
│   │   ├── archetype_classifier.py  # 4-archetype classification
│   │   └── nba_engine.py       # YAML rule engine + revenue calculator
│   ├── clv/
│   │   └── clv_calculator.py   # Discounted CLV calculator
│   ├── business_impact/
│   │   └── impact_engine.py    # ROI simulation engine
│   └── pipeline.py             # Master end-to-end orchestrator
│
├── 📂 api/
│   └── main.py                 # FastAPI REST API (predict/recommend/explain)
│
├── 📂 dashboards/
│   └── streamlit/
│       └── app.py              # 6-page Streamlit executive dashboard
│
├── 📂 models/
│   ├── artifacts/              # Saved model + scaler + meta
│   ├── metrics/                # Model comparison CSV
│   └── reports/                # Model comparison Markdown
│
├── 📂 reports/
│   ├── shap/                   # Global importance + individual explanations
│   └── nba/                    # Archetype distribution + action reports
│
├── 📂 tests/
│   ├── unit/                   # Unit tests (features, CLV, NBA, archetypes)
│   └── data_validation/        # Schema + distribution tests
│
├── 📂 deployment/
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── github_actions/
│       └── ci_cd.yml           # Full CI/CD pipeline
│
├── 📂 config/
│   ├── config.yaml             # Master configuration
│   └── nba_rules.yaml          # Configurable NBA rules
│
├── 📂 docs/
│   └── powerbi_assets.md       # Star schema + DAX measures
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourname/nba-churn-retention-engine.git
cd nba-churn-retention-engine
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
make install            # full ML + API + dashboard stack
# (dashboard only, e.g. for deploy:  make install-dashboard)
```

### 2. Run the Full Pipeline

```bash
make pipeline           # generate data → features → train → SHAP → archetypes → NBA
make pipeline-tune      # with hyperparameter tuning (slower, better model)
make pipeline-db        # read from a real database instead of synthetic data
```

### 3. Launch the Dashboard

```bash
make dashboard
# Open http://localhost:8501
```

### 4. Start the API

```bash
make api
# Swagger UI at http://localhost:8000/docs
```

### 5. Docker (All Services)

```bash
make docker-up
# API: http://localhost:8000  ·  Dashboard: http://localhost:8501  ·  MLflow: http://localhost:5000
```

### 6. Run Tests

```bash
make test       # 78 tests + coverage report
```

### 7. Deploy to Streamlit Cloud (free)

Push to GitHub, then at [share.streamlit.io](https://share.streamlit.io) point to:
**Main file path** → `dashboards/streamlit/app.py`. The slim root `requirements.txt` keeps the cloud build fast.

---

## 📖 Data Dictionary

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `customer_id` | str | Unique customer identifier | CUST-XXXXXX |
| `subscription_type` | cat | Plan tier | Starter/Pro/Business/Enterprise |
| `monthly_revenue` | float | MRR in USD | $29–$5,000 |
| `tenure_months` | int | Months since first subscription | 1–84 |
| `sessions_30d` | int | Product sessions in last 30 days | 0–120 |
| `sessions_90d` | int | Product sessions in last 90 days | 0–300 |
| `days_since_last_login` | int | Recency of last login | 0–90 |
| `days_since_last_core_action` | int | Recency of last value-creating action | 0–90 |
| `login_frequency_change` | float | Month-over-month login frequency change | -1 to +1 |
| `ticket_count` | int | Support tickets in 30 days | 0–20 |
| `support_sentiment` | float | NLP sentiment of support interactions | -1 to +1 |
| `error_rate` | float | API/product error rate | 0–0.3 |
| `failed_api_calls` | int | Failed API calls in 30 days | 0–50 |
| `rage_click_count` | int | Detected rage clicks (UI frustration) | 0–30 |
| `billing_page_visits` | int | Pricing/billing page visits | 0–20 |
| `downgrade_page_visits` | int | Downgrade/cancel page visits | 0–10 |
| `tier_capacity_utilization` | float | % of plan capacity used | 0.05–1.0 |
| `api_limit_usage` | float | % of API quota consumed | 0–1.0 |
| `seat_limit_usage` | float | % of seat limit used | 0.05–1.0 |
| `churned` | int | Binary churn label (target) | 0/1 |

---

## ⚙️ Feature Engineering

13 advanced features are derived from raw signals:

| Feature | Formula | Purpose |
|---------|---------|---------|
| `velocity_30d_vs_90d` | sessions_30d / (sessions_90d/3) | Engagement trend direction |
| `engagement_decay_score` | Weighted: login recency + velocity + frequency change | Ghost detection |
| `friction_score` | Weighted: tickets + errors + rage clicks + sentiment | Frustrated Pro detection |
| `pricing_sensitivity_score` | Weighted: billing visits + downgrade visits + invoice downloads | Price-sensitive detection |
| `growth_pressure_score` | Weighted: capacity + API + seat utilization | Outgrown User detection |
| `customer_health_score` | Inverse composite of engagement + friction + pricing | Overall health |
| `retention_risk_score` | Composite risk aggregation | Overall risk signal |
| `ticket_density_30d` | Raw ticket count (normalised) | Support load signal |
| `support_ticket_sentiment` | Pass-through of NLP sentiment | Friction signal |
| `subscription_tier` | Ordinal encoding of plan tier | Plan context |
| `tenure_bucket` | Bucketed tenure: 0–3m, 3–12m, 12–24m, 24m+ | Lifecycle stage |
| `log_api_usage` | log1p(api_usage) | Stabilise heavy tail |
| `log_monthly_revenue` | log1p(monthly_revenue) | Stabilise heavy tail |

---

## 🤖 Model Selection & Performance

Five models are trained and compared. The champion is auto-selected by **ROC-AUC**.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----|---------|--------|
| Logistic Regression | ~0.82 | ~0.61 | ~0.58 | ~0.59 | ~0.87 | ~0.62 |
| Random Forest | ~0.86 | ~0.71 | ~0.64 | ~0.67 | ~0.91 | ~0.70 |
| **XGBoost** | **~0.88** | **~0.74** | **~0.68** | **~0.71** | **~0.93** | **~0.74** |
| LightGBM | ~0.87 | ~0.73 | ~0.67 | ~0.70 | ~0.93 | ~0.73 |
| CatBoost | ~0.87 | ~0.72 | ~0.66 | ~0.69 | ~0.92 | ~0.72 |

*Metrics are approximate; vary by random seed and dataset generation.*

**Selection logic:** `metrics_df["roc_auc"].idxmax()` — fully automated.

---

## 🧠 Explainable AI (SHAP)

### Global Importance (Top 10 Typical Features)
```
engagement_decay_score      ████████████████  0.142
days_since_last_login       ██████████████    0.121
friction_score              ████████████      0.098
pricing_sensitivity_score   ██████████        0.084
retention_risk_score        █████████         0.076
support_sentiment           ████████          0.068
billing_page_visits         ███████           0.059
ticket_count                ██████            0.051
error_rate                  █████             0.044
velocity_30d_vs_90d         █████             0.041
```

### Individual Customer Explanation Example
```
Customer CUST-007823 has an 87% churn probability.
Top drivers:
  • ticket_density_30d (SHAP=+0.241) — increases churn risk
  • support_sentiment (SHAP=+0.198) — increases churn risk
  • billing_page_visits (SHAP=+0.156) — increases churn risk
  • engagement_decay_score (SHAP=+0.134) — increases churn risk
  • velocity_30d_vs_90d (SHAP=-0.089) — decreases churn risk

Archetype: 😤 Frustrated Professional
Action: Priority Support & CS Outreach
Expected Revenue Saved: $2,868 (12-month horizon)
```

---

## 🎯 NBA Logic

Rules are stored in `config/nba_rules.yaml` for business-user editability.

```yaml
# Example rule
- id: "B1"
  archetype: "B"
  priority: 1
  conditions:
    support_sentiment_lte: -0.2
    ticket_count_gte: 3
  action: "Priority Support & CS Outreach"
  channel: "phone + email"
  urgency: "critical"
  expected_conversion_rate: 0.28
  cost_usd: 25.00
```

**Decision flow:**
1. Filter customers with churn_probability ≥ threshold
2. Compute archetype affinity scores (4 scores per customer)
3. Assign winning archetype (argmax)
4. Evaluate rules by archetype in priority order
5. Return first matching rule → recommended action
6. Calculate expected revenue saved and ROI

---

## 💰 CLV & Business Impact

### CLV Formula
```
CLV = Σ [ MRR × (1 - churn_prob)^t / (1 + monthly_discount)^t ]
      for t in 1..36 months
```

### Customer Tiers
| Tier | CLV Threshold | Typical Profile |
|------|--------------|-----------------|
| Enterprise | ≥ ₹4 L | Large accounts, dedicated CSM |
| High | ≥ ₹1.25 L | Growth accounts |
| Medium | ≥ ₹40 K | Core SMB |
| Low | < ₹40 K | Self-serve / Starter |

*Dashboard renders all monetary values in **INR (₹)** with Indian lakh/crore formatting; the display rate is configurable via `USD_TO_INR`.*

### Impact Formula
```
Revenue Saved = P(saved) × MRR × 12
P(saved) = churn_probability × conversion_rate × reach_rate
Net Profit = Revenue Saved × gross_margin − campaign_cost × overhead
ROI = Net Profit / campaign_cost
```

---

## 📊 Dashboard Guide

Launch: `streamlit run dashboards/streamlit/app.py`

| Page | Purpose | Key Visualizations |
|------|---------|-------------------|
| 🏠 Executive Overview | C-suite summary | KPI cards, archetype bar, CLV tier donut |
| 🔮 Churn Prediction | Real-time scoring | Gauge meter, driver bars |
| 🔍 Customer Explorer | Filterable table | Multi-filter data grid, CSV export |
| 🧠 SHAP Explainability | Model transparency | Global importance, individual narrative |
| 🎯 NBA Recommendations | Action queue | Action distribution, urgency donut |
| 💰 Revenue Simulator | Scenario planning | ROI sliders, sensitivity curve |

---

## 🔌 API Reference

Base URL: `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model status |
| `/predict` | POST | Single customer churn probability |
| `/recommend` | POST | Full NBA recommendation |
| `/batch_predict` | POST | Bulk scoring (up to 1000 customers) |
| `/archetypes` | GET | Archetype definitions |
| `/model_info` | GET | Champion model metadata |
| `/docs` | GET | Swagger UI |

### Example Request
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST-001234",
    "subscription_type": "Professional",
    "monthly_revenue": 199.0,
    "tenure_months": 14,
    "sessions_30d": 4,
    "days_since_last_login": 18,
    "ticket_count": 5,
    "support_sentiment": -0.6,
    "billing_page_visits": 4,
    "downgrade_page_visits": 2,
    "tier_capacity_utilization": 0.55
  }'
```

### Example Response
```json
{
  "customer_id": "CUST-001234",
  "churn_probability": 0.83,
  "churn_archetype": "B",
  "archetype_name": "Frustrated Professional",
  "primary_driver": "friction_score",
  "recommended_action": "Priority Support & CS Outreach",
  "channel": "phone + email",
  "urgency": "critical",
  "expected_conversion_rate": 0.28,
  "expected_revenue_saved": 534.46,
  "clv": 2876.40,
  "clv_tier": "Medium"
}
```

---

## 🚢 Deployment Guide

### Local (Development)
```bash
python src/pipeline.py        # Run full pipeline
uvicorn api.main:app --reload  # API server
streamlit run dashboards/streamlit/app.py  # Dashboard
```

### Docker (Production)
```bash
cd deployment/docker
docker compose up --build -d
```

### Environment Variables
```bash
PYTHONPATH=/app
LOG_LEVEL=INFO
MLFLOW_TRACKING_URI=http://mlflow:5000
```

### CI/CD (GitHub Actions)
The workflow at `deployment/github_actions/ci_cd.yml` runs:
1. Code quality (flake8 + black)
2. Unit + data validation tests with coverage
3. Docker build + push to GHCR
4. Auto-deploy to staging on `main` branch

---

## 🧪 Testing

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Just unit tests
pytest tests/unit/ -v

# Just data validation
pytest tests/data_validation/ -v
```

**Test coverage targets:**
- Feature engineering: >95%
- Archetype classifier: >90%
- CLV calculator: >95%
- NBA engine: >90%
- Data quality: >85%

---

## 📈 ROI Analysis

### Scenario: 50,000 Customer SaaS Platform

| Metric | Value |
|--------|-------|
| Annual Churn Rate | 18% |
| Customers Churning | ~9,000 |
| Average MRR | ₹15,400 |
| Annual Revenue at Risk | **~₹166 Cr** |
| High-Risk Identified (P>50%) | ~9,500 |
| Campaign Reach Rate | 85% |
| Blended Conversion Rate | 19% |
| **Customers Retained** | **~1,530** |
| **Annual Revenue Saved** | **~₹28 Cr** |
| Total Campaign Cost | ~₹1 Cr |
| **Net Incremental Profit** | **~₹19 Cr** |
| **ROI** | **~19×** |

---

## 🔮 Future Improvements

- [ ] **Real-time scoring** via Kafka event stream integration
- [ ] **Multi-touch attribution** for retention campaign measurement
- [ ] **A/B testing framework** for NBA action experiments
- [ ] **LLM-powered explanations** (GPT-4 narrative generation)
- [ ] **Graph neural network** for network-effect churn propagation
- [ ] **Causal inference** uplift modelling (treatment vs control)
- [ ] **Snowflake / BigQuery** connector for production data
- [ ] **Slack/Salesforce integration** for automatic action triggering
- [ ] **Time-series features** (rolling windows, seasonality)
- [ ] **Federated learning** for multi-tenant privacy-preserving training

---

## 🏆 Portfolio Highlights

### Why This Project Demonstrates Senior-Level Analytics Engineering

| Dimension | Evidence |
|-----------|---------|
| **Business Acumen** | Translates churn prediction into actionable revenue impact with ROI simulation |
| **Technical Depth** | 5-model ensemble, SHAP explainability, sklearn pipelines, FastAPI |
| **System Design** | Two-stage pipeline, YAML-configurable rules, MLOps-ready structure |
| **Analytics Maturity** | Goes beyond prediction → explanation → segmentation → prescription |
| **Decision Intelligence** | Rule engine maps ML output to business actions with cost-benefit analysis |
| **Executive Reporting** | 6-page Streamlit dashboard + Power BI star schema + DAX measures |
| **Production Readiness** | Docker, CI/CD, API with Swagger, logging, error handling, tests |
| **Explainability** | SHAP global + waterfall + force plots + natural language narratives |
| **CLV Integration** | Discounted CLV calculator driving action prioritisation |

### Why This Resembles a Real SaaS Retention Platform
- Archetypes mirror real customer success frameworks used at companies like Gainsight, ChurnZero, and Salesforce
- NBA rule engine mirrors the decision logic in enterprise retention tools
- SHAP explanations replicate what modern ML-driven CS platforms surface to account managers
- Revenue impact simulation matches the ROI models used by CS leadership for budget justification
- The YAML-configurable rules allow business users (not just engineers) to update logic without code changes — a real production requirement

---

## 📄 License

MIT License — free to use for portfolio, learning, and commercial projects.

---

*Built with ❤️ as an industry-grade portfolio project demonstrating end-to-end data science, ML engineering, and business analytics at a professional SaaS standard.*
