# Changelog — NBA Churn Retention Engine

All notable changes to this project are documented in this file.
Format: [Semantic Versioning](https://semver.org/)

---

## [1.0.0] — 2024-06-14 — Initial Release

### Added
**Data Layer**
- Synthetic SaaS dataset generator (50K records, realistic distributions)
- Class imbalance (~18-22% churn rate) with configurable noise and outliers
- 27 raw features across engagement, support, billing, and capacity domains

**Feature Engineering**
- 13 derived features: velocity, decay, friction, pricing sensitivity, growth pressure scores
- sklearn-compatible `FeatureEngineer` transformer (fit/transform API)
- Composite health and risk scores for holistic customer assessment
- Log-transformations for heavy-tailed MRR and API usage

**Churn Prediction (Stage 1)**
- 5-model comparison: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- Auto-selection of champion model by ROC-AUC
- Stratified train/test split with cross-validation (5-fold)
- Optional hyperparameter tuning via `RandomizedSearchCV`
- Full metrics export: accuracy, precision, recall, F1, ROC-AUC, PR-AUC

**Explainability (SHAP)**
- `SHAPExplainer` class with TreeExplainer + KernelExplainer fallback
- Global feature importance (mean |SHAP| ranking)
- Individual waterfall plots and force plots
- Automated natural language narrative generation per customer
- Batch explanation export to CSV

**Churn Archetypes**
- 4 archetypes: Ghost (A), Frustrated Professional (B), Price-Sensitive (C), Outgrown (D)
- Archetype affinity scoring (argmax assignment with confidence)
- Archetype distribution reporting

**NBA Engine (Stage 2)**
- YAML-configurable rule engine (`config/nba_rules.yaml`)
- 12 prioritised rules across 4 archetypes
- Condition evaluator supporting ≥, ≤, =, >, < operators
- Fallback action for unmatched conditions
- Revenue impact calculation per customer

**CLV Calculator**
- Discounted CLV over 36-month horizon
- 4 CLV tiers: Low, Medium, High, Enterprise
- CLV-weighted action prioritisation

**Business Impact Engine**
- Campaign ROI simulation with configurable assumptions
- Sensitivity analysis (reach rate, conversion rate, margin)
- By-archetype and by-urgency breakdowns
- Payback period calculation

**Dashboards**
- 6-page Streamlit executive dashboard
- Executive Overview, Churn Prediction, Customer Explorer
- SHAP Explainability, NBA Recommendations, Revenue Impact Simulator
- Plotly interactive visualisations with file upload support

**API**
- FastAPI REST service with Pydantic validation
- `/predict`, `/recommend`, `/batch_predict`, `/archetypes`, `/model_info`
- Swagger UI at `/docs`, ReDoc at `/redoc`
- Health check endpoint

**Monitoring**
- `DriftMonitor` class: PSI + KS-test feature drift detection
- Data quality checks (null inflation, out-of-range values)
- Overall health status: HEALTHY / MONITOR CLOSELY / RETRAIN REQUIRED
- Monitoring report export to CSV + text

**MLOps**
- MLflow experiment tracking integration
- Docker + docker-compose (API, Dashboard, MLflow services)
- GitHub Actions CI/CD (lint → test → build → deploy)
- DVC pipeline configuration for data + model versioning
- Structured logging with file + console handlers

**Testing**
- 40+ test cases across unit, integration, data validation, and monitoring
- pytest configuration with coverage reporting
- `test_model_validation.py`: quality thresholds + stability + feature importance
- `test_monitoring.py`: PSI, KS, drift detection, data quality

**Documentation**
- Enterprise-grade README (5,000+ words)
- Data Dictionary (`docs/data_dictionary.md`)
- Architecture Guide (`docs/architecture_guide.md`)
- Power BI Assets: star schema + DAX measures (`docs/powerbi_assets.md`)
- Portfolio Showcase (`docs/portfolio_showcase.md`)

**Notebooks**
- `01_eda_and_pipeline.ipynb`: Full EDA + pipeline walkthrough
- `02_shap_explainability.ipynb`: SHAP deep dive
- `03_business_impact_simulation.ipynb`: Revenue simulation + sensitivity
- `04_archetype_deep_dive.ipynb`: Per-archetype profiling + visualisations

---

## [Unreleased] — Planned Improvements

### Planned
- Real-time Kafka streaming integration for live customer scoring
- Causal inference / uplift modelling (treatment vs control)
- A/B testing framework for action experiments
- LLM-powered explanation narratives (GPT-4 integration)
- Snowflake / BigQuery connector for production data
- Slack + Salesforce action triggering webhook
- Graph neural network for network-effect churn propagation
- Multi-tenant support with per-tenant model training
- Feast feature store integration
- Evidently AI monitoring dashboard
