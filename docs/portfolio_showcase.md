# Portfolio Showcase — NBA Churn Retention Engine

## Why This Project Stands Out

This is not a standard churn prediction notebook. It is a **production-grade, end-to-end retention analytics platform** that demonstrates the complete analytical value chain from raw data to revenue impact.

---

## Skill Dimensions Demonstrated

### 1. Business Acumen
- Translates abstract ML output into dollar-value business decisions
- Implements revenue impact simulation with configurable assumptions
- Designs customer archetypes grounded in real customer success psychology
- Quantifies ROI, payback period, and incremental profit — the metrics C-suites actually care about

**Evidence:** `src/business_impact/impact_engine.py` | `reports/nba/executive_summary.csv`

---

### 2. Data Science & Machine Learning
- **5-model ensemble** with automated champion selection by ROC-AUC
- **Imbalanced class handling** via class weights and stratified sampling
- **Hyperparameter tuning** via RandomizedSearchCV
- **Cross-validation** for unbiased performance estimates
- Proper train/test split discipline with stratification

**Evidence:** `src/models/train.py` | `models/reports/model_comparison.md`

---

### 3. Feature Engineering
- **13 derived features** that encode domain expertise in signals
- Composite scores that capture psychological patterns (decay, friction, pricing, growth)
- Log-transformations for heavy-tailed distributions
- Ordinal encoding, bucketing, velocity calculations
- sklearn-compatible `BaseEstimator` / `TransformerMixin` pipeline

**Evidence:** `src/features/feature_engineering.py`

---

### 4. Explainable AI (XAI)
- **SHAP TreeExplainer** with KernelExplainer fallback for robustness
- Global feature importance (mean |SHAP| ranking)
- Per-customer SHAP waterfall plots with narratives
- Archetype-level SHAP analysis
- **Natural language explanation generation**: "Customer X is at 87% risk because..."

**Evidence:** `src/explainability/shap_engine.py` | `notebooks/02_shap_explainability.ipynb`

---

### 5. Decision Intelligence & Prescriptive Analytics
- Goes beyond prediction to **prescription**: what action should we take?
- YAML-configurable rule engine (business-user editable without code changes)
- Priority-ordered rule matching with fallback logic
- Multi-channel, urgency-tiered action recommendations
- 12 rules across 4 archetypes with realistic conversion rate assumptions

**Evidence:** `config/nba_rules.yaml` | `src/nba_engine/nba_engine.py`

---

### 6. Analytics Engineering
- **Star schema design** for Power BI with fact + dimension tables
- **DAX measures** for KPIs, trend calculations, and cohort analysis
- Data dictionary with full field definitions
- Reproducible data generation with realistic distributions, noise, and outliers

**Evidence:** `docs/powerbi_assets.md` | `docs/data_dictionary.md`

---

### 7. Software Engineering & MLOps
- **FastAPI REST API** with Pydantic validation and Swagger documentation
- **Docker** + **docker-compose** for reproducible multi-service deployment
- **GitHub Actions CI/CD** with quality gates, test coverage, and auto-deploy
- **MLflow** integration for experiment tracking and model registry
- Structured logging, error handling, config management
- 30+ unit, integration, and data validation tests

**Evidence:** `api/main.py` | `deployment/` | `tests/`

---

### 8. Dashboarding & Executive Communication
- **6-page Streamlit dashboard** with Plotly visualisations
- Real-time churn scoring with gauge meter UI
- Revenue impact simulator with interactive sliders
- Customer explorer with multi-filter grid and CSV export
- Professional color theme, KPI cards, and responsive layout

**Evidence:** `dashboards/streamlit/app.py`

---

## How This Compares to Real SaaS Tools

| Feature | This Project | Gainsight | ChurnZero | Totango |
|---------|-------------|-----------|-----------|---------|
| Churn prediction | ✅ ML model | ✅ | ✅ | ✅ |
| Explanation engine | ✅ SHAP | Partial | ❌ | ❌ |
| Behavioral archetypes | ✅ 4 archetypes | Custom | ❌ | ❌ |
| NBA recommendations | ✅ Rule engine | ✅ (Cockpit) | ✅ | ✅ |
| Revenue impact calc | ✅ ROI sim | Partial | Partial | ❌ |
| CLV integration | ✅ Discounted CLV | ✅ | Partial | ❌ |
| BI-ready schema | ✅ Star schema + DAX | ✅ | ❌ | ❌ |
| Open & configurable | ✅ YAML rules | ❌ (closed) | ❌ | ❌ |

---

## Key Technical Decisions & Rationale

### Why YAML for NBA rules?
Business rules change frequently. Hardcoding them in Python requires a developer each time. YAML allows a Customer Success Manager or Product Manager to update rules and thresholds without touching code — a real production requirement.

### Why SHAP over feature importance?
Feature importance from tree models is biased toward high-cardinality features and doesn't show directionality. SHAP is game-theoretically grounded, additive, and supports both global and individual-level explanation — critical for regulatory compliance and customer-facing narratives.

### Why 4 archetypes?
Grounded in real customer success practice. Most enterprise CS teams segment at-risk customers this way: disengaged users, frustrated users, price-concerned users, and capacity-constrained users. Each requires a fundamentally different intervention strategy. More segments increase complexity without proportional business value.

### Why LightGBM as likely champion?
LightGBM handles class imbalance well with `class_weight='balanced'`, is fast on tabular data, produces well-calibrated probabilities, and natively supports SHAP via TreeExplainer. It consistently outperforms Logistic Regression and is competitive with XGBoost at lower training cost.

### Why CLV in action prioritization?
Not all customers are equal. A 70% churn-risk customer paying $29/month should receive a different intervention (and budget) than a 60% churn-risk customer paying $2,000/month. CLV ensures the system allocates expensive interventions (CS calls, discounts) to high-value accounts.

---

## Metrics That Would Matter in Production

If deployed at a real SaaS company, the following would be tracked:

| KPI | Measurement Method |
|-----|--------------------|
| Model AUC stability | Monthly retraining comparison vs baseline |
| Archetype distribution drift | Chi-squared test on monthly archetype mix |
| Intervention conversion rate | A/B test: treated vs holdout group |
| Revenue saved (actual) | Track MRR 6 months post-intervention |
| Campaign ROI (actual) | Actual retained revenue / actual campaign cost |
| Time to first action | % of at-risk customers actioned within 48h |
| False positive cost | Manual CS outreach cost for customers who wouldn't have churned |

---

## Estimated Build Complexity

| Component | Lines of Code | Time (Senior Engineer) |
|-----------|--------------|----------------------|
| Data generation | ~150 | 2h |
| Feature engineering | ~200 | 4h |
| Model training | ~180 | 3h |
| SHAP engine | ~180 | 4h |
| Archetype classifier | ~140 | 3h |
| NBA rule engine | ~160 | 4h |
| CLV calculator | ~80 | 1.5h |
| Business impact engine | ~130 | 2.5h |
| FastAPI | ~200 | 3h |
| Streamlit dashboard | ~400 | 6h |
| Tests (30+ cases) | ~350 | 5h |
| Docker + CI/CD | ~100 | 2h |
| Documentation | ~800 | 6h |
| **Total** | **~3,070** | **~46h** |

This represents approximately **1.5 weeks of focused senior-level work** — a scope appropriate for a significant portfolio piece.
