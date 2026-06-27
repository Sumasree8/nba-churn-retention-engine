# Architecture & Deployment Guide

## System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                               │
│                                                                           │
│   SaaS Product DB ──→ ETL/ELT ──→ Raw Features Table                    │
│   (Postgres / Snowflake / BigQuery)                                       │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────┐
│                       FEATURE ENGINEERING LAYER                           │
│                                                                           │
│   src/features/feature_engineering.py                                     │
│   • 13 derived features (decay, friction, pricing, growth)               │
│   • sklearn Pipeline (fit on train, transform on all)                    │
│   • Artifact: models/artifacts/feature_pipeline.pkl                      │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────┐
│                      CHURN PREDICTION LAYER (Stage 1)                     │
│                                                                           │
│   src/models/train.py                                                     │
│   • 5 models trained & compared (LR, RF, XGB, LGBM, CatBoost)           │
│   • Auto-selected champion by ROC-AUC                                    │
│   • Artifacts: champion_model.pkl, scaler.pkl, model_meta.json           │
│   • Tracking: MLflow experiment logs                                      │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │ High-risk customers (P ≥ 0.50)
┌───────────────────────────────▼──────────────────────────────────────────┐
│                       NBA ENGINE LAYER (Stage 2)                          │
│                                                                           │
│   ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│   │  SHAP Engine    │  │ Archetype        │  │ NBA Rule Engine      │  │
│   │                 │  │ Classifier       │  │                      │  │
│   │ shap_engine.py  │  │archetype_        │  │ nba_engine.py        │  │
│   │ • TreeExplainer │  │classifier.py     │  │ • YAML rules         │  │
│   │ • Global SHAP   │  │ • A: Ghost       │  │ • 12 rules × 4       │  │
│   │ • Individual    │  │ • B: Frustrated  │  │   archetypes         │  │
│   │   narratives    │  │ • C: Price-sens  │  │ • Priority ordering  │  │
│   └────────┬────────┘  │ • D: Outgrown   │  │ • Fallback action    │  │
│            │           └────────┬─────────┘  └──────────┬───────────┘  │
│            └────────────────────┼──────────────────────┘              │
└───────────────────────────────┬─┘                                        │
                                │                                          │
┌───────────────────────────────▼──────────────────────────────────────────┐
│                    CLV & BUSINESS IMPACT LAYER                            │
│                                                                           │
│   clv_calculator.py + impact_engine.py                                   │
│   • Discounted CLV (36-month horizon)                                    │
│   • Revenue saved = P(churn) × P(saved) × MRR × 12                     │
│   • ROI = (Revenue×margin − Cost×overhead) / Cost                       │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
┌─────────▼───────┐  ┌──────────▼──────────┐  ┌──────▼──────────────────┐
│  Streamlit      │  │  FastAPI REST API   │  │  Power BI / Tableau     │
│  Dashboard      │  │                     │  │                         │
│  (port 8501)    │  │  (port 8000)        │  │  Star schema + DAX      │
│                 │  │  /predict           │  │  measures               │
│  6 pages        │  │  /recommend         │  │                         │
│  Interactive    │  │  /batch_predict     │  │  3 report pages         │
└─────────────────┘  └─────────────────────┘  └─────────────────────────┘
```

---

## Local Development Setup

### Prerequisites
- Python 3.11+
- Git
- (Optional) Docker Desktop

### Step-by-Step

```bash
# 1. Clone
git clone https://github.com/yourname/nba-churn-retention-engine.git
cd nba-churn-retention-engine

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate.bat     # Windows

# 3. Dependencies
pip install -r requirements.txt

# 4. Generate data (50K records, ~30 seconds)
make data

# 5. Run full pipeline (train + SHAP + NBA + impact)
make pipeline

# 6. Launch everything in separate terminals:
make api           # Terminal 1: API at http://localhost:8000/docs
make dashboard     # Terminal 2: Dashboard at http://localhost:8501
make mlflow        # Terminal 3: MLflow at http://localhost:5000
```

---

## Docker Deployment

### Single Command (All Services)

```bash
make docker-up
```

This starts:
- **API container** on port 8000 with health checks
- **Dashboard container** on port 8501
- **MLflow container** on port 5000 with SQLite backend

### Service Ports

| Service | Port | URL |
|---------|------|-----|
| FastAPI | 8000 | http://localhost:8000/docs |
| Streamlit | 8501 | http://localhost:8501 |
| MLflow UI | 5000 | http://localhost:5000 |

### Production Docker Compose

```yaml
# Override for production — add to docker-compose.override.yml
services:
  api:
    restart: always
    environment:
      - LOG_LEVEL=WARNING
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'
```

---

## CI/CD Pipeline (GitHub Actions)

The pipeline at `.github/workflows/ci_cd.yml` runs on every push:

```
Push to main/develop
    │
    ├── Quality checks (flake8 + black)
    │
    ├── Test suite (pytest + coverage)
    │       └── Unit tests
    │       └── Data validation tests
    │       └── Integration tests
    │
    ├── [main branch only]
    │   ├── Docker build + push to GHCR
    │   └── Deploy to staging
    │
    └── Coverage report (Codecov)
```

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `STAGING_HOST` | Staging server IP/hostname |
| `STAGING_USER` | SSH username |
| `STAGING_SSH_KEY` | Private SSH key |

---

## MLflow Experiment Tracking

Every training run is logged with:
- Model parameters (all hyperparameters)
- Evaluation metrics (accuracy, F1, ROC-AUC, PR-AUC)
- Model artifact (sklearn/XGBoost/LightGBM format)
- Feature names JSON

```python
# View in MLflow UI
mlflow ui --port 5000

# Or in code
import mlflow
runs = mlflow.search_runs(experiment_names=["nba-churn-retention-engine"])
best = runs.sort_values("metrics.roc_auc", ascending=False).iloc[0]
print(f"Best model: {best['params.model']} | AUC: {best['metrics.roc_auc']:.4f}")
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `MLFLOW_EXPERIMENT_NAME` | `nba-churn-retention-engine` | Experiment name |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8000` | FastAPI bind port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `RAW_DATA_PATH` | `data/raw/saas_churn_dataset.csv` | Input data path |

---

## Scaling Considerations

| Concern | Recommendation |
|---------|---------------|
| Data > 1M rows | Use Spark/Dask for feature engineering |
| Real-time scoring | Deploy model to SageMaker/Vertex AI endpoint |
| Daily batch jobs | Orchestrate with Airflow / Prefect / Dagster |
| Multi-tenant | Add `tenant_id` to data + partition models per tier |
| A/B testing | Add `treatment_group` flag to output, track via Amplitude/Mixpanel |
| Model drift | Monitor input distributions with Evidently AI |
| Feature store | Move to Feast / Tecton for reusable features |
