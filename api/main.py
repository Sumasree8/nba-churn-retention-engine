"""
FastAPI — Churn Prediction & NBA Recommendation API
=====================================================
Endpoints:
  GET  /health
  POST /predict
  POST /recommend
  POST /explain
  POST /batch_predict

Swagger docs at /docs
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Lazy imports — only used if model loaded
try:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.clv.clv_calculator import add_clv_columns
    from src.features.feature_engineering import MODEL_FEATURES, FeatureEngineer
    from src.nba_engine.archetype_classifier import ARCHETYPES, classify_archetypes
    from src.nba_engine.nba_engine import NBAEngine, add_revenue_impact

    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    logging.warning(f"Could not import project modules: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_assets()
    yield


app = FastAPI(
    title="NBA Churn Retention Engine API",
    description="Predict churn, explain drivers, classify archetypes, and recommend next-best-actions.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading ──────────────────────────────────────────────────────────────

MODEL_PATH = Path("models/artifacts/champion_model.pkl")
SCALER_PATH = Path("models/artifacts/scaler.pkl")
META_PATH = Path("models/artifacts/model_meta.json")

_model = None
_scaler = None
_feature_engineer = None
_nba_engine = None


def load_assets():
    global _model, _scaler, _feature_engineer, _nba_engine
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        logger.info("Champion model loaded")
    if SCALER_PATH.exists():
        _scaler = joblib.load(SCALER_PATH)
        logger.info("Scaler loaded")
    if MODULES_LOADED:
        _feature_engineer = FeatureEngineer()
        _nba_engine = NBAEngine()
        logger.info("Feature engineer and NBA engine ready")


# ── Schemas ────────────────────────────────────────────────────────────────────


class CustomerFeatures(BaseModel):
    customer_id: str = "CUST-000001"
    subscription_type: str = "Professional"
    monthly_revenue: float = 199.0
    tenure_months: int = 14
    country: str = "US"
    industry: str = "SaaS"
    sessions_30d: int = 8
    sessions_90d: int = 35
    days_since_last_login: int = 10
    days_since_last_core_action: int = 12
    login_frequency_change: float = -0.3
    session_duration_change: float = -0.15
    ticket_count: int = 4
    support_sentiment: float = -0.4
    error_rate: float = 0.12
    failed_api_calls: int = 5
    rage_click_count: int = 2
    billing_page_visits: int = 3
    downgrade_page_visits: int = 2
    invoice_download_frequency: int = 2
    trial_expiry_days: int = -1
    tier_capacity_utilization: float = 0.55
    export_frequency: int = 4
    api_usage: int = 1200
    api_limit_usage: float = 0.12
    seat_limit_usage: float = 0.40


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_risk_level: str
    model_version: str


class RecommendationResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_archetype: str
    archetype_name: str
    primary_driver: str
    recommended_action: str
    channel: str
    urgency: str
    expected_conversion_rate: float
    expected_revenue_saved: float
    clv: float
    clv_tier: str


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]


# ── Helpers ────────────────────────────────────────────────────────────────────


def risk_level(prob: float) -> str:
    if prob >= 0.75:
        return "CRITICAL"
    if prob >= 0.50:
        return "HIGH"
    if prob >= 0.30:
        return "MEDIUM"
    return "LOW"


def customer_to_df(customer: CustomerFeatures) -> pd.DataFrame:
    return pd.DataFrame([customer.model_dump()])


def predict_churn(df: pd.DataFrame) -> np.ndarray:
    if not MODULES_LOADED or _model is None:
        raise HTTPException(503, "Model not loaded. Run the pipeline first.")
    fe = FeatureEngineer()
    fe.fit(df)
    df_eng = fe.transform(df)
    X = df_eng[MODEL_FEATURES].fillna(0).values
    if _scaler:
        X = _scaler.transform(X)
    return _model.predict_proba(X)[:, 1]


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "scaler_loaded": _scaler is not None,
        "modules_loaded": MODULES_LOADED,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    df = customer_to_df(customer)
    probs = predict_churn(df)
    prob = float(probs[0])

    meta = {}
    if META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)

    return PredictionResponse(
        customer_id=customer.customer_id,
        churn_probability=round(prob, 4),
        churn_risk_level=risk_level(prob),
        model_version=meta.get("champion_model", "unknown"),
    )


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(customer: CustomerFeatures):
    df = customer_to_df(customer)
    probs = predict_churn(df)
    prob = float(probs[0])
    df["churn_probability"] = prob

    if MODULES_LOADED:
        fe = FeatureEngineer()
        fe.fit(df)
        df_eng = fe.transform(df)
        df_eng["churn_probability"] = prob
        df_eng = classify_archetypes(df_eng)
        df_eng = add_clv_columns(df_eng)
        df_eng = _nba_engine.recommend_batch(df_eng)
        df_eng = add_revenue_impact(df_eng)
        row = df_eng.iloc[0]

        return RecommendationResponse(
            customer_id=customer.customer_id,
            churn_probability=round(prob, 4),
            churn_archetype=str(row.get("churn_archetype", "A")),
            archetype_name=str(row.get("archetype_name", "Unknown")),
            primary_driver=str(row.get("primary_driver", "engagement_decay_score")),
            recommended_action=str(row.get("action", "General Retention Email")),
            channel=str(row.get("channel", "email")),
            urgency=str(row.get("urgency", "medium")),
            expected_conversion_rate=float(row.get("expected_conversion_rate", 0.1)),
            expected_revenue_saved=float(row.get("expected_revenue_saved", 0.0)),
            clv=float(row.get("clv", 0.0)),
            clv_tier=str(row.get("clv_tier", "Medium")),
        )
    raise HTTPException(503, "Modules not loaded")


@app.post("/batch_predict")
def batch_predict(request: BatchPredictionRequest):
    results = []
    df = pd.DataFrame([c.model_dump() for c in request.customers])
    probs = predict_churn(df)
    for i, customer in enumerate(request.customers):
        prob = float(probs[i])
        results.append(
            {
                "customer_id": customer.customer_id,
                "churn_probability": round(prob, 4),
                "churn_risk_level": risk_level(prob),
            }
        )
    return {"predictions": results, "count": len(results)}


@app.get("/archetypes")
def get_archetypes():
    return ARCHETYPES if MODULES_LOADED else {}


@app.get("/model_info")
def model_info():
    if META_PATH.exists():
        with open(META_PATH) as f:
            return json.load(f)
    return {"status": "model not trained yet"}
