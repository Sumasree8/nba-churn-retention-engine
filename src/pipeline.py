"""
Master Pipeline Orchestrator
==============================
End-to-end pipeline: data → features → churn model → SHAP → archetypes → NBA → impact.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.business_impact.impact_engine import print_impact_report, simulate_business_impact
from src.clv.clv_calculator import add_clv_columns

# Internal modules
from src.data.db_loader import load_from_db
from src.data.generate_data import generate_dataset
from src.explainability.shap_engine import SHAPExplainer
from src.features.feature_engineering import MODEL_FEATURES, engineer_features
from src.models.train import load_champion_model, train_all_models
from src.nba_engine.archetype_classifier import classify_archetypes
from src.nba_engine.nba_engine import run_nba_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger("pipeline")

ARTIFACTS_DIR = Path("models/artifacts")
DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")


def run_full_pipeline(
    n_records: int = 50000,
    generate_new_data: bool = True,
    train_new_model: bool = True,
    tune_hyperparams: bool = False,
    churn_threshold: float = 0.50,
    data_source: str = "synthetic",
):
    logger.info("=" * 70)
    logger.info("  NBA CHURN RETENTION ENGINE — Full Pipeline")
    logger.info("=" * 70)

    # ── STAGE 0: Data Acquisition ──────────────────────────────────────
    raw_data_path = DATA_DIR / "raw" / "saas_churn_dataset.csv"
    if data_source == "db":
        logger.info("[Stage 0] Loading data from production database …")
        df_raw = load_from_db(
            require_label=train_new_model,
            output_path=str(raw_data_path),
        )
    elif generate_new_data or not raw_data_path.exists():
        logger.info(f"[Stage 0] Generating {n_records:,} synthetic records …")
        df_raw = generate_dataset(n=n_records, output_path=str(raw_data_path))
    else:
        logger.info(f"[Stage 0] Loading existing dataset from {raw_data_path}")
        df_raw = pd.read_csv(raw_data_path)
    logger.info(f"Raw data shape: {df_raw.shape}")

    # ── STAGE 1: Feature Engineering ──────────────────────────────────
    logger.info("[Stage 1] Feature engineering …")
    df_eng = engineer_features(df_raw)
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
    df_eng.to_csv(DATA_DIR / "processed" / "engineered_features.csv", index=False)
    logger.info(f"Engineered data shape: {df_eng.shape}")

    # ── STAGE 2: Model Training ────────────────────────────────────────
    X = df_eng[MODEL_FEATURES].fillna(0).values
    y = df_eng["churned"].values
    customer_ids = df_eng["customer_id"].tolist()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(X)), test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")

    if train_new_model or not (ARTIFACTS_DIR / "champion_model.pkl").exists():
        logger.info("[Stage 2] Training churn prediction models …")
        best_model, metrics_df = train_all_models(
            X_train_s,
            y_train,
            X_test_s,
            y_test,
            feature_names=MODEL_FEATURES,
            tune_best=tune_hyperparams,
        )
        logger.info(f"\n{metrics_df.to_string()}")
    else:
        logger.info("[Stage 2] Loading existing champion model …")
        best_model = load_champion_model()

    # ── STAGE 3: Churn Scoring ─────────────────────────────────────────
    logger.info("[Stage 3] Scoring all customers …")
    X_all_s = scaler.transform(X)
    churn_probs = best_model.predict_proba(X_all_s)[:, 1]
    df_eng["churn_probability"] = churn_probs

    # ── STAGE 4: SHAP Explanations ─────────────────────────────────────
    logger.info("[Stage 4] Running SHAP explanations …")
    shap_explainer = SHAPExplainer(best_model, feature_names=MODEL_FEATURES)
    # Use training sample as background
    shap_explainer.fit(X_train_s[:2000])
    high_risk_mask = churn_probs >= churn_threshold
    X_high_risk = X_all_s[high_risk_mask]
    hr_customer_ids = [customer_ids[i] for i in np.where(high_risk_mask)[0]]
    logger.info(f"High-risk customers: {high_risk_mask.sum():,} ({high_risk_mask.mean():.1%})")

    shap_explainer.explain(X_high_risk)
    shap_df = shap_explainer.explain_batch(
        hr_customer_ids, X_high_risk, churn_probs[high_risk_mask], churn_threshold=0.0
    )
    (REPORTS_DIR / "shap").mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(REPORTS_DIR / "shap" / "shap_explanations.csv", index=False)

    # Save global importance
    shap_explainer.save_global_importance(REPORTS_DIR / "shap" / "global_importance.csv")

    # ── Build full customer output table ───────────────────────────────
    df_high_risk = df_eng[high_risk_mask].copy().reset_index(drop=True)
    df_high_risk = df_high_risk.merge(
        shap_df[["customer_id", "primary_driver"]], on="customer_id", how="left"
    )

    # ── STAGE 5: Archetype Classification ─────────────────────────────
    logger.info("[Stage 5] Classifying churn archetypes …")
    df_high_risk = classify_archetypes(df_high_risk)

    # ── STAGE 6: CLV Calculation ───────────────────────────────────────
    logger.info("[Stage 6] Computing CLV …")
    df_high_risk = add_clv_columns(df_high_risk)

    # ── STAGE 7: NBA Recommendations ──────────────────────────────────
    logger.info("[Stage 7] Generating NBA recommendations …")
    df_nba = run_nba_pipeline(df_high_risk)

    # ── STAGE 8: Business Impact ───────────────────────────────────────
    logger.info("[Stage 8] Calculating business impact …")
    impact_report, df_nba = simulate_business_impact(df_nba)
    print_impact_report(impact_report)

    # ── Save outputs ───────────────────────────────────────────────────
    output_cols = [
        "customer_id",
        "subscription_type",
        "monthly_revenue",
        "tenure_months",
        "churn_probability",
        "primary_driver",
        "churn_archetype",
        "archetype_name",
        "action",
        "action_code",
        "channel",
        "urgency",
        "expected_conversion_rate",
        "action_cost_usd",
        "expected_revenue_saved",
        "campaign_roi",
        "clv",
        "clv_tier",
        "customer_health_score",
        "retention_risk_score",
    ]
    available_cols = [c for c in output_cols if c in df_nba.columns]
    df_final = df_nba[available_cols].sort_values("churn_probability", ascending=False)

    (DATA_DIR / "sample_outputs").mkdir(parents=True, exist_ok=True)
    df_final.to_csv(DATA_DIR / "sample_outputs" / "nba_recommendations.csv", index=False)
    logger.info(f"Final output saved: {len(df_final):,} rows")

    # ── Archetype distribution report ──────────────────────────────────
    arch_dist = df_nba["churn_archetype"].value_counts().reset_index()
    arch_dist.columns = ["archetype", "count"]
    arch_dist.to_csv(REPORTS_DIR / "nba" / "archetype_distribution.csv", index=False)

    logger.info("=" * 70)
    logger.info("  Pipeline complete!")
    logger.info("=" * 70)

    return df_final, impact_report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--no-retrain", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument(
        "--source",
        choices=["synthetic", "db"],
        default="synthetic",
        help="Where to read raw data from: synthetic generator or production database.",
    )
    args = parser.parse_args()

    run_full_pipeline(
        n_records=args.n,
        train_new_model=not args.no_retrain,
        tune_hyperparams=args.tune,
        data_source=args.source,
    )
