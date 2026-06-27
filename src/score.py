"""
Batch Scoring Script
====================
Score a new CSV of customers through the full NBA pipeline.

Usage:
    python src/score.py --input path/to/customers.csv --output path/to/output.csv
    python src/score.py --input data/raw/saas_churn_dataset.csv --threshold 0.5
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.business_impact.impact_engine import print_impact_report, simulate_business_impact
from src.clv.clv_calculator import add_clv_columns
from src.features.feature_engineering import MODEL_FEATURES, FeatureEngineer
from src.nba_engine.archetype_classifier import classify_archetypes
from src.nba_engine.nba_engine import run_nba_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("models/artifacts")

OUTPUT_COLUMNS = [
    "customer_id",
    "subscription_type",
    "monthly_revenue",
    "tenure_months",
    "churn_probability",
    "churn_risk_level",
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


def risk_level(prob: float) -> str:
    if prob >= 0.75:
        return "CRITICAL"
    if prob >= 0.50:
        return "HIGH"
    if prob >= 0.30:
        return "MEDIUM"
    return "LOW"


def score_customers(
    input_path: str,
    output_path: str = None,
    threshold: float = 0.50,
    model_path: str = None,
    scaler_path: str = None,
) -> pd.DataFrame:
    """Load model + scaler, score a CSV, return NBA recommendations DataFrame."""

    # Load artifacts
    _model_path = model_path or ARTIFACTS_DIR / "champion_model.pkl"
    _scaler_path = scaler_path or ARTIFACTS_DIR / "scaler.pkl"

    if not Path(_model_path).exists():
        raise FileNotFoundError(
            f"Champion model not found at {_model_path}. "
            "Run `python src/pipeline.py` first to train."
        )

    model = joblib.load(_model_path)
    scaler = joblib.load(_scaler_path) if Path(_scaler_path).exists() else None
    logger.info(f"Model loaded: {type(model).__name__}")

    # Load input data
    df_raw = pd.read_csv(input_path)
    logger.info(f"Input: {df_raw.shape[0]:,} customers from {input_path}")

    # Feature engineering
    fe = FeatureEngineer()
    fe.fit(df_raw)
    df_eng = fe.transform(df_raw)

    # Score
    X = df_eng[MODEL_FEATURES].fillna(0).values
    if scaler:
        X = scaler.transform(X)
    churn_probs = model.predict_proba(X)[:, 1]
    df_eng["churn_probability"] = churn_probs
    df_eng["churn_risk_level"] = [risk_level(p) for p in churn_probs]
    df_eng["primary_driver"] = "engagement_decay_score"  # placeholder; run SHAP for real values

    logger.info(
        f"Scored {len(df_eng):,} customers. "
        f"High-risk (≥{threshold}): {(churn_probs >= threshold).sum():,} "
        f"({(churn_probs >= threshold).mean():.1%})"
    )

    # NBA pipeline on high-risk subset
    df_hr = df_eng[df_eng["churn_probability"] >= threshold].copy().reset_index(drop=True)
    if len(df_hr) == 0:
        logger.warning(f"No customers above threshold {threshold}. Lower the threshold.")
        return df_eng

    df_hr = classify_archetypes(df_hr)
    df_hr = add_clv_columns(df_hr)
    df_nba = run_nba_pipeline(df_hr)

    # Business impact
    report, df_impact = simulate_business_impact(df_nba)
    print_impact_report(report)

    # Final output
    available = [c for c in OUTPUT_COLUMNS if c in df_impact.columns]
    df_final = df_impact[available].sort_values("churn_probability", ascending=False)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_path, index=False)
        logger.info(f"Output saved to {output_path}")

    return df_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch NBA churn scoring")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument(
        "--output", default="data/sample_outputs/batch_nba_output.csv", help="Path to output CSV"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Churn probability threshold for NBA (default 0.50)",
    )
    parser.add_argument("--model", help="Override champion model path")
    parser.add_argument("--scaler", help="Override scaler path")
    args = parser.parse_args()

    df_out = score_customers(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        model_path=args.model,
        scaler_path=args.scaler,
    )
    print("\nTop 5 highest-risk customers:")
    preview_cols = [
        "customer_id",
        "churn_probability",
        "churn_risk_level",
        "archetype_name",
        "action",
        "expected_revenue_saved",
    ]
    print(df_out[[c for c in preview_cols if c in df_out.columns]].head(5).to_string(index=False))
