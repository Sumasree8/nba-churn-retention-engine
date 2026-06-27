"""
MLflow Experiment Tracking Integration
=======================================
Wraps training pipeline with MLflow logging for experiment tracking,
model registry, and artifact storage.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "nba-churn-retention-engine"


def setup_mlflow(tracking_uri: str = "http://localhost:5000"):
    """Configure MLflow tracking server."""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
    except Exception as e:
        logger.warning(f"MLflow setup failed (offline mode): {e}")


def log_training_run(
    model_name: str,
    model,
    metrics: Dict[str, float],
    params: Dict[str, Any] = None,
    feature_names: list = None,
):
    """Log a single model training run to MLflow."""
    try:
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            if params:
                mlflow.log_params(params)
            else:
                try:
                    model_params = model.get_params()
                    mlflow.log_params({k: str(v)[:250] for k, v in model_params.items()})
                except Exception:
                    pass

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            try:
                if "xgboost" in model_name.lower() or "xgb" in type(model).__name__.lower():
                    mlflow.xgboost.log_model(model, artifact_path="model")
                elif "lightgbm" in model_name.lower() or "lgbm" in type(model).__name__.lower():
                    mlflow.lightgbm.log_model(model, artifact_path="model")
                else:
                    mlflow.sklearn.log_model(model, artifact_path="model")
            except Exception as e:
                logger.warning(f"Could not log model artifact: {e}")

            # Log feature names as artifact
            if feature_names:
                features_path = Path("/tmp/feature_names.json")
                features_path.write_text(json.dumps(feature_names))
                mlflow.log_artifact(str(features_path), "features")

            logger.info(f"Logged run for {model_name}: ROC-AUC={metrics.get('roc_auc', 'N/A')}")

    except Exception as e:
        logger.warning(f"MLflow logging skipped (server unavailable): {e}")


def log_pipeline_run(
    champion_model_name: str,
    champion_metrics: Dict[str, float],
    all_metrics_df,
    n_records: int,
    churn_rate: float,
):
    """Log the full pipeline run as a parent MLflow run."""
    try:
        with mlflow.start_run(run_name="full_pipeline"):
            mlflow.log_param("n_records", n_records)
            mlflow.log_param("churn_rate", round(churn_rate, 4))
            mlflow.log_param("champion_model", champion_model_name)
            mlflow.log_metrics(
                {
                    f"champion_{k}": v
                    for k, v in champion_metrics.items()
                    if isinstance(v, (int, float))
                }
            )

            # Log comparison table
            comparison_path = Path("/tmp/model_comparison.csv")
            all_metrics_df.to_csv(comparison_path)
            mlflow.log_artifact(str(comparison_path))

            logger.info("Pipeline run logged to MLflow")

    except Exception as e:
        logger.warning(f"MLflow pipeline logging skipped: {e}")
