"""
Churn Prediction Model Training Pipeline
==========================================
Trains, evaluates, and selects the best churn prediction model
from: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
METRICS_DIR = MODELS_DIR / "metrics"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"


# ── Model definitions ──────────────────────────────────────────────────────────


def get_model_zoo() -> Dict[str, Any]:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=0.5,
            reg_lambda=2.0,
            gamma=0.5,
            scale_pos_weight=5,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=3,
            num_leaves=8,
            learning_rate=0.03,
            min_child_samples=60,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            depth=3,
            learning_rate=0.03,
            l2_leaf_reg=6.0,
            subsample=0.8,
            random_strength=1.0,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
        ),
    }


# ── Hyperparameter search spaces ───────────────────────────────────────────────

PARAM_GRIDS = {
    "XGBoost": {
        "n_estimators": [150, 200, 300],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.01, 0.03, 0.05],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [5, 10, 20],
        "reg_lambda": [1.0, 2.0, 5.0],
    },
    "LightGBM": {
        "n_estimators": [150, 200, 300],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.01, 0.03, 0.05],
        "num_leaves": [4, 8, 15],
        "min_child_samples": [40, 60, 100],
        "reg_lambda": [0.5, 1.0, 2.0],
    },
}


# ── Evaluation helpers ─────────────────────────────────────────────────────────


def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray, model_name: str
) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
    }
    logger.info(
        f"{model_name}: ROC-AUC={metrics['roc_auc']} | F1={metrics['f1']} | PR-AUC={metrics['pr_auc']}"
    )
    return metrics


def cross_validate_model(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        k.replace("test_", "") + "_cv_mean": round(float(np.mean(v)), 4)
        for k, v in results.items()
        if k.startswith("test_")
    }


# ── Main training loop ─────────────────────────────────────────────────────────


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    tune_best: bool = True,
) -> Tuple[Any, pd.DataFrame]:
    """Train all models, return (best_model, metrics_df)."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    trained_models = {}

    for name, model in get_model_zoo().items():
        logger.info(f"Training {name} …")
        model.fit(X_train, y_train)
        trained_models[name] = model

        test_metrics = evaluate_model(model, X_test, y_test, name)
        cv_metrics = cross_validate_model(model, X_train, y_train)
        combined = {**test_metrics, **cv_metrics}
        all_metrics.append(combined)

        # Save individual model
        joblib.dump(model, ARTIFACTS_DIR / f"{name.lower()}_model.pkl")

    metrics_df = pd.DataFrame(all_metrics).set_index("model")
    metrics_df.to_csv(METRICS_DIR / "model_comparison.csv")

    # ── Select best model by ROC-AUC ──────────────────────────────────
    best_name = metrics_df["roc_auc"].idxmax()
    best_model = trained_models[best_name]
    logger.info(f"Best model: {best_name} (ROC-AUC={metrics_df.loc[best_name, 'roc_auc']})")

    # ── Optional hyperparameter tuning of best model ───────────────────
    if tune_best and best_name in PARAM_GRIDS:
        logger.info(f"Hyperparameter tuning {best_name} …")
        search = RandomizedSearchCV(
            get_model_zoo()[best_name],
            PARAM_GRIDS[best_name],
            n_iter=20,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        tuned_metrics = evaluate_model(best_model, X_test, y_test, f"{best_name}_tuned")
        logger.info(f"Tuned {best_name}: {tuned_metrics}")

    # ── Save champion model ────────────────────────────────────────────
    joblib.dump(best_model, ARTIFACTS_DIR / "champion_model.pkl")
    meta = {
        "champion_model": best_name,
        "feature_names": feature_names,
        "roc_auc": float(metrics_df.loc[best_name, "roc_auc"]),
        "f1": float(metrics_df.loc[best_name, "f1"]),
    }
    with open(ARTIFACTS_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("All models saved.")
    return best_model, metrics_df


def load_champion_model(path: str = "models/artifacts/champion_model.pkl"):
    return joblib.load(path)


def generate_model_report(metrics_df: pd.DataFrame) -> str:
    """Return a Markdown-formatted model comparison report."""
    lines = [
        "# Model Comparison Report\n",
        metrics_df.to_markdown(),
        "\n\n## Champion Model\n",
        f"**{metrics_df['roc_auc'].idxmax()}** selected based on highest ROC-AUC.\n",
    ]
    report = "\n".join(lines)
    report_path = Path("models/reports/model_comparison.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    return report


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from sklearn.model_selection import train_test_split

    from src.data.generate_data import generate_dataset
    from src.features.feature_engineering import MODEL_FEATURES, engineer_features

    logging.basicConfig(level=logging.INFO)

    df = generate_dataset(n=10000)
    df_eng = engineer_features(df)
    X = df_eng[MODEL_FEATURES].fillna(0).values
    y = df_eng["churned"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    best, metrics = train_all_models(
        X_train_s, y_train, X_test_s, y_test, feature_names=MODEL_FEATURES, tune_best=False
    )
    print(metrics)
