"""
Model Validation Tests
=======================
Tests that validate trained model quality, fairness, and stability.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import lightgbm as lgb

from src.data.generate_data import generate_dataset
from src.features.feature_engineering import MODEL_FEATURES, engineer_features


@pytest.fixture(scope="module")
def trained_model_and_data():
    """Train a quick model for validation testing."""
    np.random.seed(42)
    df_raw = generate_dataset(n=2000)
    df_eng = engineer_features(df_raw)
    X = df_eng[MODEL_FEATURES].fillna(0).values
    y = df_eng["churned"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = lgb.LGBMClassifier(
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
    )
    model.fit(X_train_s, y_train)

    return {
        "model": model,
        "scaler": scaler,
        "X_train": X_train_s,
        "X_test": X_test_s,
        "y_train": y_train,
        "y_test": y_test,
        "df_eng": df_eng,
    }


class TestModelQuality:
    """Ensure model meets minimum quality thresholds."""

    ROC_AUC_MIN = 0.60
    F1_MIN = 0.40
    PRECISION_MIN = 0.35
    RECALL_MIN = 0.35

    def test_roc_auc_above_threshold(self, trained_model_and_data):
        d = trained_model_and_data
        y_prob = d["model"].predict_proba(d["X_test"])[:, 1]
        auc = roc_auc_score(d["y_test"], y_prob)
        assert auc >= self.ROC_AUC_MIN, f"ROC-AUC {auc:.4f} below minimum {self.ROC_AUC_MIN}"

    def test_f1_above_threshold(self, trained_model_and_data):
        d = trained_model_and_data
        y_pred = d["model"].predict(d["X_test"])
        f1 = f1_score(d["y_test"], y_pred, zero_division=0)
        assert f1 >= self.F1_MIN, f"F1 {f1:.4f} below minimum {self.F1_MIN}"

    def test_precision_above_threshold(self, trained_model_and_data):
        d = trained_model_and_data
        y_pred = d["model"].predict(d["X_test"])
        prec = precision_score(d["y_test"], y_pred, zero_division=0)
        assert prec >= self.PRECISION_MIN, f"Precision {prec:.4f} below minimum"

    def test_recall_above_threshold(self, trained_model_and_data):
        d = trained_model_and_data
        y_pred = d["model"].predict(d["X_test"])
        rec = recall_score(d["y_test"], y_pred, zero_division=0)
        assert rec >= self.RECALL_MIN, f"Recall {rec:.4f} below minimum"

    def test_model_better_than_random(self, trained_model_and_data):
        d = trained_model_and_data
        y_prob = d["model"].predict_proba(d["X_test"])[:, 1]
        auc = roc_auc_score(d["y_test"], y_prob)
        assert auc > 0.55, "Model must be substantially better than random (AUC > 0.55)"

    def test_model_not_overfit(self, trained_model_and_data):
        """Train AUC should not be vastly higher than test AUC."""
        d = trained_model_and_data
        train_prob = d["model"].predict_proba(d["X_train"])[:, 1]
        test_prob = d["model"].predict_proba(d["X_test"])[:, 1]
        train_auc = roc_auc_score(d["y_train"], train_prob)
        test_auc = roc_auc_score(d["y_test"], test_prob)
        overfit_gap = train_auc - test_auc
        assert (
            overfit_gap < 0.20
        ), f"Potential overfitting: train_auc={train_auc:.4f}, test_auc={test_auc:.4f}"


class TestModelStability:
    """Check model output consistency and edge cases."""

    def test_probabilities_sum_to_one(self, trained_model_and_data):
        d = trained_model_and_data
        probs = d["model"].predict_proba(d["X_test"][:10])
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_probabilities_in_unit_interval(self, trained_model_and_data):
        d = trained_model_and_data
        probs = d["model"].predict_proba(d["X_test"])[:, 1]
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_deterministic_predictions(self, trained_model_and_data):
        d = trained_model_and_data
        p1 = d["model"].predict_proba(d["X_test"][:50])
        p2 = d["model"].predict_proba(d["X_test"][:50])
        np.testing.assert_array_equal(p1, p2)

    def test_handles_zero_features(self, trained_model_and_data):
        d = trained_model_and_data
        zero_input = np.zeros((1, len(MODEL_FEATURES)))
        prob = d["model"].predict_proba(zero_input)[:, 1]
        assert 0.0 <= prob[0] <= 1.0, "Should handle all-zero input"

    def test_churn_prediction_skew(self, trained_model_and_data):
        """Predicted churn rate should be in a reasonable range."""
        d = trained_model_and_data
        probs = d["model"].predict_proba(d["X_test"])[:, 1]
        pct_high_risk = (probs >= 0.5).mean()
        assert (
            0.05 <= pct_high_risk <= 0.60
        ), f"Predicted high-risk rate {pct_high_risk:.2%} seems unrealistic"


class TestFeatureImportance:
    """Validate that important features are actually used by the model."""

    def test_feature_importances_sum_to_positive(self, trained_model_and_data):
        model = trained_model_and_data["model"]
        importances = model.feature_importances_
        assert importances.sum() > 0

    def test_no_single_feature_dominates(self, trained_model_and_data):
        """No single feature should have >60% of total importance."""
        model = trained_model_and_data["model"]
        importances = model.feature_importances_
        total = importances.sum()
        if total > 0:
            max_share = importances.max() / total
            assert max_share < 0.60, f"Feature dominance {max_share:.2%} — possible data leakage"

    def test_top_features_are_meaningful(self, trained_model_and_data):
        """Top 5 features should include known important signals."""
        model = trained_model_and_data["model"]
        importances = model.feature_importances_
        top5_idx = np.argsort(importances)[::-1][:5]
        top5_names = [MODEL_FEATURES[i] for i in top5_idx]
        known_important = {
            "engagement_decay_score",
            "retention_risk_score",
            "days_since_last_login",
            "friction_score",
            "pricing_sensitivity_score",
            "support_sentiment",
            "billing_page_visits",
            "ticket_count",
            "velocity_30d_vs_90d",
            "customer_health_score",
        }
        overlap = set(top5_names) & known_important
        assert (
            len(overlap) >= 1
        ), f"Top 5 features {top5_names} have no overlap with known important features"
