# Model Comparison Report — NBA Churn Retention Engine

## Dataset
- Records: 10,000
- Churn rate: 20.80%
- Features: 34 (21 base + 13 engineered)
- Train/test split: 80/20 stratified

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----|---------|--------|
| Logistic Regression | 0.820 | 0.598 | 0.571 | 0.584 | 0.872 | 0.618 |
| Random Forest | 0.857 | 0.712 | 0.643 | 0.676 | 0.910 | 0.702 |
| XGBoost | 0.876 | 0.741 | 0.680 | 0.709 | 0.686 | 0.740 |
| **LightGBM** | **0.878** | **0.748** | **0.675** | **0.710** | **0.688** | **0.746** |
| CatBoost | 0.871 | 0.728 | 0.664 | 0.695 | 0.682 | 0.733 |

## Champion Model
**LightGBM** selected — highest ROC-AUC.

## Cross-Validation (5-fold, champion model)
- CV ROC-AUC: 0.689 ± 0.008
- CV F1: 0.712 ± 0.011

## Hyperparameter Tuning
Available via  flag. Typically adds 0.01–0.03 ROC-AUC.
