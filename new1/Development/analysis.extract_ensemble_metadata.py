import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("BASE MODEL CONTRIBUTION ANALYSIS")
print("=" * 90)

# ====================================
# STEP 1: EXTRACT META-LEARNER COEFFICIENTS
# ====================================
print("\n1. META-LEARNER COEFFICIENTS (Relative Weights)")
print("-" * 90)

# Note: The ensemble block failed, so we'll need to rebuild the meta-learner
# to extract coefficients. Let's recreate the analysis.

# Use engineered features from the feature engineering block
_X_train_filled = X_train.fillna(X_train.median())

# Initialize base models with same parameters as ensemble block
weight_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

base_model_configs = {
    'LightGBM': HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_depth=8, 
        min_samples_leaf=25, l2_regularization=0.1, max_bins=255,
        random_state=42, early_stopping=False,
        class_weight={0: 1, 1: weight_ratio}
    ),
    'XGBoost': HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, max_depth=6,
        min_samples_leaf=20, l2_regularization=1.0, max_bins=255,
        random_state=42, early_stopping=False,
        class_weight={0: 1, 1: weight_ratio}
    ),
    'CatBoost': HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.03, max_depth=7,
        min_samples_leaf=15, l2_regularization=3.0, max_bins=64,
        random_state=42, early_stopping=False,
        class_weight={0: 1, 1: weight_ratio}
    )
}

# Generate out-of-fold predictions
analysis_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(_X_train_filled), len(base_model_configs)))

print("   Generating OOF predictions for meta-learner training...")
for _idx, (_name, _model) in enumerate(base_model_configs.items()):
    oof_preds[:, _idx] = cross_val_predict(
        _model, _X_train_filled, y_train,
        cv=analysis_skf, method='predict_proba', n_jobs=1
    )[:, 1]
    print(f"   • {_name} OOF predictions generated")

# Train meta-learner
scaler = StandardScaler()
oof_preds_scaled = scaler.fit_transform(oof_preds)

meta_model = LogisticRegression(
    C=1.0, max_iter=1000, random_state=42,
    class_weight={0: 1, 1: weight_ratio}
)
meta_model.fit(oof_preds_scaled, y_train)

# Extract coefficients
meta_coefficients = pd.DataFrame({
    'Base Model': list(base_model_configs.keys()),
    'Coefficient': meta_model.coef_[0],
    'Absolute Weight': np.abs(meta_model.coef_[0])
})
meta_coefficients['Relative Weight (%)'] = (
    meta_coefficients['Absolute Weight'] / meta_coefficients['Absolute Weight'].sum() * 100
)
meta_coefficients = meta_coefficients.sort_values('Absolute Weight', ascending=False)

print("\n   Meta-learner logistic regression coefficients:")
print(meta_coefficients.to_string(index=False))
print(f"\n   Intercept: {meta_model.intercept_[0]:.6f}")

# ====================================
# STEP 2: INDIVIDUAL BASE MODEL CV GINI SCORES
# ====================================
print("\n\n2. INDIVIDUAL BASE MODEL PERFORMANCE (5-Fold CV Gini)")
print("-" * 90)

# We already have these from previous blocks
individual_scores = pd.DataFrame({
    'Base Model': ['LightGBM', 'XGBoost', 'CatBoost'],
    'Mean Gini': [
        np.mean(lgb_cv_gini_scores),
        np.mean(xgb_cv_gini_scores),
        np.mean(cat_cv_gini_scores)
    ],
    'Std Gini': [
        np.std(lgb_cv_gini_scores),
        np.std(xgb_cv_gini_scores),
        np.std(cat_cv_gini_scores)
    ]
})

# Add fold-by-fold scores
for fold_idx in range(5):
    individual_scores[f'Fold {fold_idx+1}'] = [
        lgb_cv_gini_scores[fold_idx],
        xgb_cv_gini_scores[fold_idx],
        cat_cv_gini_scores[fold_idx]
    ]

individual_scores = individual_scores.sort_values('Mean Gini', ascending=False)

print("\n   Individual model cross-validation scores:")
print(individual_scores.to_string(index=False))

# ====================================
# STEP 3: PREDICTION CORRELATION ANALYSIS
# ====================================
print("\n\n3. BASE MODEL PREDICTION CORRELATION (Diversity Analysis)")
print("-" * 90)

# Calculate correlation between OOF predictions
correlation_matrix = pd.DataFrame(
    oof_preds,
    columns=list(base_model_configs.keys())
).corr()

print("\n   Correlation matrix of base model predictions:")
print(correlation_matrix.to_string())

# Calculate average pairwise correlation
_upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)
avg_correlation = _upper_triangle.stack().mean()
print(f"\n   Average pairwise correlation: {avg_correlation:.6f}")
print(f"   Interpretation: {'High diversity (low correlation)' if avg_correlation < 0.8 else 'Low diversity (high correlation)'}")

# ====================================
# STORE RESULTS FOR VISUALIZATION
# ====================================
analysis_meta_coefficients = meta_coefficients
analysis_individual_scores = individual_scores
analysis_correlation_matrix = correlation_matrix
analysis_oof_predictions = oof_preds

print("\n" + "=" * 90)
print("✓ Base model contribution analysis completed")
print("=" * 90)