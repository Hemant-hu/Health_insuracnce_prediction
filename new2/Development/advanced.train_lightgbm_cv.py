from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

# ===============================
# 1. CLASS IMBALANCE RATIO
# ===============================
lgb_weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()

print(f"Positive class weight: {lgb_weight_ratio:.2f}")
print("Training LightGBM-style model with 5-fold CV...\n")

# ===============================
# 2. STRATIFIED K-FOLD
# ===============================
lgb_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_cv_auc_scores = []
lgb_cv_gini_scores = []

# ===============================
# 3. CROSS-VALIDATION
# ===============================
for _lgb_fold_num, (_lgb_train_idx, _lgb_val_idx) in enumerate(lgb_skf.split(X_train, y_train), 1):

    _X_lgb_fold_train = X_train.iloc[_lgb_train_idx]
    _y_lgb_fold_train = y_train.iloc[_lgb_train_idx]
    _X_lgb_fold_val = X_train.iloc[_lgb_val_idx]
    _y_lgb_fold_val = y_train.iloc[_lgb_val_idx]

    # Sample weights (CORRECT imbalance handling)
    _lgb_sample_weights = np.where(
        _y_lgb_fold_train == 1, lgb_weight_ratio, 1.0
    ).astype(float)

    # New model per fold (BEST PRACTICE)
    _lgb_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=200,
        max_depth=7,
        l2_regularization=1.0,
        random_state=42
    )

    # Train
    _lgb_model.fit(
        _X_lgb_fold_train,
        _y_lgb_fold_train,
        sample_weight=_lgb_sample_weights
    )

    # Predict probabilities
    _y_lgb_pred_proba = _lgb_model.predict_proba(_X_lgb_fold_val)[:, 1]

    # Metrics
    _lgb_fold_auc = roc_auc_score(_y_lgb_fold_val, _y_lgb_pred_proba)
    _lgb_fold_gini = 2 * _lgb_fold_auc - 1

    lgb_cv_auc_scores.append(_lgb_fold_auc)
    lgb_cv_gini_scores.append(_lgb_fold_gini)

    print(f"Fold {_lgb_fold_num} → AUC: {_lgb_fold_auc:.4f}, Gini: {_lgb_fold_gini:.4f}")

# ===============================
# 4. FINAL CV RESULTS
# ===============================
print("\n✓ LightGBM-style CV completed")
print(f"Mean Gini: {np.mean(lgb_cv_gini_scores):.6f} ± {np.std(lgb_cv_gini_scores, ddof=1):.6f}")